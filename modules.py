from typing import Optional, Union
import math
import torch
import torch.nn as nn
from transformers import Cache, EncoderDecoderCache, apply_chunking_to_forward
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.roberta.modeling_roberta import RobertaAttention, RobertaEncoder, RobertaLayer

class HapbertaAxialEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([HapbertaAxialLayer(config) for i in range(config.num_hidden_layers)])
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    def forward(
        self,
        hidden_states: torch.Tensor,  # (batch_size, n_haps, n_snps, hidden_size)
        attention_mask: Optional[torch.FloatTensor] = None,
        distances: Optional[torch.FloatTensor] = None,
    ) -> Union[tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:

        for i, layer_module in enumerate(self.layer):
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                distances,
            )

            hidden_states = layer_outputs

        hidden_states = self.layer_norm(hidden_states)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
        )
    

class HapbertaAxialLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        row_attn = RowSelfAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.attention_probs_dropout_prob,
        )
        col_attn = ColumnSelfAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.attention_probs_dropout_prob,
        )
        ff_layer = FeedForwardNetwork(config)

        self.row_attn = NormalizedResidualBlock(row_attn, config)
        self.col_attn = NormalizedResidualBlock(col_attn, config)
        self.ff_layer = NormalizedResidualBlock(ff_layer, config)

    def forward(
        self,
        hidden_states: torch.Tensor,  # (batch_size, n_haps, n_snps, hidden_size)
        attention_mask: Optional[torch.FloatTensor] = None,
        distances = None,
    ) -> tuple[torch.Tensor]:
        batch_size, n_haps, n_snps, hidden_size = hidden_states.shape
        
        # Reshape for axial attention: (n_haps, n_snps, batch_size, hidden_size)
        x = hidden_states.permute(1, 2, 0, 3)
        
        # Create padding mask from attention mask
        self_attn_padding_mask = None
        # if attention_mask is not None:
            # attention_mask: (batch_size, n_haps, n_snps)
            # Convert to padding mask: (batch_size, n_haps, n_snps) where 0 = padded
            # print(attention_mask)
            # self_attn_padding_mask = (attention_mask == 0).permute(1, 2, 0)  # (n_haps, n_snps, batch_size)
        
        # Row attention (over haplotypes for each SNP)
        x, row_attn = self.row_attn(
            x,
            distances,
        )
        
        # Column attention (over SNPs for each haplotype)
        x, col_attn = self.col_attn(
            x,
            distances, 
        )
        
        x = self.ff_layer(x)

        # Reshape back: (batch_size, n_haps, n_snps, hidden_size)
        x = x.permute(2, 0, 1, 3)
        
        return x


# from the msa-transformer repository
class RowSelfAttention(nn.Module):
    """Compute self-attention over rows of a 2D input."""

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        max_tokens_per_msa: int = 2 ** 16,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.max_tokens_per_msa = max_tokens_per_msa
        self.attn_shape = "nrhij"

        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)

        self.dist_bias = RelativePosAttnBias(num_heads)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout_module = nn.Dropout(dropout)

    def align_scaling(self, q):
        num_rows = q.size(0)
        return self.scaling / math.sqrt(num_rows)

    def compute_attention_weights(
        self,
        x,
        distances,
        scaling: float,
        self_attn_mask=None,
        self_attn_padding_mask=None,
    ):
        num_rows, num_cols, batch_size, embed_dim = x.size()
        q = self.q_proj(x).view(
            num_rows, num_cols, batch_size, self.num_heads, self.head_dim
        )
        k = self.k_proj(x).view(
            num_rows, num_cols, batch_size, self.num_heads, self.head_dim
        )
        q *= scaling
        if self_attn_padding_mask is not None:
            # Zero out any padded aligned positions - this is important since
            # we take a sum across the alignment axis.
            # q *= 1 - self_attn_padding_mask.permute(1, 2, 0).unsqueeze(3).unsqueeze(
            #     4
            # ).to(q)
            q *= (~self_attn_padding_mask).unsqueeze(-1).unsqueeze(-1).to(q)

        attn_weights = torch.einsum(f"rinhd,rjnhd->{self.attn_shape}", q, k)

        if self_attn_mask is not None:
            raise NotImplementedError
            # Mask Size: [B x R x C], Weights Size: [H x B x C x C]

        if self_attn_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                self_attn_padding_mask[:, 0].unsqueeze(0).unsqueeze(2),
                -10000,
            )

        # add distance bias
        relative_pos_bias = self.dist_bias(distances)
        attn_weights += relative_pos_bias

        return attn_weights

    def compute_attention_update(
        self,
        x,
        attn_probs,
    ):
        num_rows, num_cols, batch_size, embed_dim = x.size()
        v = self.v_proj(x).view(
            num_rows, num_cols, batch_size, self.num_heads, self.head_dim
        )
        context = torch.einsum(f"{self.attn_shape},rjnhd->rinhd", attn_probs, v)
        context = context.contiguous().view(num_rows, num_cols, batch_size, embed_dim)
        output = self.out_proj(context)
        return output

    def forward(
        self,
        x,
        distances,
        self_attn_mask=None,
        self_attn_padding_mask=None,
    ):
        num_rows, num_cols, batch_size, embed_dim = x.size()
        
        scaling = self.align_scaling(x)
        attn_weights = self.compute_attention_weights(
            x, distances, scaling, self_attn_mask, self_attn_padding_mask
        )
        attn_probs = attn_weights.softmax(-1)
        attn_probs = self.dropout_module(attn_probs)
        output = self.compute_attention_update(x, attn_probs)
        return output, attn_probs

# from the msa-transformer repository
class ColumnSelfAttention(nn.Module):
    """Compute self-attention over columns of a 2D input."""

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        max_tokens_per_msa: int = 2 ** 16,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.max_tokens_per_msa = max_tokens_per_msa

        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout_module = nn.Dropout(dropout)

    def compute_attention_update(
        self,
        x,
        distances,
        self_attn_mask=None,
        self_attn_padding_mask=None,
    ):
        num_rows, num_cols, batch_size, embed_dim = x.size()
        if num_rows == 1:
            # if there is only 1 position, this is equivalent and doesn't break with
            # padding
            attn_probs = torch.ones(
                self.num_heads,
                num_cols,
                batch_size,
                num_rows,
                num_rows,
                device=x.device,
                dtype=x.dtype,
            )
            output = self.out_proj(self.v_proj(x))
        else:
            q = self.q_proj(x).view(
                num_rows, num_cols, batch_size, self.num_heads, self.head_dim
            )
            k = self.k_proj(x).view(
                num_rows, num_cols, batch_size, self.num_heads, self.head_dim
            )
            v = self.v_proj(x).view(
                num_rows, num_cols, batch_size, self.num_heads, self.head_dim
            )
            q *= self.scaling

            attn_weights = torch.einsum("icnhd,jcnhd->hcnij", q, k)

            if self_attn_mask is not None:
                raise NotImplementedError
            if self_attn_padding_mask is not None:
                attn_weights = attn_weights.masked_fill(
                    self_attn_padding_mask.permute(2, 0, 1).unsqueeze(0).unsqueeze(3),
                    -10000,
                )

            attn_probs = attn_weights.softmax(-1)
            attn_probs = self.dropout_module(attn_probs)
            context = torch.einsum("hcnij,jcnhd->icnhd", attn_probs, v)
            context = context.contiguous().view(
                num_rows, num_cols, batch_size, embed_dim
            )
            output = self.out_proj(context)
        return output, attn_probs

    def forward(
        self,
        x,
        distances,
        self_attn_mask=None,
        self_attn_padding_mask=None,
    ):
        num_rows, num_cols, batch_size, embed_dim = x.size()
    
        return self.compute_attention_update(
            x, distances, self_attn_mask, self_attn_padding_mask
        )


class RelativePosAttnBias(nn.Module):
    """T5-style relative position bias for genomic sequences with distance information."""
    
    def __init__(self, num_heads, num_buckets=32, max_distance=50000):
        super().__init__()
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        
        # Embedding table for relative position biases
        self.relative_attention_bias = nn.Embedding(
            self.num_buckets, self.num_heads
        )
    
    def _relative_position_bucket(self, distances):
        """Convert relative positions to bucket indices."""
        abs_dist = distances.abs().float()
        buckets = torch.zeros_like(abs_dist, dtype=torch.long)
        nonzero = abs_dist >= 1
        if nonzero.any():
            log_ratio = torch.log(abs_dist[nonzero]) / (
                torch.log(torch.tensor(self.max_distance, device=abs_dist.device, dtype=abs_dist.dtype))
            )
            log_buckets = (log_ratio * (self.num_buckets - 2)).long() + 1
            log_buckets = torch.clamp(log_buckets, min=1, max=self.num_buckets - 1)
            buckets[nonzero] = log_buckets
        return buckets
    
    def forward(self, distances):
        """
        Args:
            distances: (batch_size, seq_len, seq_len) matrix of distances
        Returns:
            bias: (batch_size, num_heads, seq_len, seq_len) attention bias
        """

        # Convert distances to relative position buckets

        # Get bias values
        if distances.dim() == 3:
            batch_size, seq_len, _ = distances.shape
            relative_buckets = self._relative_position_bucket(distances)
            bias = self.relative_attention_bias(relative_buckets)  # (batch_size, seq_len, seq_len, num_heads)
            # print(bias.shape)
            return bias.permute(0, 3, 1, 2)  # (batch_size, num_heads, seq_len, seq_len)

        if distances.dim() == 4:
            b, r, l, _ = distances.shape
            flat = distances.view(b * r, l, l)
            rel_buckets = self._relative_position_bucket(flat)
            bias = self.relative_attention_bias(rel_buckets)  # (B*R,L,L,H)
            bias = bias.permute(0, 3, 1, 2).view(b, r, self.num_heads, l, l)  # (B,R,H,L,L)
            return bias
        

# other modules from git:facebookresearch/esm
class FeedForwardNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
        
class NormalizedResidualBlock(nn.Module):
    def __init__(self, layer: nn.Module, config):
        super().__init__()
        self.layer = layer

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    
    def forward(self, x, *args, **kwargs):
        residual = x
        x = self.layer_norm(x)
        outputs = self.layer(x, *args, **kwargs)
        if isinstance(outputs, tuple):
            x, *out = outputs
        else:
            x, out = outputs, None
        
        x = self.dropout(x)
        x = residual + x

        return x if out is None else (x,) + tuple(out)
