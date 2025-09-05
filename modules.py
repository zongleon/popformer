from typing import Optional, Union
import math
import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

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
        return_attentions: bool = False,
    ) -> Union[tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:

        if return_attentions:
            row_attns = []
            col_attns = []

        for i, layer_module in enumerate(self.layer):
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                distances,
                return_attentions=return_attentions,
            )

            hidden_states = layer_outputs[0]
            
            if return_attentions:
                row_attns.append(layer_outputs[1])
                col_attns.append(layer_outputs[2])

        hidden_states = self.layer_norm(hidden_states)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            attentions=(row_attns, col_attns) if return_attentions else None
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
        return_attentions = False,
    ) -> tuple[torch.Tensor]:
        batch_size, n_haps, n_snps, hidden_size = hidden_states.shape
        
        # compute attention mask for all heads (roberta impl)
        extended_attention_mask = None
        if attention_mask is not None:
            dtype = hidden_states.dtype
            extended_attention_mask = attention_mask[None, :, :, :].to(dtype=dtype)
            extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min

        # Reshape for axial attention: (n_haps, n_snps, batch_size, hidden_size)
        x = hidden_states.permute(1, 2, 0, 3)
        
        # Row attention (over haplotypes for each SNP)
        x, row_attn = self.row_attn(
            x,
            distances,
            extended_attention_mask
        )
        
        # Column attention (over SNPs for each haplotype)
        x, col_attn = self.col_attn(
            x,
        )
        
        x = self.ff_layer(x)

        # Reshape back: (batch_size, n_haps, n_snps, hidden_size)
        x = x.permute(2, 0, 1, 3)
        
        if return_attentions:
            return x, row_attn, col_attn
        return (x,)


# from the msa-transformer repository
class RowSelfAttention(nn.Module):
    """Compute self-attention over rows of a 2D input."""

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.attn_shape = "hnij"

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
        attn_mask: Optional[torch.FloatTensor] = None
    ):
        num_rows, num_cols, batch_size, embed_dim = x.size()
        q = self.q_proj(x).view(
            num_rows, num_cols, batch_size, self.num_heads, self.head_dim
        )
        k = self.k_proj(x).view(
            num_rows, num_cols, batch_size, self.num_heads, self.head_dim
        )
        q *= scaling
        
        attn_weights = torch.einsum(f"rinhd,rjnhd->{self.attn_shape}", q, k)

        if attn_mask is not None:
            # print(attn_weights.size())
            # print(attn_mask.size())
            attn_weights += attn_mask

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
        attn_mask: Optional[torch.FloatTensor] = None
    ):
        num_rows, num_cols, batch_size, embed_dim = x.size()
        
        scaling = self.align_scaling(x)
        attn_weights = self.compute_attention_weights(
            x, distances, scaling, attn_mask=attn_mask
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
    ):
        super().__init__()

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout_module = nn.Dropout(dropout)

    def compute_attention_update(
        self,
        x,
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
    ):
        num_rows, num_cols, batch_size, embed_dim = x.size()
    
        return self.compute_attention_update(
            x,
        )


class RelativePosAttnBias(nn.Module):
    """T5-style relative position bias for genomic sequences with distance information."""
    
    def __init__(self, num_heads, num_buckets=64, max_distance=50000):
        super().__init__()
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        
        # Embedding table for relative position biases
        self.relative_attention_bias = nn.Embedding(
            self.num_buckets, self.num_heads
        )
        # log_buckets = torch.logspace(
        #     0,
        #     math.log10(self.max_distance),
        #     self.num_buckets - 1
        # )
        # buckets = torch.cat([log_buckets])
        lin_buckets = torch.linspace(
            0,
            self.max_distance,
            self.num_buckets - 1
        )
        buckets = torch.cat([lin_buckets])

        self.register_buffer('buckets', buckets)
    
    
    def _relative_position_bucket(self, distances):
        """Convert relative positions to bucket indices."""
        abs_dist = distances.abs()
        bucket_indices = torch.bucketize(abs_dist, self.buckets, right=False)
        return bucket_indices

    def forward(self, distances):
        """
        Args:
            distances: (batch_size, seq_len, seq_len) matrix of distances
        Returns:
            bias: (batch_size, num_heads, seq_len, seq_len) attention bias
        """

        # Convert distances to relative position buckets

        # Get bias values
        batch_size, seq_len, _ = distances.shape
        relative_buckets = self._relative_position_bucket(distances)
        bias = self.relative_attention_bias(relative_buckets)  # (batch_size, seq_len, seq_len, num_heads)
        # print(bias.shape)
        return bias.permute(3, 0, 1, 2)  # (num_heads, batch_size, seq_len, seq_len)


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
