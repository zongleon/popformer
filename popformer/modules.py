from typing import Optional, Union
import math
import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

# want to add typing
from jaxtyping import Int, Float, Bool
from . import typed


class PopformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList(
            [AxialAttentionLayer(config) for i in range(config.num_hidden_layers)]
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    @typed
    def forward(
        self,
        hidden_states: Float[torch.Tensor, "batch n_haps n_snps hidden"],
        attention_mask: Bool[torch.Tensor, "batch n_haps n_snps"] | None = None,
        distances: Int[torch.Tensor, "batch n_snps n_snps"] | None = None,
        return_attentions: bool = False,
    ) -> Union[BaseModelOutputWithPastAndCrossAttentions,]:

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
            attentions=(row_attns, col_attns) if return_attentions else None,
        )


class AxialAttentionLayer(nn.Module):
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

    @staticmethod
    @typed
    def _build_axial_attention_masks(
        token_mask: Bool[torch.Tensor, "batch n_haps n_snps"],
        dtype: torch.dtype,
    ) -> tuple[
        Float[torch.Tensor, "1 batch 1 n_snps"],
        Float[torch.Tensor, "1 1 batch 1 n_haps"],
        Float[torch.Tensor, "n_haps 1 batch 1"],
    ]:
        # For each sample, use one non-padded row for row attention and one
        # non-padded column for column attention. This matches the collator
        # contract where non-padded rows/cols share the same mask pattern.
        batch_size = token_mask.size(0)
        batch_idx = torch.arange(batch_size, device=token_mask.device)

        valid_rows = token_mask.any(dim=2)
        first_valid_row = valid_rows.float().argmax(dim=1)
        row_mask = token_mask[batch_idx, first_valid_row, :]
        row_mask = row_mask & valid_rows.any(dim=1, keepdim=True)

        valid_cols = token_mask.any(dim=1)
        first_valid_col = valid_cols.float().argmax(dim=1)
        col_mask = token_mask[batch_idx, :, first_valid_col]
        col_mask = col_mask & valid_cols.any(dim=1, keepdim=True)

        neg_inf = torch.finfo(dtype).min
        snp_attn_mask = (~row_mask).to(dtype=dtype)[None, :, None, :] * neg_inf
        hap_attn_mask = (~col_mask).to(dtype=dtype)[None, None, :, None, :] * neg_inf
        hap_query_mask = col_mask.transpose(0, 1)[:, None, :, None].to(dtype=dtype)
        return snp_attn_mask, hap_attn_mask, hap_query_mask

    @typed
    def forward(
        self,
        hidden_states: Float[torch.Tensor, "batch n_haps n_snps hidden"],
        attention_mask: Bool[torch.Tensor, "batch n_haps n_snps"] | None = None,
        distances: Int[torch.Tensor, "batch n_snps n_snps"] | None = None,
        return_attentions: bool = False,
    ) -> Union[
        tuple[Float[torch.Tensor, "batch n_haps n_snps hidden"]],
        tuple[
            Float[torch.Tensor, "batch n_haps n_snps hidden"],
            Float[torch.Tensor, "num_heads batch n_snps n_snps"],
            Float[torch.Tensor, "num_heads n_snps batch n_haps n_haps"],
        ],
    ]:
        token_mask = None
        snp_attn_mask = None
        hap_attn_mask = None
        hap_query_mask = None
        if attention_mask is not None:
            dtype = hidden_states.dtype
            token_mask = attention_mask.bool()
            snp_attn_mask, hap_attn_mask, hap_query_mask = (
                self._build_axial_attention_masks(token_mask, dtype)
            )

        # Reshape for axial attention: (n_haps, n_snps, batch_size, hidden_size)
        x = hidden_states.permute(1, 2, 0, 3)

        # Row attention (over haplotypes for each SNP)
        x, row_attn = self.row_attn(
            x,
            distances,
            attn_mask=snp_attn_mask,
            row_mask=hap_query_mask,
        )

        # Column attention (over SNPs for each haplotype)
        x, col_attn = self.col_attn(
            x,
            attn_mask=hap_attn_mask,
            query_mask=hap_query_mask,
        )

        x = self.ff_layer(x)

        if token_mask is not None:
            x = x * token_mask.permute(1, 2, 0).unsqueeze(-1).to(x.dtype)

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
        self.scaling = self.head_dim**-0.5
        self.attn_shape = "hnij"

        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)

        self.dist_bias = RelativePosAttnBias(num_heads)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout_module = nn.Dropout(dropout)

    @typed
    def align_scaling(
        self, q: Float[torch.Tensor, "n_rows n_cols batch embed"]
    ) -> float:
        num_rows = q.size(0)
        return self.scaling / math.sqrt(num_rows)

    @typed
    def compute_attention_weights(
        self,
        x: Float[torch.Tensor, "n_rows n_cols batch embed"],
        distances: Int[torch.Tensor, "batch n_snps n_snps"],
        scaling: float,
        attn_mask: Optional[Float[torch.Tensor, "1 batch 1 n_snps"]] = None,
        row_mask: Optional[Float[torch.Tensor, "n_haps 1 batch 1"]] = None,
    ) -> Float[torch.Tensor, "num_heads batch n_snps n_snps"]:
        num_rows, num_cols, batch_size, embed_dim = x.size()
        q = self.q_proj(x).view(
            num_rows, num_cols, batch_size, self.num_heads, self.head_dim
        )
        k = self.k_proj(x).view(
            num_rows, num_cols, batch_size, self.num_heads, self.head_dim
        )
        if row_mask is not None:
            mask = row_mask.to(q.dtype).unsqueeze(-1)
            q = q * mask
            k = k * mask
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

    @typed
    def compute_attention_update(
        self,
        x: Float[torch.Tensor, "n_haps n_snps batch hidden"],
        attn_probs: Float[torch.Tensor, "num_heads batch n_snps n_snps"],
        row_mask: Optional[Float[torch.Tensor, "n_haps 1 batch 1"]] = None,
    ) -> Float[torch.Tensor, "n_haps n_snps batch hidden"]:
        num_rows, num_cols, batch_size, embed_dim = x.size()
        v = self.v_proj(x).view(
            num_rows, num_cols, batch_size, self.num_heads, self.head_dim
        )
        if row_mask is not None:
            v = v * row_mask.to(v.dtype).unsqueeze(-1)
        context = torch.einsum(f"{self.attn_shape},rjnhd->rinhd", attn_probs, v)
        context = context.contiguous().view(num_rows, num_cols, batch_size, embed_dim)
        output = self.out_proj(context)
        if row_mask is not None:
            output = output * row_mask.to(output.dtype)
        return output

    @typed
    def forward(
        self,
        x: Float[torch.Tensor, "n_haps n_snps batch hidden"],
        distances: Int[torch.Tensor, "batch n_snps n_snps"],
        attn_mask: Optional[Float[torch.Tensor, "1 batch 1 n_snps"]] = None,
        row_mask: Optional[Float[torch.Tensor, "n_haps 1 batch 1"]] = None,
    ) -> tuple[
        Float[torch.Tensor, "n_haps n_snps batch hidden"],
        Float[torch.Tensor, "num_heads batch n_snps n_snps"],
    ]:
        scaling = self.align_scaling(x)
        attn_weights = self.compute_attention_weights(
            x, distances, scaling, attn_mask=attn_mask, row_mask=row_mask
        )
        attn_probs = attn_weights.softmax(-1)
        attn_probs = self.dropout_module(attn_probs)
        output = self.compute_attention_update(x, attn_probs, row_mask=row_mask)
        return output, attn_probs


# from the msa-transformer repository
# adjusted to remove attention probes since we don't have a task like that
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
        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout_module = nn.Dropout(dropout)

    @typed
    def compute_attention_update(
        self,
        x: Float[torch.Tensor, "n_haps n_snps batch hidden"],
        attn_mask: Optional[Float[torch.Tensor, "1 1 batch 1 n_haps"]] = None,
        query_mask: Optional[Float[torch.Tensor, "n_haps 1 batch 1"]] = None,
    ) -> tuple[
        Float[torch.Tensor, "n_haps n_snps batch hidden"],
        Float[torch.Tensor, "num_heads n_snps batch n_haps n_haps"],
    ]:
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
            if query_mask is not None:
                mask = query_mask.to(q.dtype).unsqueeze(-1)
                q = q * mask
                k = k * mask
                v = v * mask
            q *= self.scaling

            attn_weights = torch.einsum("icnhd,jcnhd->hcnij", q, k)
            if attn_mask is not None:
                attn_weights = attn_weights + attn_mask

            attn_probs = attn_weights.softmax(-1)
            attn_probs = self.dropout_module(attn_probs)
            context = torch.einsum("hcnij,jcnhd->icnhd", attn_probs, v)
            context = context.contiguous().view(
                num_rows, num_cols, batch_size, embed_dim
            )
            output = self.out_proj(context)
        if query_mask is not None:
            output = output * query_mask.to(output.dtype)
        return output, attn_probs

    @typed
    def forward(
        self,
        x: Float[torch.Tensor, "n_haps n_snps batch hidden"],
        attn_mask: Optional[Float[torch.Tensor, "1 1 batch 1 n_haps"]] = None,
        query_mask: Optional[Float[torch.Tensor, "n_haps 1 batch 1"]] = None,
    ) -> tuple[
        Float[torch.Tensor, "n_haps n_snps batch hidden"],
        Float[torch.Tensor, "num_heads n_snps batch n_haps n_haps"],
    ]:
        return self.compute_attention_update(
            x,
            attn_mask=attn_mask,
            query_mask=query_mask,
        )


class RelativePosAttnBias(nn.Module):
    """T5-style relative position bias for genomic sequences with distance information."""

    def __init__(self, num_heads, num_buckets=64, max_distance=50000):
        super().__init__()
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance

        # Embedding table for relative position biases
        self.relative_attention_bias = nn.Embedding(self.num_buckets, self.num_heads)
        lin_buckets = torch.linspace(0, self.max_distance, self.num_buckets - 1)
        buckets = torch.cat([lin_buckets])

        self.register_buffer("buckets", buckets)

    @typed
    def _relative_position_bucket(
        self, distances: Int[torch.Tensor, "batch n_snps n_snps"]
    ) -> Int[torch.Tensor, "batch n_snps n_snps"]:
        """Convert relative positions to bucket indices."""
        abs_dist = distances.abs()
        bucket_indices = torch.bucketize(abs_dist, self.buckets, right=False)
        return bucket_indices

    @typed
    def forward(
        self, distances: Int[torch.Tensor, "batch n_snps n_snps"]
    ) -> Float[torch.Tensor, "num_heads batch n_snps n_snps"]:
        # Convert distances to relative position buckets

        # Get bias values
        batch_size, seq_len, _ = distances.shape
        relative_buckets = self._relative_position_bucket(distances)
        bias = self.relative_attention_bias(
            relative_buckets
        )  # (batch_size, seq_len, seq_len, num_heads)
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

    @typed
    def forward(
        self, x: Float[torch.Tensor, "n_haps n_snps batch hidden"]
    ) -> Float[torch.Tensor, "n_haps n_snps batch hidden"]:
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
