from typing import Optional, Union
import math
import torch
import torch.nn as nn
from transformers import Cache, EncoderDecoderCache, apply_chunking_to_forward
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.roberta.modeling_roberta import RobertaAttention, RobertaEncoder, RobertaLayer

# Copied from transformers.models.bert.modeling_bert.BertEncoder with Bert->Roberta
class HapbertaEncoder(RobertaEncoder):
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx)
        if getattr(config, "axial", False):
            self.layer = nn.ModuleList([HapbertaLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        else:
            self.layer = nn.ModuleList([HapbertaAxialLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        distances: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[tuple[tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Union[tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False

        return_legacy_cache = False
        if use_cache and self.config.is_decoder and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            past_key_values = EncoderDecoderCache.from_legacy_cache(past_key_values)

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                distances,
                layer_head_mask,
                encoder_hidden_states,  # as a positional argument for gradient checkpointing
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                cache_position=cache_position,
            )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if return_legacy_cache:
            past_key_values = past_key_values.to_legacy_cache()

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    past_key_values,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


# Copied from transformers.models.bert.modeling_bert.BertLayer with Bert->Roberta
class HapbertaLayer(RobertaLayer):
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        distances = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_value  = None,
        output_attentions: Optional[bool] = False,
        cache_position: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor]:
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            distances=distances,
            head_mask=head_mask,
            output_attentions=output_attentions,
            past_key_value=past_key_value,
            cache_position=cache_position,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
    
class HapbertaAxialLayer(RobertaLayer):
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx=layer_idx)
        self.row_attn = RowSelfAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.attention_probs_dropout_prob,
        )
        self.col_attn = ColumnSelfAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.attention_probs_dropout_prob,
        )
        # we will use alternating row / col attention
        delattr(self, "attention")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        distances = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_value  = None,
        output_attentions: Optional[bool] = False,
        cache_position: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor]:
        
        attention_output, row_attn = self.row_attn(
            hidden_states,
            distances,
            self_attention_mask=attention_mask,
        )
        
        attention_output, col_attn = self.col_attn(
            hidden_states,
            distances, 
            self_attention_mask=attention_mask,
        )
        # outputs = col_attn[1:]  # add self attentions if we output attention weights
        outputs = None

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
    

class HapbertaAttention(RobertaAttention):
    def __init__(self, config, relative_bias_module):
        super().__init__(config)
        self.dist_bias = relative_bias_module

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor = None,
        distances: torch.Tensor = None,
        head_mask: torch.FloatTensor = None,
        encoder_hidden_states: torch.FloatTensor = None,
        past_key_value = None,
        cache_position = None,
    ) -> tuple[torch.Tensor]:
        batch_size, seq_length, _ = hidden_states.shape
        query_layer = self.self.query(hidden_states)
        query_layer = query_layer.view(batch_size, -1, self.self.num_attention_heads, self.self.attention_head_size).transpose(
            1, 2
        )

        is_cross_attention = encoder_hidden_states is not None
        if past_key_value is not None:
            if isinstance(past_key_value, EncoderDecoderCache):
                is_updated = past_key_value.is_updated.get(self.layer_idx)
                if is_cross_attention:
                    # after the first generated id, we can subsequently re-use all key/value_layer from cache
                    curr_past_key_value = past_key_value.cross_attention_cache
                else:
                    curr_past_key_value = past_key_value.self_attention_cache
            else:
                curr_past_key_value = past_key_value

        current_states = encoder_hidden_states if is_cross_attention else hidden_states
        if is_cross_attention and past_key_value is not None and is_updated:
            # reuse k,v, cross_attentions
            key_layer = curr_past_key_value.layers[self.layer_idx].keys
            value_layer = curr_past_key_value.layers[self.layer_idx].values
        else:
            key_layer = self.self.key(current_states)
            key_layer = key_layer.view(batch_size, -1, self.self.num_attention_heads, self.self.attention_head_size).transpose(
                1, 2
            )
            value_layer = self.self.value(current_states)
            value_layer = value_layer.view(
                batch_size, -1, self.self.num_attention_heads, self.self.attention_head_size
            ).transpose(1, 2)

            if past_key_value is not None:
                # save all key/value_layer to cache to be re-used for fast auto-regressive generation
                cache_position = cache_position if not is_cross_attention else None
                key_layer, value_layer = curr_past_key_value.update(
                    key_layer, value_layer, self.layer_idx, {"cache_position": cache_position}
                )
                # set flag that curr layer for cross-attn is already updated so we can re-use in subsequent calls
                if is_cross_attention:
                    past_key_value.is_updated[self.layer_idx] = True

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.self.position_embedding_type == "haplo":
            # Use Haplo-specific relative position bias
            relative_position_bias = self.dist_bias(distances)
            attention_scores = attention_scores + relative_position_bias

        attention_scores = attention_scores / math.sqrt(self.self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        return context_layer, attention_probs


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
        self.attn_shape = "hnij"

        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)

        self.dist_bias = RelativePosAttnBias()
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout_module = nn.Dropout(dropout)

    def align_scaling(self, q):
        num_rows = q.size(0)
        return self.scaling / math.sqrt(num_rows)

    def _batched_forward(
        self,
        x,
        distances,
        self_attn_mask=None,
        self_attn_padding_mask=None,
    ):
        num_rows, num_cols, batch_size, embed_dim = x.size()
        max_rows = max(1, self.max_tokens_per_msa // num_cols)
        attns = 0
        scaling = self.align_scaling(x)
        for start in range(0, num_rows, max_rows):
            attn_weights = self.compute_attention_weights(
                x[start : start + max_rows],
                scaling,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask[
                    :, start : start + max_rows
                ]
                if self_attn_padding_mask is not None
                else None,
            )
            attns += attn_weights
        attn_probs = attns.softmax(-1)
        attn_probs = self.dropout_module(attn_probs)

        outputs = []
        for start in range(0, num_rows, max_rows):
            output = self.compute_attention_update(
                x[start : start + max_rows], distances, attn_probs
            )
            outputs.append(output)

        output = torch.cat(outputs, 0)
        return output, attn_probs

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
            q *= 1 - self_attn_padding_mask.permute(1, 2, 0).unsqueeze(3).unsqueeze(
                4
            ).to(q)

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
        if (
            num_rows * num_cols > self.max_tokens_per_msa
        ) and not torch.is_grad_enabled():
            return self._batched_forward(x, self_attn_mask, self_attn_padding_mask)
        else:
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

        self.dist_bias = RelativePosAttnBias()
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout_module = nn.Dropout(dropout)

    def _batched_forward(
        self,
        x,
        distances,
        self_attn_mask=None,
        self_attn_padding_mask=None,
    ):
        num_rows, num_cols, batch_size, embed_dim = x.size()
        max_cols = max(1, self.max_tokens_per_msa // num_rows)
        outputs = []
        attns = []
        for start in range(0, num_cols, max_cols):
            output, attn = self(
                x[:, start : start + max_cols],
                distances,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask[
                    :, :, start : start + max_cols
                ]
                if self_attn_padding_mask is not None
                else None,
            )
            outputs.append(output)
            attns.append(attn)
        output = torch.cat(outputs, 1)
        attns = torch.cat(attns, 1)
        return output, attns

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

            rel_pos_bias = self.dist_bias(distances)
            attn_weights += rel_pos_bias

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
        # if False and num_rows * num_cols > 2 ** 14 and not torch.is_grad_enabled():
        if (
            num_rows * num_cols
        ) > self.max_tokens_per_msa and not torch.is_grad_enabled():
            return self._batched_forward(
                x,
                distances,
                self_attn_mask,
                self_attn_padding_mask,
            )
        else:
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
        ret = torch.zeros_like(distances, dtype=torch.long)
        n = distances.abs()
        
        # Half buckets for small distances (linear)
        max_exact = self.num_buckets // 2
        is_small = n < max_exact

        ret[is_small] = n[is_small].long()
        
        # log buckets for large distances
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) 
            / torch.log(torch.tensor(self.max_distance / max_exact)) *
            (self.num_buckets - max_exact - 1)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, self.num_buckets - 1))
        ret[~is_small] = val_if_large[~is_small]

        return ret
    
    def forward(self, distances):
        """
        Args:
            distances: (batch_size, seq_len, seq_len) matrix of distances
        Returns:
            bias: (batch_size, num_heads, seq_len, seq_len) attention bias
        """
        batch_size, seq_len, _ = distances.shape

        # Convert distances to relative position buckets
        relative_buckets = self._relative_position_bucket(distances)

        # Get bias values
        bias = self.relative_attention_bias(relative_buckets)  # (batch_size, seq_len, seq_len, num_heads)
        return bias.permute(0, 3, 1, 2)  # (batch_size, num_heads, seq_len, seq_len)
