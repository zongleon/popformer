from typing import Optional
import torch
import torch.nn as nn
from transformers import RobertaForSequenceClassification
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.models.roberta.modeling_roberta import RobertaForMaskedLM, RobertaModel

from .modules import PopformerEncoder

class PopformerForMaskedLM(RobertaForMaskedLM):
    """RobertaForMaskedLM that accepts distances in forward pass."""
    def __init__(self, config):
        super().__init__(config)
        self.roberta = PopformerModel(config, add_pooling_layer=False)
        self.post_init()

    def forward(self, input_ids, distances, attention_mask, 
                labels=None, 
                return_hidden_states=False, 
                return_attentions=False,
                **kwargs):
        # Pass distances through to the model
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            distances=distances,
            return_attentions=return_attentions,
        )

        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if return_hidden_states or return_attentions:
            return {
                "loss": masked_lm_loss,
                "logits": prediction_scores,
                "hidden_states": sequence_output if return_hidden_states else None,
                "attentions": outputs[1] if return_attentions else None,
            }
        return {
            "loss": masked_lm_loss,
            "logits": prediction_scores,
        }


class PopformerClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        # x = features[:, 0, 0, :]  # take <s> token (equiv. to [CLS])
        x = features.mean(dim=(1, 2)) # take mean across haps, snps
        # x = features[:, :, 0, :].mean(dim=1)  # mean across haplotypes at bos SNP token
        x = self.layer_norm(x)
        # x = features[:, 0, :].mean(dim=1) # take mean across [CLS] token
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class PopformerForWindowClassification(RobertaForSequenceClassification):
    """RobertaForSequenceClassification that accepts distances in forward pass."""
    def __init__(self, config):
        super().__init__(config)
        if getattr(config, "axial", False):
            self.roberta = PopformerModel(config, add_pooling_layer=False)

        # test a simple logistic regression head
        self.classifier = PopformerClassificationHead(config)
        self.post_init()

    def forward(self, input_ids=None, distances=None, attention_mask=None, labels=None, 
                return_hidden_states=False, **kwargs):
        # Pass distances through to the model
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            distances=distances,
        )

        output = outputs[0] # unpooled
        # print(output.mean(), output.std())
        logits = self.classifier(output)

        loss = None
        if labels is not None:
            if self.config.num_labels == 1:
                loss_fct = torch.nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = torch.nn.CrossEntropyLoss()
                # print softmax'd logits and labels
                # print(logits.softmax(dim=-1), labels)
                # print(logits, labels)
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        if return_hidden_states:
            return {
                "loss": loss,
                "logits": logits,
                "hidden_states": output,
            }
        return {
            "loss": loss,
            "logits": logits,
            # "hidden_states": outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            # "attentions": outputs.attentions if hasattr(outputs, 'attentions') else None,
        }


class PopformerSNPClassificationHead(nn.Module):
    """Head for SNP-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        # Pool only over haplotypes, keep SNP dimension
        x = features.mean(dim=1)  # (batch_size, n_snps, hidden_size)
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)  # (batch_size, n_snps, num_labels)
        return x


class PopformerForSNPClassification(RobertaForSequenceClassification):
    """SNP-level (column-wise) classification model."""
    def __init__(self, config):
        super().__init__(config)
        self.roberta = PopformerModel(config, add_pooling_layer=False)

        # Use the column-specific head
        self.classifier = PopformerSNPClassificationHead(config)
        self.post_init()

    def forward(self, input_ids=None, distances=None, attention_mask=None, labels=None, 
                return_hidden_states=False, **kwargs):
        # Pass distances through to the model
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            distances=distances,
        )

        output = outputs[0]  # (batch_size, n_haps, n_snps, hidden_size)
        logits = self.classifier(output)  # (batch_size, n_snps, num_labels)

        loss = None
        if labels is not None:
            if self.config.num_labels == 1:
                loss_fct = torch.nn.MSELoss(reduction='none')
                loss = loss_fct(logits.view(-1), labels.view(-1))
                mask = labels.view(-1) != -100
                loss = loss[mask].mean() if mask.any() else torch.tensor(0.0, device=logits.device)
            else:
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        if return_hidden_states:
            return {
                "loss": loss,
                "logits": logits,
                "hidden_states": output,
            }
        return {
            "loss": loss,
            "logits": logits,
        }


class PopformerModel(RobertaModel):
    """
    Base Popformer model. Subclasses RobertaModel but generally only for the saving/loading logic,
    basically every module is replaced.
    """
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer=add_pooling_layer)
        # remove absolute position embeddings
        self.embeddings.position_embeddings = None
        self.encoder = PopformerEncoder(config)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        distances: Optional[torch.Tensor] = None,
        return_attentions: bool = False,
    ):
        batch_size, n_haps, n_snps = input_ids.size()
        input_shape = (batch_size, n_haps * n_snps)  # Flatten for embeddings
        device = input_ids.device

        # Flatten input_ids for embedding lookup
        flattened_input_ids = input_ids.view(batch_size, -1)
        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # Get embeddings and reshape back to 2D
        embedding_output = self.embeddings(
            input_ids=flattened_input_ids,
            token_type_ids=token_type_ids,
        )

        embedding_output = embedding_output.view(batch_size, n_haps, n_snps, -1)
        # print(embedding_output.size())

        # Create attention mask for 2D input
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, n_haps, n_snps), device=device)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            distances=distances,
            return_attentions=return_attentions,
        )

        sequence_output = encoder_outputs[0]

        # For pooling, we might want to pool over haplotypes or use a different strategy
        # mean over haplotypes
        # pooled_output = sequence_output.mean(dim=1)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        attentions = encoder_outputs[1] if return_attentions else None
        # if self.pooler is not None:
        #     # Simple pooling: take mean over haplotypes and first SNP
        #     pooled_input = sequence_output.mean(dim=1)[:, 0, :]  # (batch_size, hidden_size)
        #     pooled_output = self.pooler(pooled_input.unsqueeze(1)).squeeze(1)

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            attentions=attentions,
        )

