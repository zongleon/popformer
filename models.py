from typing import Optional
import torch
from transformers import RobertaForSequenceClassification
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.models.roberta.modeling_roberta import RobertaForMaskedLM, RobertaModel

from modules import *

class HapbertaForMaskedLM(RobertaForMaskedLM):
    """RobertaForMaskedLM that accepts distances in forward pass."""
    def __init__(self, config):
        super().__init__(config)
        if getattr(config, "axial", False):
            self.roberta = HapbertaAxialModel(config, add_pooling_layer=False)
        self.post_init()

        # lm_head = self.lm_head
        # print("Bias:", lm_head.decoder.bias)
        # print("Weight std:", lm_head.decoder.weight.std(dim=1))

    def forward(self, input_ids=None, distances=None, attention_mask=None, labels=None, **kwargs):
        # Pass distances through to the model
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            distances=distances,
        )
        
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)
        
        masked_lm_loss = None
        if labels is not None:
            # weight = torch.tensor(
            #     [0.1707489937543869, 1.2902734279632568, 5.4285712242126465, 5.4285712242126465, 1, 1, 1]
            # ).to(input_ids.device)
            loss_fct = torch.nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        
        return {
            "loss": masked_lm_loss,
            "logits": prediction_scores,
            # "hidden_states": outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            # "attentions": outputs.attentions if hasattr(outputs, 'attentions') else None,
        }


class HapbertaClassificationHead(nn.Module):
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
        x = self.layer_norm(x)
        # x = features[:, 0, :].mean(dim=1) # take mean across [CLS] token
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class HapbertaForSequenceClassification(RobertaForSequenceClassification):
    """RobertaForSequenceClassification that accepts distances in forward pass."""
    def __init__(self, config):
        super().__init__(config)
        if getattr(config, "axial", False):
            self.roberta = HapbertaAxialModel(config, add_pooling_layer=False)
        
        # test a simple logistic regression head
        self.classifier = HapbertaClassificationHead(config)
        self.post_init()

        # for param in self.roberta.parameters():
        #     param.requires_grad = False

    def forward(self, input_ids=None, distances=None, attention_mask=None, labels=None, **kwargs):
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

        return {
            "loss": loss,
            "logits": logits,
            # "hidden_states": outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            # "attentions": outputs.attentions if hasattr(outputs, 'attentions') else None,
        }


class HapbertaAxialModel(RobertaModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer=add_pooling_layer)
        # remove absolute position embeddings
        self.embeddings.position_embeddings = None
        self.encoder = HapbertaAxialEncoder(config)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        distances: Optional[torch.Tensor] = None,
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
        )
        
        sequence_output = encoder_outputs[0]
        
        # For pooling, we might want to pool over haplotypes or use a different strategy
        # mean over haplotypes
        # pooled_output = sequence_output.mean(dim=1)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        # if self.pooler is not None:
        #     # Simple pooling: take mean over haplotypes and first SNP
        #     pooled_input = sequence_output.mean(dim=1)[:, 0, :]  # (batch_size, hidden_size)
        #     pooled_output = self.pooler(pooled_input.unsqueeze(1)).squeeze(1)

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
        )