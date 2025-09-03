import copy
from typing import Optional
import torch
import torch.nn as nn
from transformers import RobertaForSequenceClassification
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions, BaseModelOutput
from transformers.models.roberta.modeling_roberta import RobertaForMaskedLM, RobertaModel

from modules import HapbertaAxialDecoder, HapbertaAxialEncoder

class HapbertaForMaskedLM(RobertaForMaskedLM):
    """RobertaForMaskedLM that accepts distances in forward pass."""
    def __init__(self, config):
        super().__init__(config)
        self.axial = getattr(config, "axial", False) 
        self.mae = getattr(config, "mae", False)
        self.encoder_only = getattr(config, "encoder_only", False)
        self.mae_mask = getattr(config, "mae_mask", 0.0)
        if self.axial:
            if self.mae:
                self.roberta = HapbertaMAEModel(config, decoder_depth=2,
                                                encoder_only=self.encoder_only,
                                                mask_ratio=self.mae_mask,
                                                )
            else:
                self.roberta = HapbertaAxialModel(config, add_pooling_layer=False,)
        self.post_init()

        # lm_head = self.lm_head
        # print("Bias:", lm_head.decoder.bias)
        # print("Weight std:", lm_head.decoder.weight.std(dim=1))

    def forward(self, input_ids, distances, attention_mask, 
                labels=None, 
                return_hidden_states=False, 
                return_attentions=False,
                input_mask=None,
                **kwargs):
        # Pass distances through to the model
        if getattr(self.config, "mae", False):
            outputs, labels2 = self.roberta(
                input_ids=input_ids,
                distances=distances,
                return_attentions=return_attentions,
                input_mask=input_mask
            )
            labels = labels2 if labels2 is not None else labels
        else:
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
            # print("pred scores: ", prediction_scores.view(-1, self.config.vocab_size).size())
            # print("labels: ", labels.view(-1).size())
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


class HapbertaMAEModel(RobertaModel):
    """
    Axial attention layers in masked autoencoders task
    Input should not be masked! And don't include
    special tokens.
    """

    def __init__(self, config, decoder_depth=2, mask_ratio=0.75, encoder_only=False):
        super().__init__(config, add_pooling_layer=False)
        self.encoder_config = copy.deepcopy(config)
        self.decoder_config = copy.deepcopy(config)
        self.decoder_config.num_hidden_layers = decoder_depth
        self.mask_ratio = mask_ratio
        self.encoder_only = encoder_only

        # remove absolute position embeddings
        self.embeddings.position_embeddings = None
        self.encoder = HapbertaAxialEncoder(self.encoder_config)
        self.decoder = HapbertaAxialDecoder(self.decoder_config)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.embeddings_decoder = nn.Linear(config.hidden_size, config.hidden_size)

        torch.nn.init.normal_(self.mask_token, std=.02)

    def random_masking(self, x, distances, ids_input = None):
        """
        From https://github.com/facebookresearch/mae/blob/main/models_mae.py#L123
        Perform column random masking by per-sample shuffling.
        Column shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        ids_keep, ids_restore = None, None
        if ids_input is not None:
            ids_keep, ids_restore = ids_input

        batch_size, n_haps, n_snps = x.shape  # batch, length, dim
        len_keep = int(n_snps * (1 - self.mask_ratio))
        
        noise = torch.rand(batch_size, n_snps, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove

        # keep the first subset
        if ids_keep is None:
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=2, index=ids_keep.unsqueeze(1).repeat(1, n_haps, 1))
        
        distances_masked = torch.gather(
            distances, 1, ids_keep.unsqueeze(-1).expand(-1, -1, distances.size(-1))
        )  # (batch_size, len_keep, n_snps)
        distances_masked = torch.gather(
            distances_masked, 2, ids_keep.unsqueeze(1).expand(-1, distances_masked.size(1), -1)
        )  # (batch_size, len_keep, len_keep)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([batch_size, n_snps], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore).unsqueeze(1).repeat(1, n_haps, 1)

        labels = torch.full((batch_size, n_haps, n_snps), -100, device=x.device)
        labels[mask == 0] = x[mask == 0]

        # Convert tensors to numpy arrays and save them
        # print("masked input ids size ", x_masked.size())
        # print("masked distances size ", distances_masked.size())
        # print("mask size ", mask.size())
        # print("restore ids ", ids_restore.size())
        return x_masked, distances_masked, mask, labels, ids_restore

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        distances: Optional[torch.Tensor] = None,
        input_mask: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        return_attentions: bool = False,
    ):
        do_mask = self.mask_ratio > 0 or input_mask is not None
        if do_mask:
            # masking here
            input_masked, distances_masked, mask, labels, ids_restore = self.random_masking(input_ids, distances, input_mask)
            batch_size, n_haps, n_snps = input_masked.size()
        else:
            batch_size, n_haps, n_snps = input_ids.size()   

        input_shape_encoder = (batch_size, n_haps * n_snps)
        device = input_ids.device

        # Flatten input_ids for embedding lookup
        flattened_input_ids = input_ids.view(input_shape_encoder) if not do_mask else input_masked.view(input_shape_encoder)
        token_type_ids = torch.zeros(input_shape_encoder, dtype=torch.long, device=device)
        
        # Get embeddings and reshape back to 2D
        embedding_output = self.embeddings(
            input_ids=flattened_input_ids,
            token_type_ids=token_type_ids,
        )
        embedding_output = embedding_output.view(batch_size, n_haps, n_snps, -1)

        encoder_outputs = self.encoder(
            embedding_output,
            distances=distances if not do_mask else distances_masked,
            return_attentions=return_attentions,
        )

        encoder_output = encoder_outputs[0]
        if self.encoder_only:
            return BaseModelOutput(
                last_hidden_state=encoder_output,
                attentions=encoder_outputs[1] if return_attentions else None
            ), None
        # print("encoder output shape: ", encoder_output.size())

        if do_mask:
            # append mask tokens to sequence
            mask_tokens = self.mask_token.repeat(batch_size, n_haps, ids_restore.shape[1] - encoder_output.shape[2], 1)
            encoder_output = torch.cat([encoder_output, mask_tokens], dim=2)
            ids_restore_expanded = ids_restore.unsqueeze(1).unsqueeze(-1).expand(-1, n_haps, -1, encoder_output.shape[-1])
            decoder_input = torch.gather(encoder_output, dim=2, index=ids_restore_expanded)
        else:
            decoder_input = encoder_output

        decoder_input = self.embeddings_decoder(decoder_input)

        decoder_output = self.decoder(
            hidden_states=decoder_input,
            distances=distances,
            return_attentions=return_attentions,
        )

        # print("decoder output shape: ", decoder_output[0].size())

        attentions = encoder_outputs[1] if return_attentions else None

        return BaseModelOutput(
            last_hidden_state=decoder_output[0],
            attentions=attentions,
        ), labels if do_mask else None
