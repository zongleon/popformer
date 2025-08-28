import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import DataCollatorForLanguageModeling

class HaploSimpleDataCollator:
    """
    Data collator for axial attention models.
    Expects examples as dicts with:
      - input_ids: List[List[int]] (ragged, n_haps x n_snps)
      - distances: List[List[int]] (ragged, n_haps x n_snps)
    Pads both axes to max in batch.
    """
    subsample: int = 128
    mlm_probability: float = 0.15

    def __init__(self, subsample=128, mlm_probability=0.15):
        self.subsample = subsample
        self.mlm_probability = mlm_probability

    def _torch_mask_tokens(self, inputs):
        """
        Custom masking: oversample 1s for masking.
        Special tokens are any token not 0 or 1.
        """
        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        # Oversample 1s: set higher probability for tokens == 1
        probability_matrix[inputs == 1] = 0.5  # e.g., 50% chance for 1s

        # Custom special tokens mask: True where token is NOT 0 or 1
        special_tokens_mask = ~((labels == 0) | (labels == 1))
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # Only compute loss on masked tokens

        # 80% of the time, replace masked input tokens with [MASK]
        mask_token_id = 4
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 1.0)).bool() & masked_indices
        inputs[indices_replaced] = mask_token_id

        # 10% of the time, replace masked input tokens with random word
        # indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        # random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        # inputs[indices_random] = random_words[indices_random]

        # The rest 10% keep original

        return inputs, labels

    def __call__(self, examples):
        batch_input_ids = []
        batch_distances = []
        batch_attention_masks = []

        for idx, ex in enumerate(examples):
            # list of list of input_ids
            # list of distances
            input_ids = ex["input_ids"]
            distances = ex["distances"]

            # subsample strategy: randomly select
            if len(input_ids) > self.subsample:
                indices = torch.randperm(len(input_ids))[:self.subsample]

            input_ids_2d = torch.tensor(input_ids, dtype=torch.long)[indices]
            # print(input_ids_2d.shape)
            attention_masks_2d = torch.ones_like(input_ids_2d)

            batch_input_ids.append(input_ids_2d)
            batch_attention_masks.append(attention_masks_2d)

            # add a haplotype thats just <s> <pad> * max_n_snps </s>
            # haplotype = [self.tokenizer.bos_token_id] + [self.tokenizer.pad_token_id] * (max_n_snps - 2) + [self.tokenizer.eos_token_id]
            # input_ids_2d.insert(0, haplotype)
            # attention_masks_2d.insert(0, [1] + [0] * (max_n_snps - 2) + [1])

            # Vectorized cumulative distance matrix
            cumulative_dists = torch.cumsum(torch.tensor(ex["distances"]), dim=0)
            dist_matrix = torch.abs(cumulative_dists.unsqueeze(0) - cumulative_dists.unsqueeze(1))
            batch_distances.append(dist_matrix)

        # Convert to tensors: (batch, n_haps, n_snps)
        input_ids = torch.stack(batch_input_ids)
        batch_attention_masks = torch.stack(batch_attention_masks)
        distances = torch.stack(batch_distances)

        # Flatten for MLM masking, then reshape back
        flat_input_ids = input_ids.view(input_ids.size(0), -1)
        inputs, labels = self._torch_mask_tokens(flat_input_ids)
        masked_input_ids = inputs.view_as(input_ids)
        labels = labels.view_as(input_ids)

        return {
            "input_ids": masked_input_ids,
            "attention_mask": batch_attention_masks,
            "distances": distances,
            "labels": labels,
        }


class HaploSimpleNormalDataCollator:
    """
    Data collator for axial attention models.
    Expects examples as dicts with:
      - input_ids: List[List[int]] (n_haps x n_snps)
      - distances: List[List[int]] (n_haps x n_snps)
    Pads both axes to max in batch.
    """

    subsample: int = 128
    label_dtype: torch.dtype = torch.long

    def __init__(self, subsample=128, label_dtype=torch.long):
        self.subsample = subsample
        self.label_dtype = label_dtype

    def __call__(self, examples):
        batch_input_ids = []
        batch_attention_masks = []
        batch_distances = []
        batch_labels = []

        for idx, ex in enumerate(examples):
            # list of list of input_ids
            # list of distances
            input_ids = ex["input_ids"]
            distances = ex["distances"]

            # subsample strategy: randomly select
            if len(input_ids) > self.subsample:
                indices = torch.randperm(len(input_ids))[:self.subsample]
            else:
                indices = torch.arange(len(input_ids))

            input_ids_2d = torch.tensor(input_ids, dtype=torch.long)[indices]
            # print(input_ids_2d.shape)
            attention_masks_2d = torch.ones_like(input_ids_2d)
            # add a haplotype thats just <s> <pad> * max_n_snps </s>
            # haplotype = [self.tokenizer.bos_token_id] + [self.tokenizer.pad_token_id] * (max_n_snps - 2) + [self.tokenizer.eos_token_id]
            # input_ids_2d.insert(0, haplotype)
            # attention_masks_2d.insert(0, [1] + [0] * (max_n_snps - 2) + [1])

            batch_input_ids.append(input_ids_2d)
            batch_attention_masks.append(attention_masks_2d)
            
            if "label" in ex:
                batch_labels.append(ex["label"])

            # Vectorized cumulative distance matrix
            cumulative_dists = torch.cumsum(torch.tensor(ex["distances"]), dim=0)
            dist_matrix = torch.abs(cumulative_dists.unsqueeze(0) - cumulative_dists.unsqueeze(1))
            batch_distances.append(dist_matrix)

        # Convert to tensors: (batch, n_haps, n_snps)
        input_ids = torch.stack(batch_input_ids)
        batch_attention_masks = torch.stack(batch_attention_masks)
        distances = torch.stack(batch_distances)
        batch_labels = torch.tensor(batch_labels, dtype=self.label_dtype)

        return {
            "input_ids": input_ids,
            "attention_mask": batch_attention_masks,
            "distances": distances,
            "labels": batch_labels,
        }

