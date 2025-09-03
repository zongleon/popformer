import torch


class HaploSimpleDataCollator:
    """
    Data collator for axial attention models.

    Parameters:
      - subsample : int, number of SNPs to subsample
      - mlm_probability : float, probability of masking random tokens
      - whole_snp_mask_probability : float, probability of masking whole SNPs
      - label_dtype : torch.dtype, dtype for the labels
    """

    subsample: int = 32
    mlm_probability: float = 0.15
    whole_snp_mask_probability: float = 0.75
    return_input_mask = False
    label_dtype: torch.dtype = None

    bos_token_id = 2
    eos_token_id = 3
    pad_token_id = 5

    def __init__(
        self,
        subsample=32,
        mlm_probability=0.05,
        whole_snp_mask_probability=0.15,
        return_input_mask = False,
        label_dtype=None,
    ):
        self.subsample = subsample
        self.mlm_probability = mlm_probability
        self.whole_snp_mask_probability = whole_snp_mask_probability
        self.return_input_mask = return_input_mask
        self.label_dtype = label_dtype

    def _torch_mask_tokens(self, inputs):
        """
        Custom masking: oversample 1s for masking.
        Special tokens are any token not 0 or 1.
        Applies both random token masking (mlm_probability) and whole SNP (column) masking (whole_snp_mask_probability).
        """
        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        
        if self.mlm_probability > 0:
            # Oversample 1s: set higher probability for tokens == 1
            probability_matrix[inputs == 1] = 0.5  # e.g., 50% chance for 1s

        # --- Whole SNP (column) masking ---
        # For each column, decide whether to mask the whole column
        if self.whole_snp_mask_probability > 0:
            n_cols = inputs.shape[2]
            col_mask = torch.bernoulli(
                torch.full((n_cols,), self.whole_snp_mask_probability)
            ).bool()
            # Broadcast col_mask to all rows
            probability_matrix[:, :, col_mask] = 1.0  # force masking for these columns

        # Custom special tokens mask: True where token is NOT 0 or 1
        special_tokens_mask = ~((labels == 0) | (labels == 1))
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # Only compute loss on masked tokens

        mask_token_id = 4
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 1.0)).bool() & masked_indices
        )
        inputs[indices_replaced] = mask_token_id

        return inputs, labels

    def __call__(self, examples):
        batch_input_ids = []
        batch_distances = []
        batch_attention_masks = []
        batch_labels = []

        # Find the index (position) of the maximum eos_token_id in the batch
        # we'll pad to this, rather than a max size like 512
        max_len = max(
            [
            torch.where(torch.tensor(ex["input_ids"]) == self.eos_token_id)[1].max().item()
            for ex in examples
            ]
        ) + 1
        if max_len % 8 != 0:
            max_len = ((max_len // 8) + 1) * 8

        for idx, ex in enumerate(examples):
            # list of list of input_ids
            # list of distances
            input_ids = ex["input_ids"]
            distances = ex["distances"]
            if self.label_dtype is not None:
                labels = ex["label"]
                batch_labels.append(labels)

            input_ids = torch.tensor(input_ids, dtype=torch.long)
            distances = torch.tensor(distances)

            # Identify non-padded haplotypes (rows not all pad_token_id)
            non_pad_mask = ~(input_ids == self.pad_token_id).all(dim=1)
            non_pad_indices = non_pad_mask.nonzero(as_tuple=True)[0]

            # Subsample only from non-padded haplotypes
            if len(non_pad_indices) >= self.subsample:
                selected = non_pad_indices[
                    torch.randperm(len(non_pad_indices))[: self.subsample]
                ]
            else:
                raise RuntimeError(
                    f"Not enough non-padded haplotypes ({len(non_pad_indices)}) to subsample {self.subsample}"
                )
            input_ids = input_ids[selected, :max_len]
            distances = distances[:max_len]
            
            n_snps = input_ids.shape[1]
            attention_masks = torch.ones((n_snps, n_snps), dtype=torch.long)

            pad_cols = (input_ids == self.pad_token_id).all(dim=0)
            if pad_cols.any():
                attention_masks[:, pad_cols] = 0

            # print("input ids shape ", input_ids.size())

            batch_input_ids.append(input_ids)
            batch_attention_masks.append(attention_masks)

            # add a haplotype thats just <s> <pad> * max_n_snps </s>
            # haplotype = [self.tokenizer.bos_token_id] + [self.tokenizer.pad_token_id] * (max_n_snps - 2) + [self.tokenizer.eos_token_id]
            # input_ids_2d.insert(0, haplotype)
            # attention_masks_2d.insert(0, [1] + [0] * (max_n_snps - 2) + [1])

            # Vectorized cumulative distance matrix
            cumulative_dists = torch.cumsum(distances, dim=0)
            dist_matrix = torch.abs(
                cumulative_dists.unsqueeze(0) - cumulative_dists.unsqueeze(1)
            )
            batch_distances.append(dist_matrix)

        # Convert to tensors: (batch, n_haps, n_snps)
        input_ids = torch.stack(batch_input_ids)
        batch_attention_masks = torch.stack(batch_attention_masks)
        distances = torch.stack(batch_distances)

        out = {
            "input_ids": input_ids,
            "attention_mask": batch_attention_masks,
            "distances": distances,
        }

        if self.label_dtype is not None:
            labels = torch.tensor(batch_labels, dtype=self.label_dtype)
        elif self.mlm_probability > 0 or self.whole_snp_mask_probability > 0:
            input_ids, labels = self._torch_mask_tokens(input_ids)
            if self.return_input_mask:
                print(torch.where((input_ids == 4).all(axis=1)))
                ids_keep = torch.tensor([torch.where((input_ids == 4).all(axis=1))[0]])
                ids_restore = torch.tensor([torch.arange(input_ids.shape[2])])
                out["input_mask"] = (ids_keep, ids_restore)
        else:
            labels = torch.ones_like(input_ids)

        out["labels"] = labels
        return out 