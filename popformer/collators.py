import torch


class RawMatrixCollator:
    """
    simple collator that:
      - finds the single BOS and EOS columns,
      - returns columns strictly between them,
      - removes any all-pad rows,
      - trims distances and attention_mask to the same column range.
    
    Returns lists of per-example tensors (no padding across the batch).
    """
    bos_token_id = 2
    eos_token_id = 3
    pad_token_id = 5

    def __call__(self, examples):
        batch_inputs = []
        batch_dists = []
        batch_attns = []
        batch_labels = []

        for ex in examples:
            x = torch.as_tensor(ex["input_ids"], dtype=torch.long)  # (n_haps, n_cols)

            # Find BOS/EOS column indices (guaranteed single columns)
            
            bos_idx = torch.any(x == self.bos_token_id, dim=0).nonzero(as_tuple=False).squeeze()
            eos_idx = torch.any(x == self.eos_token_id, dim=0).nonzero(as_tuple=False).squeeze()
            start = int(bos_idx.item()) + 1
            end = int(eos_idx.item())  # exclusive

            # Trim SNP columns strictly between BOS and EOS
            x = x[:, start:end]

            # Remove rows that are entirely PAD
            keep_rows = ~(x == self.pad_token_id).all(dim=1)
            x = x[keep_rows]
            batch_inputs.append(x)

            # Trim distances if present (assumed 1D over SNP columns)
            if "distances" in ex:
                d = torch.as_tensor(ex["distances"])
                d = d[start:end]
                batch_dists.append(d)

            # Trim attention_mask if present (either [n_cols] or [n_cols, n_cols])
            if "attention_mask" in ex and ex["attention_mask"] is not None:
                am = torch.as_tensor(ex["attention_mask"])
                if am.dim() == 2 and am.shape[0] == am.shape[1]:
                    am = am[start:end, start:end]
                else:
                    am = am[start:end]
                batch_attns.append(am)

            if "label" in ex:
                batch_labels.append(ex["label"])

        out = {"input_ids": batch_inputs,
               "distances": batch_dists,
               "attention_mask": batch_attns,
               "labels": batch_labels
               }
        
        return out


class HaploSimpleDataCollator:
    """
    Data collator for axial attention models.

    Parameters:
      - subsample : int, number of SNPs to subsample
      - mlm_probability : float, probability of masking random tokens
      - whole_snp_mask_probability : float, probability of masking whole SNPs
      - label_dtype : torch.dtype, dtype for the labels
    """

    bos_token_id = 2
    eos_token_id = 3
    mask_token_id = 4
    pad_token_id = 5
    subsample: tuple[int, int] | None = (32, 32)
    mlm_probability: float = 0.
    whole_snp_mask_probability: float = 0.
    span_mask_probability: float = 0.
    oversample_ones = False
    pad_batch = False
    label_dtype: torch.dtype = None
    subsample_type: str = "random"

    def __init__(
        self,
        subsample: tuple[int, int] | None = None,
        mlm_probability=0.,
        whole_snp_mask_probability=0.,
        span_mask_probability=0.,
        oversample_ones = False,
        pad_batch=True,
        label_dtype=None,
        subsample_type: str = "random",
    ):
        if subsample is not None and subsample[0] > subsample[1]:
            raise ValueError("subsample[0] must be <= subsample[1]")
        self.subsample = subsample
        self.mlm_probability = mlm_probability
        self.subsample_type = subsample_type
        self.whole_snp_mask_probability = whole_snp_mask_probability
        self.span_mask_probability = span_mask_probability
        self.oversample_ones = oversample_ones
        self.pad_batch = pad_batch
        self.label_dtype = label_dtype

    def _hamming_dist_all(self, A_bool, row_bool):
        """Compute Hamming distance from each row of A_bool to row_bool."""
        # A_bool, row_bool are boolean arrays
        return torch.count_nonzero(A_bool != row_bool, dim=1)

    def select_diverse_rows(
        self,
        X: torch.Tensor,
        k: int,
        seed: int | None = None,
    ):
        """
        Select k row indices from a binary matrix X (shape: [n, m]) to maximize diversity
        using greedy farthest-first traversal w.r.t. Hamming distance (max-min criterion).

        - Picks a random start row.
        - Repeatedly adds the row that maximizes the minimum Hamming distance to the selected set.

        Args:
            X: Binary matrix (values in {0,1}); shape (n, m). dtype can be bool or int.
            k: Number of rows to select; 1 <= k <= n.
            seed: Optional random seed for reproducibility.

        Returns:
            np.ndarray of shape (k,) with selected row indices in selection order.

        Complexity:
            O(k * n * m) time, O(n) extra memory.
        """
        n, m = X.shape
        # Convert to boolean for fast XOR/compare
        Xb = X.bool()
        
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)
        selected = []

        # Start with a random row
        first = torch.randint(0, n, (1,), generator=rng).item()
        selected.append(first)

        # Track which rows remain candidates
        candidate_mask = torch.ones(n, dtype=torch.bool)
        candidate_mask[first] = False

        # min_dists[i] = min Hamming distance from row i to any selected row
        min_dists = self._hamming_dist_all(Xb, Xb[first]).to(torch.long)

        while len(selected) < k:
            # Consider only candidates; pick one with max min-distance (random tie-break)
            candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).squeeze()
            if candidate_indices.numel() == 0:
                break  # nothing left to pick
            cand_dists = min_dists[candidate_indices]
            max_val = cand_dists.max()
            # Random tie-break among equals
            tied = candidate_indices[cand_dists == max_val]
            choice = int(tied[torch.randint(0, len(tied), (1,), generator=rng)])
            selected.append(choice)
            candidate_mask[choice] = False

            # Update min distances with the newly selected row
            d_new = self._hamming_dist_all(Xb, Xb[choice])
            torch.minimum(min_dists, d_new, out=min_dists)

        return torch.tensor(selected, dtype=torch.long)

    def _torch_mask_tokens(self, inputs):
        """
        Custom masking: oversample 1s for masking.
        Special tokens are any token not 0 or 1.
        Applies both random token masking (mlm_probability) and whole SNP (column) masking (whole_snp_mask_probability).
        """
        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if self.mlm_probability > 0 and self.oversample_ones:
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

        if self.span_mask_probability > 0:
            batch_size, n_rows, n_cols = inputs.shape
            span_mask = torch.zeros_like(inputs, dtype=torch.bool)
            mean_len = 8
            for b in range(batch_size):
                n_spans = int(self.span_mask_probability * n_cols / mean_len)
                for _ in range(n_spans):
                    start = torch.randint(0, n_cols, (1,))
                    # geometric length
                    length = torch.distributions.Geometric(1.0 / mean_len).sample().long()
                    end = min(start + length, n_cols)
                    span_mask[b, :, start:end] = True
            probability_matrix[span_mask] = 1.0

        # Custom special tokens mask: True where token is NOT 0 or 1
        special_tokens_mask = ~((labels == 0) | (labels == 1))
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # Only compute loss on masked tokens

        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 1.0)).bool() & masked_indices
        )
        inputs[indices_replaced] = self.mask_token_id

        return inputs, labels

    def __call__(self, examples):
        batch_input_ids = []
        batch_distances = []
        batch_attention_masks = []
        batch_labels = []

        # Find the index (position) of the maximum eos_token_id in the batch
        # we'll pad to this, rather than a max size like 512
        if self.pad_batch:
            max_len = max(
                [
                torch.where(torch.tensor(ex["input_ids"]) == self.eos_token_id)[1].max().item()
                for ex in examples
                ]
            ) + 1
            if max_len % 8 != 0:
                max_len = ((max_len // 8) + 1) * 8
        else:
            max_len = 512
        
        # subsample storage
        # find the position of the maximum number of non-padded haplotypes
        max_n_non_pad = max(
            len((~(torch.tensor(ex["input_ids"]) == self.pad_token_id).all(dim=1)).nonzero(as_tuple=True)[0])
            for ex in examples
        )
        if self.subsample is not None:
            subs = torch.randint(self.subsample[0], min(self.subsample[1], max_n_non_pad) + 1, (1,)).item()
        else:
            subs = max_n_non_pad

        for idx, ex in enumerate(examples):
            # list of list of input_ids
            # list of distances
            input_ids = ex["input_ids"]
            distances = ex["distances"]
            if self.label_dtype is not None:
                labels = ex["label"]
                if isinstance(labels, list):
                    labels = labels[:max_len] + [-100] * (max_len - len(labels))
                    labels = labels[:514]
                batch_labels.append(labels)

            input_ids = torch.tensor(input_ids, dtype=torch.long)
            distances = torch.tensor(distances)

            # Identify non-padded haplotypes (rows not all pad_token_id)
            non_pad_mask = ~(input_ids == self.pad_token_id).all(dim=1)
            non_pad_indices = non_pad_mask.nonzero(as_tuple=True)[0]
            n_non_pad = len(non_pad_indices)

            # Subsample only from non-padded haplotypes
            if n_non_pad == subs:
                selected = non_pad_indices
            elif n_non_pad > subs:
                if self.subsample_type == "diverse":
                    selected = self.select_diverse_rows(
                        input_ids[non_pad_indices, :max_len],
                        subs,
                        seed=idx,
                    )
                    selected = non_pad_indices[selected]
                elif self.subsample_type == "random":
                    selected = non_pad_indices[
                        torch.randperm(n_non_pad)[: subs]
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
        if self.label_dtype is not None:
            labels = torch.tensor(batch_labels, dtype=self.label_dtype)
        elif self.mlm_probability > 0 or self.whole_snp_mask_probability > 0 or self.span_mask_probability > 0:
            input_ids, labels = self._torch_mask_tokens(input_ids)
        else:
            labels = None

        return {
            "input_ids": input_ids,
            "attention_mask": batch_attention_masks,
            "distances": distances,
            "labels": labels,
        }
