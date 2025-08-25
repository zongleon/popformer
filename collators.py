import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import DataCollatorForLanguageModeling

class HaploDataCollator:
    """Custom data collator that handles both tokens and genomic distances."""
    
    def __init__(self, tokenizer, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.mask_token_id = tokenizer.mask_token_id

    def __call__(self, examples):
        # Extract token_ids and distances from examples
        token_ids = [ex["token_ids"] for ex in examples]
        distances_list = [ex["distances"] for ex in examples]
        
        # Pad sequences to same length
        max_len = max(len(ids) for ids in token_ids)
        
        input_ids = []
        labels = []
        distance_matrices = []
        attention_masks = []
        
        for ids, dists in zip(token_ids, distances_list):
            # Pad token ids
            padded_ids = ids + [self.tokenizer.pad_token_id] * (max_len - len(ids))
            
            # Create attention mask
            attn_mask = [1] * len(ids) + [0] * (max_len - len(ids))
            
            # Apply MLM masking
            masked_ids = padded_ids.copy()
            label_ids = [-100] * max_len  # -100 is ignored in loss computation
            
            for i in range(len(ids)):  # Only mask non-padded tokens
                if torch.rand(1).item() < self.mlm_probability:
                    label_ids[i] = padded_ids[i]  # Store original token for loss
                    masked_ids[i] = self.mask_token_id  # Replace with mask token
            
            # Create distance matrix
            seq_len = len(ids)
            dist_matrix = torch.zeros(max_len, max_len)
            
            if len(dists) > 0:
                # Build cumulative distance matrix
                cumulative_dists = torch.cumsum(torch.tensor([0.0] + dists + [0.0]), dim=0)
                for i in range(seq_len):
                    for j in range(seq_len):
                        dist_matrix[i, j] = abs(cumulative_dists[i] - cumulative_dists[j])
            
            input_ids.append(masked_ids)
            labels.append(label_ids)
            distance_matrices.append(dist_matrix)
            attention_masks.append(attn_mask)
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "distances": torch.stack(distance_matrices),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        }
    
class HaploAxialDataCollator(DataCollatorForLanguageModeling):
    """
    Data collator for axial attention models.
    Expects examples as dicts with:
      - input_ids: List[List[int]] (ragged, n_haps x n_snps)
      - distances: List[List[int]] (ragged, n_haps x n_snps)
    Pads both axes to max in batch.
    """
    def __call__(self, examples):
        # Find max n_haps and n_snps in batch
        max_n_snps = max([len(ex["token_ids"][i]["input_ids"]) for ex in examples for i in range(len(ex["token_ids"]))])
        def pad_2d(list_of_lists, pad_value, dictkey="input_ids"):
            # Pad each haplotype to max_n_snps
            if isinstance(list_of_lists[0], dict):
                list_of_lists = [list_of_lists[i][dictkey] for i in range(len(list_of_lists))]
            padded = []
            for hap_list in list_of_lists:
                padded.append(hap_list + [pad_value] * (max_n_snps - len(hap_list)))
                # print("padding ", max_n_snps - len(hap_list))
            # print(padded)
            return padded

        batch_input_ids = []
        batch_distances = []
        batch_attention_masks = []

        for idx, ex in enumerate(examples):
            input_ids_2d = pad_2d(ex["token_ids"], self.tokenizer.pad_token_id, dictkey="input_ids")
            attention_masks_2d = pad_2d(ex["token_ids"], 0, dictkey="attention_mask")
            distances_2d = pad_2d(ex["distances"], 0)
            batch_input_ids.append(input_ids_2d)
            batch_attention_masks.append(attention_masks_2d)
            batch_distances.append([])

            for dist in distances_2d:
                dist_matrix = torch.zeros(max_n_snps, max_n_snps)
                
                # Build cumulative distance matrix
                cumulative_dists = torch.cumsum(torch.tensor([0.0] + ex["distances"][0] + [0.0]), dim=0)
                for i in range(max_n_snps):
                    for j in range(max_n_snps):
                        dist_matrix[i, j] = abs(cumulative_dists[i] - cumulative_dists[j])

                batch_distances[idx].append(dist_matrix)
            
            batch_distances[idx] = torch.stack(batch_distances[idx])

        # Convert to tensors: (batch, n_haps, n_snps)
        input_ids = torch.tensor(batch_input_ids, dtype=torch.long)
        distances = torch.stack(batch_distances)
        batch_attention_masks = torch.tensor(batch_attention_masks, dtype=torch.long)

        # Flatten for MLM masking, then reshape back
        flat_input_ids = input_ids.view(input_ids.size(0), -1)
        masked = self.torch_mask_tokens(flat_input_ids)
        masked_input_ids = masked[0].view_as(input_ids)
        labels = masked[1].view_as(input_ids)

        for i, bd in enumerate(batch_distances):
            if bd is None or (isinstance(bd, list) and len(bd) == 0):
                raise ValueError(f"Bad batch_distances at index {i}: {bd}")


        return {
            "input_ids": masked_input_ids,
            "attention_mask": batch_attention_masks,
            "distances": distances,
            "labels": labels,
        }


class HaploAxialNormalDataCollator(DataCollatorForLanguageModeling):
    """
    Data collator for axial attention models.
    Expects examples as dicts with:
      - input_ids: List[List[int]] (ragged, n_haps x n_snps)
      - distances: List[List[int]] (ragged, n_haps x n_snps)
    Pads both axes to max in batch.
    """
    def __call__(self, examples):
        # Find max n_haps and n_snps in batch
        max_n_snps = max([len(ex["token_ids"][i]["input_ids"]) for ex in examples for i in range(len(ex["token_ids"]))])
        def pad_2d(list_of_lists, pad_value, dictkey="input_ids"):
            # Pad each haplotype to max_n_snps
            if isinstance(list_of_lists[0], dict):
                list_of_lists = [list_of_lists[i][dictkey] for i in range(len(list_of_lists))]
            padded = []
            for hap_list in list_of_lists:
                padded.append(hap_list + [pad_value] * (max_n_snps - len(hap_list)))
                # print("padding ", max_n_snps - len(hap_list))
            # print(padded)
            return padded

        batch_input_ids = []
        batch_distances = []
        batch_attention_masks = []
        batch_labels = []

        for idx, ex in enumerate(examples):
            input_ids_2d = pad_2d(ex["token_ids"], self.tokenizer.pad_token_id, dictkey="input_ids")
            attention_masks_2d = pad_2d(ex["token_ids"], 0, dictkey="attention_mask")
            distances_2d = pad_2d(ex["distances"], 0)
            batch_input_ids.append(input_ids_2d)
            batch_attention_masks.append(attention_masks_2d)
            batch_labels.append(ex["label"])            
            batch_distances.append([])

            for dist in distances_2d:
                dist_matrix = torch.zeros(max_n_snps, max_n_snps)
                
                # Build cumulative distance matrix
                cumulative_dists = torch.cumsum(torch.tensor([0.0] + ex["distances"][0] + [0.0]), dim=0)
                for i in range(max_n_snps):
                    for j in range(max_n_snps):
                        dist_matrix[i, j] = abs(cumulative_dists[i] - cumulative_dists[j])

                batch_distances[idx].append(dist_matrix)
            
            batch_distances[idx] = torch.stack(batch_distances[idx])

        # Convert to tensors: (batch, n_haps, n_snps)
        input_ids = torch.tensor(batch_input_ids, dtype=torch.long)
        distances = torch.stack(batch_distances)
        batch_attention_masks = torch.tensor(batch_attention_masks, dtype=torch.long)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": batch_attention_masks,
            "distances": distances,
            "labels": batch_labels
        }


class HaploSimpleDataCollator(DataCollatorForLanguageModeling):
    """
    Data collator for axial attention models.
    Expects examples as dicts with:
      - input_ids: List[List[int]] (ragged, n_haps x n_snps)
      - distances: List[List[int]] (ragged, n_haps x n_snps)
    Pads both axes to max in batch.
    """
    def _torch_mask_tokens(self, inputs):
        """
        Custom masking: oversample 1s for masking.
        """
        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        # Oversample 1s: set higher probability for tokens == 1
        probability_matrix[inputs == 1] = 0.5  # e.g., 50% chance for 1s
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # Only compute loss on masked tokens

        # 80% of the time, replace masked input tokens with [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 1.0)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, replace masked input tokens with random word
        # indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        # random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        # inputs[indices_random] = random_words[indices_random]

        # The rest 10% keep original

        return inputs, labels

    def __call__(self, examples):
        # Find max n_haps and n_snps in batch
        max_n_snps = max([len(ex["token_ids"][0]["input_ids"]) for ex in examples])
        def get_2d(list_of_lists, dictkey="input_ids"):
            list_of_lists = [list_of_lists[i][dictkey] for i in range(len(list_of_lists))]
            return list_of_lists
            

        batch_input_ids = []
        batch_attention_masks = []
        batch_distances = []

        for idx, ex in enumerate(examples):
            input_ids_2d = get_2d(ex["token_ids"], dictkey="input_ids")
            attention_masks_2d = get_2d(ex["token_ids"], dictkey="attention_mask")
            
            # add a haplotype thats just <s> <pad> * max_n_snps </s>
            # haplotype = [self.tokenizer.bos_token_id] + [self.tokenizer.pad_token_id] * (max_n_snps - 2) + [self.tokenizer.eos_token_id]
            # input_ids_2d.insert(0, haplotype)
            # attention_masks_2d.insert(0, [1] + [0] * (max_n_snps - 2) + [1])

            # add to batch
            batch_input_ids.append(input_ids_2d)
            batch_attention_masks.append(attention_masks_2d)
            batch_distances.append([])

            dist_matrix = torch.zeros(max_n_snps, max_n_snps)
            
            # Build cumulative distance matrix
            cumulative_dists = torch.cumsum(torch.tensor(ex["distances"][0]), dim=0)
            for i in range(max_n_snps):
                for j in range(i + 1):
                    dist_matrix[i, j] = abs(cumulative_dists[i] - cumulative_dists[j])
                    dist_matrix[j, i] = dist_matrix[i, j]

            batch_distances[idx].append(dist_matrix)
            
            batch_distances[idx] = torch.stack(batch_distances[idx])

        # Convert to tensors: (batch, n_haps, n_snps)
        input_ids = torch.tensor(batch_input_ids, dtype=torch.long)
        distances = torch.stack(batch_distances)
        batch_attention_masks = torch.tensor(batch_attention_masks, dtype=torch.long)

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


class HaploSimpleNormalDataCollator(DataCollatorForLanguageModeling):
    """
    Data collator for axial attention models.
    Expects examples as dicts with:
      - input_ids: List[List[int]] (n_haps x n_snps)
      - distances: List[List[int]] (n_haps x n_snps)
    Pads both axes to max in batch.
    """
    def __call__(self, examples):
        # Find max n_haps and n_snps in batch
        max_n_snps = max([len(ex["token_ids"][0]["input_ids"]) for ex in examples])
        def get_2d(list_of_lists, dictkey="input_ids"):
            list_of_lists = [list_of_lists[i][dictkey] for i in range(len(list_of_lists))]
            return list_of_lists


        batch_input_ids = []
        batch_attention_masks = []
        batch_distances = []
        batch_labels = []

        for idx, ex in enumerate(examples):
            input_ids_2d = get_2d(ex["token_ids"], dictkey="input_ids")
            attention_masks_2d = get_2d(ex["token_ids"], dictkey="attention_mask")

            # add a haplotype thats just <s> <pad> * max_n_snps </s>
            # haplotype = [self.tokenizer.bos_token_id] + [self.tokenizer.pad_token_id] * (max_n_snps - 2) + [self.tokenizer.eos_token_id]
            # input_ids_2d.insert(0, haplotype)
            # attention_masks_2d.insert(0, [1] + [0] * (max_n_snps - 2) + [1])

            batch_input_ids.append(input_ids_2d)
            batch_attention_masks.append(attention_masks_2d)
            batch_distances.append([])
            batch_labels.append(ex["label"])

            dist_matrix = torch.zeros(max_n_snps, max_n_snps)
            
            # Build cumulative distance matrix
            cumulative_dists = torch.cumsum(torch.tensor(ex["distances"][0]), dim=0)
            for i in range(max_n_snps):
                for j in range(i + 1):
                    dist_matrix[i, j] = abs(cumulative_dists[i] - cumulative_dists[j])
                    dist_matrix[j, i] = dist_matrix[i, j]

            batch_distances[idx].append(dist_matrix)
            
            batch_distances[idx] = torch.stack(batch_distances[idx])

        # Convert to tensors: (batch, n_haps, n_snps)
        input_ids = torch.tensor(batch_input_ids, dtype=torch.long)
        distances = torch.stack(batch_distances)
        batch_attention_masks = torch.tensor(batch_attention_masks, dtype=torch.long)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": batch_attention_masks,
            "distances": distances,
            "labels": batch_labels,
        }