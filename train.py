from tokenizers import Tokenizer
from transformers import RobertaConfig, RobertaTokenizerFast
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from models import HapbertaForMaskedLM, RelativePosAttnBias, HapbertaAttention
from datasets import load_from_disk
import torch
import numpy as np

def replace_attention_with_genomic_bias(model, bias_module_cls, **bias_kwargs):
    """Replace all RobertaAttention layers with bias-augmented version."""
    for layer in model.encoder.layer:
        old_attn = layer.attention
        new_attn = HapbertaAttention(
            model.config,
            bias_module_cls(**bias_kwargs),
        )
        # copy weights from old attention so we don't lose initialization
        new_attn.load_state_dict(old_attn.state_dict(), strict=False)
        layer.attention = new_attn
    return model

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
                cumulative_dists = torch.cumsum(torch.tensor([0.0] + [0.0] + dists + [0.0]), dim=0)
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


# tokenizer, and add special tokens
tokenizer = RobertaTokenizerFast(vocab_file="tokenizer/vocab.json", 
                                 merges_file="tokenizer/merges.txt")

# load dataset
dataset = load_from_disk("dataset-CEU/tokenized")

# Split dataset
dataset = dataset.train_test_split(test_size=0.1)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

print(train_dataset)

# model configuration
config = RobertaConfig(
    vocab_size=tokenizer.vocab_size + 5,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    max_position_embeddings=512,
    position_embedding_type="haplo",
)

# Create model for masked LM
model = HapbertaForMaskedLM(config)

# Remove absolute position embeddings
model.roberta.embeddings.position_embeddings = None

# Replace attention layers with genomic bias
model.roberta = replace_attention_with_genomic_bias(
    model.roberta,
    bias_module_cls=RelativePosAttnBias,
    num_buckets=32,
    max_distance=50000,
    num_heads=config.num_attention_heads,
)

print(model)

# data collator
data_collator = HaploDataCollator(tokenizer, mlm_probability=0.15)

# test data collator
# print(data_collator([train_dataset[0], train_dataset[1]]))

# test training args (on cpu)
# training_args = TrainingArguments(
#     output_dir="./hapberta",
#     overwrite_output_dir=True,
#     num_train_epochs=1,  # Single epoch for testing
#     per_device_train_batch_size=2,  # Small batch size for CPU
#     per_device_eval_batch_size=2,
#     max_steps=10,  # Limit steps for quick testing
#     warmup_steps=2,
#     weight_decay=0.01,
#     logging_dir="./logs",
#     logging_steps=2,  # Log frequently for testing
#     save_steps=5,
#     eval_steps=5,
#     eval_strategy="steps",
#     save_strategy="steps",
#     load_best_model_at_end=False,  # Skip for testing
#     dataloader_pin_memory=False,  # Disable for CPU
#     dataloader_num_workers=0,  # Single threaded for CPU
#     use_cpu=True,  # Force CPU usage
#     remove_unused_columns=False,
#     report_to="none",  # Disable wandb/tensorboard for testing
# )

# training arguments
training_args = TrainingArguments(
    output_dir="./hapberta",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    save_steps=1000,
    eval_steps=1000,
    eval_strategy="steps",
    save_strategy="steps",
    save_total_limit=4,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    dataloader_pin_memory=True,
    dataloader_num_workers=4,
    fp16=True,
    remove_unused_columns=False,
)

# trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

# train the model
trainer.train()

# save the final model
trainer.save_model("./hapberta")
tokenizer.save_pretrained("./hapberta/")
