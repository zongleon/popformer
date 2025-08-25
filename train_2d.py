from transformers import RobertaConfig, RobertaTokenizerFast
from transformers import Trainer, TrainingArguments
from models import HapbertaForMaskedLM
from datasets import load_from_disk
from collators import HaploAxialDataCollator, HaploSimpleDataCollator
from transformers import TrainerCallback
import torch

# tokenizer, and add special tokens
tokenizer = RobertaTokenizerFast(vocab_file="tokenizer_simple/vocab.json", 
                                 merges_file="tokenizer_simple/merges.txt")

# load dataset
dataset = load_from_disk("dataset-CEU/tokenized_simple")

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
    axial=True,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)

# Create model for masked LM
model = HapbertaForMaskedLM(config)

print(model)

# data collator
data_collator = HaploSimpleDataCollator(tokenizer, mlm_probability=0.15)

# test data collator
# print(data_collator([train_dataset[0], train_dataset[1]]))

# test training args (on cpu)
# training_args = TrainingArguments(
#     output_dir="./hapberta2d",
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
    output_dir="./hapberta2d_simple",
    overwrite_output_dir=True,
    num_train_epochs=10,
    # max_steps=50,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=100,
    eval_steps=100,
    eval_strategy="steps",
    save_strategy="steps",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    dataloader_pin_memory=True,
    dataloader_num_workers=4,
    # fp16=True,
    bf16=True,
    remove_unused_columns=False,
    learning_rate=1e-4
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
trainer.save_model("./hapberta2d_simple")
tokenizer.save_pretrained("./hapberta2d_simple/")
