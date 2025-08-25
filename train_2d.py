from transformers import RobertaConfig
from transformers import Trainer, TrainingArguments
from models import HapbertaForMaskedLM
from datasets import load_from_disk
from collators import HaploSimpleDataCollator

# load dataset
dataset = load_from_disk("dataset/tokenized")

# Split dataset
dataset = dataset.train_test_split(test_size=0.1)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

print(train_dataset)

# model configuration
config = RobertaConfig(
    vocab_size=6,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    max_position_embeddings=512,
    position_embedding_type="haplo",
    axial=True,
    bos_token_id=2,
    eos_token_id=3,
    pad_token_id=4,
)

# Create model for masked LM
model = HapbertaForMaskedLM(config)

print(model)

# data collator
data_collator = HaploSimpleDataCollator()

# training arguments
training_args = TrainingArguments(
    output_dir="./models/hapberta2d",
    overwrite_output_dir=True,
    num_train_epochs=3,
    # max_steps=50,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    save_steps=500,
    eval_steps=500,
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
trainer.save_model("./models/hapberta2d_simple")
