from transformers import RobertaConfig
from transformers import Trainer, TrainingArguments
from models import HapbertaForMaskedLM
from datasets import load_from_disk
from collators import HaploSimpleDataCollator
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Train HapbertaForMaskedLM model")
parser.add_argument("--dataset_path", type=str, default="", help="Path to tokenized dataset")
parser.add_argument("--mlm_probability", type=float, default=0.15, help="MLM probability")
parser.add_argument("--span_mask_probability", type=float, default=0, help="Span mask probability")
parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training and evaluation")
parser.add_argument("--output_path", type=str, default="./models/pt", help="Output path for model checkpoints")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")

args = parser.parse_args()

# load dataset
dataset = load_from_disk(args.dataset_path)

# Split dataset
dataset = dataset.train_test_split(test_size=0.05)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# model configuration
config = RobertaConfig(
    vocab_size=6,
    # hidden_size=768,
    hidden_size=512,
    num_hidden_layers=4,
    num_attention_heads=4,
    # intermediate_size=3072,
    intermediate_size=1024,
    max_position_embeddings=512,
    position_embedding_type="haplo",
    axial=True,
    bos_token_id=2,
    eos_token_id=3,
    pad_token_id=5,
)

# Create model for masked LM
model = HapbertaForMaskedLM(config)

# data collator
data_collator = HaploSimpleDataCollator(subsample=(32, 128),
                                        mlm_probability=args.mlm_probability,
                                        span_mask_probability=args.span_mask_probability)

# ex = data_collator([train_dataset[0]])
# print(ex["input_ids"][0])
# np.savetxt("test.txt", ex["input_ids"][0].cpu().numpy(), fmt="%d")

# training arguments
training_args = TrainingArguments(
    output_dir=args.output_path,
    overwrite_output_dir=True,
    num_train_epochs=args.num_epochs,
    # max_steps=50,
    # use_cpu=True,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    save_steps=500,
    eval_steps=100,
    eval_strategy="steps",
    save_strategy="steps",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    bf16=True,
    ddp_find_unused_parameters=False,
    remove_unused_columns=False,
    learning_rate=args.learning_rate
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
trainer.save_model(args.output_path)

