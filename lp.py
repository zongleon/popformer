import torch
from transformers import Trainer, TrainingArguments
from models import HapbertaForSequenceClassification
from datasets import load_from_disk
from collators import HaploSimpleDataCollator
import argparse
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error

parser = argparse.ArgumentParser(description="Linear probe for sel2 classification")
parser.add_argument("--mode", type=str, default="sel2", help="Mode: realsim/sel/sel2/pop")
parser.add_argument("--dataset_path", type=str, default="./dataset4/ft_selbin_fixwindow_tkns", help="Path to tokenized dataset")
parser.add_argument("--output_path", type=str, default="./models/lp_sel_bin", help="Output path for model checkpoints")
parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training and evaluation")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for classifier head")
args = parser.parse_args()

if args.mode == "sel":
    num_labels = 1 # continuous
    typ = torch.float32
elif args.mode == "sel2":
    num_labels = 2
    typ = torch.long

# Load dataset
dataset = load_from_disk(args.dataset_path)
dataset = dataset.shuffle()
train_dataset = dataset["train"]
eval_dataset = dataset["test"].take(512)

# Load pretrained model
model = HapbertaForSequenceClassification.from_pretrained(
    "./models/pt",
    num_labels=num_labels,
    classifier_dropout=0
)

# Freeze all layers except classification head
for name, param in model.named_parameters():
    if "classifier" not in name:
        param.requires_grad = False

collator = HaploSimpleDataCollator(subsample=32, label_dtype=typ)

training_args = TrainingArguments(
    output_dir=args.output_path,
    overwrite_output_dir=True,
    num_train_epochs=args.num_epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    save_steps=100,
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
    learning_rate=args.learning_rate,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if args.mode != "sel":
        preds = logits.argmax(axis=-1)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds)
        return {"accuracy": acc, "f1": f1}
    else:
        mse = mean_squared_error(labels, logits)
        mae = mean_absolute_error(labels, logits)
        return {"mse": mse, "mae": mae}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=collator,
)

trainer.train()
trainer.save_model(args.output_path)