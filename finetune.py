import torch
from models import HapbertaForSequenceClassification
from transformers import RobertaConfig, TrainingArguments, Trainer
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, confusion_matrix
from collators import HaploSimpleDataCollator
import argparse

parser = argparse.ArgumentParser(description="finetune")
parser.add_argument("--mode", type=str, default="sel2", help="Mode: realsim/sel/sel2/pop")
parser.add_argument("--dataset_path", type=str, default="", help="Path to tokenized dataset")
parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training and evaluation")
parser.add_argument("--output_path", type=str, default="", help="Output path for model checkpoints")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--from_init", action="store_true", help="Whether to train from scratch")

args = parser.parse_args()

MODE = args.mode
dataset_path = args.dataset_path
output_path = args.output_path

if MODE == "realsim":
    num_labels = 2
    typ = torch.long
elif MODE == "sel":
    num_labels = 1 # continuous
    typ = torch.float32
elif MODE == "sel2":
    num_labels = 2
    typ = torch.long
elif MODE == "pop":
    num_labels = 3
    typ = torch.long
else:
    raise ValueError("Incorrect mode selected")

dataset = load_from_disk(dataset_path).shuffle(42)

if MODE == "pop":
    # dataset = dataset.shuffle().select(range(10000))

    lbl2id = {
        "CEU": 0,
        "CHB": 1,
        "YRI": 2,
    }
    def process_pop(example):
        example["label"] = lbl2id[example["pop"]]
        return example

    dataset = dataset.map(process_pop, keep_in_memory=True)

# Split dataset
train_dataset = dataset["train"]
eval_dataset = dataset["test"].take(1024)

if args.from_init:
    model = HapbertaForSequenceClassification(RobertaConfig(
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
        pad_token_id=5,
        num_labels=num_labels,
        classifier_dropout=0
    ))
else:
    model = HapbertaForSequenceClassification.from_pretrained("./models/pt",
                                                                classifier_dropout=0,
                                                                num_labels=num_labels,
    )

collator = HaploSimpleDataCollator(subsample=32, label_dtype=typ)

# training arguments
training_args = TrainingArguments(
    output_dir=output_path,
    overwrite_output_dir=True,
    num_train_epochs=args.num_epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
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
    bf16=True,
    ddp_find_unused_parameters=False,
    remove_unused_columns=False,
    learning_rate=args.learning_rate,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if MODE != "sel":
        preds = logits.argmax(axis=-1)
        if MODE == "pop":
            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, average="micro")
            matr = confusion_matrix(labels, preds)
            return {"accuracy": acc, "f1": f1, 
                    "cm_0": matr[0].tolist(), "cm_1": matr[1].tolist(), "cm_2": matr[2].tolist()}
        else:
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

# Train
trainer.train()

# Save model and tokenizer
trainer.save_model(output_path)

