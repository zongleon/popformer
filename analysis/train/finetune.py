import torch
from popformer.models import PopformerForSNPClassification, PopformerForWindowClassification
from transformers import TrainingArguments, Trainer
from datasets import load_from_disk
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
)
from popformer.collators import HaploSimpleDataCollator
import argparse

parser = argparse.ArgumentParser(description="finetune")
parser.add_argument(
    "--mode", type=str, default="selbin", help="Mode: realsim/selbin/selreg/ancientx"
)
parser.add_argument(
    "--dataset_path", type=str, default="", help="Path to tokenized dataset"
)
parser.add_argument(
    "--num_epochs", type=int, default=5, help="Number of training epochs"
)
parser.add_argument(
    "--batch_size", type=int, default=16, help="Batch size for training and evaluation"
)
parser.add_argument(
    "--output_path", type=str, default="", help="Output path for model checkpoints"
)
parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
parser.add_argument(
    "--freeze_layers_up_to", type=int, default=0, help="Number of layers to freeze from the bottom"
)

args = parser.parse_args()

MODE = args.mode
dataset_path = args.dataset_path
output_path = args.output_path

model = PopformerForWindowClassification
if MODE == "realsim":
    num_labels = 2
    typ = torch.long
elif MODE == "selreg":
    num_labels = 1  # continuous
    typ = torch.float16
elif MODE == "selbin":
    num_labels = 2
    typ = torch.long
elif MODE == "pop":
    num_labels = 3
    typ = torch.long
elif MODE == "ancientx":
    num_labels = 1
    typ = torch.float16
    model = PopformerForSNPClassification
else:
    raise ValueError("Unknown mode selected")

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
# train_dataset = dataset.filter(lambda ex: ex["chrom"] != 21 and ex["chrom"] != 22)
# eval_dataset = dataset.filter(lambda ex: ex["chrom"] == 21)

# dataset = dataset.filter(lambda ex: ex["chrom"] % 2 != 0)
# dataset = dataset.train_test_split(0.1, shuffle=True)
# train_dataset = dataset["train"]
# eval_dataset = dataset["test"]

# test_dataset = dataset["test"].take(512)
dataset = dataset.train_test_split(0.1, shuffle=True)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]
print(f"Labels distribution in train: {train_dataset['label'].count(0)}, {train_dataset['label'].count(1)}")
print(f"Labels distribution in eval: {eval_dataset['label'].count(0)}, {eval_dataset['label'].count(1)}")
# eval_dataset = test_dataset

model = model.from_pretrained(
    "./models/old/popf-small",
    classifier_dropout=0,
    num_labels=num_labels,
    torch_dtype=torch.bfloat16,
)
if args.freeze_layers_up_to > 0:
    for param in model.roberta.embeddings.parameters():
        param.requires_grad = False
    for i in range(args.freeze_layers_up_to):
        # freeze from the bottom
        for param in model.roberta.encoder.layer[i].parameters():
            param.requires_grad = False

collator = HaploSimpleDataCollator(subsample=(32, 128), label_dtype=typ)

# training arguments
training_args = TrainingArguments(
    output_dir=output_path,
    overwrite_output_dir=True,
    num_train_epochs=args.num_epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    # gradient_accumulation_steps=4,
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    save_steps=100,
    eval_steps=100,
    eval_strategy="steps",
    save_strategy="steps",
    save_total_limit=4,
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
    if MODE != "sel" and MODE != "ancientx":
        preds = logits.argmax(axis=-1)
        if MODE == "pop":
            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, average="micro")
            matr = confusion_matrix(labels, preds)
            return {
                "accuracy": acc,
                "f1": f1,
                "cm_0": matr[0].tolist(),
                "cm_1": matr[1].tolist(),
                "cm_2": matr[2].tolist(),
            }
        else:
            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds)
            return {"accuracy": acc, "f1": f1}
    else:
        if MODE == "ancientx":
            logits, labels = logits.reshape(-1), labels.reshape(-1)
            mask = labels != -100
            logits = logits[mask]
            labels = labels[mask]
        mse = mean_squared_error(labels, logits)
        mae = mean_absolute_error(labels, logits)
        r2 = r2_score(labels, logits)
        return {"mse": mse, "mae": mae, "r2": r2}


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
