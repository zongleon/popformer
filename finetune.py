import torch
from models import HapbertaForSequenceClassification
from transformers import TrainingArguments, Trainer
import datasets
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, confusion_matrix
from collators import HaploSimpleNormalDataCollator

MODE = "pop"

if MODE == "realsim":
    dataset_path = "dataset/tokenizedrealsim"
    output_path = "./models/hapberta2d_realsim"
    num_labels = 2
    typ = torch.long
elif MODE == "sel":
    dataset_path = "dataset/tokenizedsel"
    output_path = "./models/hapberta2d_sel"
    num_labels = 1 # continuous
    typ = torch.float32
elif MODE == "sel2":
    dataset_path = "dataset/tokenizedsel2"
    output_path = "./models/hapberta2d_sel_binary"
    num_labels = 2
    typ = torch.long
elif MODE == "pop":
    dataset_path = "dataset2/tokenized"
    output_path = "./models/hapberta2d_pop"
    num_labels = 3
    typ = torch.long
else:
    raise ValueError("Incorrect mode selected")

dataset = load_from_disk(dataset_path)

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
dataset = dataset.train_test_split(test_size=0.1)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

model = HapbertaForSequenceClassification.from_pretrained("./models/hapberta2d",
                                                            classifier_dropout=0,
                                                            num_labels=num_labels,
)

collator = HaploSimpleNormalDataCollator(subsample=32, label_dtype=typ)

# training arguments
training_args = TrainingArguments(
    output_dir=output_path,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
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
    
    remove_unused_columns=False,
    learning_rate=1e-5,
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

