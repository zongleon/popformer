import torch
from models import HapbertaForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from collators import HaploSimpleNormalDataCollator

MODE = "realsim"
# MODE = "sel"

if MODE == "realsim":
    dataset_path = "dataset/tokenizedrealsim"
    output_path = "./models/hapberta2d_realsim"
else:
    dataset_path = "dataset/tokenizedsel"
    output_path = "./models/hapberta2d_sel"

dataset = load_from_disk(dataset_path)

# Split dataset
dataset = dataset.train_test_split(test_size=0.1)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

model = HapbertaForSequenceClassification.from_pretrained("./models/hapberta2d",
                                                            classifier_dropout=0,
                                                            num_labels=2 if MODE == "realsim" else 1,
                                                            )

collator = HaploSimpleNormalDataCollator(label_dtype=torch.float32 if MODE == "sel" else torch.long)

# training arguments
training_args = TrainingArguments(
    output_dir=output_path,
    overwrite_output_dir=True,
    num_train_epochs=1,
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
    bf16=True,
    remove_unused_columns=False,
    learning_rate=1e-5,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if MODE == "realsim":
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

# Train
trainer.train()

# Save model and tokenizer
trainer.save_model(output_path)
