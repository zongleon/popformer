import torch
from models import HapbertaForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from collators import HaploSimpleNormalDataCollator

# MODE = "realsim"
MODE = "sel"
MODE = "sel2"

if MODE == "realsim":
    dataset_path = "dataset/tokenizedrealsim"
    output_path = "./models/hapberta2d_realsim"
elif MODE == "sel":
    dataset_path = "dataset/tokenizedsel"
    output_path = "./models/hapberta2d_sel"
else:
    dataset_path = "dataset/tokenizedsel2"
    output_path = "./models/hapberta2d_sel_binary"

dataset = load_from_disk(dataset_path)

# Split dataset
dataset = dataset.train_test_split(test_size=0.1)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

model = HapbertaForSequenceClassification.from_pretrained("./models/hapberta2d",
                                                            classifier_dropout=0,
                                                            num_labels=1 if MODE == "sel" else 2,
                                                            )

collator = HaploSimpleNormalDataCollator(label_dtype=torch.float32 if MODE == "sel" else torch.long)

# training arguments
training_args = TrainingArguments(
    output_dir=output_path,
    overwrite_output_dir=True,
    num_train_epochs=20,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
>>>>>>> f4ed8db (parcc training updates)
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
    bf16=True,
    remove_unused_columns=False,
    learning_rate=3e-5,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if MODE != "sel":
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

