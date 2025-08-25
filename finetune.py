import numpy as np
from models import HapbertaForSequenceClassification
from transformers import AutoTokenizer, TrainingArguments, Trainer
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score
from collators import HaploSimpleNormalDataCollator

dataset = load_from_disk(f"dataset-CEU/tokenizedft")

# Split dataset
dataset = dataset.train_test_split(test_size=0.1)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

tokenizer = AutoTokenizer.from_pretrained("./hapberta2d_simple")

model = HapbertaForSequenceClassification.from_pretrained("./hapberta2d_simple",
                                                          classifier_dropout=0)

collator = HaploSimpleNormalDataCollator(tokenizer)

# training arguments
training_args = TrainingArguments(
    output_dir="./hapberta2d_simple-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=10,
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
    fp16=True,
    remove_unused_columns=False,
    learning_rate=1e-5,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

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
trainer.save_model("./hapberta2d_simple-finetuned")
tokenizer.save_pretrained("./hapberta2d_simple-finetuned")