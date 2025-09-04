import torch
from models import HapbertaForSequenceClassification
from transformers import RobertaConfig, TrainingArguments, Trainer
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, confusion_matrix
from collators import HaploSimpleDataCollator

MODE = "sel2"

if MODE == "realsim":
    dataset_path = "dataset/tokenizedrealsim"
    output_path = "./models/hapberta2d_realsim"
    num_labels = 2
    typ = torch.long
elif MODE == "sel":
    dataset_path = "dataset2/tokenizedsel"
    output_path = "./models/hapberta2d_sel"
    num_labels = 1 # continuous
    typ = torch.float32
elif MODE == "sel2":
    dataset_path = "dataset2/tokenizedsel2"
    output_path = "./models/hapberta2d_sel_binary_from_init"
    num_labels = 2
    typ = torch.long
elif MODE == "pop":
    dataset_path = "dataset2/tokenized"
    output_path = "./models/hapberta2d_pop_from_init"
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

# model = HapbertaForSequenceClassification.from_pretrained("./models/hapberta2d",
#                                                             classifier_dropout=0,
#                                                             num_labels=num_labels,
# )
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

collator = HaploSimpleDataCollator(subsample=32, mlm_probability=0.0, label_dtype=typ)

# training arguments
training_args = TrainingArguments(
    output_dir=output_path,
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
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
    # torch_compile=True,
    ddp_find_unused_parameters=False,
    remove_unused_columns=False,
    learning_rate=1e-4,
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

