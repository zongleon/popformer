from transformers import RobertaConfig
from transformers import Trainer, TrainingArguments
from models import HapbertaForMaskedLM
from datasets import load_from_disk
from collators import HaploSimpleDataCollator
import numpy as np

OUTPUT = "./models/hapberta2d_mae"

# load dataset
dataset = load_from_disk("dataset2/tokenized")

# Split dataset
dataset = dataset.train_test_split(test_size=0.1)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

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
    mae=True,
    mae_mask=0.0,
    bos_token_id=2,
    eos_token_id=3,
    pad_token_id=4,
    mask_token_id=5,
)

# Create model for masked LM
model = HapbertaForMaskedLM(config)

# data collator
data_collator = HaploSimpleDataCollator(subsample=32,
                                        mlm_probability=0.0,
                                        whole_snp_mask_probability=0.6,
                                        return_input_mask=True)


ex = data_collator([train_dataset[0]])
np.savetxt("test.txt", ex["input_ids"][0].cpu().numpy(), fmt="%d")

# training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT,
    overwrite_output_dir=True,
    num_train_epochs=10,
    # max_steps=500,
    # use_cpu=True,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    save_steps=500,
    eval_steps=100,
    eval_strategy="steps",
    save_strategy="steps",
    save_total_limit=2,
    # load_best_model_at_end=True,
    # metric_for_best_model="eval_loss",
    # greater_is_better=False,
    # fp16=True,
    bf16=True,
    # torch_compile=True,
    ddp_find_unused_parameters=False,
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
trainer.save_model(OUTPUT)

