import numpy as np
from transformers import AutoConfig, AutoTokenizer
from models import HapbertaForMaskedLM, HapbertaForSequenceClassification
from datasets import load_from_disk
from collators import HaploSimpleDataCollator, HaploSimpleNormalDataCollator


################################################
tokenizer = AutoTokenizer.from_pretrained("./hapberta2d_simple")
model = HapbertaForMaskedLM.from_pretrained(
    "./hapberta2d_simple/"
)

# model  = HapbertaForMaskedLM(AutoConfig.from_pretrained("./hapberta2d_simple/"))

ds = load_from_disk("dataset-CEU/tokenized_simple")
collator = HaploSimpleDataCollator(tokenizer)

# make a batch
inputs = collator([ds[0]])

# print masked haps and unmask token
haps = inputs["input_ids"].numpy()

print(haps)
print({i: (inputs["labels"][haps == tokenizer.mask_token_id] == i).sum() for i in range(7)})


# forward
outputs = model(**inputs)

# print the predicted haps
# print(outputs["logits"].size())

# print the count of predicted labels (vocab size 7)
counts = outputs["logits"].argmax(dim=-1).cpu().numpy()
print(counts)
print({i: (counts == i).sum() for i in range(7)})

# print the real vs predicted mask labels
lbls = inputs["labels"][haps == tokenizer.mask_token_id].numpy()
predlbls = counts[haps == tokenizer.mask_token_id]

print(lbls)
print(predlbls)
print((lbls == predlbls).mean())
