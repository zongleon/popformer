import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer
from models import HapbertaForMaskedLM, HapbertaForSequenceClassification
from datasets import load_from_disk
from collators import HaploSimpleDataCollator, HaploSimpleNormalDataCollator

################################################
tokenizer = AutoTokenizer.from_pretrained("./hapberta2d_simple")
model = HapbertaForSequenceClassification.from_pretrained(
    "./hapberta2d_simple-finetuned/"
)
model.eval()

# model  = HapbertaForMaskedLM(AutoConfig.from_pretrained("./hapberta2d_simple/"))

ds = load_from_disk("dataset-CEU/tokenizedft")
collator = HaploSimpleNormalDataCollator(tokenizer)

# make a batch
for i in range(10):
    inputs = collator([ds[4 * i + j] for j in range(4)])

    # print masked haps and unmask token
    haps = inputs["input_ids"].numpy()

    # print(haps)
    # print({i: haps[haps == int(i)].shape[0] for i in range(0, 7)})

    # forward
    with torch.no_grad():
        outputs = model(**inputs)
        # outputs = model.roberta(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], 
        #                         distances=inputs["distances"])
        # print(outputs[0].mean(), outputs[0].std())

    # print the predicted haps
        print(inputs["labels"])
        print(outputs["logits"].argmax(-1))

    # print mean and sd of logits
        # logits = outputs["logits"].detach().numpy()
        # print(logits.mean())
        # print(logits.std())

    # print(model.classifier.dense.weight.mean())
    # print(model.classifier.dense.weight.std())
    # print(model.classifier.dense.bias)
