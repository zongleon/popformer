import numpy as np
import torch
from models import HapbertaForMaskedLM, HapbertaForSequenceClassification
from datasets import load_from_disk
from collators import HaploSimpleDataCollator, HaploSimpleNormalDataCollator


def test_masked_lm():
    model = HapbertaForMaskedLM.from_pretrained(
        "./models/hapberta2d/"
    )

    ds = load_from_disk("dataset/tokenized")
    collator = HaploSimpleDataCollator()

    # make a batch
    inputs = collator([ds[0]])

    # print(inputs)

    # print masked haps and unmask token
    haps = inputs["input_ids"].numpy()

    print(haps)
    print({i: (inputs["labels"][haps == 4] == i).sum() for i in range(7)})

    # forward
    outputs = model(**inputs)

    # print the predicted haps
    # print(outputs["logits"].size())

    # print the count of predicted labels (vocab size 7)
    counts = outputs["logits"].argmax(dim=-1).cpu().numpy()
    print(counts)
    print({i: (counts == i).sum() for i in range(7)})

    # print the real vs predicted mask labels
    lbls = inputs["labels"][haps == 4].numpy()
    predlbls = counts[haps == 4]

    print(lbls[:20])
    print(predlbls[:20])
    print((lbls == predlbls).mean())

def test_realsim_ft():
    model = HapbertaForSequenceClassification.from_pretrained(
        "./models/hapberta2d_realsim/"
    )

    ds = load_from_disk("dataset/tokenizedrealsim").shuffle()

    collator = HaploSimpleNormalDataCollator()

    # make a batch
    inputs = collator([ds[i] for i in range(4)])

    haps = inputs["input_ids"].numpy()

    print(haps[0])
    print("Labels: ", inputs["labels"].numpy())

    outputs = model(**inputs)
    preds = outputs["logits"].argmax(dim=-1).cpu().numpy()
    print(outputs["logits"])
    print(preds)


def test_sel_ft():
    model = HapbertaForSequenceClassification.from_pretrained(
        "./models/hapberta2d_sel/",
        num_labels=1,
    )

    ds = load_from_disk("dataset/tokenizedsel").shuffle()

    collator = HaploSimpleNormalDataCollator(label_dtype=torch.float32)

    # make a batch
    inputs = collator([ds[i] for i in range(4)])

    haps = inputs["input_ids"].numpy()

    print(haps[0])
    print("Labels: ", inputs["labels"].numpy())

    outputs = model(**inputs)
    print(outputs["logits"].detach())
    # print(preds)


if __name__ == "__main__":
    # test_masked_lm()
    # test_realsim_ft()
    test_sel_ft()