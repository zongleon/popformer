import numpy as np
import torch
from transformers import RobertaConfig
from models import HapbertaForMaskedLM, HapbertaForSequenceClassification
from datasets import load_from_disk
from collators import HaploSimpleDataCollator
from scipy.spatial.distance import cdist

def test_model():
    print("=" * 30)
    print("Test: Show model")
    model = HapbertaForSequenceClassification.from_pretrained(
        "./models/hapberta2d_realsim/"
    )

    print(model)
def test_masked_lm():
    print("=" * 30)
    print("Test: Masked performance")
    # Load data
    model = HapbertaForMaskedLM.from_pretrained(
        "./models/hapberta2d/"
    )

    ds = load_from_disk("dataset2/tokenized")
    collator = HaploSimpleDataCollator()

    # make a batch
    inputs = collator([ds[0]])

    # print(inputs)

    # print masked haps and unmask token
    haps = inputs["input_ids"].numpy()

    print("Example input haps:")
    print(haps)

    print("Counts of masked tokens:")
    print({i: (inputs["labels"][haps == 4] == i).sum() for i in range(7)})

    # forward
    outputs = model(**inputs)

    # print the predicted haps
    # print(outputs["logits"].size())

    # print the count of predicted labels (vocab size 7)
    counts = outputs["logits"].argmax(dim=-1).cpu().numpy()
    print("Example predicted tokens:")
    print(counts)
    print("Counts of predicted tokens:")
    print({i: (counts[haps == 4] == i).sum() for i in range(7)})

    # print the real vs predicted mask labels
    lbls = inputs["labels"][haps == 4].numpy()
    predlbls = counts[haps == 4]

    print("Comparing first few tokens:")
    print(lbls[:20])
    print(predlbls[:20])
    print((lbls == predlbls).mean())

def test_mae():
    print("=" * 30)
    print("Test: MAE performance")
    # Load data
    config = RobertaConfig.from_pretrained(
        "./models/hapberta2d_mae/"
    )
    config.mae_mask = 0.0
    model = HapbertaForMaskedLM.from_pretrained(
        "./models/hapberta2d_mae/"
    )

    ds = load_from_disk("dataset2/tokenized")
    collator = HaploSimpleDataCollator(subsample=64, mlm_probability=0.0, whole_snp_mask_probability=0.5)

    # make a batch
    inputs = collator([ds[0]])

    # print(inputs)

    # print masked haps and unmask token
    haps = inputs["input_ids"].numpy()

    print("Example input haps:")
    print(haps)

    print("Counts of masked tokens:")
    print({i: (inputs["labels"][haps == 4] == i).sum() for i in range(7)})

    ids_keep = torch.tensor([np.where((haps[0] == 4).all(axis=0))[0]])
    ids_restore = torch.tensor([np.arange(haps.shape[2])])
    print(ids_keep)
    print(ids_restore)

    # forward
    outputs = model(inputs["input_ids"], inputs["distances"], inputs["attention_mask"], 
                    input_mask=(ids_keep, ids_restore))

    # print the predicted haps
    # print(outputs["logits"].size())

    # print the count of predicted labels (vocab size 7)
    counts = outputs["logits"].argmax(dim=-1).cpu().numpy()
    print("Example predicted tokens:")
    print(counts)
    print("Counts of predicted tokens:")
    print({i: (counts[haps == 4] == i).sum() for i in range(7)})

    # print the real vs predicted mask labels
    lbls = inputs["labels"][haps == 4].numpy()
    predlbls = counts[haps == 4]

    print("Comparing first few tokens:")
    print(lbls[:20])
    print(predlbls[:20])
    print((lbls == predlbls).mean())
def test_realsim_ft():
    print("=" * 30)
    print("Test: Finetuning on real/sim task")
    model = HapbertaForSequenceClassification.from_pretrained(
        "./models/hapberta2d_realsim/"
    )

    ds = load_from_disk("dataset/tokenizedrealsim").shuffle()

    collator = HaploSimpleNormalDataCollator()

    # make a batch
    inputs = collator([ds[i] for i in range(4)])

    haps = inputs["input_ids"].numpy()

    print("Example input haps:")
    print(haps[0])
    print("Labels: ", inputs["labels"].numpy())

    outputs = model(**inputs)
    preds = outputs["logits"].argmax(dim=-1).cpu().numpy()
    print("Output logits")
    print(outputs["logits"])
    print("Pred labels: ", preds)


def test_sel_ft():
    print("=" * 30)
    print("Test: Finetuning on selection task")
    model = HapbertaForSequenceClassification.from_pretrained(
        "./models/hapberta2d_sel/",
        num_labels=1,
    )

    ds = load_from_disk("dataset/tokenizedsel").shuffle()

    collator = HaploSimpleNormalDataCollator(label_dtype=torch.float32)

    # make a batch
    inputs = collator([ds[i] for i in range(4)])

    haps = inputs["input_ids"].numpy()

    print("Example input haps:")
    print(haps[0])
    print("Labels: ", inputs["labels"].numpy())

    outputs = model(**inputs)
    print("Output logits")
    print(outputs["logits"].detach())
    # print(preds)


def test_baseline():
    print("=" * 30)
    print("Test: Baseline column frequency approach")
    
    # Load data
    ds = load_from_disk("dataset2/tokenized")
    collator = HaploSimpleDataCollator()
    
    inputs = collator([ds[0]])
    haps = inputs["input_ids"].numpy()
    
    print("Example input haps:")
    print(haps)
    
    masked_positions = (haps == 4)
    baseline_predictions = np.copy(haps)
    
    for i in range(haps.shape[0]):  # batch dimension
        for hap in range(haps.shape[1]):  # sequence dimension
            for snp in range(haps.shape[2]):
                if haps[i, hap, snp] == 4:
                    cnts = np.bincount(haps[i, :, snp][haps[i, :, snp] != 4])
                    if cnts.shape[0] == 0:
                        # whole column was masked, predict 0
                        cnts = np.array([0])
                    baseline_predictions[i, hap, snp] = cnts.argmax()

    print("Baseline predicted tokens:")
    print(baseline_predictions)
    
    # Compare with ground truth
    lbls = inputs["labels"][masked_positions].numpy()
    pred_lbls = baseline_predictions[masked_positions]
    
    print("Comparing first few tokens:")
    print("True:     ", lbls[:20])
    print("Baseline: ", pred_lbls[:20])
    print("Baseline accuracy:", (lbls == pred_lbls).mean())
    
    print("Counts of true tokens:")
    print({i: (lbls == i).sum() for i in range(7)})
    print("Counts of baseline predicted tokens:")
    print({i: (pred_lbls == i).sum() for i in range(7)})


def test_baseline2():
    print("=" * 30)
    print("Test: Baseline nearest neighbor approach")
    
    # Load data
    ds = load_from_disk("dataset2/tokenized")
    collator = HaploSimpleDataCollator()
    
    inputs = collator([ds[0]])
    haps = inputs["input_ids"].numpy()
    
    print("Example input haps:")
    print(haps)
    
    masked_positions = (haps == 4)
    baseline_predictions = np.copy(haps)
    
    # get most similar sequences (excluding masks)
    for i in range(haps.shape[0]):  # batch dimension
        dists = cdist(haps[i, :, :], haps[i, :, :], metric="hamming")
        for hap in range(haps.shape[1]):  # sequence dimension
            for snp in range(haps.shape[2]):
                if haps[i, hap, snp] == 4:
                    # only copy if not masked
                    most_similar = dists[hap].argsort()
                    idx = 0
                    while baseline_predictions[i, hap, snp] == 4:
                        if idx == haps.shape[1]:
                            baseline_predictions[i, hap, snp] = 0
                            break
                        baseline_predictions[i, hap, snp] = haps[i, most_similar[idx], snp]
                        idx += 1

    print("Baseline predicted tokens:")
    print(baseline_predictions)
    
    # Compare with ground truth
    lbls = inputs["labels"][masked_positions].numpy()
    pred_lbls = baseline_predictions[masked_positions]
    
    print("Comparing first few tokens:")
    print("True:     ", lbls[:20])
    print("Baseline: ", pred_lbls[:20])
    print("Baseline accuracy:", (lbls == pred_lbls).mean())
    
    print("Counts of true tokens:")
    print({i: (lbls == i).sum() for i in range(7)})
    print("Counts of baseline predicted tokens:")
    print({i: (pred_lbls == i).sum() for i in range(7)})


if __name__ == "__main__":
    # test_model()
    # test_baseline()
    # test_baseline2()
    # test_masked_lm()
    test_mae()
    # test_realsim_ft()
    # test_sel_ft()

