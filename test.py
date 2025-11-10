import math
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import torch
from popformer.models import PopformerForMaskedLM, PopformerForWindowClassification
from datasets import load_from_disk
from popformer.collators import HaploSimpleDataCollator
from scipy.spatial.distance import cdist
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

def test_model():
    print("=" * 30)
    print("Test: Show model")
    model = PopformerForWindowClassification.from_pretrained(
        "./models/hapberta2d_realsim/"
    )

    print(model)
def test_masked_lm(mpath, mlm, snpmlm, spanmlm):
    print("=" * 30)
    print("Test: Masked performance")
    # Load data
    model = PopformerForMaskedLM.from_pretrained(
        mpath
    )

    ds = load_from_disk("dataset4/pt_snpwindow_tkns")
    collator = HaploSimpleDataCollator(subsample=32, 
                                       mlm_probability=mlm,
                                       whole_snp_mask_probability=snpmlm,
                                       span_mask_probability=spanmlm)

    # make a batch
    inputs = collator([ds[0]])

    # print(inputs)

    # print masked haps and unmask token
    haps = inputs["input_ids"].numpy()

    print("Counts of masked tokens:")
    print({i: (inputs["labels"][haps == 4] == i).sum() for i in range(7)})

    # forward
    outputs = model(**inputs)

    # print the count of predicted labels (vocab size 7)
    counts = outputs["logits"].argmax(dim=-1).cpu().numpy()
    print("Counts of predicted tokens:")
    print({i: (counts[haps == 4] == i).sum() for i in range(7)})

    # print the real vs predicted mask labels
    lbls = inputs["labels"][haps == 4].numpy()
    print("Classweights: ", compute_class_weight('balanced', classes=np.unique(lbls), y=lbls))
    predlbls = counts[haps == 4]

    print("Comparing first few tokens:")
    print(lbls[:40])
    print(predlbls[:40])
    print("Accuracy: ", (lbls == predlbls).mean())

    # input_ids: (batch, haps, snps)
    ax0: Axes
    ax1: Axes
    ax2: Axes
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(6, 10))
    
    def color(img):
        # img = img[:50]
        # Create a color image: 0->white, 1->black, 4->red
        color_img = np.stack([img, img, img], axis=-1).astype(float)
        # Set all to white
        color_img[:] = 1.0
        # Set 0 to white, 1 to black, 4 to red
        color_img[img == 0] = [1, 1, 1]
        color_img[img == 1] = [0, 0, 0]
        color_img[img == 2] = [0, 1, 0]
        color_img[img == 3] = [0, 0, 1]
        color_img[img == 4] = [1, 0, 0]
        color_img[img == 5] = [0, 0, 0]
        return color_img

    ax0.imshow(color(haps[0]), aspect='auto', interpolation="none")
    ax0.set_title("masked")
    ax0.set_ylabel("Haplotypes")

    pr_img = haps.copy()
    mask = (pr_img == 4)
    pr_img[mask] = counts[mask]
    ax1.imshow(color(pr_img[0]), aspect='auto', cmap='Greys', interpolation="none")
    ax1.set_title("predicted")

    # Show ground truth: input_ids with masked id 4 replaced by labels
    gt_img = haps.copy()
    mask = (gt_img == 4)
    gt_img[mask] = inputs["labels"][mask]
    ax2.imshow(color(gt_img[0]), aspect='auto', cmap='Greys', interpolation="none")
    ax2.set_title("ground truth")

    plt.savefig("figs/ex_maskedrecreation_new.png", dpi=300, bbox_inches="tight")


def test_baseline():
    print("=" * 30)
    print("Test: Baseline column frequency approach")
    
    # Load data
    ds = load_from_disk("dataset4/pt_snpwindow_tkns")
    collator = HaploSimpleDataCollator(mlm_probability=0.15, span_mask_probability=0.15)
    
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
    ds = load_from_disk("dataset4/pt_snpwindow_tkns")
    collator = HaploSimpleDataCollator(mlm_probability=0.15, span_mask_probability=0.15)
    
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

def test_distance():
    # Embedding table for relative position biases
    lin_buckets = torch.linspace(
        0,
        50000,# math.log10(50000), 
        32 - 1
    ).cpu().numpy()
    log_buckets = torch.logspace(
        0,
        math.log10(50000), 
        32 - 1
    ).cpu().numpy()
    print(log_buckets)
    print(lin_buckets)

    ds = load_from_disk("dataset2/tokenized")
    collator = HaploSimpleDataCollator()
    inputs = collator([ds[i] for i in range(5)])
    distances = inputs["distances"].numpy()
    # Assign distances to buckets
    np.savetxt("test_distances.out", distances[0], fmt="%05d")
    
    bucket_indices = np.digitize(distances, lin_buckets).flatten()
    plt.figure(figsize=(8, 4))
    plt.hist(bucket_indices, bins=np.arange(len(lin_buckets)+2)-0.5, edgecolor='black')
    plt.xlabel("Bucket index")
    plt.ylabel("Count")
    plt.title("Histogram of pairwise distances in log buckets")
    plt.tight_layout()
    plt.savefig("figs/distance_histogram_lin.png", dpi=300)

def test_selmodel(train=True):

    if train:
        data = load_from_disk("dataset4/ft_selbin_snpwindow64_tkns")
    else:
        data = load_from_disk("GHIST/ghist_samples_singlesweep_64")
    data = data.shuffle()

    model = PopformerForWindowClassification.from_pretrained(
        "models/ft_sel_bin_long",
        torch_dtype=torch.bfloat16,
        num_labels=2
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    model.to(device)
    model.eval()
    # model.compile()

    collator = HaploSimpleDataCollator(subsample=32)

    batch_size = 32
    preds = []
    lbls = []

    with torch.no_grad():
        for i in tqdm(range(0, 1000, batch_size)):
            batch_samples = [data[j] for j in range(i, min(i + batch_size, len(data)))]
            batch = collator(batch_samples)
            # Move tensors to device
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device=device)
            
            output = model(batch["input_ids"], batch["distances"], batch["attention_mask"])
            preds.append(output["logits"])
            if train:
                lbls = lbls + [smp["label"] for smp in batch_samples]

    all_preds = torch.cat(preds, dim=0).to(torch.float16).cpu().numpy().squeeze()
    plt.figure(figsize=(8, 4))
    if train:
        lbls_np = np.array(lbls)
        for label in np.unique(lbls_np):
            plt.hist(all_preds[lbls_np == label, 1], bins=30, alpha=0.5, label=f"Label {label}")
    else:
        plt.hist(all_preds[:, 1], bins=30)
    plt.legend()
    plt.xlabel("Predicted logits")
    plt.ylabel("Count")
    plt.title("Histogram of predicted logits")
    plt.savefig("figs/preds_histogram.png", dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    # test_model()
    test_baseline()
    test_baseline2()
    test_masked_lm("models/pt", 0.15, 0., 0.15)
    # test_realsim_ft()
    # test_selmodel(False)

    # test_distance()

