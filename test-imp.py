import sys
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from models import HapbertaForMaskedLM
from collators import HaploSimpleDataCollator
from datasets import load_from_disk
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from cyvcf2 import VCF

def test_masked_lm(model_path, dataset):
    print("=" * 30)
    print("Test: Masked performance")
    # Load data
    model = HapbertaForMaskedLM.from_pretrained(
        model_path
    )

    ds = load_from_disk(dataset)
    collator = HaploSimpleDataCollator(subsample=None)

    # make a batch
    inputs = collator([ds[0]])

    # print(inputs)

    # print masked haps and unmask token
    haps = inputs["input_ids"].numpy()
    
    print("Counts of tokens:")
    print({i: (haps == i).sum() for i in range(7)})

    # forward
    outputs = model(inputs["input_ids"], inputs["distances"], inputs["attention_mask"])

    # print the count of predicted labels (vocab size 7)
    counts = outputs["logits"].argmax(dim=-1).cpu().numpy()
    print("Counts of predicted tokens:")
    print({i: (counts[haps == 4] == i).sum() for i in range(7)})

    # input_ids: (batch, haps, snps)
    ax0: Axes
    ax1: Axes
    # ax2: Axes
    fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(6, 10))
    
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

    # # Show ground truth: input_ids with masked id 4 replaced by labels
    # gt_img = haps.copy()
    # mask = (gt_img == 4)
    # gt_img[mask] = inputs["labels"][mask]
    # ax2.imshow(color(gt_img[0]), aspect='auto', cmap='Greys', interpolation="none")
    # ax2.set_title("ground truth")

    plt.savefig(f"figs/imp_{model_path.split("/")[-1]}_{dataset.split("/")[-1]}.png", dpi=300, bbox_inches="tight")


def test(model, dataset, save_preds_path=None):
    data = load_from_disk(dataset)

    model = HapbertaForMaskedLM.from_pretrained(
        model,
        torch_dtype=torch.float16
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    model.to(device)
    model.eval()

    collator = HaploSimpleDataCollator(subsample=None)

    loader = DataLoader(
        data,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
        collate_fn=collator,
    )
    preds = []

    with torch.inference_mode():
        for batch in tqdm(loader):
            # Move tensors to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=True)
            
            output = model(batch["input_ids"], 
                            batch["distances"], 
                            batch["attention_mask"])
            # only store output logits for masked positions
            # logits = output["logits"] 
            # mask = batch["input_ids"] == 4  # assuming 4 is the mask token id
            # preds.append(logits[mask].detach().cpu())
            preds.append(output["logits"].detach().cpu())

    # Concatenate all logits, move to CPU, convert to numpy
    preds = torch.cat(preds, dim=0).numpy()
    print(f"Predictions shape: {preds.shape}")

    np.save(save_preds_path, preds)


def compute_metrics(preds_path, dataset, labels_path):
    preds = np.load(preds_path)
    labels = pd.read_csv(labels_path)

    data = load_from_disk(dataset)
    positions = np.array(data["positions"]).flatten()
    
    data = np.array(data["input_ids"])
    data = data.transpose(1, 0, 2)
    data = data[:, :, 1:-1]
    data = data.reshape(data.shape[0], -1)

    mask = (data == 4).any(axis=0)

    positions = positions[mask]
    np.savetxt("testpos.out", positions, "%d")
    np.savetxt("testpos2.out", labels["pos"], "%d")

    # Find indices in labels["pos"] that are not in positions
    mask_labels = ~labels["pos"].isin(positions)
    labels = labels[~mask_labels].reset_index(drop=True)

    preds = preds[:, :, 1:-1, :]
    pred_labels = torch.softmax(torch.tensor(preds), dim=-1).numpy()
    pred_labels = pred_labels.transpose(1, 0, 2, 3)  # shape: (haps, batch, snps, 6)
    pred_labels = pred_labels.reshape(pred_labels.shape[0], -1, pred_labels.shape[-1])  # shape: (haps, batch*snps, 6)
    pred_labels = pred_labels[-len(labels["genotypes"].iloc[0]):, mask]
    print(pred_labels.shape)

    # print out first 10 preds and corresponding labels
    for i in range(10):
        true = labels["genotypes"].iloc[i]
        pred = pred_labels[:, i, :2].argmax(axis=-1).astype(str).tolist()
        print(f"True: {"".join([str(t) for t in true])}")
        print(f"Pred: {"".join(pred)}") #, Prob: {pred_labels[0, i, pred]:.4f}")

    # preprocess true labels
    true = labels["genotypes"].apply(lambda x: [int(c) for c in x]).tolist()
    true = np.array(true).T
    
    # Convert true and pred_labels (shape: n_haps, n_snps) to genotypes
    # Each consecutive pair of haps is summed to produce one genotype (0, 1, or 2)

    def haps_to_genotypes(haps):
        # haps: (n_haps, n_snps)
        # group every two haps and sum along axis 0
        return haps.reshape(-1, 2, haps.shape[1]).sum(axis=1)

    # Convert true haplotypes to genotypes
    true_genotypes = haps_to_genotypes(true)
    # Convert predicted haplotypes to genotypes
    pred_haps = pred_labels[:, :, :2].argmax(axis=-1)  # shape: (n_haps, n_snps)
    pred_genotypes = haps_to_genotypes(pred_haps)

    true_flat = true_genotypes.flatten()
    pred_flat = pred_genotypes.flatten()

    # np.savetxt("true_pred_flat.txt", np.vstack([true_flat, pred_flat]).T, fmt="%d", header="true\tpred")

    r, _ = pearsonr(true_flat, pred_flat)
    r2 = r ** 2

    # Compute error rate (fraction of mismatches)
    error_rate = (true_genotypes != pred_genotypes).mean()

    print(f"r: {r:.4f}")
    print(f"r^2: {r2:.4f}")
    print(f"Error rate: {error_rate:.4f}")


def test_impute():
    vcf = VCF("IMP/impute5.out.vcf.gz")
    imputeds = []
    for record in vcf:
        try:
            # was imputed
            record.INFO["IMP"]
        except KeyError:
            continue

        chrom = record.CHROM
        pos = record.POS
        gts = record.genotypes
        gts = "".join([str(gt[0]) + str(gt[1]) for gt in gts])
        imputeds.append(gts)

    labels = pd.read_csv("IMP/masked_snps.csv")
    true = labels["genotypes"].apply(lambda x: [int(c) for c in x]).tolist()
    true = np.array(true).T
    imps = pd.Series(imputeds).apply(lambda x: [int(c) for c in x]).tolist()
    imps = np.array(imps).T

    print(true.shape, imps.shape)
        
    true_flat = true.flatten()
    imps_flat = imps.flatten()

    # np.savetxt("true_pred_flat.txt", np.vstack([true_flat, pred_flat]).T, fmt="%d", header="true\tpred")

    r, _ = pearsonr(true_flat, imps_flat)
    r2 = r ** 2

    # Compute error rate (fraction of mismatches)
    error_rate = (true_flat != imps_flat).mean()

    print(f"r: {r:.4f}")
    print(f"r^2: {r2:.4f}")
    print(f"Error rate: {error_rate:.4f}")

if __name__ == "__main__":
    model = sys.argv[1]
    n_snps = sys.argv[2]
    test_masked_lm(model, f"IMP/infmasked_{n_snps}")
    test(model, f"IMP/infmasked_{n_snps}", f"IMP/preds_pt4_{n_snps}.npy")
    compute_metrics(f"IMP/preds_pt5_{n_snps}.npy", f"IMP/infmasked_{n_snps}", "IMP/masked_snps.csv")
    # test_impute()