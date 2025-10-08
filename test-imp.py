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
from scipy.spatial.distance import cdist
import subprocess
import time

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

    plt.savefig(f"figs/imp_{model_path.split('/')[-1]}_{dataset.split('/')[-1]}.png", dpi=300, bbox_inches="tight")


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

    np.save(save_preds_path, preds)


def test_baseline(dataset, save_preds_path):
    ds = load_from_disk(dataset)
    collator = HaploSimpleDataCollator(subsample=None)

    loader = DataLoader(
        ds,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
        collate_fn=collator,
    )
    preds = []

    for batch in tqdm(loader):
        input_ids = batch["input_ids"].numpy()  # (batch, haps, snps)
        batch_preds = np.zeros((input_ids.shape[0], input_ids.shape[1], input_ids.shape[2], 7))

        for i in range(input_ids.shape[0]):
            for hap in range(input_ids.shape[1]):
                for snp in range(input_ids.shape[2]):
                    if input_ids[i, hap, snp] == 4:
                        cnts = np.bincount(input_ids[i, :, snp][input_ids[i, :, snp] != 4])
                        if cnts.shape[0] == 0:
                            predicted = 0
                        else:
                            predicted = cnts.argmax()
                        batch_preds[i, hap, snp, predicted] = 1  # One-hot for predicted token

        preds.append(torch.tensor(batch_preds))

    preds = torch.cat(preds, dim=0).numpy()
    np.save(save_preds_path, preds)


def test_baseline2(dataset, save_preds_path):
    ds = load_from_disk(dataset)
    collator = HaploSimpleDataCollator(subsample=None)

    loader = DataLoader(
        ds,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
        collate_fn=collator,
    )
    preds = []

    for batch in tqdm(loader):
        input_ids = batch["input_ids"].numpy()  # (batch, haps, snps)
        batch_preds = np.zeros((input_ids.shape[0], input_ids.shape[1], input_ids.shape[2], 7))

        for i in range(input_ids.shape[0]):
            dists = cdist(input_ids[i, :, :], input_ids[i, :, :], metric="hamming")
            for hap in range(input_ids.shape[1]):
                for snp in range(input_ids.shape[2]):
                    if input_ids[i, hap, snp] == 4:
                        most_similar = dists[hap].argsort()
                        idx = 0
                        predicted = 4
                        while predicted == 4:
                            if idx == input_ids.shape[1]:
                                predicted = 0
                                break
                            predicted = input_ids[i, most_similar[idx], snp]
                            idx += 1
                        batch_preds[i, hap, snp, predicted] = 1  # One-hot for predicted token

        preds.append(torch.tensor(batch_preds))

    preds = torch.cat(preds, dim=0).numpy()
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

    # Binned by MAF
    bins = [0, 0.05, 0.1, 0.2, 0.5, 1.0]
    bin_labels = ["<0.05", "0.05-0.1", "0.1-0.2", "0.2-0.5", ">=0.5"]
    binned_results = []
    for i in range(len(bins) - 1):
        bin_mask = (labels["MAF"] >= bins[i]) & (labels["MAF"] < bins[i + 1])
        if bin_mask.sum() == 0:
            continue
        true_bin = true_genotypes[:, bin_mask]
        pred_bin = pred_genotypes[:, bin_mask]
        true_flat_bin = true_bin.flatten()
        pred_flat_bin = pred_bin.flatten()
        if len(true_flat_bin) == 0:
            continue
        r_bin, _ = pearsonr(true_flat_bin, pred_flat_bin)
        r2_bin = r_bin ** 2
        error_rate_bin = (true_bin != pred_bin).mean()
        binned_results.append((bin_labels[i], r_bin, r2_bin, error_rate_bin))

    return r, r2, error_rate, binned_results


def test_impute(labels_path):
    vcf = VCF("IMP/preds_impute5.vcf.gz")
    imputeds = []
    for record in vcf:
        try:
            # was imputed
            record.INFO["IMP"]
        except KeyError:
            continue

        gts = record.genotypes
        gts = "".join([str(gt[0]) + str(gt[1]) for gt in gts])
        imputeds.append(gts)

    labels = pd.read_csv(labels_path)
    true = labels["genotypes"].apply(lambda x: [int(c) for c in x]).tolist()
    true = np.array(true).T
    imps = pd.Series(imputeds).apply(lambda x: [int(c) for c in x]).tolist()
    imps = np.array(imps).T

    true_flat = true.flatten()
    imps_flat = imps.flatten()

    print(true, imps)
            
    # np.savetxt("true_pred_flat.txt", np.vstack([true_flat, pred_flat]).T, fmt="%d", header="true\tpred")

    r, _ = pearsonr(true_flat, imps_flat)
    r2 = r ** 2

    # Compute error rate (fraction of mismatches)
    error_rate = (true_flat != imps_flat).mean()

    # Binned by MAF
    bins = [0, 0.05, 0.1, 0.2, 0.5, 1.0]
    bin_labels = ["<0.05", "0.05-0.1", "0.1-0.2", "0.2-0.5", ">=0.5"]
    binned_results = []
    for i in range(len(bins) - 1):
        bin_mask = (labels["MAF"] >= bins[i]) & (labels["MAF"] < bins[i + 1])
        if bin_mask.sum() == 0:
            continue
        true_bin = true[:, bin_mask]
        imps_bin = imps[:, bin_mask]
        true_flat_bin = true_bin.flatten()
        imps_flat_bin = imps_bin.flatten()
        if len(true_flat_bin) == 0:
            continue
        r_bin, _ = pearsonr(true_flat_bin, imps_flat_bin)
        r2_bin = r_bin ** 2
        error_rate_bin = (true_flat_bin != imps_flat_bin).mean()
        binned_results.append((bin_labels[i], r_bin, r2_bin, error_rate_bin))

    return r, r2, error_rate, binned_results


if __name__ == "__main__":
    model = sys.argv[1]
    dataset = "IMP/KHV.chr20.64.256"
    labels_path = "IMP/KHV.chr20.64_snps.csv"
    model_preds_path = "IMP/preds_pt_64_256.npy"
    baseline1_preds_path = "IMP/preds_baseline1_64_256.npy"
    baseline2_preds_path = "IMP/preds_baseline2_64_256.npy"

    test_masked_lm(model, dataset)

    start = time.time()
    test(model, dataset, model_preds_path)
    model_time = time.time() - start

    start = time.time()
    test_baseline(dataset, baseline1_preds_path)
    baseline1_time = time.time() - start

    start = time.time()
    test_baseline2(dataset, baseline2_preds_path)
    baseline2_time = time.time() - start

    start = time.time()
    subprocess.run(["bcftools", "index", "-f", "IMP/KHV.chr20.64_ref.vcf.gz"], check=True)
    subprocess.run(["bcftools", "index", "-f", "IMP/KHV.chr20.64_tgt.vcf.gz"], check=True)
    subprocess.run(["./IMP/impute5/impute5_v1.2.0_static", 
                    "--h", "IMP/KHV.chr20.64_ref.vcf.gz", 
                    "--g", "IMP/KHV.chr20.64_tgt.vcf.gz", 
                    "--r", "20:30000-63000000", 
                    "--buffer-region", "20:0-63500000", 
                    "--o", "IMP/preds_impute5.vcf.gz"], check=True)
    impute_time = time.time() - start

    impute_r, impute_r2, impute_err, impute_binned = test_impute(labels_path)

    model_r, model_r2, model_err, model_binned = compute_metrics(model_preds_path, dataset, labels_path)
    baseline1_r, baseline1_r2, baseline1_err, baseline1_binned = compute_metrics(baseline1_preds_path, dataset, labels_path)
    baseline2_r, baseline2_r2, baseline2_err, baseline2_binned = compute_metrics(baseline2_preds_path, dataset, labels_path)

    # Print a nice table
    table = [
        ["popformer", f"{model_r:.4f}", f"{model_r2:.4f}", f"{model_err:.4f}", f"{model_time:.2f}s"],
        ["impute5", f"{impute_r:.4f}", f"{impute_r2:.4f}", f"{impute_err:.4f}", f"{impute_time:.2f}s"],
        ["column freq baseline", f"{baseline1_r:.4f}", f"{baseline1_r2:.4f}", f"{baseline1_err:.4f}", f"{baseline1_time:.2f}s"],
        ["nearest neighbor baseline", f"{baseline2_r:.4f}", f"{baseline2_r2:.4f}", f"{baseline2_err:.4f}", f"{baseline2_time:.2f}s"],
    ]

    print("{:<30} {:>8} {:>8} {:>10} {:>6}".format("Method", "r", "r^2", "Error rate", "Runtime"))
    print("-" * 70)
    for row in table:
        print("{:<30} {:>8} {:>8} {:>10} {:>6}".format(*row))

    # Print binned tables
    # methods = ["popformer", "impute5", "column freq baseline", "nearest neighbor baseline"]
    # binned_all = [model_binned, impute_binned, baseline1_binned, baseline2_binned]
    # bin_labels = ["<0.05", "0.05-0.1", "0.1-0.2", "0.2-0.5", ">=0.5"]
    # for i, bin_label in enumerate(bin_labels):
    #     print(f"\nBin: {bin_label}")
    #     table_bin = []
    #     for method, binned in zip(methods, binned_all):
    #         if i < len(binned) and binned[i][0] == bin_label:
    #             _, r, r2, err = binned[i]
    #             table_bin.append([method, f"{r:.4f}", f"{r2:.4f}", f"{err:.4f}"])
    #         else:
    #             table_bin.append([method, "N/A", "N/A", "N/A"])
    #     print("{:<30} {:>8} {:>8} {:>10}".format("Method", "r", "r^2", "Error rate"))
    #     print("-" * 60)
    #     for row in table_bin:
    #         print("{:<30} {:>8} {:>8} {:>10}".format(*row))
