import os
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import theme
import torch
from cyvcf2 import VCF
from matplotlib.axes import Axes
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from tqdm import tqdm

from popformer.collators import HaploSimpleDataCollator
from popformer.dataset import Tokenizer, parse_files_imputation
from popformer.models import PopformerForMaskedLM

MAF_BINS = np.linspace(0, 0.5, 11)


def test_masked_lm(model_path, dataset):
    print("=" * 30)
    print("Test: Masked performance")
    # Load data
    model = PopformerForMaskedLM.from_pretrained(model_path)

    collator = HaploSimpleDataCollator(subsample=None)

    # make a batch
    inputs = collator([dataset[0]])

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

    # print the accuracy of the round-trip (accuracy of predicting non-masked tokens)
    accuracy = (counts[haps != 4] == haps[haps != 4]).mean()
    print(f"Round-trip accuracy: {accuracy:.4f}")

    # input_ids: (batch, haps, snps)
    ax0: Axes
    ax1: Axes
    # ax2: Axes
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(6, 20))

    def color(img):
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

    ax0.imshow(color(haps[0]), aspect="auto", interpolation="none")
    ax0.set_title("masked")
    ax0.set_ylabel("Haplotypes")

    pr_img = haps.copy()
    mask = pr_img == 4
    pr_img[mask] = counts[mask]
    ax1.imshow(color(pr_img[0]), aspect="auto", cmap="Greys", interpolation="none")
    ax1.set_title("predicted")

    roundtrip_img = counts.copy()
    ax2.imshow(
        color(roundtrip_img[0]), aspect="auto", cmap="Greys", interpolation="none"
    )
    ax2.set_title("round-trip")

    # # Show ground truth: input_ids with masked id 4 replaced by labels
    # gt_img = haps.copy()
    # mask = (gt_img == 4)
    # gt_img[mask] = inputs["labels"][mask]
    # ax2.imshow(color(gt_img[0]), aspect='auto', cmap='Greys', interpolation="none")
    # ax2.set_title("ground truth")

    plt.savefig("figs/imputation/example.pdf", dpi=300, bbox_inches="tight")


def test(model, dataset):
    model = PopformerForMaskedLM.from_pretrained(model, torch_dtype=torch.float16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    model.to(device)
    model.eval()

    collator = HaploSimpleDataCollator(subsample=None)

    loader = DataLoader(
        dataset,
        batch_size=4,
        collate_fn=collator,
    )
    preds = []

    with torch.inference_mode():
        for batch in tqdm(loader):
            # Move tensors to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=True)

            output = model(
                batch["input_ids"], batch["distances"], batch["attention_mask"]
            )
            preds.append(output["logits"].detach().cpu())

    # Concatenate all logits, move to CPU, convert to numpy
    # softmax too
    preds = torch.cat(preds, dim=0)
    preds = torch.softmax(preds, dim=-1).numpy()

    return preds


def test_baseline(dataset):
    collator = HaploSimpleDataCollator(subsample=None)

    loader = DataLoader(
        dataset,
        batch_size=4,
        collate_fn=collator,
    )
    preds = []

    for batch in tqdm(loader):
        input_ids = batch["input_ids"].numpy()  # (batch, haps, snps)
        batch_preds = np.zeros(
            (input_ids.shape[0], input_ids.shape[1], input_ids.shape[2], 7)
        )

        for i in range(input_ids.shape[0]):
            for hap in range(input_ids.shape[1]):
                for snp in range(input_ids.shape[2]):
                    if input_ids[i, hap, snp] == 4:
                        cnts = np.bincount(
                            input_ids[i, :, snp][input_ids[i, :, snp] != 4]
                        )
                        if cnts.shape[0] == 0:
                            predicted = 0
                        else:
                            predicted = cnts.argmax()
                        batch_preds[i, hap, snp, predicted] = (
                            1  # One-hot for predicted token
                        )

        preds.append(torch.tensor(batch_preds))

    preds = torch.cat(preds, dim=0).numpy()

    return preds


def test_baseline2(dataset):
    collator = HaploSimpleDataCollator(subsample=None)

    loader = DataLoader(
        dataset,
        batch_size=4,
        collate_fn=collator,
    )
    preds = []

    for batch in tqdm(loader):
        input_ids = batch["input_ids"].numpy()  # (batch, haps, snps)
        batch_preds = np.zeros(
            (input_ids.shape[0], input_ids.shape[1], input_ids.shape[2], 7)
        )

        for i in range(input_ids.shape[0]):
            # Find all masked positions upfront
            mask_matrix = input_ids[i] == 4

            # Skip if nothing is masked
            if not mask_matrix.any():
                continue

            # For each haplotype with masked positions
            haps_with_masks = np.where(mask_matrix.any(axis=1))[0]

            for hap_idx in haps_with_masks:
                # Get the masked SNP positions for this haplotype
                masked_snps = np.where(mask_matrix[hap_idx])[0]

                # Compute distance only between this haplotype and all others
                # Only on non-masked positions
                hap_data = input_ids[i, hap_idx].copy()
                other_data = input_ids[i].copy()

                # Create mask for valid comparison positions (non-masked in query)
                valid_positions = hap_data != 4

                # Compute distances only on valid positions
                # Count mismatches for each other haplotype
                distances = np.zeros(input_ids.shape[1])
                for j in range(input_ids.shape[1]):
                    if j == hap_idx:
                        distances[j] = np.inf
                        continue
                    # Only compare on positions that are not masked in the query haplotype
                    valid_mask = valid_positions & (other_data[j] != 4)
                    if valid_mask.sum() == 0:
                        distances[j] = np.inf
                    else:
                        distances[j] = (
                            hap_data[valid_mask] != other_data[j, valid_mask]
                        ).sum()

                # Sort neighbors by distance once
                nearest_neighbors = np.argsort(distances)

                # For each masked SNP in this haplotype, find nearest neighbor value
                for snp_idx in masked_snps:
                    predicted = 4
                    for neighbor_idx in nearest_neighbors:
                        if distances[neighbor_idx] == np.inf:
                            break
                        candidate = input_ids[i, neighbor_idx, snp_idx]
                        if candidate != 4:
                            predicted = candidate
                            break

                    if predicted == 4:
                        predicted = 0

                    batch_preds[i, hap_idx, snp_idx, predicted] = 1

        preds.append(torch.tensor(batch_preds))

    preds = torch.cat(preds, dim=0).numpy()
    return preds


def compute_metrics(preds, dataset, labels_path):
    labels = pd.read_csv(
        labels_path, dtype={"pos": int, "MAF": float, "genotypes": str}
    )

    positions = np.array(dataset["positions"]).flatten()
    flippeds = np.array(dataset["major_allele_flipped"]).flatten()

    data = np.array(dataset["input_ids"])
    data = data.transpose(1, 0, 2)
    data = data[:, :, 1:-1]
    preds = preds[:, :, 1:-1, :]
    data = data.reshape(data.shape[0], -1)

    mask = (data == 4).any(axis=0)

    rt_labels = data[:64, :]
    rt_preds = preds.transpose(1, 0, 2, 3)[:64]
    rt_preds = rt_preds.reshape(rt_preds.shape[0], -1, rt_preds.shape[-1])
    rt_preds = rt_preds.argmax(axis=-1)
    # rt_preds = rt_preds[:, :, 1]
    # have to calculate maf for rt_labels
    rt_maf = rt_labels.mean(axis=0)

    positions = positions[mask]
    flippeds = flippeds[mask]

    # Find indices in labels["pos"] that are not in positions
    mask_labels = ~labels["pos"].isin(positions)
    labels = labels[~mask_labels].reset_index(drop=True)

    pred_labels = preds.transpose(1, 0, 2, 3)  # shape: (haps, batch, snps, 6)
    pred_labels = pred_labels.reshape(
        pred_labels.shape[0], -1, pred_labels.shape[-1]
    )  # shape: (haps, batch*snps, 6)

    pred_labels = pred_labels[-len(labels["genotypes"].iloc[0]) :, mask]

    # preprocess true labels
    true = labels["genotypes"].apply(lambda x: [int(c) for c in x]).tolist()
    true = np.array(true).T

    # Convert true and pred_labels (shape: n_haps, n_snps) to genotypes
    # def haps_to_genotypes(haps):
    #     # haps: (n_haps, n_snps)
    #     # group every two haps and sum along axis 0
    #     return haps.reshape(-1, 2, haps.shape[1]).sum(axis=1)

    # Convert predicted haplotypes to genotypes
    pred_haps = pred_labels[:, :, :2].argmax(axis=-1)  # shape: (n_haps, n_snps)
    pred_hap_probs = pred_labels[:, :, 1]

    # flip predicted haplotypes where major allele was flipped
    for snp_idx in range(pred_haps.shape[1]):
        if flippeds[snp_idx]:
            pred_haps[:, snp_idx] = 1 - pred_haps[:, snp_idx]
            pred_hap_probs[:, snp_idx] = 1 - pred_hap_probs[:, snp_idx]

    r, _ = pearsonr(true.flatten(), pred_hap_probs.flatten())
    r2 = r**2

    # Compute error rate (fraction of mismatches)
    error_rate = (true != pred_haps).mean()

    # Binned by MAF
    binned_results = []
    for i in range(len(MAF_BINS) - 1):
        bin_mask = (labels["MAF"] >= MAF_BINS[i]) & (labels["MAF"] < MAF_BINS[i + 1])
        if bin_mask.sum() == 0:
            continue
        true_bin = true[:, bin_mask]
        pred_bin = pred_haps[:, bin_mask]
        probs_bin = pred_hap_probs[:, bin_mask]
        if len(true_bin) == 0:
            continue
        r_bin, _ = pearsonr(true_bin.flatten(), probs_bin.flatten())
        r2_bin = r_bin**2

        error_rate_bin = (true_bin != pred_bin).mean()

        # TODO roundtrip accuracy
        rt_maf_mask = (rt_maf > MAF_BINS[i]) & (rt_maf < MAF_BINS[i + 1])
        roundtrip_accuracy_bin = (
            rt_labels[:, rt_maf_mask] == rt_preds[:, rt_maf_mask]
        ).mean()

        binned_results.append(
            (MAF_BINS[i], r_bin, r2_bin, error_rate_bin, roundtrip_accuracy_bin)
        )

    results = (r, r2, error_rate)

    return results, binned_results


def test_impute(vcf_path, labels_path):
    vcf = VCF(vcf_path)
    imputeds, probs = [], []
    for record in vcf:
        try:
            # was imputed
            record.INFO["IMP"]
        except KeyError:
            continue

        gts = record.genotypes
        gts = "".join([str(gt[0]) + str(gt[1]) for gt in gts])
        imputeds.append(gts)

        prob = record.format("AP").flatten()
        probs.append(prob)

    labels = pd.read_csv(
        labels_path, dtype={"pos": int, "MAF": float, "genotypes": str}
    )
    true = labels["genotypes"].apply(lambda x: [int(c) for c in x]).tolist()
    true = np.array(true).T
    imps = pd.Series(imputeds).apply(lambda x: [int(c) for c in x]).tolist()
    imps = np.array(imps).T
    probs = np.array(probs).T

    r, _ = pearsonr(true.flatten(), probs.flatten())
    r2 = r**2

    # Compute error rate (fraction of mismatches)
    error_rate = (true != imps).mean()

    # Binned by MAF
    binned_results = []
    for i in range(len(MAF_BINS) - 1):
        bin_mask = (labels["MAF"] >= MAF_BINS[i]) & (labels["MAF"] < MAF_BINS[i + 1])
        if bin_mask.sum() == 0:
            continue
        true_bin = true[:, bin_mask]
        imps_bin = imps[:, bin_mask]
        probs_bin = probs[:, bin_mask]
        if len(true_bin) == 0:
            continue
        r_bin, _ = pearsonr(true_bin.flatten(), probs_bin.flatten())
        r2_bin = r_bin**2
        error_rate_bin = (true_bin != imps_bin).mean()
        binned_results.append((MAF_BINS[i], r_bin, r2_bin, error_rate_bin, None))

    results = (r, r2, error_rate)

    return results, binned_results


def run(seeds, mask_ratios):
    # Store results: {mask_ratio: {method: [(r, r2, err, time), ...]}}
    plot_data = []
    maf_plot_data = []
    tokenizer = Tokenizer(max_haps=256, num_snps=512, major_minor_flip=False)

    # Run predictions
    model = "models/popf-small"

    for mr in mask_ratios:
        for seed in seeds:
            print(f"\nSeed: {seed}")

            ref_vcf = f"data/imputation/masked/KHV_{mr}_{seed}_ref.h5"
            tgt_vcf = f"data/imputation/masked/KHV_{mr}_{seed}_tgt.h5"
            labels_path = f"data/imputation/masked/KHV_{mr}_{seed}_snps.csv"
            dataset = parse_files_imputation(ref_vcf, tgt_vcf, tokenizer)

            start = time.time()
            model_preds = test(model, dataset)
            model_time = time.time() - start

            start = time.time()
            baseline1_preds = test_baseline(dataset)
            baseline1_time = time.time() - start

            start = time.time()
            baseline2_preds = test_baseline2(dataset)
            baseline2_time = time.time() - start

            start = time.time()
            out_vcf = f"data/imputation/imputed/KHV_{mr}_{seed}.vcf.gz"

            subprocess.run(
                ["bcftools", "index", "-f", ref_vcf.replace(".h5", ".vcf.gz")],
                check=True,
            )
            subprocess.run(
                ["bcftools", "index", "-f", tgt_vcf.replace(".h5", ".vcf.gz")],
                check=True,
            )
            subprocess.run(
                [
                    "./analysis/scripts/impute5/impute5_v1.2.0_static",
                    "--h",
                    ref_vcf.replace(".h5", ".vcf.gz"),
                    "--g",
                    tgt_vcf.replace(".h5", ".vcf.gz"),
                    "--r",
                    "20:30000-63000000",
                    "--buffer-region",
                    "20:0-63500000",
                    "--out-ap-field",
                    "--o",
                    out_vcf,
                ],
                check=True,
            )

            impute_time = time.time() - start

            # Compute metrics
            results, model_binned = compute_metrics(model_preds, dataset, labels_path)
            baseline1_results, baseline1_binned = compute_metrics(
                baseline1_preds, dataset, labels_path
            )
            baseline2_results, baseline2_binned = compute_metrics(
                baseline2_preds, dataset, labels_path
            )
            impute_results, impute_binned = test_impute(out_vcf, labels_path)

            for res, bin_res, t, method in zip(
                [results, impute_results, baseline1_results, baseline2_results],
                [model_binned, impute_binned, baseline1_binned, baseline2_binned],
                [model_time, impute_time, baseline1_time, baseline2_time],
                [
                    "popformer",
                    "impute5",
                    "column freq baseline",
                    "nearest neighbor baseline",
                ],
            ):
                r, r2, err = res
                plot_data.append(
                    {
                        "Method": method,
                        "Mask Ratio": int(mr),
                        "Seed": int(seed),
                        "r": r,
                        "r2": r2,
                        "Error Rate": err,
                        "Time": t,
                    }
                )

                for bin in bin_res:
                    maf_bin, r_bin, r2_bin, err_bin, rt_bin = bin
                    maf_plot_data.append(
                        {
                            "Method": method,
                            "Mask Ratio": int(mr),
                            "Seed": int(seed),
                            "MAF Bin": maf_bin,
                            "r": r_bin,
                            "r2": r2_bin,
                            "Error Rate": err_bin,
                            "Roundtrip Accuracy": rt_bin,
                        }
                    )

    return plot_data, maf_plot_data


if __name__ == "__main__":
    RUN = os.environ.get("RUN") is not None
    EXAMPLE = os.environ.get("EXAMPLE") is not None
    os.makedirs("figs/imputation", exist_ok=True)
    # Define seeds and mask ratios to test
    seeds = [0, 1, 2]
    mask_ratios = reversed([20, 40, 60, 80])
    maf_summary_path = Path("imputation_results_maf_summary.csv")

    if EXAMPLE:
        model = "models/popf-small"
        ref_vcf = "data/imputation/masked/KHV_80_0_ref.h5"
        tgt_vcf = "data/imputation/masked/KHV_80_0_tgt.h5"
        labels_path = "data/imputation/masked/KHV_80_0_snps.csv"
        tokenizer = Tokenizer(max_haps=256, num_snps=512, major_minor_flip=False)
        dataset = parse_files_imputation(ref_vcf, tgt_vcf, tokenizer)
        test_masked_lm(model, dataset)

    if RUN:
        plot_data, maf_plot_data = run(seeds, mask_ratios)

        df_plot = pd.DataFrame(plot_data)
        df_plot.to_csv("imputation_results_summary.csv", index=False)

        df_maf_plot = pd.DataFrame(maf_plot_data)
        df_maf_plot.to_csv(maf_summary_path, index=False)
    else:
        df_plot = pd.read_csv("imputation_results_summary.csv")
        if maf_summary_path.exists():
            df_maf_plot = pd.read_csv(maf_summary_path)
        else:
            df_maf_plot = pd.DataFrame()
            print(
                "Warning: imputation_results_maf_summary.csv not found, skipping MAF-binned plots."
            )

    for metric in ["Error Rate", "r2", "Time"]:
        df_plot = df_plot[df_plot["Method"] != "column freq baseline"]
        df_plot["Method"] = df_plot["Method"].replace(
            {
                "impute5": "IMPUTE5",
                "nearest neighbor baseline": "Nearest Neighbor",
                "column freq baseline": "Column Frequency",
            },
        )
        plt.figure(figsize=(8, 6))
        sns.pointplot(
            data=df_plot,
            x="Mask Ratio",
            y=metric,
            hue="Method",
            palette=theme.model_color_map,
            errorbar=("sd"),
        )
        if metric == "Error Rate":
            plt.ylim(0, 0.1)
        if metric == "r2":
            plt.ylim(0.5, 1)
        plt.tight_layout()
        plt.savefig(f"figs/imputation/{metric.replace(' ', '_').lower()}.pdf", dpi=300)
        plt.close()

    if not df_maf_plot.empty:
        df_maf_plot = df_maf_plot[
            df_maf_plot["Method"] != "column freq baseline"
        ].copy()
        df_maf_plot["Method"] = df_maf_plot["Method"].replace(
            {
                "impute5": "IMPUTE5",
                "nearest neighbor baseline": "Nearest Neighbor",
                "column freq baseline": "Column Frequency",
            },
        )

        df_maf_plot = df_maf_plot[df_maf_plot["Mask Ratio"] == 80].copy()

        for metric in ["Error Rate", "r2", "Roundtrip Accuracy"]:
            plot_df = df_maf_plot.copy()
            if metric == "Roundtrip Accuracy":
                plot_df = df_maf_plot[df_maf_plot["Method"] == "popformer"]
            plt.figure(figsize=(8, 6))
            sns.pointplot(
                data=plot_df,
                x="MAF Bin",
                y=metric,
                hue="Method",
                # err_style="bars",
                palette=theme.model_color_map,
                errorbar="sd",
            )
            if metric == "Error Rate":
                plt.ylim(0, 0.2)
            elif metric == "Roundtrip Accuracy":
                pass
            else:
                plt.ylim(0.0, 1)

            plt.tight_layout()
            plt.savefig(
                f"figs/imputation/{metric.replace(' ', '_').lower()}_vs_maf_mask80.pdf",
                dpi=300,
            )
            plt.close()

    # Print summary tables
    print(f"\n{'=' * 80}")
    print("SUMMARY: Mean ± Std across seeds")
    print(f"{'=' * 80}\n")
    print(
        "{:<30} {:>15} {:>15} {:>18} {:>15}".format(
            "Method", "r", "r^2", "Error rate", "Runtime (s)"
        )
    )
    print("-" * 95)

    for mr in mask_ratios:
        print(f"\nMask Ratio: {mr}%")
        mr_results = df_plot[df_plot["Mask Ratio"] == mr]
        for method_name in mr_results["Method"].unique():
            method_results = mr_results[mr_results["Method"] == method_name]

            aggregated = method_results.agg(
                {
                    "r": ["mean", "std"],
                    "r2": ["mean", "std"],
                    "Error Rate": ["mean", "std"],
                    "Time": ["mean", "std"],
                }
            )

            print(
                "{:<30} {:>15} {:>15} {:>18} {:>15}".format(
                    method_name,
                    f"{aggregated['r']['mean']:.4f}±{aggregated['r']['std']:.4f}",
                    f"{aggregated['r2']['mean']:.4f}±{aggregated['r2']['std']:.4f}",
                    f"{aggregated['Error Rate']['mean']:.4f}±{aggregated['Error Rate']['std']:.4f}",
                    f"{aggregated['Time']['mean']:.2f}±{aggregated['Time']['std']:.2f}",
                )
            )
