from matplotlib.axes import Axes
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from popformer.models import PopformerForMaskedLM
from popformer.collators import HaploSimpleDataCollator
from popformer.dataset import parse_files_imputation, Tokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from cyvcf2 import VCF
import subprocess
import time
import seaborn as sns

plt.style.use("seaborn-v0_8-talk")

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

    ax0.imshow(color(haps[0]), aspect="auto", interpolation="none")
    ax0.set_title("masked")
    ax0.set_ylabel("Haplotypes")

    pr_img = haps.copy()
    mask = pr_img == 4
    pr_img[mask] = counts[mask]
    ax1.imshow(color(pr_img[0]), aspect="auto", cmap="Greys", interpolation="none")
    ax1.set_title("predicted")

    # # Show ground truth: input_ids with masked id 4 replaced by labels
    # gt_img = haps.copy()
    # mask = (gt_img == 4)
    # gt_img[mask] = inputs["labels"][mask]
    # ax2.imshow(color(gt_img[0]), aspect='auto', cmap='Greys', interpolation="none")
    # ax2.set_title("ground truth")

    plt.savefig("figs/imp_testimp.png", dpi=300, bbox_inches="tight")


def test(model, dataset):
    model = PopformerForMaskedLM.from_pretrained(model, torch_dtype=torch.float16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    model.to(device)
    model.eval()

    collator = HaploSimpleDataCollator(subsample=None)

    # np savetxt on dataset[0]["input_ids"]
    # np.savetxt("test_input_ids.txt", dataset[0]["input_ids"], fmt="%d")
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
            # only store output logits for masked positions
            # logits = output["logits"]
            # mask = batch["input_ids"] == 4  # assuming 4 is the mask token id
            # preds.append(logits[mask].detach().cpu())
            preds.append(output["logits"].detach().cpu())

    # Concatenate all logits, move to CPU, convert to numpy
    preds = torch.cat(preds, dim=0).numpy()

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
    data = data.reshape(data.shape[0], -1)

    mask = (data == 4).any(axis=0)

    positions = positions[mask]
    flippeds = flippeds[mask]
    # np.savetxt("testpos.out", positions, "%d")
    # np.savetxt("testpos2.out", labels["pos"], "%d")

    # Find indices in labels["pos"] that are not in positions
    mask_labels = ~labels["pos"].isin(positions)
    labels = labels[~mask_labels].reset_index(drop=True)

    preds = preds[:, :, 1:-1, :]
    pred_labels = torch.softmax(torch.tensor(preds), dim=-1).numpy()
    pred_labels = pred_labels.transpose(1, 0, 2, 3)  # shape: (haps, batch, snps, 6)
    pred_labels = pred_labels.reshape(
        pred_labels.shape[0], -1, pred_labels.shape[-1]
    )  # shape: (haps, batch*snps, 6)
    pred_labels = pred_labels[-len(labels["genotypes"].iloc[0]) :, mask]

    # print out first 10 preds and corresponding labels
    print()
    for i in range(10):
        true = labels["genotypes"].iloc[i]
        pred = pred_labels[:, i, :2].argmax(axis=-1).astype(str).tolist()
        print(f"True: {''.join([str(t) for t in true])}")
        print(f"Pred: {''.join(pred)}")  # , Prob: {pred_labels[0, i, pred]:.4f}")

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

    # flip predicted haplotypes where major allele was flipped
    for snp_idx in range(pred_haps.shape[1]):
        if flippeds[snp_idx]:
            pred_haps[:, snp_idx] = 1 - pred_haps[:, snp_idx]

    pred_genotypes = haps_to_genotypes(pred_haps)

    true_flat = true_genotypes.flatten()
    pred_flat = pred_genotypes.flatten()

    r, _ = pearsonr(true_flat, pred_flat)
    r2 = r**2

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
        r2_bin = r_bin**2
        error_rate_bin = (true_bin != pred_bin).mean()
        binned_results.append((bin_labels[i], r_bin, r2_bin, error_rate_bin))

    return r, r2, error_rate, binned_results


def test_impute(vcf_path, labels_path):
    vcf = VCF(vcf_path)
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

    labels = pd.read_csv(
        labels_path, dtype={"pos": int, "MAF": float, "genotypes": str}
    )
    true = labels["genotypes"].apply(lambda x: [int(c) for c in x]).tolist()
    true = np.array(true).T
    imps = pd.Series(imputeds).apply(lambda x: [int(c) for c in x]).tolist()
    imps = np.array(imps).T

    true_flat = true.flatten()
    imps_flat = imps.flatten()

    r, _ = pearsonr(true_flat, imps_flat)
    r2 = r**2

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
        r2_bin = r_bin**2
        error_rate_bin = (true_flat_bin != imps_flat_bin).mean()
        binned_results.append((bin_labels[i], r_bin, r2_bin, error_rate_bin))

    return r, r2, error_rate, binned_results


def run(seeds, mask_ratios, models):
    # Store results: {mask_ratio: {method: [(r, r2, err, time), ...]}}
    results = {
        mr: {
            "popformer-base": [],
            "impute5": [],
            "baseline1": [],
            "baseline2": [],
        }
        for mr in mask_ratios
    }

    for mr in mask_ratios:
        for seed in seeds:
            print(f"\nSeed: {seed}")

            ref_vcf = f"data/imputation/masked/KHV_{mr}_{seed}_ref.h5"
            tgt_vcf = f"data/imputation/masked/KHV_{mr}_{seed}_tgt.h5"
            labels_path = f"data/imputation/masked/KHV_{mr}_{seed}_snps.csv"
            tokenizer = Tokenizer(max_haps=256, num_snps=256)
            dataset = parse_files_imputation(ref_vcf, tgt_vcf, tokenizer)

            # Run predictions
            model = "models/old/popf-small"

            if seed == seeds[0]:
                test_masked_lm(model, dataset)

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
                    "--o",
                    out_vcf,
                ],
                check=True,
            )

            impute_time = time.time() - start

            # Compute metrics
            impute_r, impute_r2, impute_err, _ = test_impute(out_vcf, labels_path)
            model_r, model_r2, model_err, _ = compute_metrics(
                model_preds, dataset, labels_path
            )
            # model_r_large, model_r2_large, model_err_large, _ = compute_metrics(
            #     model_preds_large, dataset, labels_path
            # )
            baseline1_r, baseline1_r2, baseline1_err, _ = compute_metrics(
                baseline1_preds, dataset, labels_path
            )
            baseline2_r, baseline2_r2, baseline2_err, _ = compute_metrics(
                baseline2_preds, dataset, labels_path
            )

            # Store results
            results[mr]["popformer-base"].append((model_r, model_r2, model_err, model_time))
            # results["popformer-large"].append(
            #     (model_r_large, model_r2_large, model_err_large, model_time_large)
            # )
            results[mr]["impute5"].append((impute_r, impute_r2, impute_err, impute_time))
            results[mr]["baseline1"].append(
                (baseline1_r, baseline1_r2, baseline1_err, baseline1_time)
            )
            results[mr]["baseline2"].append(
                (baseline2_r, baseline2_r2, baseline2_err, baseline2_time)
            )

            # print intermediate results
            # print(
            #     f"Popformer: r={model_r:.4f}, r2={model_r2:.4f}, err={model_err:.4f}, time={model_time:.2f}s"
            # )
            # print(
            #     f"Impute5:   r={impute_r:.4f}, r2={impute_r2:.4f}, err={impute_err:.4f}, time={impute_time:.2f}s"
            # )
            # print(
            #     f"Baseline1: r={baseline1_r:.4f}, r2={baseline1_r2:.4f}, err={baseline1_err:.4f}, time={baseline1_time:.2f}s"
            # )
            # print(f"Baseline2: r={baseline2_r:.4f}, r2={baseline2_r2:.4f}, err={baseline2_err:.4f}, time={baseline2_time:.2f}s")
    return results

if __name__ == "__main__":
    RUN = False
    # Define seeds and mask ratios to test
    seeds = [0, 1, 2]
    mask_ratios = [20, 40, 60, 80]

    if RUN:
        results = run(seeds, mask_ratios, models=None)

        # Prepare data for plotting
        plot_data = []
        for mr in mask_ratios:
            for method_name, method_key in [
                ("popformer-base", "popformer-base"),
                # ("popformer-large", "popformer-large"),
                ("impute5", "impute5"),
                ("column freq baseline", "baseline1"),
                ("nearest neighbor baseline", "baseline2"),
            ]:
                for result in results[mr][method_key]:
                    plot_data.append(
                        {
                            "Method": method_name,
                            "Mask Ratio": int(mr),
                            "r": result[0],
                            "r2": result[1],
                            "Error Rate": result[2],
                            "Time (s)": result[3],
                        }
                    )

        df_plot = pd.DataFrame(plot_data)
        df_plot.to_csv("imputation_results_summary.csv", index=False)
    else: 
        df_plot = pd.read_csv("imputation_results_summary.csv")
    
    for metric in ["Error Rate", "r2"]:
        df_plot_remove_col = df_plot[df_plot["Method"] != "column freq baseline"]
        plt.figure(figsize=(8, 6))
        sns.pointplot(
            data=df_plot_remove_col,
            x="Mask Ratio",
            y=metric,
            hue="Method",
            errorbar=("sd"),
        )
        plt.tight_layout()
        plt.savefig(f"figs/imp_{metric.replace(' ', '_').lower()}.png", dpi=300)



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
        for method_name, method_key in [
            ("popformer-base", "popformer-base"),
            # ("popformer-large", "popformer-large"),
            ("impute5", "impute5"),
            ("column freq baseline", "baseline1"),
            ("nearest neighbor baseline", "baseline2"),
        ]:
            method_results = mr_results[mr_results["Method"] == method_name]

            aggregated = method_results.agg(
                {
                    "r": ["mean", "std"],
                    "r2": ["mean", "std"],
                    "Error Rate": ["mean", "std"],
                    "Time (s)": ["mean", "std"],
                }
            )

            print(
                "{:<30} {:>15} {:>15} {:>18} {:>15}".format(
                    method_name,
                    f"{aggregated['r']['mean']:.4f}±{aggregated['r']['std']:.4f}",
                    f"{aggregated['r2']['mean']:.4f}±{aggregated['r2']['std']:.4f}",
                    f"{aggregated['Error Rate']['mean']:.4f}±{aggregated['Error Rate']['std']:.4f}",
                    f"{aggregated['Time (s)']['mean']:.2f}±{aggregated['Time (s)']['std']:.2f}",
                )
            )
