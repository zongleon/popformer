"""Evaluate selection-detection models on real (1000 Genomes) genome data.

Produces genome-scan region plots, boxplots of scores in known-selection
vs background windows, FDR-vs-threshold curves, positive-vs-negative
called-region curves, popformer ↔ summary-stat correlations, and
null-distribution histograms from permutation tests.
"""

import numpy as np
import pandas as pd

from evaluation.models import popformer, summary_stat
from evaluation.evaluators import genome_classification
from selection_config import (
    FORCE,
    make_nn_models,
    make_summary_stat_models,
    score_transform,
    normalize,
    aggregate_windows,
    run_all,
    collect_region_data,
    sort_models,
)

# ---------------------------------------------------------------------------
# Datasets & evaluators
# ---------------------------------------------------------------------------
GENOME_DATASETS = [
    "data/dataset/genome_CEU",
    "data/dataset/genome_CHB",
    "data/dataset/genome_YRI",
]
KNOWN_POS_PATH = "data/SEL/sel.csv"
KNOWN_NEG_PATH = "data/SEL/reichsel_negs.csv"

AGG_WINDOW_N = 1


def build_evaluators(dataset_paths):
    import os

    pos_df = pd.read_csv(KNOWN_POS_PATH)
    neg_df = pd.read_csv(KNOWN_NEG_PATH)
    evaluators = []
    for path in dataset_paths:
        evaluators.append(
            genome_classification.GenomeClassificationEvaluator(
                path,
                known_selection_region_df=pos_df,
                dataset_name=os.path.basename(path) + "_pos",
            )
        )
        evaluators.append(
            genome_classification.GenomeClassificationEvaluator(
                path,
                known_selection_region_df=neg_df,
                dataset_name=os.path.basename(path) + "_neg",
            )
        )

    return evaluators


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def save_and_plot_regions(results, datasets):
    for ds in datasets:
        rd = collect_region_data(results, ds)
        if rd is None:
            continue
        model_names, preds_list, start_pos, end_pos, chrom = rd

        # Save per-model scan data
        for name, preds in zip(model_names, preds_list):
            np.savez(
                f"preds/scans/{ds}_{name}_region_plot_data.npz",
                chrom=chrom,
                start_pos=start_pos,
                end_pos=end_pos,
                preds=preds,
            )


def plot_boxplots(results, datasets):
    for ds in datasets:
        items = [
            (m, res["sig_mask"], res["preds"])
            for (m, d), res in results.items()
            if d == ds and "sig_mask" in res
        ]
        if not items:
            continue
        ordered = sort_models([m for m, _, _ in items])
        item_map = {m: (mask, p) for m, mask, p in items}
        model_names = ordered
        y_preds = [normalize(score_transform(m, item_map[m][1])) for m in ordered]
        sig_mask = item_map[model_names[0]][0]
        genome_classification.plot_boxplot(
            y_preds=y_preds,
            model_names=model_names,
            sig_mask=sig_mask,
            save_path=f"figs/{ds}_boxplot.png",
        )


def plot_rate(results, datasets, suffix=""):
    for ds in datasets:
        rate_series = []
        for (m, d), res in results.items():
            if d != ds or "preds" not in res or "sig_mask" not in res:
                continue
            preds = score_transform(m, res["preds"])
            sig_mask = res["sig_mask"]
            valid = np.isfinite(preds)
            if not np.any(valid):
                continue
            preds, sig_mask = preds[valid], sig_mask[valid]
            preds, sig_mask = aggregate_windows(preds, sig_mask, AGG_WINDOW_N)
            preds = normalize(preds)

            thresholds = np.unique(preds)
            fnr = np.array(
                [
                    (np.sum((preds < t) & sig_mask) / max(np.sum(sig_mask), 1))
                    for t in thresholds
                ]
            )
            tpr = 1 - fnr
            frac_called = np.array(
                [np.sum(preds >= t) / len(preds) for t in thresholds]
            )
            rate_series.append((m, frac_called, tpr))

        if rate_series:
            genome_classification.plot_rate_vs_threshold(
                rate_series,
                ds,
                rate_name="TPR",
                save_path=f"figs/{ds}_tpr_vs_threshold{'' if suffix == '' else '_' + suffix}.png",
            )


def plot_enrichment(results, datasets, suffix=""):
    """Plot fold-enrichment at top-k fraction for every dataset."""
    for ds in datasets:
        enrichment_series = []
        for (m, d), res in results.items():
            if d != ds or "preds" not in res or "sig_mask" not in res:
                continue
            preds = score_transform(m, res["preds"])
            sig_mask = res["sig_mask"]
            valid = np.isfinite(preds)
            if not np.any(valid):
                continue
            preds, sig_mask = preds[valid], sig_mask[valid]
            preds, sig_mask = aggregate_windows(preds, sig_mask, AGG_WINDOW_N)

            n = len(preds)
            base_rate = sig_mask.mean()
            if base_rate == 0:
                continue

            order = np.argsort(preds)[::-1]  # descending
            sig_sorted = sig_mask[order]
            cum_sel = np.cumsum(sig_sorted)

            ks = np.arange(1, n + 1)
            k_fracs = ks / n
            enrichment = (cum_sel / ks) / base_rate

            enrichment_series.append((m, k_fracs, enrichment))

        if enrichment_series:
            genome_classification.plot_enrichment_at_k(
                enrichment_series,
                ds,
                save_path=f"figs/{ds}_enrichment_at_k{'' if suffix == '' else '_' + suffix}.png",
            )


def plot_pos_vs_neg_called(results, datasets, suffix=""):
    """ROC-like curves: #pos regions called vs #neg regions falsely called."""
    base_datasets = sorted(
        {
            ds[:-4]
            for ds in datasets
            if ds.endswith("_pos") and f"{ds[:-4]}_neg" in datasets
        }
    )
    for base_ds in base_datasets:
        pos_ds = f"{base_ds}_pos"
        neg_ds = f"{base_ds}_neg"

        model_names = sort_models(
            [m for (m, d) in results if d == pos_ds and (m, neg_ds) in results]
        )

        called_series = []
        for m in model_names:
            pos_res = results.get((m, pos_ds), {})
            neg_res = results.get((m, neg_ds), {})
            if (
                "preds" not in pos_res
                or "sig_mask" not in pos_res
                or "preds" not in neg_res
                or "sig_mask" not in neg_res
            ):
                continue

            pos_preds = score_transform(m, pos_res["preds"])
            neg_preds = score_transform(m, neg_res["preds"])
            pos_mask = pos_res["sig_mask"]
            neg_mask = neg_res["sig_mask"]

            pos_valid = np.isfinite(pos_preds)
            neg_valid = np.isfinite(neg_preds)
            if not np.any(pos_valid) or not np.any(neg_valid):
                continue

            pos_preds, pos_mask = pos_preds[pos_valid], pos_mask[pos_valid]
            neg_preds, neg_mask = neg_preds[neg_valid], neg_mask[neg_valid]
            pos_preds, pos_mask = aggregate_windows(pos_preds, pos_mask, AGG_WINDOW_N)
            neg_preds, neg_mask = aggregate_windows(neg_preds, neg_mask, AGG_WINDOW_N)

            both_preds = np.concatenate([pos_preds, neg_preds])
            lo, hi = np.min(both_preds), np.max(both_preds)
            if hi > lo:
                pos_preds = (pos_preds - lo) / (hi - lo)
                neg_preds = (neg_preds - lo) / (hi - lo)
            else:
                pos_preds = pos_preds * 0.0
                neg_preds = neg_preds * 0.0

            thresholds = np.unique(np.concatenate([pos_preds, neg_preds]))[::-1]
            pos_called = np.array(
                [np.sum((pos_preds >= t) & pos_mask) for t in thresholds]
            )
            neg_called = np.array(
                [np.sum((neg_preds >= t) & neg_mask) for t in thresholds]
            )

            called_series.append((m, neg_called, pos_called))

        if called_series:
            genome_classification.plot_called_pos_vs_neg(
                called_series,
                base_ds,
                save_path=f"figs/{base_ds}_pos_vs_neg_called{'' if suffix == '' else '_' + suffix}.png",
            )


def plot_correlations(results, models, genome_ds):
    """Scatter popformer-ft scores against each summary stat."""
    model = next((m for m in models if "popformer" in m), None)
    if model is None or (model, genome_ds) not in results:
        return
    popf_preds = results[(model, genome_ds)]["preds"]

    stats = [
        ("Tajima's D", "tajimas_d"),
        ("SFS[1]", "sfs_1"),
        ("SFS[2]", "sfs_2"),
        ("Number of SNPs", "n_snps"),
    ]
    for stat_label, stat_key in stats:
        if (stat_key, genome_ds) not in results:
            continue
        stat_preds = score_transform(stat_key, results[(stat_key, genome_ds)]["preds"])
        valid = ~np.isnan(stat_preds)
        genome_classification.plot_correlation(
            popf_preds[valid],
            stat_preds[valid],
            y1lab=f"{model} score",
            y2lab=stat_label,
            save_path=f"figs/genome_correlation_{stat_label.replace(' ', '_').lower()}_popf.png",
        )


def plot_null_distributions(results):
    for (m, ds), res in results.items():
        if m != "popformer-ft-0.05" or "obs" not in res or "null" not in res:
            continue
        genome_classification.plot_histogram_with_line(
            res["null"],
            res["obs"],
            save_path=f"figs/{ds}_null_distribution.png",
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    all_models = (
        make_nn_models(train_ds="pan2CEU_train", test_sizes=[0.05], suffix="CEU")
        + make_summary_stat_models()
    )
    evaluators = build_evaluators(GENOME_DATASETS)

    results, df = run_all(all_models, evaluators, force=FORCE)
    models = sort_models(df["model"].unique().tolist())
    datasets = df["dataset"].unique().tolist()

    if "obs" in df.columns:
        cols = ["model", "dataset", "obs", "null_mean", "ci", "p_emp"]
        print(df[cols].dropna().to_string())

    popf_models = [m for m in models if m.startswith("popformer")]
    unused_stat_models = ["sfs_1", "sfs_2", "n_snps"]
    all_models = list(set(models) - set(popf_models) - set(unused_stat_models)) + [
        model for model in models if model.startswith("popformer-lp")
    ]

    pos_datasets = [ds for ds in datasets if ds.endswith("_pos")]
    CEU_datasets = [ds for ds in datasets if "CEU" in ds]

    for model_list, suffix in [(all_models, "all"), (popf_models, "popformer")]:
        subset_res = {(m, d): res for (m, d), res in results.items() if m in model_list}
        plot_enrichment(subset_res, pos_datasets, suffix=suffix)
        plot_rate(subset_res, pos_datasets, suffix=suffix)

        plot_pos_vs_neg_called(subset_res, CEU_datasets, suffix=suffix)

    save_and_plot_regions(results, pos_datasets)
    plot_boxplots(results, pos_datasets)
    plot_correlations(results, models, pos_datasets[0])
    plot_null_distributions(results)
