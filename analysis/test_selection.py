"""Evaluate selection-detection models on simulated datasets.

Produces per-dataset ROC / PR / score-distribution plots, a faceted
ROC-by-s grid for pan_test, accuracy-vs-train-size curves, and
popformer ↔ summary-stat correlation scatter plots.
"""

import numpy as np
import matplotlib.pyplot as plt

import theme
from evaluation.evaluators import random_classification, genome_classification
from selection_config import (
    make_nn_models,
    make_summary_stat_models,
    run_all,
    sort_models,
)


SIMULATED_DATASETS = [
    "data/dataset/pan2CEU_test",
    "data/dataset/pan2CHB_test",
    "data/dataset/pan2YRI_test",
    "data/dataset/pan_4_test",
    "data/dataset/pan2_test_50000",
    # "data/dataset/pan_3_demoid-0_balanced",
    # "data/dataset/pan_3_demoid-1_balanced",
]


def build_evaluators(dataset_paths):
    evaluators = []
    for path in dataset_paths:
        evaluators.append(random_classification.RandomClassificationEvaluator(path))
    return evaluators


def plot_classification_curves(results, models, datasets, final_suffix=""):
    """ROC, PR, and score-distribution plots for every dataset with labels."""
    for ds in datasets:
        trues = [results[(m, ds)].get("trues") for m in models]
        scores = [results[(m, ds)].get("preds_for_metrics") for m in models]
        if trues[0] is None:
            continue
        for typ in ["roc", "pr"]:
            random_classification.plot_curves(
                trues,
                scores,
                models,
                curve_type=typ,
                save_path=f"figs/{ds}_{typ}{'_' + final_suffix if final_suffix else ''}.png",
            )


def plot_roc_by_s(results, models, dataset_name="pan_test"):
    """Per-s ROC/PR and faceted ROC grid for a single dataset."""
    res = results.get((models[0], dataset_name), {})
    y_trues, s_vals, shoulder_vals = (
        np.array(res["trues"]),
        np.array(res["s"]),
        np.array(res["shoulder"]),
    )

    if np.unique(s_vals).shape[0] <= 5:
        unique_s = [v for v in np.unique(s_vals) if v != 0]
        s_masks = {f"s={v:.2f}": ((s_vals == v) | (y_trues == 0)) for v in unique_s}
    else:
        # bin s into 5 bins if there are too many unique values
        unique_s = np.linspace(np.min(s_vals), np.max(s_vals), 6)
        s_masks = {
            f"s=({lo:.2f}, {hi:.2f})": ((s_vals > lo) & (s_vals <= hi) | (y_trues == 0))
            for lo, hi in zip(unique_s[:-1], unique_s[1:])
        }

    # Faceted ROC grid (columns = s bins)
    all_trues = [results[(m, dataset_name)]["trues"] for m in models]
    all_scores = [results[(m, dataset_name)]["preds_for_metrics"] for m in models]

    fig, axes = plt.subplots(
        2,
        len(s_masks),
        layout="constrained",
        figsize=(4 * len(s_masks), 8),
        sharex=True,
        sharey=True,
    )
    axes = axes.flatten()
    for shoulder_val in [0, 1]:
        for idx, (label, mask) in enumerate(s_masks.items()):
            ax_idx = shoulder_val * len(s_masks) + idx
            ax = axes[ax_idx]
            if ax_idx < len(s_masks):
                ax.set_title(f"{label}", fontsize=14)
            if ax_idx % len(s_masks) == 0:
                ax.set_ylabel(
                    "with shoulders" if shoulder_val == 1 else "without shoulders",
                    fontsize=14,
                )
            m = mask & ((shoulder_vals == shoulder_val) | (y_trues == 1))
            random_classification.plot_curves(
                [np.array(t)[m] for t in all_trues],
                [np.array(s)[m] for s in all_scores],
                models,
                curve_type="roc",
                ax=ax,
                baseify_model_names=True,
                add_ax_labels=False,
            )
    fig.supxlabel("False Positive Rate", fontsize=16)
    fig.supylabel("True Positive Rate", fontsize=16)
    plt.savefig(f"figs/{dataset_name}_roc_facet_grid.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_acc_vs_train_size(df, dataset_name="pan2CEU"):
    df_acc = df[df["dataset"] == dataset_name].copy()
    df_acc["train_size"] = df_acc["model"].apply(
        lambda x: 1 - float(x.split("-")[-1]) if "-" in x else 0.0
    )
    df_acc["model"] = df_acc["model"].apply(
        lambda x: "-".join(x.split("-")[:-1]) if "-" in x else x
    )
    df_acc = df_acc[["model", "train_size", "auprc"]].query(
        "model not in ['sfs_1', 'sfs_1_count', 'sfs_2', 'n_snps']"
    )
    random_classification.plot_y_by_x(
        df_acc,
        y="auprc",
        x="train_size",
        save_path=f"figs/{dataset_name}_auprc_vs_train_size.png",
    )


def plot_popf_vs_summary_stats(results, models, dataset_name="pan_test"):
    """Scatter popformer-ft scores against each summary stat, colored by s."""
    popf_ft = next((m for m in models if m.startswith("popformer-ft")), None)
    if popf_ft is None or (popf_ft, dataset_name) not in results:
        return
    popf_res = results[(popf_ft, dataset_name)]
    s, popf_scores = popf_res.get("s"), popf_res.get("preds")
    if s is None or popf_scores is None:
        return

    for stat in ["tajimas_d", "sfs_1", "sfs_1_count", "sfs_2", "n_snps"]:
        stat_scores = results.get((stat, dataset_name), {}).get("preds")
        if stat_scores is None:
            continue
        valid = np.isfinite(popf_scores) & np.isfinite(stat_scores) & np.isfinite(s)
        if not np.any(valid):
            continue
        genome_classification.plot_correlation(
            popf_scores[valid],
            stat_scores[valid],
            y1lab=f"{theme.get_model_base_name(popf_ft)} score",
            y2lab=stat,
            color_by=s[valid],
            color_by_label="s",
            save_path=f"figs/{dataset_name}_correlation_{stat}_popf_ft_colored_by_s.png",
        )


if __name__ == "__main__":
    all_models = (
        make_nn_models(
            train_ds="pan2CEU_train",
            test_sizes=[0.05, 0.5, 0.9, 0.95, 0.99],
            suffix="CEU",
        )
        + make_summary_stat_models()
    )
    evaluators = build_evaluators(SIMULATED_DATASETS)

    results, df = run_all(all_models, evaluators, force=False)
    models = sort_models(df["model"].unique().tolist())
    datasets = df["dataset"].unique().tolist()

    # Print summary tables
    if "accuracy" in df.columns:
        cols = ["model", "dataset", "accuracy", "precision", "recall", "auroc", "auprc"]
        print(df[cols].dropna().to_string())
    if "obs" in df.columns:
        cols = ["model", "dataset", "obs", "null_mean", "ci", "p_emp"]
        print(df[cols].dropna().to_string())

    # All plots
    popf_models = [m for m in models if m.startswith("popformer") and "0.05" in m]
    all_models = [m for m in models if "0.05" in m] + ["tajimas_d"]

    for model_list, suffix in [
        (all_models, "all_models"),
        (popf_models, "popformer_models"),
    ]:
        plot_classification_curves(results, model_list, datasets, final_suffix=suffix)

    for dataset_name in ["pan2CEU_test", "pan2CHB_test", "pan2YRI_test"]:
        plot_acc_vs_train_size(df, dataset_name=dataset_name)
        plot_roc_by_s(results, all_models, dataset_name=dataset_name)
    plot_popf_vs_summary_stats(results, models)
