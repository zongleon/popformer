"""Evaluate selection-detection models on simulated datasets.

Produces per-dataset ROC / PR / score-distribution plots, a faceted
ROC-by-s grid for pan_test, accuracy-vs-train-size curves, and
popformer ↔ summary-stat correlation scatter plots.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from evaluation.evaluators import random_classification, genome_classification
from selection_config import (
    FORCE,
    INVERT_SCORE_MODELS,
    make_nn_models,
    make_summary_stat_models,
    run_all,
    score_transform,
    collect_region_data,
    sort_models,
)

# ---------------------------------------------------------------------------
# Datasets & evaluators
# ---------------------------------------------------------------------------
SIMULATED_DATASETS = [
    "data/dataset/pan2CEU_test",
    "data/dataset/pan2CHB_test",
    "data/dataset/pan2YRI_test",
    "data/dataset/pan_4_test",
    # "data/dataset/pan_test",
    # "data/dataset/pan_test_50000",
    # "data/dataset/pan2_test_50000",
    # "data/dataset/pan_3_demoid-0_balanced",
    # "data/dataset/pan_3_demoid-1_balanced",
    # "data/dataset/len200_ghist_const1",
    # "data/dataset/len200_ghist_const2",
    # "data/dataset/neutral_chr20_CEU",
    # "data/dataset/simhumanity_chr20",
]


def build_evaluators(dataset_paths):
    evaluators = []
    for path in dataset_paths:
        if "pan" in path:
            evaluators.append(random_classification.RandomClassificationEvaluator(path))
        else:
            label_path = "data/SEL/sel.csv"
            ds_name = (
                os.path.basename(path)
                + "_"
                + os.path.basename(label_path).split(".")[0]
            )
            if "ghist" in path:
                label_path = f"data/matrices/bigregions/{os.path.basename(path)}.csv"
                ds_name = None
            evaluators.append(
                genome_classification.GenomeClassificationEvaluator(
                    path,
                    known_selection_region_df=(
                        pd.read_csv(label_path) if os.path.exists(label_path) else None
                    ),
                    dataset_name=ds_name,
                )
            )
    return evaluators


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def plot_classification_curves(results, models, datasets, final_suffix=""):
    """ROC, PR, and score-distribution plots for every dataset with labels."""
    for ds in datasets:
        trues = [results[(m, ds)].get("trues") for m in models]
        scores = [score_transform(m, results[(m, ds)].get("preds")) for m in models]
        if trues[0] is None:
            continue
        for fn, suffix in [
            (random_classification.plot_roc_curves, "roc_curves"),
            (random_classification.plot_pr_curves, "pr_curves"),
            # (random_classification.plot_score_distributions, "score_distributions"),
        ]:
            fn(
                trues,
                scores,
                models,
                save_path=f"figs/{ds}_{suffix}{'_' + final_suffix if final_suffix else ''}.png",
            )


def plot_roc_by_s(results, models, dataset_name="pan_test"):
    """Per-s ROC/PR and faceted ROC grid for a single dataset."""
    model = next((m for m in models if m.startswith("popformer")), None)
    if model is None or (model, dataset_name) not in results:
        return
    res = results[(model, dataset_name)]
    y_trues, y_scores, s_vals = res["trues"], res["preds"], res["s"]

    # Per-s-value curves
    if np.unique(s_vals).shape[0] <= 5:
        unique_s = [v for v in np.unique(s_vals) if v != 0]
        s_masks = {
            f"s={v:.3f}": (s_vals == v) | (np.array(y_trues) == 0) for v in unique_s
        }
    else:
        # bin s into 5 bins if there are too many unique values
        unique_s = np.linspace(np.min(s_vals), np.max(s_vals), 6)
        s_masks = {
            f"s=({lo:.3f}, {hi:.3f})": (s_vals > lo) & (s_vals <= hi)
            | (np.array(y_trues) == 0)
            for lo, hi in zip(unique_s[:-1], unique_s[1:])
        }
    colors = plt.cm.cividis(np.linspace(0, 1, len(s_masks)))
    for fn, suffix in [
        (random_classification.plot_roc_curves, "roc_curves_by_s"),
        (random_classification.plot_pr_curves, "pr_curves_by_s"),
    ]:
        fn(
            [np.array(y_trues)[m] for m in s_masks.values()],
            [np.array(y_scores)[m] for m in s_masks.values()],
            list(s_masks.keys()),
            colors=colors,
            save_path=f"figs/{dataset_name}_{suffix}.png",
        )

    # Faceted ROC grid (columns = s bins)
    all_trues = [results[(m, dataset_name)]["trues"] for m in models]
    all_scores = [
        score_transform(m, results[(m, dataset_name)]["preds"]) for m in models
    ]

    fig, axes = plt.subplots(
        1,
        len(s_masks),
        figsize=(4 * len(s_masks), 4),
        layout="constrained",
        sharex=True,
        sharey=True,
    )
    if len(s_masks) == 1:
        axes = [axes]
    for ax, (label, mask) in zip(axes, s_masks.items()):
        ax.set_title(f"{label}", fontsize=16)
        random_classification.plot_roc_curves(
            [np.array(t)[mask] for t in all_trues],
            [np.array(s)[mask] for s in all_scores],
            models,
            ax=ax,
            add_ax_labels=False,
        )
    plt.savefig(f"figs/{dataset_name}_roc_facet_grid.png", dpi=300)
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
    df_acc["auprc"] = df_acc.apply(
        lambda row: (
            1 - row["auprc"] if row["model"] in INVERT_SCORE_MODELS else row["auprc"]
        ),
        axis=1,
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
        stat_scores = score_transform(stat, stat_scores)
        valid = np.isfinite(popf_scores) & np.isfinite(stat_scores) & np.isfinite(s)
        if not np.any(valid):
            continue
        genome_classification.plot_correlation(
            popf_scores[valid],
            stat_scores[valid],
            y1lab=f"{popf_ft} score",
            y2lab=stat,
            color_by=s[valid],
            color_by_label="s",
            save_path=f"figs/{dataset_name}_correlation_{stat}_popf_ft_colored_by_s.png",
        )


def plot_region_plots(results, datasets):
    for ds in datasets:
        rd = collect_region_data(results, ds)
        if rd is None:
            continue
        model_names, preds_list, start_pos, end_pos, _ = rd
        genome_classification.plot_region(
            preds_list=preds_list,
            model_names=model_names,
            start_pos=start_pos,
            end_pos=end_pos,
            save_path=f"figs/{ds}_region_plot.png",
            line=False,
            window=1,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    all_models = (
        make_nn_models(
            train_ds="pan2CEU_train",
            # test_sizes=[0.05, 0.5, 0.9, 0.95, 0.99],
            test_sizes=[0.05],
            suffix="CEU",
        )
        + make_nn_models(
            train_ds="pan2_train_50000",
            test_sizes=[0.05],
            suffix="2",
        )
        + make_summary_stat_models()
    )
    evaluators = build_evaluators(SIMULATED_DATASETS)

    results, df = run_all(all_models, evaluators, force=FORCE)
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
    popf_models = [m for m in models if m.startswith("popformer")]
    all_models = list(set(models) - set(popf_models)) + [
        model for model in models if model.startswith("popformer-ft")
    ]
    some_models = [m for m in models if "0.05" in m]

    for model_list, suffix in [
        (all_models, "all_models"),
        (popf_models, "popformer_models"),
    ]:
        plot_classification_curves(results, model_list, datasets, final_suffix=suffix)

    for dataset_name in ["pan2CEU_test", "pan2CHB_test", "pan2YRI_test"]:
        plot_acc_vs_train_size(df, dataset_name=dataset_name)
        plot_roc_by_s(results, some_models, dataset_name=dataset_name)
    plot_popf_vs_summary_stats(results, models)
    plot_region_plots(results, datasets)
