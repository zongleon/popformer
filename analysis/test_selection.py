"""Evaluate selection-detection models on simulated datasets.

Produces per-dataset ROC / PR / score-distribution plots, a faceted
ROC-by-s grid for pan_test, accuracy-vs-train-size curves, and
popformer ↔ summary-stat correlation scatter plots.
"""

import argparse
from signal import Handlers

import matplotlib.pyplot as plt
import numpy as np
import theme
from evaluation.evaluators import genome_classification, random_classification
from matplotlib.ticker import LogLocator
from selection_config import (
    make_summary_stat_models,
    models_to_models,
    run_all,
)


def build_evaluators(dataset_paths):
    evaluators = []
    for path in dataset_paths:
        evaluators.append(random_classification.RandomClassificationEvaluator(path))
    return evaluators


def plot_classification_curves(
    results, models, datasets, roc_only=False, final_suffix=""
):
    """ROC, PR, and score-distribution plots for every dataset with labels."""
    for ds in datasets:
        trues = [results[(m, ds)].get("trues") for m in models]
        scores = [results[(m, ds)].get("preds_for_metrics") for m in models]
        if trues[0] is None:
            continue
        for typ in ["roc", "pr"]:
            if roc_only and typ != "roc":
                continue
            random_classification.plot_curves(
                trues,
                scores,
                models,
                dataset=ds,
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
                legend_fontsize=10,
            )
    fig.supxlabel("False Positive Rate", fontsize=16)
    fig.supylabel("True Positive Rate", fontsize=16)
    plt.savefig(f"figs/{dataset_name}_roc_facet_grid.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc_by_s_f(results, models, dataset_name="pan_test"):
    """Per-s ROC/PR and faceted ROC grid for a single dataset."""
    res = results.get((models[0], dataset_name), {})
    y_trues, s_vals, f_vals = (
        np.array(res["trues"]),
        np.array(res["s"]),
        np.array(res["f"]),
    )

    unique_s = [v for v in np.unique(s_vals) if v != 0]
    s_masks = {f"s={v:.3f}": ((s_vals == v) | (y_trues == 0)) for v in unique_s}

    unique_f = [v for v in np.unique(f_vals) if v != 0]
    f_masks = {f"f={v:.2f}": ((f_vals == v) | (y_trues == 0)) for v in unique_f}

    # Faceted ROC grid (columns = s bins)
    all_trues = [results[(m, dataset_name)]["trues"] for m in models]
    all_scores = [results[(m, dataset_name)]["preds_for_metrics"] for m in models]

    fig, axes = plt.subplots(
        len(f_masks),
        len(s_masks),
        layout="constrained",
        figsize=(4 * len(s_masks), 4 * len(f_masks)),
        sharex=True,
        sharey=True,
    )
    axes = axes.flatten()
    for idx_f, (label_f, mask_f) in enumerate(f_masks.items()):
        for idx_s, (label, mask) in enumerate(s_masks.items()):
            ax_idx = idx_f * len(s_masks) + idx_s
            ax = axes[ax_idx]
            if ax_idx < len(s_masks):
                ax.set_title(f"{label}", fontsize=14)
            if ax_idx % len(s_masks) == 0:
                ax.set_ylabel(
                    f"{label_f}",
                    fontsize=14,
                )
            m = mask & mask_f
            random_classification.plot_curves(
                [np.array(t)[m] for t in all_trues],
                [np.array(s)[m] for s in all_scores],
                models,
                curve_type="pr",
                ax=ax,
                baseify_model_names=True,
                add_ax_labels=False,
                legend_fontsize=10,
            )
    fig.supxlabel("False Positive Rate", fontsize=16)
    fig.supylabel("True Positive Rate", fontsize=16)
    plt.savefig(f"figs/{dataset_name}_roc_facet_grid.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_acc_by_x(
    df,
    x: str,
    trained_on_line=None,
    pretrained_on_line=None,
    x_func=None,
    flip_x=False,
    log_x=False,
    metric="auprc",
):
    df_acc = df[["model", "dataset", x, metric]].copy()
    df_acc = df_acc[~df_acc[x].isna()]
    df_acc = df_acc.query("model not in ['sfs_1', 'sfs_1_count', 'sfs_2', 'n_snps']")
    fig, ax = plt.subplots(figsize=(8, 6))
    lines = []
    if trained_on_line is not None:
        line1 = ax.axvline(
            trained_on_line,
            color="black",
            linewidth=1,
            label="Training",
        )
        lines.append(line1)
    if pretrained_on_line is not None:
        line2 = ax.axvline(
            pretrained_on_line,
            color="black",
            linestyle="--",
            linewidth=1,
            label="Pre-training",
        )
        lines.append(line2)
    random_classification.plot_y_by_x(
        df_acc, y=metric, x=x, save_path=f"figs/{metric}_vs_{x}.png", ax=ax, logx=log_x
    )
    if x_func:
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        ax.xaxis.set_major_formatter(x_func)
    if flip_x:
        ax.invert_xaxis()
    if trained_on_line is not None or pretrained_on_line is not None:
        legend = ax.legend(handles=lines, loc="upper right")
        ax.add_artist(legend)
    # add another legend for the models
    # get the lines for the models from the plot except ones that start with '_'
    # kinda hacky a lil
    model_lines = [
        line
        for line in ax.get_lines()
        if line not in lines and not line.get_label().startswith("_")
    ]
    ax.legend(handles=model_lines, loc="lower left")
    plt.savefig(f"figs/{metric}_vs_{x}.png", dpi=300, bbox_inches="tight")


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
            save_path=f"figs/{dataset_name}_correlation_{stat}.png",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate selection-detection models.")
    parser.add_argument(
        "--models", nargs="+", help="List of model names to evaluate", required=True
    )
    parser.add_argument(
        "--names", nargs="+", help="List of display names for the models", required=True
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="List of dataset names to evaluate on",
        required=True,
    )
    parser.add_argument("--rocs", action="store_true", help="Plot ROC/PR curves")
    parser.add_argument(
        "--varying", action="store_true", help="Plot varying bottlenecks/Ns"
    )
    args = parser.parse_args()

    all_models = models_to_models(args.models, args.names) + make_summary_stat_models()
    evaluators = build_evaluators(args.datasets)

    results, df = run_all(all_models, evaluators, force=False)
    models = df["model"].unique().tolist()
    datasets = df["dataset"].unique().tolist()

    # Print summary tables
    if "accuracy" in df.columns:
        cols = ["model", "dataset", "accuracy", "precision", "recall", "auroc", "auprc"]
        print(df[cols].dropna().to_string())

    if args.rocs:
        models = [
            model
            for model in models
            if model not in ["sfs_1", "sfs_1_count", "sfs_2", "n_snps"]
        ]
        plot_classification_curves(results, models, datasets, roc_only=False)

    if args.varying:

        def btl_labeller(x, pos):
            val = x / 10000 * 100
            val = round(val)
            return f"{val:d}%"

        df["bottleneck"] = df["dataset"].apply(
            lambda x: int(x.split("_")[-1]) if "discoal_bottlenecks" in x else np.nan
        )
        df["N"] = df["dataset"].apply(
            lambda x: int(x.split("_")[-1]) if "discoal_consts" in x else np.nan
        )

        if df["bottleneck"].notna().any():
            plot_acc_by_x(
                df,
                "bottleneck",
                trained_on_line=10000,
                pretrained_on_line=1000,
                x_func=btl_labeller,
                flip_x=True,
                log_x=True,
            )
        if df["N"].notna().any():
            plot_acc_by_x(
                df,
                "N",
                trained_on_line=10000,
                x_labels=None,
                flip_x=False,
                log_x=True,
            )
