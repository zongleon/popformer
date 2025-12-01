import os
from ..core import BaseHFEvaluator
import numpy as np
import pandas as pd
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import matplotlib.pyplot as plt
import seaborn as sns

class RandomClassificationEvaluator(BaseHFEvaluator):
    """
    Evaluator that makes classifications for simulated windows.
    Computes the following metrics:
        - Accuracy
        - Precision
        - Recall
        - AUROC
        - AUPRC

    And generates plots for:
        - ROC Curve
        - Precision-Recall Curve
        - Accuracy vs (s) bins (if s available)
        - Distribution of (s) (if s available)
        - Regioned predictions if positions are available
    """
    def evaluate(self, predictions):
        results = {}
        pos_preds = predictions[:, 1]
        binary_preds = predictions.argmax(axis=1)

        acc = accuracy_score(self.labels, binary_preds)
        aucroc = roc_auc_score(self.labels, pos_preds)
        auprc = average_precision_score(self.labels, pos_preds)
        precision = precision_score(self.labels, binary_preds)
        recall = recall_score(self.labels, binary_preds)

        if hasattr(self, "s"):
            # we'll store plotting data
            s = self.s
            lbls = self.labels
            # accuracy vs s bins per model
            bins = [
                ("s=(0,0.02]", (0.0, 0.02)),
                ("s=(0.02,0.04]", (0.02, 0.04)),
                ("s=(0.04,0.06]", (0.04, 0.06)),
                ("s=(0.06,0.08]", (0.06, 0.08)),
                ("s=(0.08,0.1]", (0.08, 0.1)),
            ]

            s_masks = []
            for label, (a, b) in bins:
                mask = ((s > a) & (s <= b)) | (s == 0)
                if np.any(mask):
                    s_masks.append(
                        {
                            "s_bin": label,
                            "mask": mask,
                        }
                    )
            
            results["s_masks"] = s_masks


            # --- AUROC vs s with faceting by other variables ---
            facet_vars = [
                v for v in ["growth", "low_mut", "has_dfe", "onset_time", "min_freq"] if hasattr(self, v)
            ]

            facet_rows = []

            # helper to compute auroc for a mask safely
            def safe_auroc(mask):
                y_true_sub = lbls[mask]
                # need at least one positive and one negative
                if y_true_sub.sum() == 0 or y_true_sub.sum() == len(y_true_sub):
                    return np.nan
                try:
                    return roc_auc_score(y_true_sub, pos_preds[mask])
                except ValueError:
                    return np.nan

            # Prepare categorical s_bin ordering
            s_bin_labels = [b[0] for b in bins]

            # Precompute s_bin masks list for reuse
            s_bin_masks = [(label, ((s > a) & (s <= b)) | (s == 0)) for label, (a, b) in bins]

            for var in facet_vars:
                values = getattr(self, var)
                # Normalize / bin if too many unique values
                unique_vals = np.unique(values)
                # If numeric and too many unique, bin into quartiles
                if (
                    np.issubdtype(unique_vals.dtype, np.number)
                    and len(unique_vals) > 8
                ):
                    # create quartile bins
                    try:
                        # pandas qcut for nicer labels
                        q = pd.qcut(values, 4, duplicates="drop")
                        value_labels = q.astype(str)
                    except ValueError:
                        # fallback: just use raw values
                        value_labels = values.astype(str)
                else:
                    value_labels = values.astype(str)

                for val_label in np.unique(value_labels):
                    val_mask = value_labels == val_label
                    for s_label, s_mask in s_bin_masks:
                        combined_mask = val_mask & s_mask
                        if combined_mask.sum() < 5:  # skip tiny groups
                            continue
                        auroc_bin = safe_auroc(combined_mask)
                        facet_rows.append(
                            {
                                "facet_var": var,
                                "facet_value": val_label,
                                "s_bin": s_label,
                                "auroc": auroc_bin,
                                "model": getattr(self, "model_name", "model"),
                            }
                        )

            if facet_rows:
                df_facets = pd.DataFrame(facet_rows)
                df_facets["s_bin"] = pd.Categorical(df_facets["s_bin"], categories=s_bin_labels, ordered=True)
                results["auroc_s_facets_df"] = df_facets

            # plot the distribution of s
            # fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
            # sns.histplot(s[s > 0], bins=50, kde=False, ax=ax)
            # ax.set_xlabel("Selection coefficient s")
            # ax.set_ylabel("Count")
            # ax.grid(True, axis="y", alpha=0.3, linestyle="--")
            # plt.savefig("figs/lp_s_dist.png", dpi=300)
            # plt.close()

        results.update({
            "model_name": self.model_name,
            "accuracy": acc,
            "auroc": aucroc,
            "auprc": auprc,
            "precision": precision,
            "recall": recall,
        })

        return results


def plot_roc_curves(y_trues, y_scores, model_names, colors=None, save_path="figs/roc_curves.png"):
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    for i, (y_true, y_score, model_name) in enumerate(zip(y_trues, y_scores, model_names)):
        if y_true is None:
            continue
        RocCurveDisplay.from_predictions(
            y_true,
            y_score,
            name=model_name,
            ax=ax,
            curve_kwargs=None if colors is None else {"color": colors[i]},
        )
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True, axis="y", alpha=0.3, linestyle="--")
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_pr_curves(y_trues, y_scores, model_names, colors=None, save_path="figs/pr_curves.png"):
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    for i, (y_true, y_score, model_name) in enumerate(zip(y_trues, y_scores, model_names)):
        if y_true is None:
            continue
        PrecisionRecallDisplay.from_predictions(
            y_true,
            y_score,
            name=model_name,
            ax=ax,
            color=None if colors is None else colors[i],
        )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True, axis="y", alpha=0.3, linestyle="--")
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_acc_by_s(df_s_acc, save_path="figs/lp_acc_vs_s.png"):
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    sns.lineplot(
        data=df_s_acc,
        x="s_bin",
        y="acc",
        hue="model",
        markers=True,
        ax=ax,
    )
    ax.set_xlabel("s bin")
    ax.set_ylabel("Accuracy")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_auroc_vs_s_facets(df_facets, save_dir="figs", filename_prefix="auroc_vs_s"):
    """Plot AUROC vs s bins with separate facet grids for each variable.

    Expects DataFrame with columns:
        facet_var, facet_value, s_bin, auroc, model
    Generates one PNG per facet_var.
    """
    os.makedirs(save_dir, exist_ok=True)

    facet_vars = df_facets["facet_var"].unique()
    saved_paths = []
    for var in facet_vars:
        sub = df_facets[df_facets["facet_var"] == var]
        # Ensure ordering preserved
        if isinstance(sub["s_bin"].dtype, pd.CategoricalDtype):
            s_order = sub["s_bin"].cat.categories
        else:
            s_order = sorted(sub["s_bin"].unique())
        g = sns.FacetGrid(sub, col="facet_value", col_wrap=4, sharey=True, height=3)
        def _lineplot(data, color=None):
            sns.lineplot(
                data=data,
                x="s_bin",
                y="auroc",
                hue="model",
                hue_order=sorted(sub["model"].unique()),
                marker="o",
            )
        g.map_dataframe(_lineplot)
        g.set_axis_labels("s bin", "AUROC")
        g.add_legend(title="Model")
        for ax in g.axes.flatten():
            ax.grid(True, axis="y", alpha=0.3, linestyle="--")
            ax.set_xticks(range(len(s_order)))
            ax.set_xticklabels(s_order, rotation=45)
        g.fig.subplots_adjust(top=0.85)
        g.fig.suptitle(f"AUROC vs s faceted by {var}")
        out_path = os.path.join(save_dir, f"{filename_prefix}__{var}.png")
        g.savefig(out_path, dpi=300)
        plt.close(g.fig)
        saved_paths.append(out_path)
    return saved_paths

