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
        facet_vars = {
            k: v for k, v in {
                "growth": "bin",
                "low_mut": "binary", 
                "has_dfe": "binary",
                "onset_time": "zero_bin",
                "min_freq": "zero_bin"
            }.items() if hasattr(self, k)
        }
        
        for var, typ in facet_vars.items():
            # add masks for each facet variable
            if typ == "binary":
                facet_values = [0, 1]
            elif typ == "bin":
                # bin into 5 equal sized bins
                var_values = getattr(self, var)
                facet_values = pd.qcut(var_values, q=5, duplicates="drop").unique()
                facet_values = sorted(facet_values, key=lambda x: x.left)
            elif typ == "zero_bin":
                # bin into a bin for 0 and 5 bins for non-zero values
                var_values = getattr(self, var)
                facet_values = [0] + sorted(list(pd.qcut(var_values[var_values != 0], q=5, duplicates="drop").unique()), key=lambda x: x.left)
            else:
                continue

            facet_masks = []
            for val in facet_values:
                if typ == "binary":
                    mask = getattr(self, var) == val
                else:
                    # masks for binned variables
                    mask = pd.Series(getattr(self, var)).apply(lambda x: x in val if not val == 0 else x == 0).values

                print(f"Facet variable '{var}' value '{val}' has {np.sum(mask)} samples.")
                facet_masks.append(
                    {
                        "facet_value": val,
                        "mask": mask,
                    }
                )
            results[var + "_masks"] = facet_masks

        results.update({
            "model_name": self.model_name,
            "accuracy": acc,
            "auroc": aucroc,
            "auprc": auprc,
            "precision": precision,
            "recall": recall,
        })

        return results


def plot_roc_curves(y_trues, y_scores, model_names, colors=None, ax=None, add_ax_labels=False, save_path="figs/roc_curves.png"):
    new = False
    if not ax:
        new = True
        fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")

    if not add_ax_labels:
        old_xlabel = ax.get_xlabel()
        old_ylabel = ax.get_ylabel()

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

    if not add_ax_labels:
        ax.set_xlabel(old_xlabel)
        ax.set_ylabel(old_ylabel)

    if new:
        if add_ax_labels:
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
        plt.grid(True, axis="y", alpha=0.3, linestyle="--")
        plt.savefig(save_path, dpi=300)
        plt.close()


def plot_pr_curves(y_trues, y_scores, model_names, colors=None, ax=None, add_ax_labels=False, save_path="figs/pr_curves.png"):
    new = False
    if not ax:
        new = True
        fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")

    if not add_ax_labels:
        old_xlabel = ax.get_xlabel()
        old_ylabel = ax.get_ylabel()
    
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
    
    if not add_ax_labels:
        ax.set_xlabel(old_xlabel)
        ax.set_ylabel(old_ylabel)

    if new:
        if add_ax_labels:
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
