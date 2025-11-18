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
                ("s=0", (0.0, 0.0)),
                ("(0,0.02]", (0.0, 0.02)),
                ("(0.02,0.04]", (0.02, 0.04)),
                ("(0.04,0.06]", (0.04, 0.06)),
                ("(0.06,0.08]", (0.06, 0.08)),
                ("(0.08,0.1]", (0.08, 0.1)),
            ]

            acc_by_s = []
            for label, (a, b) in bins:
                if a == 0.0 and b == 0.0:
                    mask = np.isclose(s, 0.0)
                else:
                    mask = (s > a) & (s <= b)
                if np.any(mask):
                    acc = accuracy_score(lbls[mask], binary_preds[mask])
                    acc_by_s.append(
                        {
                            "s_bin": label,
                            "acc": acc,
                            "model": self.model_name,
                        }
                    )
            
            results["acc_by_s"] = acc_by_s

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


def plot_roc_curves(y_trues, y_scores, model_names, save_path="figs/roc_curves.png"):
    fig, ax = plt.figure(figsize=(8, 6), layout="constrained")
    for y_true, y_score, model_name in zip(y_trues, y_scores, model_names):
        RocCurveDisplay.from_predictions(
            y_true,
            y_score,
            name=model_name,
            alpha=0.7,
        ).plot(ax=ax)
    plt.grid(True, axis="y", alpha=0.3, linestyle="--")
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_pr_curves(y_trues, y_scores, model_names, save_path="figs/pr_curves.png"):
    fig, ax = plt.figure(figsize=(8, 6), layout="constrained")
    for y_true, y_score, model_name in zip(y_trues, y_scores, model_names):
        PrecisionRecallDisplay.from_predictions(
            y_true,
            y_score,
            name=model_name,
            alpha=0.7,
        ).plot(ax=ax)
    plt.grid(True, axis="y", alpha=0.3, linestyle="--")
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_acc_by_s(acc_by_s_list, model_names, save_path="figs/lp_acc_vs_s.png"):
    df_s_acc = pd.DataFrame(acc_by_s_list)

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

