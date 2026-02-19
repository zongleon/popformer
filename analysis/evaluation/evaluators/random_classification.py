from ..core import BaseHFEvaluator
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
import theme


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

        facet_vars = [
            k
            for k in [
                "s",
                "growth",
                "low_mut",
                "has_dfe",
                "onset_time",
                "min_freq",
                "goal_freq",
            ]
            if hasattr(self, k)
        ]

        for var in facet_vars:
            results[var] = getattr(self, var)

        results.update(
            {
                "model_name": self.model_name,
                "accuracy": acc,
                "auroc": aucroc,
                "auprc": auprc,
                "precision": precision,
                "recall": recall,
                "preds": pos_preds,
                "trues": self.labels,
            }
        )

        return results


def plot_score_distributions(
    y_trues,
    y_scores,
    model_names,
    colors=None,
    save_path="figs/score_distributions.png",
):
    fig, axs = plt.subplots(
        len(model_names), 1, figsize=(8, 6 * len(model_names)), layout="constrained"
    )

    for i, (y_true, y_score, model_name) in enumerate(
        zip(y_trues, y_scores, model_names)
    ):
        ax = axs[i] if len(model_names) > 1 else axs
        if y_true is None:
            continue
        sns.histplot(
            x=y_score,
            hue=y_true,
            ax=ax,
        )

    ax.set_xlabel("Predicted Score")
    ax.set_ylabel("Density")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_roc_curves(
    y_trues,
    y_scores,
    model_names,
    colors=None,
    ax=None,
    add_ax_labels=True,
    save_path="figs/roc_curves.png",
):
    new = False
    if ax is None:
        new = True
        fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")

    if not add_ax_labels:
        old_xlabel = ax.get_xlabel()
        old_ylabel = ax.get_ylabel()

    for i, (y_true, y_score, model_name) in enumerate(
        zip(y_trues, y_scores, model_names)
    ):
        if y_true is None:
            continue
        RocCurveDisplay.from_predictions(
            y_true,
            y_score,
            name=model_name,
            ax=ax,
            curve_kwargs={"color": theme.model_to_color(model_name)}
            if colors is None
            else {"color": colors[i]},
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


def plot_pr_curves(
    y_trues,
    y_scores,
    model_names,
    colors=None,
    ax=None,
    add_ax_labels=True,
    save_path="figs/pr_curves.png",
):
    new = False
    if not ax:
        new = True
        fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")

    if not add_ax_labels:
        old_xlabel = ax.get_xlabel()
        old_ylabel = ax.get_ylabel()

    for i, (y_true, y_score, model_name) in enumerate(
        zip(y_trues, y_scores, model_names)
    ):
        if y_true is None:
            continue
        PrecisionRecallDisplay.from_predictions(
            y_true,
            y_score,
            name=model_name,
            ax=ax,
            # curve_kwargs={"color": theme.model_to_color(model_name)} if colors is None else {"color": colors[i]},
            color=theme.model_to_color(model_name) if colors is None else colors[i],
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


def plot_y_by_x(df_acc, y="accuracy", x="s_bin", save_path="figs/lp_acc_vs_s.png"):
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    df_acc["model_base"] = df_acc["model"].apply(theme.get_model_base_name)
    sns.lineplot(
        data=df_acc,
        x=x,
        y=y,
        hue="model_base",
        style="model_base",
        errorbar="sd",
        palette=theme.model_color_map,
        markers=True,
        ax=ax,
    )
    ax.set_xlabel(x)
    ax.set_ylabel(y.title())
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    plt.savefig(save_path, dpi=300)
    plt.close()


# def plot_acc_by_test_size()
