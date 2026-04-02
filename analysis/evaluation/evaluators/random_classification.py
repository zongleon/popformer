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
from selection_config import MODEL_ORDER
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

    def evaluate(self, predictions, **kwargs):
        results = {}
        pos_preds = predictions[:, 1]
        binary_preds = predictions.argmax(axis=1)

        pos_preds_metrics = pos_preds
        binary_preds_metrics = binary_preds
        if kwargs.get("invert_for_metrics", False):
            pos_preds_metrics = 1 - pos_preds_metrics
            binary_preds_metrics = 1 - binary_preds_metrics

        acc = accuracy_score(self.labels, binary_preds_metrics)
        aucroc = roc_auc_score(self.labels, pos_preds_metrics)
        auprc = average_precision_score(self.labels, pos_preds_metrics)
        precision = precision_score(self.labels, binary_preds_metrics)
        recall = recall_score(self.labels, binary_preds_metrics)

        facet_vars = [
            k
            for k in [
                "s",
                "shoulder",
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
                "preds_for_metrics": pos_preds_metrics,
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
    fig, axs = plt.subplots(len(model_names), 1, figsize=(8, 6 * len(model_names)))

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
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_curves(
    y_trues,
    y_scores,
    model_names,
    dataset=None,
    curve_type="roc",
    ax=None,
    add_ax_labels=True,
    baseify_model_names=True,
    save_path="figs/roc_curves.png",
    legend_fontsize=None,
):
    new = False
    if ax is None:
        new = True
        fig, ax = plt.subplots(figsize=(8, 6))

    if not add_ax_labels:
        old_xlabel = ax.get_xlabel()
        old_ylabel = ax.get_ylabel()

    if curve_type == "roc":
        plot_fn = RocCurveDisplay.from_predictions
        xlab = "False Positive Rate"
        ylab = "True Positive Rate"
    elif curve_type == "pr":
        plot_fn = PrecisionRecallDisplay.from_predictions
        xlab = "Recall"
        ylab = "Precision"
    else:
        raise ValueError(f"Unsupported curve type: {curve_type}. Choose 'roc' or 'pr'.")

    def color_arg(model_name):
        if curve_type == "roc":
            return {"curve_kwargs": {"color": theme.model_to_color(model_name)}}
        else:
            return {"color": theme.model_to_color(model_name)}

    for i, (y_true, y_score, model_name) in enumerate(
        zip(y_trues, y_scores, model_names)
    ):
        if y_true is None:
            continue
        plot_fn(
            y_true,
            y_score,
            name=theme.get_model_base_name(model_name)
            if baseify_model_names
            else model_name,
            ax=ax,
            **color_arg(model_name),
        )

        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        if legend_fontsize is not None:
            ax.legend(fontsize=legend_fontsize)
    
    if dataset is not None:
        ax.set_title(theme.dataset_rename_map.get(dataset, dataset))

    if not add_ax_labels:
        ax.set_xlabel(old_xlabel)
        ax.set_ylabel(old_ylabel)

    if new:
        if add_ax_labels:
            plt.xlabel(xlab)
            plt.ylabel(ylab)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()


def plot_y_by_x(df_acc, y="accuracy", x="s_bin", save_path="figs/lp_acc_vs_s.png"):
    fig, ax = plt.subplots(figsize=(8, 6))
    df_acc["model"] = df_acc["model"].apply(theme.get_model_base_name)
    sns.lineplot(
        data=df_acc,
        x=x,
        y=y,
        hue="model",
        hue_order=[m for m in MODEL_ORDER if m in df_acc["model"].unique()],
        style="model",
        style_order=[m for m in MODEL_ORDER if m in df_acc["model"].unique()],
        errorbar="sd",
        palette=theme.model_color_map,
        markers=True,
        ax=ax,
    )
    ax.set_xlabel(x)
    ax.set_ylabel(y.title())
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# def plot_acc_by_test_size()
