from ..core import BaseHFEvaluator

import numpy as np
import matplotlib.pyplot as plt


class GenomeClassificationEvaluator(BaseHFEvaluator):
    """
    Evaluator that makes classifications for windows that are in order (e.g. a genome).
    """
    def evaluate(self, predictions):
        if hasattr(self, "start_pos"):
            # store plotting data for region predictions
            start_pos = np.array(self.start_pos)
            end_pos = np.array(self.end_pos)
            results = {}
            results["region_plot_data"] = {
                "start_pos": start_pos,
                "end_pos": end_pos,
                "preds": predictions[:, 1],
            }

        return results


def _windowed_mean(
    data: np.ndarray, window: int, window_type: str = "mean"
) -> np.ndarray:
    if window_type == "mean":
        kernel = np.ones(window, dtype=float) / window
        data = np.convolve(data, kernel, mode="same")
    elif window_type == "min":
        p_pad = np.pad(data, (window // 2, window - 1 - window // 2), mode="edge")
        data = np.array([np.min(p_pad[i : i + window]) for i in range(len(data))])
    return data


def plot_region(
    preds_list,
    model_names,
    start_pos,
    end_pos,
    save_path="figs/lp_region_preds.png",
    window=3,
    label_df=None,
    window_type="mean",
):
    ylbl = "pred. probability of selection"
    pos = (end_pos + start_pos) // 2

    preds_adj = []
    for p in preds_list:
        if window > 1:
            p = _windowed_mean(p, window=window, window_type=window_type)
        preds_adj.append(p)

    fig, axs = plt.subplots(len(preds_adj), 1, figsize=(8, 6 * len(preds_adj)), layout="constrained")
    colors = plt.cm.get_cmap("tab10").colors
    for p, label, ax, color in zip(preds_adj, model_names, axs, colors):
        ax.scatter(pos[window:-window], p[window:-window], alpha=0.4, label=label, color=color)

        if label_df is not None:
            for idx, r in label_df.iterrows():
                x0 = r["start"]
                x1 = r["end"]
                if idx == 0:
                    label = "Selection region"
                else:
                    label = None
                ax.axvspan(x0, x1, color="purple", alpha=0.4, label=label)

        ax.legend(loc="upper right")
        ax.set_xlabel("Position (bp)")
        ax.set_ylabel(ylbl)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.ticklabel_format(style="plain", axis="x", scilimits=(0, 0))
    plt.savefig(save_path, dpi=300)
