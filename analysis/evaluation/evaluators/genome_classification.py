import re
from ..core import BaseHFEvaluator

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import theme


class GenomeClassificationEvaluator(BaseHFEvaluator):
    """
    Evaluator that makes classifications for windows that are in order (e.g. a genome).
    """

    def __init__(
        self,
        dataset_path,
        labels_path_or_labels=None,
        known_selection_region_df=None,
        s_coeff_df=None,
        *,
        dataset_name=None,
        batch_size=1,
    ):
        super().__init__(
            dataset_path,
            labels_path_or_labels,
            dataset_name=dataset_name,
            batch_size=batch_size,
        )
        self.known_selection_region_df = known_selection_region_df
        self.s_coeff_df = s_coeff_df
        self.windowed_values = None

    def _get_windowed(self, windows, margin=0):
        if self.known_selection_region_df is None:
            raise ValueError(
                "known_selection_region_df must be provided to get windowed labels."
            )

        if self.windowed_values is not None:
            return self.windowed_values

        df = self.known_selection_region_df.copy()
        df = df.rename(columns={"Chromosome": "chrom", "Start": "start", "End": "end"})

        if "Population" in df.columns:
            # if this dataset has a string of 3 capital letters, filter to that population
            pop_regex = r"[A-Z]{3}"
            pop = re.search(pop_regex, self.dataset_name)
            if pop:
                pop = pop.group(0)
                print(f"Filtering known selection regions to population {pop}")
                df = df[df["Population"] == pop]
            else:
                df = df[df["Population"] == "CEU"]

        if margin > 0:
            df["start"] = df["start"] - margin
            df["end"] = df["end"] + margin

        # convert "chr1", etc
        if "chrom" in df.columns:
            if df["chrom"].dtype == object:
                df["chrom"] = df["chrom"].apply(lambda x: int(x.replace("chr", "")))

            df = df.set_index(["chrom", "start"])
        else:
            df = df.set_index(["start"])
        df = df.sort_index()

        grossman = []

        for chrom, start, end in windows:
            try:
                if "chrom" in df.index.names:
                    window_df = df.loc[(chrom, slice(start, end)), :]
                else:
                    window_df = df.loc[slice(start, end), :]
            except KeyError:
                window_df = None

            if window_df is None or window_df.empty:
                grossman.append(0)
            else:
                grossman.append(1)

        grossman = np.array(grossman)
        self.windowed_values = grossman

        return grossman

    def _permutation(self, predictions, region_mask, M=10000, seed=None, roll=False):
        rng = np.random.default_rng(seed)
        obs = predictions[region_mask].mean()
        n = len(predictions)
        null = np.empty(M)
        for b in range(M):
            if roll:
                shift = rng.integers(0, n)
                null[b] = predictions[np.roll(region_mask, shift)].mean()
            else:
                null_mask = rng.permutation(region_mask)
                null[b] = predictions[null_mask].mean()

        p_emp = (1 + np.sum(null >= obs)) / (1 + M)  # one-sided
        fe = obs / null.mean()
        ci_lo, ci_hi = np.percentile(null, [2.5, 97.5])
        ci = (ci_lo, ci_hi)
        # optionally convert to FE CI by dividing obs by null percentiles (note asymmetry)
        fe_ci = (obs / ci_hi, obs / ci_lo)

        return {
            "obs": obs,
            "p_emp": p_emp,
            "null": null,
            "null_mean": null.mean(),
            "ci": ci,
            "fe": fe,
            "fe_ci": fe_ci,
        }

    def evaluate(self, predictions, **kwargs):
        preds = predictions[:, 1]
        preds_metrics = preds
        if kwargs.get("invert_for_metrics", False):
            preds_metrics = 1 - preds
        results = {
            "preds": preds,
            "preds_for_metrics": preds_metrics,
        }
        if hasattr(self, "start_pos"):
            # store plotting data for region predictions
            start_pos = np.array(self.start_pos)
            end_pos = np.array(self.end_pos)
            chrom = np.array(self.chrom)
            results["region_plot_data"] = {
                "start_pos": start_pos,
                "end_pos": end_pos,
                "chrom": chrom,
                "preds": preds,
            }
            if (
                hasattr(self, "known_selection_region_df")
                and self.known_selection_region_df is not None
            ):
                # perform permutation test ala test-reichx
                windows = list(zip(chrom, start_pos, end_pos))
                true_windowed = self._get_windowed(windows, margin=0)
                true_windowed_sig = true_windowed == 1
                # preds = predictions[:, 1]
                # perm_results = self._permutation(
                #     preds,
                #     true_windowed_sig,
                #     M=10000,
                #     seed=42,
                #     roll=False,
                # )
                # perm_results = self._rank_enrichment(
                #     preds,
                #     true_windowed_sig,
                #     chrom,
                #     M=10000,
                #     seed=42,
                # )
                # results.update(perm_results)

                # also results should include a significant windows mask
                results["sig_mask"] = true_windowed_sig

            if hasattr(self, "s_coeff_df") and self.s_coeff_df is not None:
                # correlate predictions with s coefficients
                s_df = self.s_coeff_df.copy()
                s_df = s_df.rename(
                    columns={
                        "Chromosome": "chrom",
                        "Start": "start",
                        "End": "end",
                        "s_coeff": "s_coeff",
                    }
                )
                if "Population" in s_df.columns:
                    s_df = s_df[s_df["Population"] == "CEU"]

                s_values = []
                for chrom_i, start_i, end_i in windows:
                    try:
                        s_row = s_df[
                            (s_df["chrom"] == chrom_i)
                            & (s_df["start"] <= start_i)
                            & (s_df["end"] >= end_i)
                        ]
                        if not s_row.empty:
                            # mean if multiple
                            s_values.append(s_row["s_coeff"].mean())
                        else:
                            s_values.append(np.nan)
                    except KeyError:
                        s_values.append(np.nan)

                s_values = np.array(s_values)
                results["s_coeff"] = s_values
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
    elif window_type == "max":
        p_pad = np.pad(data, (window // 2, window - 1 - window // 2), mode="edge")
        data = np.array([np.max(p_pad[i : i + window]) for i in range(len(data))])
    elif window_type == "median":
        p_pad = np.pad(data, (window // 2, window - 1 - window // 2), mode="edge")
        data = np.array([np.median(p_pad[i : i + window]) for i in range(len(data))])
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
    line=True,
):
    ylbl = "score"
    pos = (end_pos + start_pos) // 2

    preds_adj = []
    for p in preds_list:
        if window > 1:
            p = _windowed_mean(p, window=window, window_type=window_type)
        preds_adj.append(p)

    fig, axs = plt.subplots(len(preds_adj), 1, figsize=(12, 6 * len(preds_adj)))
    if len(preds_adj) == 1:
        axs = [axs]
    for p, label, ax in zip(preds_adj, model_names, axs):
        if line:
            ax.plot(
                pos[window:-window],
                p[window:-window],
                alpha=0.7,
                label=label,
                color=theme.model_to_color(label),
            )
        else:
            ax.scatter(
                pos[window:-window],
                p[window:-window],
                alpha=0.4,
                label=label,
                color=theme.model_to_color(label),
            )

        if label_df is not None:
            for idx, r in label_df.iterrows():
                x0 = r["start"] - 100000
                x1 = r["end"] + 100000 # add some padding to make it more visible
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
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_boxplot(y_preds, model_names, sig_mask, save_path="figs/lp_boxplot.png"):
    # plot boxplot of predictions in sig vs non-sig regions, for each model
    df = []
    for preds, model_name in zip(y_preds, model_names):
        for i in range(len(preds)):
            df.append(
                {
                    "model": model_name,
                    "pred_prob": preds[i],
                    "region": "significant" if sig_mask[i] else "non-significant",
                }
            )
    df = pd.DataFrame(df)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxenplot(
        data=df,
        x="model",
        y="pred_prob",
        hue="region",
        ax=ax,
    )
    ax.set_ylabel("normalized score")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_histogram_with_line(y_pred, line_at, save_path="figs/lp_histline.png"):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(
        y_pred,
        stat="density",
        ax=ax,
    )
    ax.axvline(line_at, color="red", linestyle="--", label="")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_correlation(
    y1,
    y2,
    y1lab,
    y2lab,
    sig_mask=None,
    save_path="figs/lp_s_coeff_correlation.png",
    *,
    color_by=None,
    color_by_label="s",
    cmap="cividis",
    alpha=0.5,
    add_colorbar=True,
):
    fig, ax = plt.subplots(figsize=(8, 8))

    if color_by is None:
        sns.regplot(
            x=y1,
            y=y2,
            ax=ax,
            scatter_kws={"alpha": alpha},
            line_kws=dict(color="orange"),
        )
    else:
        # regression line (like test_selection_real) + scatter colored by `color_by`
        sns.regplot(
            x=y1,
            y=y2,
            ax=ax,
            scatter=False,
            line_kws=dict(color="orange"),
        )
        sc = ax.scatter(y1, y2, c=color_by, cmap=cmap, alpha=alpha, s=18, linewidths=0)

        if add_colorbar:
            cbar = fig.colorbar(sc, ax=ax, pad=0.01)
            cbar.set_label(color_by_label)

    if sig_mask is not None:
        y1sig = y1[sig_mask]
        y2sig = y2[sig_mask]
        ax.scatter(y1sig, y2sig, color="red", s=18, alpha=0.8)

    spearmanr = pd.Series(y1).corr(pd.Series(y2), method="spearman")
    ax.set_xlabel(y1lab)
    ax.set_ylabel(y2lab)
    ax.set_title(f"{y2lab} Spearman r = {spearmanr:.3f}")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_enrichment_at_k(
    enrichment_series,
    dataset_name,
    save_path="figs/enrichment_at_k.png",
):
    """Plot fold-enrichment at top-k fraction for each model.

    Parameters
    ----------
    enrichment_series : list of (model_name, k_fractions, enrichments)
        Each entry contains a model name, an array of top-k fractions
        (0–1), and the corresponding fold-enrichment values.
    dataset_name : str
        Used in the plot title.
    save_path : str
        Where to save the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for model_name, k_fracs, enrichment in sorted(
        enrichment_series, key=lambda x: x[0]
    ):
        ax.plot(
            k_fracs,
            enrichment,
            label=model_name,
            color=theme.model_to_color(model_name),
        )
    ax.axhline(1.0, color="grey", linestyle=":", alpha=0.6, label="baseline")
    ax.set_xlabel("Fraction of genome called")
    ax.set_ylabel("Fold enrichment")
    ax.set_title(f"Enrichment — {dataset_name}")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper right")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_rate_vs_threshold(
    fdr_series,
    dataset_name,
    rate_name="FNR",
    save_path="figs/fnr_vs_threshold.png",
):
    fig, ax = plt.subplots(figsize=(8, 6))
    for model_name, thresholds, fdr in fdr_series:
        ax.plot(
            thresholds,
            fdr,
            label=theme.get_model_base_name(model_name),
            color=theme.model_to_color(model_name),
        )
    ax.set_xlabel("Fraction of genome predicted selected")
    ax.set_ylabel(rate_name)
    ax.set_title(''.join([c for c in dataset_name if c.isupper()]))
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="lower right")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_called_pos_vs_neg(
    called_series,
    dataset_name,
    save_path="figs/pos_called_vs_neg_called.png",
):
    """Plot # positive regions called vs # negative regions called.

    Parameters
    ----------
    called_series : list of (model_name, neg_called, pos_called)
        Each entry holds arrays for x (# negatives called) and y
        (# positives called) as threshold varies.
    dataset_name : str
        Label for title.
    save_path : str
        Output figure path.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for model_name, neg_called, pos_called in called_series:
        ax.plot(
            neg_called,
            pos_called,
            label=theme.get_model_base_name(model_name),
            color=theme.model_to_color(model_name),
        )

    ax.set_xlabel("# Reich et al. negatives called")
    ax.set_ylabel("# Grossman et al. positives called")
    ax.set_title(''.join([c for c in dataset_name if c.isupper()]))
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="lower right")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
