"""Shared configuration, model builders, and runner for selection detection evaluation."""

import numpy as np
import pandas as pd
from evaluation.core import BaseEvaluator
from evaluation.models import (
    popformer,
    popformer_lp,
    fasternn,
    schrider_resnet,
    summary_stat,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRAIN_DS = "pan_train_50000"
# TEST_SIZES = [0.05, 0.5, 0.9, 0.95, 0.99]
TEST_SIZES = [0.99]
FORCE = False
INVERT_SCORE_MODELS = {"tajimas_d", "n_snps"}

# Canonical legend order — edit this list to change every plot at once.
MODEL_ORDER = [
    "popformer-0.05",
    "popformer-ft-0.05",
    "popformer-lp-0.05",
    "FASTER-NN-0.05",
    "resnet34-0.05",
    "tajimas_d",
    "sfs_1",
    "sfs_1_count",
    "sfs_2",
    "n_snps",
]


def sort_models(names: list[str]) -> list[str]:
    """Return *names* sorted according to MODEL_ORDER (unknowns go last)."""
    rank = {m: i for i, m in enumerate(MODEL_ORDER)}
    return sorted(names, key=lambda n: rank.get(n, len(MODEL_ORDER)))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def score_transform(model_name: str, scores: np.ndarray) -> np.ndarray:
    """Negate scores for statistics where lower ⇒ more selection-like."""
    return -scores if model_name in INVERT_SCORE_MODELS else scores


def normalize(x: np.ndarray) -> np.ndarray:
    lo, hi = np.min(x), np.max(x)
    return (x - lo) / (hi - lo) if hi > lo else x * 0.0


def aggregate_windows(scores: np.ndarray, sig_mask: np.ndarray, n: int):
    """Bin consecutive windows, averaging scores and OR-ing the mask."""
    if n <= 1:
        return scores, sig_mask
    k = len(scores) // n
    if k == 0:
        return scores, sig_mask
    scores = scores[: k * n].reshape(k, n).mean(axis=1)
    sig_mask = sig_mask[: k * n].reshape(k, n).any(axis=1)
    return scores, sig_mask


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------
def make_nn_models(
    train_ds: str = TRAIN_DS, test_sizes: list[float] = TEST_SIZES, suffix=""
) -> list:
    """Construct all neural-network-based selection models."""
    suffix = suffix + "-" if suffix != "" else ""
    models = []
    for ts in test_sizes:
        models += [
            popformer.PopformerModel(
                f"models/selbin-pt-sm-{train_ds}-{ts}",
                f"popformer-{suffix}{ts}",
                subsample=(64, 64),
                subsample_type="diverse",
            ),
            popformer.PopformerModel(
                f"models/selbin-ft-sm-{train_ds}-{ts}",
                f"popformer-ft-{suffix}{ts}",
                subsample=(64, 64),
                subsample_type="diverse",
            ),
            popformer_lp.PopformerLPModel(
                "models/popf-small",
                f"models/lp/{train_ds}_popf-small-{ts}_lp.pkl",
                f"popformer-lp-{suffix}{ts}",
                subsample=(64, 64),
                subsample_type="diverse",
            ),
            fasternn.FasterNNModel(
                f"models/fasternn/fasternn_{train_ds}-{ts}.pt",
                f"FASTER-NN-{suffix}{ts}",
            ),
            schrider_resnet.SchriderResnet(
                model_path=f"models/schrider_resnet/resnet_{train_ds}-{ts}.pt",
                model_name=f"resnet34-{suffix}{ts}",
            ),
        ]
    return models


def make_summary_stat_models() -> list:
    """Construct summary-statistic baseline models."""
    return [
        summary_stat.SummaryStatModel(model_name="tajimas_d", summary_stat="tajimas_d"),
        summary_stat.SummaryStatModel(
            model_name="sfs_1", summary_stat="sfs", sfs_index=1
        ),
        # summary_stat.SummaryStatModel(
        #     model_name="sfs_1_count",
        #     summary_stat="sfs",
        #     sfs_index=1,
        #     proportional=False,
        # ),
        summary_stat.SummaryStatModel(
            model_name="sfs_2", summary_stat="sfs", sfs_index=2
        ),
        summary_stat.SummaryStatModel(model_name="n_snps", summary_stat="n_snps"),
    ]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_all(
    models: list,
    evaluators: list[BaseEvaluator],
    force: bool = FORCE,
) -> tuple[dict, pd.DataFrame]:
    """Run every model,evaluator pair.  Returns (results_dict, summary_df)."""
    results = {}
    for model in models:
        for evaluator in evaluators:
            print(f"Evaluating {model.model_name} on {evaluator.dataset_name}")
            predictions = evaluator.run(model, force)
            res = evaluator.evaluate(predictions)
            results[(model.model_name, evaluator.dataset_name)] = res

    df = pd.DataFrame.from_dict(results, orient="index")
    df.index = pd.MultiIndex.from_tuples(df.index, names=["model", "dataset"])
    df = df.reset_index().sort_values(by=["dataset", "model"])
    return results, df


def collect_region_data(results: dict, dataset_name: str):
    """Return sorted (model_names, preds_list, start_pos, end_pos, chrom) for a dataset."""
    items = [
        (m, res["region_plot_data"])
        for (m, ds), res in results.items()
        if ds == dataset_name and "region_plot_data" in res
    ]
    order = {name: i for i, name in enumerate(sort_models([m for m, _ in items]))}
    items.sort(key=lambda x: order[x[0]])
    if not items:
        return None
    model_names = [m for m, _ in items]
    preds_list = [d["preds"] for _, d in items]
    start_pos = items[0][1]["start_pos"]
    end_pos = items[0][1]["end_pos"]
    chrom = items[0][1].get("chrom", None)
    return model_names, preds_list, start_pos, end_pos, chrom
