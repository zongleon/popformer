"""Shared configuration, model builders, and runner for selection detection evaluation."""

import os

import numpy as np
import pandas as pd
from evaluation.core import BaseEvaluator, BaseModel
from evaluation.models import (
    fasternn,
    popformer,
    popformer_lp,
    schrider_resnet,
    summary_stat,
)
from theme import INVERT_SCORE_MODELS


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


def models_to_models(models: list[str], model_names: list[str]) -> list[BaseModel]:
    """Convert model names to model instances."""
    out_models = []
    for model_path, model_name in zip(models, model_names):
        if "fasternn" in model_path:
            model = fasternn.FasterNNModel(model_path, model_name)
        elif "resnet" in model_path:
            model = schrider_resnet.SchriderResnet(model_path, model_name)
        elif "lp" in model_path:
            full_name = os.path.splitext(os.path.basename(model_path))[0]
            pt_model = "models/" + full_name.split("__")[0]
            model = popformer_lp.PopformerLPModel(
                pt_model,
                model_path,
                model_name,
                subsample=(64, 64),
                subsample_type="random",
            )
        else:
            model = popformer.PopformerModel(
                model_path, model_name, subsample=(64, 64), subsample_type="random"
            )

        out_models.append(model)

    return out_models


def make_summary_stat_models() -> list:
    """Construct summary-statistic baseline models."""
    return [
        summary_stat.SummaryStatModel(model_name="tajimas_d", summary_stat="tajimas_d"),
        summary_stat.SummaryStatModel(
            model_name="sfs_1", summary_stat="sfs", sfs_index=1
        ),
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
    force: bool = False,
) -> tuple[dict, pd.DataFrame]:
    """Run every model,evaluator pair.  Returns (results_dict, summary_df)."""
    results = {}
    for model in models:
        for evaluator in evaluators:
            print(f"Evaluating {model.model_name} on {evaluator.dataset_name}")
            predictions = evaluator.run(model, force)
            res = evaluator.evaluate(
                predictions, invert_for_metrics=model.model_name in INVERT_SCORE_MODELS
            )
            results[(model.model_name, evaluator.dataset_name)] = res

    df = pd.DataFrame.from_dict(results, orient="index")
    df.index = pd.MultiIndex.from_tuples(df.index, names=["model", "dataset"])
    df = df.reset_index().sort_values(by=["dataset", "model"])
    # explode the metrics columns into separate rows
    if "accuracy" in df.columns:
        df = df.explode(
            ["accuracy", "auc", "auprc", "precision", "recall", "balanced_accuracy"]
        )
    return results, df


def collect_region_data(results: dict, dataset_name: str):
    """Return sorted (model_names, preds_list, start_pos, end_pos, chrom) for a dataset."""
    items = [
        (m, res["region_plot_data"])
        for (m, ds), res in results.items()
        if ds == dataset_name and "region_plot_data" in res
    ]
    order = {name: i for i, name in enumerate([m for m, _ in items])}
    items.sort(key=lambda x: order[x[0]])
    if not items:
        return None
    model_names = [m for m, _ in items]
    preds_list = [d["preds"] for _, d in items]
    start_pos = items[0][1]["start_pos"]
    end_pos = items[0][1]["end_pos"]
    chrom = items[0][1].get("chrom", None)
    return model_names, preds_list, start_pos, end_pos, chrom
