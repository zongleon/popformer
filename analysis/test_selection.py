import numpy as np
import os
import pandas as pd
from evaluation.core import BaseEvaluator
from evaluation.models import (
    popformer,
    popformer_lp,
    fasternn,
    schrider_resnet,
    summary_stat,
)
from evaluation.evaluators import random_classification, genome_classification

import matplotlib.pyplot as plt

FORCE = False

if __name__ == "__main__":
    dataset_paths = [
        "data/dataset/pan_4_test",
        "data/dataset/pan_test",
        "data/dataset/pan_test_50000",
        "data/dataset/pan_3_demoid-0_balanced",
        "data/dataset/pan_3_demoid-1_balanced",
        "data/dataset/len200_ghist_const1",
        "data/dataset/len200_ghist_const2",
        "data/dataset/neutral_chr20_CEU",
        "data/dataset/simhumanity_chr20",
    ]
    test_sizes = [0.05]
    train_ds = "pan_train_50000"
    models = []
    for ts in test_sizes:
        models += [
            popformer.PopformerModel(
                f"models/selbin-pt-{train_ds}-{ts}",
                f"popformer-{ts}",
                subsample=(64, 64),
                subsample_type="diverse",
            ),
            popformer.PopformerModel(
                f"models/selbin-ft-{train_ds}-{ts}",
                f"popformer-ft-{ts}",
                subsample=(64, 64),
                subsample_type="diverse",
            ),
            popformer_lp.PopformerLPModel(
                "models/popf-large",
                f"models/lp/{train_ds}_popf-large-{ts}_lp.pkl",
                f"popformer-lp-{ts}",
                subsample=(64, 64),
                subsample_type="diverse",
            ),
            fasternn.FasterNNModel(
                f"models/fasternn/fasternn_{train_ds}-{ts}.pt", f"FASTER-NN-{ts}"
            ),
            schrider_resnet.SchriderResnet(
                model_path=f"models/schrider_resnet/resnet_{train_ds}-{ts}.pt",
                model_name=f"resnet34-{ts}",
            ),
        ]

    models += [
        summary_stat.SummaryStatModel(
            model_name="tajimas_d",
            summary_stat="tajimas_d",
        ),
        summary_stat.SummaryStatModel(
            model_name="sfs_1",
            summary_stat="sfs",
            sfs_index=1,
        ),
        summary_stat.SummaryStatModel(
            model_name="sfs_1_count",
            summary_stat="sfs",
            sfs_index=1,
            proportional=False,
        ),
        summary_stat.SummaryStatModel(
            model_name="sfs_2",
            summary_stat="sfs",
            sfs_index=2,
        ),
        summary_stat.SummaryStatModel(
            model_name="n_snps",
            summary_stat="n_snps",
        ),
    ]
    evaluators: list[BaseEvaluator] = []

    for dataset_path in dataset_paths:
        labels = None  # by default labels are inferred from the dataset
        if "pan" not in dataset_path:
            if "genome" in dataset_path:
                known_paths = [
                    "data/SEL/sel.csv",
                ]  # "data/SEL/reichsel.csv"]
                ds_name = [
                    os.path.basename(dataset_path)
                    + "_"
                    + os.path.basename(kp).split(".")[0]
                    for kp in known_paths
                ]
            else:
                known_paths = [
                    f"data/matrices/bigregions/{os.path.basename(dataset_path)}.csv"
                ]
                ds_name = [None]
            for known_region_path in known_paths:
                evaluator = genome_classification.GenomeClassificationEvaluator(
                    dataset_path,
                    known_selection_region_df=pd.read_csv(known_region_path)
                    if os.path.exists(known_region_path)
                    else None,
                    dataset_name=ds_name.pop(0),
                )
                evaluators.append(evaluator)
        else:
            evaluator = random_classification.RandomClassificationEvaluator(
                dataset_path, labels_path_or_labels=labels
            )
            evaluators.append(evaluator)

    results = {}
    for model in models:
        for evaluator in evaluators:
            print(f"Evaluating {model.model_name} on {evaluator.dataset_name}")
            predictions = evaluator.run(model, FORCE)
            res = evaluator.evaluate(predictions)
            results[(model.model_name, evaluator.dataset_name)] = res

    # convert results to dataframe
    df = pd.DataFrame.from_dict(results, orient="index")
    df.index = pd.MultiIndex.from_tuples(df.index, names=["model", "dataset"])
    df = df.reset_index().sort_values(by=["dataset", "model"])

    models = df["model"].unique().tolist()
    datasets = df["dataset"].unique().tolist()

    if "accuracy" in df.columns:
        df_table = df[
            ["model", "dataset", "accuracy", "precision", "recall", "auroc", "auprc"]
        ].dropna()
        print(df_table.to_string())

    if "obs" in df.columns:
        df_table_g = df[
            ["model", "dataset", "obs", "null_mean", "ci", "p_emp"]
        ].dropna()
        print(df_table_g.to_string())

    # plot roc curves
    for dataset_name in df["dataset"].unique():
        y_trues = [
            results[(model, dataset_name)].get("trues", None) for model in models
        ]
        y_scores = [
            results[(model, dataset_name)].get("preds", None) for model in models
        ]
        if y_trues[0] is None:
            continue
        random_classification.plot_roc_curves(
            y_trues,
            y_scores,
            models,
            save_path=f"figs/{dataset_name}_roc_curves.png",
        )
        random_classification.plot_pr_curves(
            y_trues,
            y_scores,
            models,
            save_path=f"figs/{dataset_name}_pr_curves.png",
        )
        random_classification.plot_score_distributions(
            y_trues,
            y_scores,
            models,
            save_path=f"figs/{dataset_name}_score_distributions.png",
        )

    # plot curves by s
    model = [m for m in models if m.startswith("popformer")][0]
    dataset_name = "pan_test"
    if (model, dataset_name) in results:
        y_trues = results[(model, dataset_name)]["trues"]
        y_scores = results[(model, dataset_name)]["preds"]
        s_masks = {
            f"s={i:.3f}": (results[(model, dataset_name)]["s"] == i)
            | (np.array(y_trues) == 0)
            for i in np.unique(results[(model, dataset_name)]["s"])
            if i != 0
        }

        colors = plt.cm.cividis(np.linspace(0, 1, len(s_masks)))

        random_classification.plot_roc_curves(
            [np.array(y_trues)[s] for s in s_masks.values()],
            [np.array(y_scores)[s] for s in s_masks.values()],
            list(s_masks.keys()),
            colors=colors,
            save_path=f"figs/{dataset_name}_roc_curves_by_s.png",
        )
        random_classification.plot_pr_curves(
            [np.array(y_trues)[s] for s in s_masks.values()],
            [np.array(y_scores)[s] for s in s_masks.values()],
            list(s_masks.keys()),
            colors=colors,
            save_path=f"figs/{dataset_name}_pr_curves_by_s.png",
        )

        # plot facet grid of ROC curves
        fig, axes = plt.subplots(
            1, 5, figsize=(20, 4), layout="constrained", sharex=True, sharey=True
        )

        # models as lines, s_bins as columns, min_freq_bins as rows
        y_trues = [results[(model, dataset_name)]["trues"] for model in models]
        y_scores = [results[(model, dataset_name)]["preds"] for model in models]

        if np.unique(results[(model, dataset_name)]["s"]).shape[0] <= 6:
            s_masks = [
                (f"{i:.3f}", results[(model, dataset_name)]["s"] == i)
                for i in np.unique(results[(model, dataset_name)]["s"])
                if i != 0
            ]
        else:
            s_masks = [
                (
                    f"({i}, {j})",
                    (results[(model, dataset_name)]["s"] > i)
                    & (results[(model, dataset_name)]["s"] <= j),
                )
                for i, j in [
                    (0, 0.02),
                    (0.02, 0.04),
                    (0.04, 0.06),
                    (0.06, 0.08),
                    (0.08, 0.1),
                ]
            ]

        for j, (s_bin_label, s_bin) in enumerate(s_masks):
            # label first row and first column
            axes[j].set_title(f"s={s_bin_label}", fontsize=16)
            combined_m = s_bin | (np.array(y_trues[0]) == 0)
            if np.any(combined_m):
                ax = axes[j]
                random_classification.plot_roc_curves(
                    [np.array(y_true)[combined_m] for y_true in y_trues],
                    [np.array(y_score)[combined_m] for y_score in y_scores],
                    models,
                    ax=ax,
                    add_ax_labels=False,
                )

        plt.savefig(f"figs/{dataset_name}_roc_facet_grid.png", dpi=300)
        plt.close()

    # plot a line plot of accuracy vs test size for each model
    df_acc = df[df["dataset"] == "pan_test"]
    df_acc["train_size"] = df_acc["model"].apply(
        lambda x: 1 - float(x.split("-")[-1]) if "-" in x else 0.0
    )
    df_acc["model"] = df_acc["model"].apply(
        lambda x: "-".join(x.split("-")[:-1]) if "-" in x else x
    )
    df_acc = df_acc[["model", "train_size", "accuracy"]].query(
        "model not in ['sfs_1', 'sfs_1_count', 'sfs_2', 'n_snps']"
    )
    random_classification.plot_acc_by_x(
        df_acc, x="train_size", save_path="figs/pan_test_acc_vs_train_size.png"
    )

    # --- Correlation scatterplots (pan_test only), colored by s ---
    dataset_name = "pan_test"
    if dataset_name in datasets:
        popf_ft = next((m for m in models if m.startswith("popformer-ft")), None)
        if popf_ft is not None and (popf_ft, dataset_name) in results:
            popf_res = results[(popf_ft, dataset_name)]
            s = popf_res.get("s", None)
            popf_scores = popf_res.get("preds", None)

            summary_stat_models = [
                "tajimas_d",
                "sfs_1",
                "sfs_1_count",
                "sfs_2",
                "n_snps",
            ]

            if s is not None and popf_scores is not None:
                for stat_model in summary_stat_models:
                    if (stat_model, dataset_name) not in results:
                        continue
                    stat_scores = results[(stat_model, dataset_name)].get("preds", None)
                    if stat_scores is None:
                        continue

                    # mask NaNs / infs consistently
                    valid = (
                        np.isfinite(popf_scores)
                        & np.isfinite(stat_scores)
                        & np.isfinite(s)
                    )
                    if not np.any(valid):
                        continue

                    genome_classification.plot_correlation(
                        popf_scores[valid],
                        stat_scores[valid],
                        y1lab=f"{popf_ft} score",
                        y2lab=stat_model,
                        color_by=s[valid],
                        color_by_label="s",
                        save_path=f"figs/{dataset_name}_correlation_{stat_model}_popf_ft_colored_by_s.png",
                        sig_mask=None,
                    )

    for dataset_name in df["dataset"].unique():
        # Gather region plot data directly from `results`
        region_data_list = sorted(
            [
                (model_name, res["region_plot_data"])
                for (model_name, ds_name), res in results.items()
                if ds_name == dataset_name and "region_plot_data" in res
            ],
            key=lambda x: x[0],
        )
        if not region_data_list:
            continue

        model_names = [m for m, _ in region_data_list]
        preds_list = [d["preds"] for _, d in region_data_list]
        chrom = region_data_list[0][1].get("chrom", None)
        start_pos = region_data_list[0][1]["start_pos"]
        end_pos = region_data_list[0][1]["end_pos"]

        genome_classification.plot_region(
            preds_list=preds_list,
            model_names=model_names,
            start_pos=start_pos,
            end_pos=end_pos,
            save_path=f"figs/{dataset_name}_region_plot.png",
            line=False,
            window=1,
        )
