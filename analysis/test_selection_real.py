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
        # "data/dataset/genome_CEU_chr1",
        "data/dataset/genome_CEU",
        # "data/dataset/genome_YRI",
        # "data/dataset/genome_CHB",
    ]
    test_sizes = [0.05]
    train_ds = "pan_train_50000"
    models = []
    for ts in test_sizes:
        models += [
            popformer.PopformerModel(
                f"models/selbin-ft-{train_ds}-{ts}",
                f"popformer-ft-{ts}",
                subsample=(64, 64),
                subsample_type="diverse",
            ),
        ]

    models += [
        summary_stat.SummaryStatModel(
            model_name="tajimas_d",
            summary_stat="tajimas_d",
        ),
        summary_stat.SummaryStatPosModel(
            model_name="recomb_max",
            variant_file="/bigdata/smathieson/1000g-share/HDF5/CEU.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.h5",
            summary_stat="recomb",
        ),
        summary_stat.SummaryStatPosModel(
            model_name="ihs_max",
            variant_file="/bigdata/smathieson/1000g-share/HDF5/CEU.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.h5",
            summary_stat="ihs",
            score="max",
        ),
        summary_stat.SummaryStatPosModel(
            model_name="ihs_score",
            variant_file="/bigdata/smathieson/1000g-share/HDF5/CEU.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.h5",
            summary_stat="ihs",
            score="score",
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
        summary_stat.SummaryStatPosModel(
            model_name="recomb_max",
            variant_file="/bigdata/smathieson/1000g-share/HDF5/CEU.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.h5",
            summary_stat="recomb",
        ),
    ]
    evaluators: list[BaseEvaluator] = []

    for dataset_path in dataset_paths:
        labels = None  # by default labels are inferred from the dataset
        known_paths = [
            "data/SEL/sel.csv",
        ]  # "data/SEL/reichsel.csv"]
        ds_name = [
            os.path.basename(dataset_path) + "_" + os.path.basename(kp).split(".")[0]
            for kp in known_paths
        ]
        for known_region_path in known_paths:
            evaluator = genome_classification.GenomeClassificationEvaluator(
                dataset_path,
                known_selection_region_df=pd.read_csv(known_region_path),
                dataset_name=ds_name.pop(0),
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

    if "obs" in df.columns:
        df_table_g = df[
            ["model", "dataset", "obs", "null_mean", "ci", "p_emp"]
        ].dropna()
        print(df_table_g.to_string())

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

        for i in range(len(model_names)):
            np.savez(
                f"preds/scans/{dataset_name}_{model_names[i]}_region_plot_data.npz",
                chrom=chrom,
                start_pos=start_pos,
                end_pos=end_pos,
                preds=preds_list[i],
            )
        genome_classification.plot_region(
            preds_list=preds_list,
            model_names=model_names,
            start_pos=start_pos,
            end_pos=end_pos,
            save_path=f"figs/{dataset_name}_region_plot.png",
            line=False,
            window=1,
        )

    sig_masks = {ds: [] for ds in df["dataset"].unique()}
    for (model_name, dataset_name), res in results.items():
        if "sig_mask" in res:
            sig_mask = res["sig_mask"]
            sig_masks[dataset_name].append((model_name, sig_mask))

    for dataset_name, sig_mask_list in sig_masks.items():
        if not sig_mask_list:
            continue

        y_preds = [
            results[(model_name, dataset_name)]["preds"]
            for model_name, _ in sig_mask_list
        ]
        genome_classification.plot_boxplot(
            # normalize y_preds
            y_preds=[(y - np.min(y)) / (np.max(y) - np.min(y)) for y in y_preds],
            # y_preds=y_preds,
            model_names=[model_name for model_name, _ in sig_mask_list],
            sig_mask=sig_mask_list[0][1],  # all sig_masks are the same
            save_path=f"figs/{dataset_name}_boxplot.png",
        )

    # for genome, plot correlations of predictions with tajima's d
    genome = "genome_CEU_sel"
    # first popformer model
    model = [m for m in models if "popformer" in m][0]
    popf_genome = [results[(model, genome)]["preds"]]

    for stat in [
        ("Tajima's D", "tajimas_d"),
        ("Max iHS", "ihs_max"),
        ("iHS Score", "ihs_score"),
        ("Recombination Rate", "recomb_max"),
        ("SFS[1]", "sfs_1"),
        ("SFS[2]", "sfs_2"),
        ("Number of SNPs", "n_snps"),
    ]:
        stat_name, stat_genome = stat
        stat_results = results[(stat_genome, genome)]["preds"]
        # mask nans
        valid_mask = ~np.isnan(stat_results)
        popf_results = popf_genome[0][valid_mask]
        stat_results = stat_results[valid_mask]
        genome_classification.plot_correlation(
            popf_results,
            stat_results,
            y1lab=f"{model} score",
            y2lab=stat_name,
            save_path=f"figs/genome_correlation_{stat_name.replace(' ', '_').lower()}_popf.png",
        )

    # for genome, plot histogram of null distribution with line at obs
    # only for popf-ft model
    for (model_name, dataset_name), res in results.items():
        if model_name != "popformer-ft-0.05":
            continue
        if "obs" not in res or "null" not in res:
            continue
        obs = res["obs"]
        p_emp = res["p_emp"]
        null = res["null"]

        genome_classification.plot_histogram_with_line(
            null,
            obs,
            save_path=f"figs/{dataset_name}_null_distribution.png",
        )
