import numpy as np
import os
import pandas as pd
from evaluation.core import BaseEvaluator
from evaluation.models import popformer, popformer_lp, fasternn, schrider_resnet, summary_stat
from evaluation.evaluators import random_classification, genome_classification

import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-talk")

FORCE = False

if __name__ == "__main__":
    dataset_paths = [
        # "data/dataset/pan_4_test_balanced",
        "data/dataset/pan_4_test_with_low_s",
        "data/dataset/pan_3_demoid-0_balanced",
        "data/dataset/pan_3_demoid-1_balanced",
        "data/dataset/len200_ghist_const1",
        "data/dataset/len200_ghist_const2",
        "data/dataset/genome_CEU",
        # "data/dataset/genome_CEU_chr1",
        # "data/dataset/genome_CEU_50000_thresh_50"
        # "data/dataset/genome_CEU_100000"
        # "data/dataset/ghist_multisweep",
        # "data/dataset/ghist_multisweep.growth_bg",
        # "data/dataset/ghist_singlesweep",
        # "data/dataset/ghist_singlesweep.growth_bg",
        # "data/dataset/ghist_multisweep_final",
        # "data/dataset/ghist_multisweep.growth_bg_final",
        # "data/dataset/ghist_singlesweep_final",
        # "data/dataset/ghist_singlesweep.growth_bg_final",
    ]
    models = [
        popformer.PopformerModel(
            "models/selbin",
            "popf-ft",
            subsample=(64, 64),
            subsample_type="diverse",
        ),
        popformer_lp.PopformerLPModel(
            "models/popf-small",
            "models/lp/pan_4_train_with_low_s_popf-small_lp.pkl",
            "popf-lp",
            subsample=(64, 64),
            subsample_type="diverse",
        ),
        fasternn.FasterNNModel("models/fasternn/fasternn.pt", "FASTER-NN"),
        schrider_resnet.SchriderResnet(
            model_path="models/schrider_resnet/resnet.pt",
            model_name="resnet34",
        ),
        # summary_stat.PopformerModel(
        #     model_name="pi",
        #     summary_stat="pi",
        # ),
        summary_stat.PopformerModel(
            model_name="tajimas_d",
            summary_stat="tajimas_d",
        ),
    ]
    evaluators: list[BaseEvaluator] = []

    for dataset_path in dataset_paths:
        labels = None  # by default labels are inferred from the dataset
        if "ghist" in dataset_path or "genome" in dataset_path:
            if "genome" in dataset_path:
                known_paths = ["data/SEL/sel.csv", "data/SEL/reichsel.csv"]
                ds_name = [os.path.basename(dataset_path) + "_" + os.path.basename(kp).split(".")[0] for kp in known_paths]
            else:
                known_paths = [f"data/matrices/bigregions/{os.path.basename(dataset_path)}.csv"]
                ds_name = [None]
            for known_region_path in known_paths:
                evaluator = genome_classification.GenomeClassificationEvaluator(
                    dataset_path,
                    known_selection_region_df=pd.read_csv(known_region_path) if os.path.exists(known_region_path) else None,
                    dataset_name=ds_name.pop(0),
                )
                evaluators.append(evaluator)
        else:
            evaluator = random_classification.RandomClassificationEvaluator(
                dataset_path, labels_path_or_labels=labels
            )
            evaluators.append(evaluator)

    results = {}
    preds = {}
    trues = {}
    for model in models:
        for evaluator in evaluators:
            print(f"Evaluating {model.model_name} on {evaluator.dataset_name}")
            predictions = evaluator.run(model, FORCE)
            res = evaluator.evaluate(predictions)
            results[(model.model_name, evaluator.dataset_name)] = res
            preds[(model.model_name, evaluator.dataset_name)] = predictions
            trues[(model.model_name, evaluator.dataset_name)] = evaluator.trues()

    # convert results to dataframe
    df = pd.DataFrame.from_dict(results, orient="index")
    df.index = pd.MultiIndex.from_tuples(df.index, names=["model", "dataset"])
    df = df.reset_index().sort_values(by=["dataset", "model"])

    if "accuracy" in df.columns:
        df_table = df[
            ["model", "dataset", "accuracy", "precision", "recall", "auroc", "auprc"]
        ].dropna()
        print(df_table.to_string())

    if "obs" in df.columns:
        df_table_g = df[["model", "dataset", "obs", "null_mean", "ci", "p_emp"]].dropna()
        print(df_table_g.to_string())

    # plot roc curves
    for dataset_name in df["dataset"].unique():
        y_trues = [trues[key] for key in preds.keys() if key[1] == dataset_name]
        y_scores = [preds[key][:, 1] for key in preds.keys() if key[1] == dataset_name]
        model_names = [key[0] for key in preds.keys() if key[1] == dataset_name]
        if y_trues[0] is None:
            continue
        random_classification.plot_roc_curves(
            y_trues,
            y_scores,
            model_names,
            save_path=f"figs/{dataset_name}_roc_curves.png",
        )
        random_classification.plot_pr_curves(
            y_trues,
            y_scores,
            model_names,
            save_path=f"figs/{dataset_name}_pr_curves.png",
        )

    # plot curves by s
    s_masks = {ds: [] for ds in df["dataset"].unique()}
    for (model_name, dataset_name), res in results.items():
        if "s_masks" in res:
            s_mask = res["s_masks"]
            min_freq_mask = res.get("min_freq_masks", None)
            s_masks[dataset_name].append((model_name, s_mask, min_freq_mask))
    for dataset_name, s_mask_list in s_masks.items():
        if "pan_4" not in dataset_name:
            continue

        for model, s_mask, min_freq_mask in s_mask_list:
            if model != "popf-ft":
                continue
            sm = [item["mask"] for item in s_mask[::-1]]
            names = [item["s_bin"] for item in s_mask[::-1]]
            y_trues = trues[(model, dataset_name)]
            y_scores = preds[(model, dataset_name)][:, 1]
            colors = plt.cm.cividis(np.linspace(0, 1, len(sm)))

            random_classification.plot_roc_curves(
                [np.array(y_trues)[s] for s in sm],
                [np.array(y_scores)[s] for s in sm],
                names,
                colors=colors,
                save_path=f"figs/{dataset_name}_roc_curves_by_s.png",
            )
            random_classification.plot_pr_curves(
                [np.array(y_trues)[s] for s in sm],
                [np.array(y_scores)[s] for s in sm],
                names,
                colors=colors,
                save_path=f"figs/{dataset_name}_pr_curves_by_s.png",
            )
    
    # plot facet grid of ROC curves
    for dataset_name, s_mask_list in s_masks.items():
        # only do for pan_4 datasets
        if "pan_4" not in dataset_name:
            continue

        fig, axes = plt.subplots(5, 5, figsize=(20, 20), layout="constrained", sharex=True, sharey=True)

        # models as lines, s_bins as columns, min_freq_bins as rows
        for model, s_mask, min_freq_mask in s_mask_list:
            y_trues = trues[(model, dataset_name)]
            y_scores = preds[(model, dataset_name)][:, 1]

            if min_freq_mask is not None:
                for i, min_freq_item in enumerate(min_freq_mask[1:]):
                    min_freq_value = min_freq_item["facet_value"]
                    min_freq_m = min_freq_item["mask"]
                    for j, s_item in enumerate(s_mask):
                        # label first row and first column
                        if i == 0:
                            axes[0, j].set_title(f"s_bin={s_item['s_bin']}", fontsize=16)
                        if j == 0:
                            axes[i, 0].set_ylabel(f"min_freq_bin={min_freq_value}", fontsize=16)
                        s_value = s_item["s_bin"]
                        s_m = s_item["mask"]
                        combined_m = (min_freq_m & s_m) | (np.array(y_trues) == 0)
                        if np.any(combined_m):
                            ax = axes[i, j]
                            random_classification.plot_pr_curves(
                                [np.array(y_trues)[combined_m]],
                                [np.array(y_scores)[combined_m]],
                                [f"{model}"],
                                ax=ax,
                            )
    
        plt.savefig(f"figs/{dataset_name}_roc_facet_grid.png", dpi=300)
        plt.close()

    # plot region predictions for genome classification
    region_plot = {ds: [] for ds in df["dataset"].unique()}
    for (model_name, dataset_name), res in results.items():
        if not ("popf" in model_name or "lp" in model_name or model_name in ["pi", "tajimas_d", "ihs"]):
            continue
        if "region_plot_data" in res:
            region_data = res["region_plot_data"]
            region_plot[dataset_name].append((model_name, region_data))
            print(
                f"{model_name} on {dataset_name} has region plot data shape {region_data['preds'].shape}"
            )
    for dataset_name, region_data_list in region_plot.items():
        if region_data_list:
            model_names = [item[0] for item in region_data_list]
            preds_list = [item[1]["preds"] for item in region_data_list]
            start_pos = region_data_list[0][1]["start_pos"]
            end_pos = region_data_list[0][1]["end_pos"]
            chrom = region_data_list[0][1]["chrom"]

            if dataset_name.startswith("genome") or "ghist" in dataset_name:
                for i in range(len(model_names)):
                    np.savez(
                        f"preds/{dataset_name}_{model_names[i]}_region_plot_data.npz",
                        chrom=chrom,
                        start_pos=start_pos,
                        end_pos=end_pos,
                        preds=preds_list[i],
                    )
                if "genome" in dataset_name:
                    continue

            genome_classification.plot_region(
                model_names=model_names,
                preds_list=preds_list,
                start_pos=start_pos,
                end_pos=end_pos,
                window=1,
                window_type="mean",
                label_df=pd.read_csv(f"data/matrices/bigregions/{dataset_name}.csv") if dataset_name.startswith("len200") else None,
                save_path=f"figs/{dataset_name}_region_plot.png",
                line=True
            )

    # plot region predictions for genome classification
    sig_masks = {ds: [] for ds in df["dataset"].unique()}
    for (model_name, dataset_name), res in results.items():
        if not ("popf" in model_name or "lp" in model_name or model_name in ["pi", "tajimas_d", "ihs"]):
            continue
        if "sig_mask" in res:
            sig_mask = res["sig_mask"]
            sig_masks[dataset_name].append((model_name, sig_mask))

    for dataset_name, sig_mask_list in sig_masks.items():
        if not sig_mask_list:
            continue
        genome_classification.plot_boxplot(
            # normalize y_preds
            y_preds = [
                (preds[(model_name, dataset_name)][:, 1] - np.min(preds[(model_name, dataset_name)][:, 1])) /
                (np.max(preds[(model_name, dataset_name)][:, 1]) - np.min(preds[(model_name, dataset_name)][:, 1]))
                for model_name, _ in sig_mask_list
            ],

            model_names=[model_name for model_name, _ in sig_mask_list],
            sig_mask=sig_mask_list[0][1],  # all sig_masks are the same
            save_path=f"figs/{dataset_name}_boxplot.png",
        )

    # for genome, plot correlations of predictions with tajima's d
    taj_d_genome = [y_scores for (model_name, dataset_name), y_scores in preds.items() if model_name == "tajimas_d" and "genome" in dataset_name]
    popf_genome = [y_trues for (model_name, dataset_name), y_trues in preds.items() if model_name == "popf-small-ft" and "genome" in dataset_name]
    
    if taj_d_genome and popf_genome:
        genome_classification.plot_correlation(
            popf_genome[0][:, 1],
            taj_d_genome[0][:, 1],
            y1lab="popf-ft score",
            y2lab="Tajima's D",
            save_path="figs/genome_correlation_tajimas_d_popf.png",
        )

    # for genome, plot correlation with s coefficients
    

