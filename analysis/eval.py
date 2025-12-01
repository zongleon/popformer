import numpy as np
import os
import pandas as pd
from evaluation.core import BaseEvaluator
from evaluation.models import popformer, popformer_lp, fasternn, schrider_resnet
from evaluation.evaluators import random_classification, genome_classification

import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-talk")

FORCE = True

if __name__ == "__main__":
    dataset_paths = [
        # "data/dataset/pan_4_test_balanced",
        # "data/dataset/pan_3_demoid-0_balanced",
        # "data/dataset/pan_3_demoid-1_balanced",
        # "data/dataset/len200_ghist_const1",
        # "data/dataset/len200_ghist_const2",
        # "data/dataset/genome_CEU",
        # "data/dataset/genome_CEU_chr1",
        # "data/dataset/genome_CEU_50000_thresh_50"
        # "data/dataset/genome_CEU_100000"
        # "data/dataset/pan_4_test_with_low_s",
        "data/dataset/ghist_multisweep",
        "data/dataset/ghist_multisweep.growth_bg",
        "data/dataset/ghist_singlesweep",
        "data/dataset/ghist_singlesweep.growth_bg",
        "data/dataset/ghist_multisweep_final",
        "data/dataset/ghist_multisweep.growth_bg_final",
        "data/dataset/ghist_singlesweep_final",
        "data/dataset/ghist_singlesweep.growth_bg_final",
    ]
    models = [
        # popformer.PopformerModel(
        #     "models/selbin",
        #     "popf-small-ft",
        #     subsample=(64, 64),
        #     subsample_type="diverse",
        # ),
        # popformer_lp.PopformerLPModel(
        #     "models/popf-small",
        #     "models/lp/pan_4_popf-small_lp.pkl",
        #     "popf-small-lp",
        #     subsample=(64, 64),
        #     subsample_type="diverse",
        # ),
        # fasternn.FasterNNModel("models/fasternn/fasternn.pt", "FASTER-NN"),
        # schrider_resnet.SchriderResnet(
        #     model_path="models/schrider_resnet/schrider_resnet.pt",
        #     model_name="resnet34",
        # ),
        # popformer.PopformerModel(
        #     "models/selbin_with_low_s",
        #     "popf-small-ft-low-s",
        #     subsample=(32, 32),
        #     subsample_type="diverse",
        # ),
        popformer_lp.PopformerLPModel(
            "models/popf-small",
            "models/lp/pan_4_train_with_low_s_popf-small_lp.pkl",
            "popf-small-lp-low-s",
            subsample=(64, 64),
            subsample_type="diverse",
        ),
    ]
    evaluators: list[BaseEvaluator] = []

    for dataset_path in dataset_paths:
        labels = None  # by default labels are inferred from the dataset
        if "ghist" in dataset_path or "genome" in dataset_path:
            known_region_path = "data/SEL/sel.csv" if "genome" in dataset_path else (
                f"data/matrices/bigregions/{os.path.basename(dataset_path)}.csv"
            )
            evaluator = genome_classification.GenomeClassificationEvaluator(
                dataset_path,
                known_selection_region_df=pd.read_csv(known_region_path) if os.path.exists(known_region_path) else None,
            )
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

    # faceted AUROC vs s plots (aggregate across models per dataset)
    faceted_datasets: dict[str, list[pd.DataFrame]] = {}
    for (model_name, dataset_name), res in results.items():
        if "auroc_s_facets_df" in res:
            df_facets = res["auroc_s_facets_df"].copy()
            faceted_datasets.setdefault(dataset_name, []).append(df_facets)

    for dataset_name, df_list in faceted_datasets.items():
        if not df_list:
            continue
        combined_df = pd.concat(df_list, ignore_index=True)
        random_classification.plot_auroc_vs_s_facets(
            combined_df,
            save_dir=f"figs/{dataset_name}",
            filename_prefix=f"{dataset_name}_auroc_vs_s",
        )

    # plot curves by s
    s_masks = {ds: [] for ds in df["dataset"].unique()}
    for (model_name, dataset_name), res in results.items():
        if "s_masks" in res:
            s_mask = res["s_masks"]
            s_masks[dataset_name].append((model_name, s_mask))
    for dataset_name, s_mask_list in s_masks.items():
        if dataset_name != "pan_4_test_balanced":
            continue
        for model, s_mask in s_mask_list:
            if model != "popf-small-ft":
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
            

    # plot region predictions for genome classification
    region_plot = {ds: [] for ds in df["dataset"].unique()}
    for (model_name, dataset_name), res in results.items():
        if not ("popf" in model_name or "lp" in model_name):
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
                save_path=f"figs/{dataset_name if 'genome' in dataset_name else f'ghist_submit/{dataset_name}'}_region_plot.png",
                line=True
            )
