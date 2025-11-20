import pandas as pd
from evaluation.core import BaseEvaluator
from evaluation.models import popformer, popformer_lp, fasternn, schrider_resnet
from evaluation.evaluators import random_classification, genome_classification

if __name__ == "__main__":
    dataset_paths = [
        "data/dataset/pan_4_test_balanced",
        "data/dataset/pan_3_demoid-0_balanced",
        "data/dataset/pan_3_demoid-1_balanced",
        "data/dataset/len200_ghist_const1",
        "data/dataset/len200_ghist_const2",
    ]
    models = [
        popformer.PopformerModel(
            "models/selbin",
            "popf-small-ft",
            subsample=(64, 64),
            subsample_type="diverse",
        ),
        popformer_lp.PopformerLPModel(
            "models/popf-small",
            "models/lp/pan_4_popf-small_lp.pkl",
            "popf-small-lp",
            subsample=(64, 64),
            subsample_type="diverse",
        ),
        fasternn.FasterNNModel("models/fasternn/fasternn.pt", "FASTER-NN"),
        schrider_resnet.SchriderResnet(
            model_path="models/schrider_resnet/schrider_resnet.pt",
            model_name="resnet34",
        ),
    ]
    evaluators: list[BaseEvaluator] = []

    for dataset_path in dataset_paths:
        labels = None  # by default labels are inferred from the dataset
        if "ghist" in dataset_path:
            evaluator = genome_classification.GenomeClassificationEvaluator(
                dataset_path
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
            predictions = evaluator.run(model)
            res = evaluator.evaluate(predictions)
            results[(model.model_name, evaluator.dataset_name)] = res
            preds[(model.model_name, evaluator.dataset_name)] = predictions
            trues[(model.model_name, evaluator.dataset_name)] = evaluator.trues()

    # convert results to dataframe
    df = pd.DataFrame.from_dict(results, orient="index")
    df.index = pd.MultiIndex.from_tuples(df.index, names=["model", "dataset"])
    df = df.reset_index()

    df_table = df[
        ["model", "dataset", "accuracy", "precision", "recall", "auroc", "auprc"]
    ].dropna()
    print(df_table.to_string())

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

    # plot accuracy by s
    acc_by_s = {ds: [] for ds in df["dataset"].unique()}
    for (model_name, dataset_name), res in results.items():
        if "acc_by_s" in res:
            s_by_acc = res["acc_by_s"]
            df_s_acc = pd.DataFrame(s_by_acc)
            acc_by_s[dataset_name].append(df_s_acc)
    for dataset_name, acc_by_s_list in acc_by_s.items():
        if acc_by_s_list:
            df_s_acc_all = pd.concat(acc_by_s_list, ignore_index=True)
            random_classification.plot_acc_by_s(
                df_s_acc_all, save_path=f"figs/{dataset_name}_acc_vs_s.png"
            )

    # plot region predictions for genome classification
    region_plot = {ds: [] for ds in df["dataset"].unique()}
    for (model_name, dataset_name), res in results.items():
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

            genome_classification.plot_region(
                model_names=model_names,
                preds_list=preds_list,
                start_pos=start_pos,
                end_pos=end_pos,
                window=9,
                label_df=pd.read_csv(f"data/matrices/bigregions/{dataset_name}.csv"),
                save_path=f"figs/{dataset_name}_region_plot.png",
            )
