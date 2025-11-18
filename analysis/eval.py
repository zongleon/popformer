from evaluation.core import BaseEvaluator
from evaluation.models import popformer, popformer_lp, fasternn
from evaluation.evaluators import random_classification, genome_classification

if __name__ == "__main__":
    dataset_paths = [
        "data/dataset/pan_4_test",
        "data/dataset/len200_ghist_const1",
        "data/dataset/len200_ghist_const2",
    ]
    models = [
        popformer.PopformerModel("models/selbin", "popf-small-ft",
                                 subsample=(64, 64), subsample_type="diverse"),
        popformer_lp.PopformerLPModel(
            "models/selbin", "models/lp/pan_4_popf-small_lp.pkl", "popf-small-lp",
            subsample=(64, 64), subsample_type="diverse"
        ),
        fasternn.FasterNNModel("models/fasternn/fasternn.pt", "FASTER-NN"),
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
    for model in models:
        for evaluator in evaluators:
            print(f"Evaluating {model.model_name} on {evaluator.dataset_name}")
            predictions = evaluator.run(model)
            res = evaluator.evaluate(predictions)
            results[(model.model_name, evaluator.dataset_name)] = res
            
    for (model_name, dataset_name), res in results.items():
        print(f"Results for {model_name} on {dataset_name}:")
        for metric, value in res.items():
            if metric != "acc_by_s":
                print(f"  {metric}: {value}")
        if "acc_by_s" in res and res["acc_by_s"] is not None:
            print("  Accuracy by s bins:")
            for entry in res["acc_by_s"]:
                print(f"    {entry['s_bin']}: {entry['acc']}")
