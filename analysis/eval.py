# import sys
import torch
import datasets
import numpy as np
import pandas as pd
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from matplotlib import pyplot as plt
import seaborn as sns

from analysis.train.lp import load_labels


def evaluate_on_test(
    preds: np.ndarray,
    metadata: pd.DataFrame,
    label_column: str = "label",
    trainset: str = "",
    model: str = "",
    testset: str = "",
) -> None:
    res = {}
    if isinstance(metadata, np.ndarray):
        metadata = pd.DataFrame({label_column: metadata})
    has_label = label_column in metadata.columns and len(metadata) == len(preds)
    has_dataset = "dataset" in metadata.columns

    print(preds.shape)
    binary_preds = np.argmax(preds, axis=1)

    if has_label:
        overall_acc = accuracy_score(metadata[label_column].values, binary_preds)
        overall_aucroc = roc_auc_score(metadata[label_column].values, preds[:, 1])
        overall_precision = precision_score(metadata[label_column].values, binary_preds)
        overall_recall = recall_score(metadata[label_column].values, binary_preds)
        overall_aucprc = average_precision_score(
            metadata[label_column].values, preds[:, 1]
        )

        prc = PrecisionRecallDisplay.from_predictions(
            metadata[label_column].values,
            preds[:, 1],
            name=f"{model} {trainset}",
        )
        roc = RocCurveDisplay.from_predictions(
            metadata[label_column].values,
            preds[:, 1],
            name=f"{model} {trainset}",
        )

        res = {
            "accuracy": overall_acc,
            "auc_roc": overall_aucroc,
            "precision": overall_precision,
            "recall": overall_recall,
            "auc_prc": overall_aucprc,
            "prc": prc,
            "roc": roc,
            "trainset": trainset,
            "model": model,
            "testset": testset,
        }
    else:
        print("\nTest labels not provided or length mismatch; skipping accuracy.")
        print("-" * 30)

    if has_dataset and has_label:
        for dataset_name in metadata["dataset"].unique():
            idx = metadata["dataset"] == dataset_name
            ds_labels = metadata.loc[idx, label_column].values
            ds_preds = binary_preds[idx]
            ds_acc = accuracy_score(ds_labels, ds_preds)
            res.update({f"accuracy_{dataset_name}": ds_acc})

    return res


def windowed_mean(data: np.ndarray, window: int, window_type: str = "mean") -> np.ndarray:
    if window_type == "mean":
        kernel = np.ones(window, dtype=float) / window
        data = np.convolve(data, kernel, mode="same")
    elif window_type == "min":
        p_pad = np.pad(data, (window // 2, window - 1 - window // 2), mode="edge")
        data = np.array([np.min(p_pad[i : i + window]) for i in range(len(data))])
    return data


def plot_region(
    preds,
    labels,
    start_pos,
    end_pos,
    outfig,
    window=3,
    label_df=None,
    window_type="mean",
):
    ylbl = "pred. probability of selection"
    pos = (end_pos + start_pos) // 2

    preds_adj = []
    for p in preds:
        if window > 1:
            p = windowed_mean(p, window=window, window_type=window_type)
        preds_adj.append(p)

    fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
    for p, label in zip(preds_adj, labels):
        ax.scatter(pos, p, alpha=0.8, label=label)

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
    plt.savefig(outfig, dpi=300)




def evaluate(preds, labels, models, trains, tests, starts, ends):
    results = []
    for pred, label, model_name, trainset, testset in zip(preds, labels, models, trains, tests):
        print(f"\n=== Evaluating Model: {model_name}, Train set: {trainset}, Test set: {testset} ===")

        # load predictions
        p = np.load(f"preds/{pred}")["preds"]

        if "ft" in model_name:
            # softmax outputs
            p = torch.softmax(torch.from_numpy(p), dim=1).numpy()

        # evaluate on test set
        res = evaluate_on_test(
            p,
            label,
            trainset=trainset,
            model=model_name,
            testset=testset,
        )

        results.append({**res, "preds": p})

    # Print summary tables
    df = pd.DataFrame(results)
    for testset, start, end in zip(tests, starts, ends):
        dft = df[df["testset"] == testset]
        print(f"\n--- Summary for test set: {testset} ---")
        print(dft[["model", "trainset", "testset", "accuracy", "auc_roc", "auc_prc"]].to_string(index=False))


        # plot prc curves
        if dft["prc"].notnull().any():
            fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
            
            for prc in dft["prc"]:
                prc.plot(ax=ax, plot_chance_level=False)
            ax.set_title(f"Precision-Recall Curve: {testset}")
            plt.savefig(f"figs/lp_prc_{testset}.png", dpi=300)
            plt.close()

        
        # plot roc curves
        if dft["roc"].notnull().any():
            fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
            for roc in dft["roc"]:
                roc.plot(ax=ax, plot_chance_level=False)
            ax.set_title(f"ROC Curve: {testset}")
            plt.savefig(f"figs/lp_roc_{testset}.png", dpi=300)
            plt.close()

        if testset == "pan_4":
            s = np.asarray(datasets.load_from_disk("data/dataset/pan_4")["s"])
            lbls = load_labels("data/dataset/pan_4/", label_column="label")

            # plot the distribution of s
            fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
            sns.histplot(s[s > 0], bins=50, kde=False, ax=ax)
            ax.set_xlabel("Selection coefficient s")
            ax.set_ylabel("Count")
            ax.grid(True, axis="y", alpha=0.3, linestyle="--")
            plt.savefig("figs/lp_s_dist.png", dpi=300)
            plt.close()

            # accuracy vs s bins per model
            bin_labels = [
                "s=0",
                "(0,0.02]",
                "(0.02,0.04]",
                "(0.04,0.06]",
                "(0.06,0.08]",
                "(0.08,0.1]",
            ]
            bin_edges = [
                (0.0, 0.0),
                (0.0, 0.02),
                (0.02, 0.04),
                (0.04, 0.06),
                (0.06, 0.08),
                (0.08, 0.1),
            ]

            rows = []
            for _, row in dft.iterrows():
                y_pred = np.argmax(row["preds"], axis=1)
                for label, (a, b) in zip(bin_labels, bin_edges):
                    if a == 0.0 and b == 0.0:
                        mask = np.isclose(s, 0.0)
                    else:
                        mask = (s > a) & (s <= b)
                    if np.any(mask):
                        acc = accuracy_score(lbls[mask], y_pred[mask])
                        rows.append(
                            {
                                "s_bin": label,
                                "acc": acc,
                                "model": row["model"],
                                "trainset": row["trainset"],
                            }
                        )

            df_sacc = pd.DataFrame(rows)

            fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
            sns.lineplot(
                data=df_sacc,
                x="s_bin",
                y="acc",
                hue="model",
                style="trainset",
                markers=True,
                ax=ax,
            )
            ax.set_xlabel("s bin")
            ax.set_ylabel("Accuracy")
            ax.grid(True, axis="y", alpha=0.3, linestyle="--")
            plt.savefig("figs/lp_acc_vs_s.png", dpi=300)
            plt.close()


        # plot region predictions
        if start is not None and end is not None:
            preds = [row["preds"][:, 1] for row in results if row["testset"] == testset]
            labels = [
                f"{row['model']} {row['trainset']}"
                for row in results
                if row["testset"] == testset
            ]
            plot_region(
                preds=preds,
                labels=labels,
                start_pos=start,
                end_pos=end,
                outfig=f"figs/lp_preds_{testset}.png",
                window=1,
                window_type="mean",
                label_df=pd.read_csv(f"data/matrices/bigregions/{testset}.csv"),
            )


if __name__ == "__main__":
    models = ["popf-small", "popf-small-ft"]
    tests = ["pan_4", "len200_ghist_const1", "len200_ghist_const2"]
    trains = ["pan_4"]
    preds_list = []
    models_list = []
    tests_list = []
    trains_list = []
    for model in models:
        for train in trains:
            for test in tests:
                preds_list.append(f"{test}_{model}_{train}.npz")
                models_list.append(model)
                trains_list.append(train)
                tests_list.append(test)

    labels = []
    starts = []
    ends = []

    for test in tests_list:
        start, end = None, None
        if "fasternn" in test:
            test_metadata = load_labels(f"FASTER_NN/{test}_test_meta.csv")
        elif "const" in test:
            sels = pd.read_csv(f"data/matrices/bigregions/{test}.csv")
            t = np.load(f"data/dataset/features/{test}_{models[0]}.npz")
            start, end = t["start_pos"], t["end_pos"]
            ls = []
            for s, e in zip(start, end):
                # label is 1 if the window overlaps with a selected region
                sel_region = sels[
                    (sels["start"] <= e) & (sels["end"] >= s)
                ]
                if len(sel_region) > 0:
                    sel_label = 1
                else:
                    sel_label = 0
                ls.append(sel_label)
            test_metadata = pd.DataFrame({"label": ls})
        else:
            try:
                test_labels = load_labels(
                    f"data/dataset/{test}/", label_column="label"
                )
                test_metadata = pd.DataFrame({"label": test_labels})
            except ValueError:
                test_metadata = None

        labels.append(test_metadata)
        starts.append(start)
        ends.append(end)

    print(preds_list)
    
    evaluate(preds_list, labels, models_list, trains_list, tests_list, starts, ends)