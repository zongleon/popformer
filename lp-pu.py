import os
import datasets
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    PrecisionRecallDisplay,
    accuracy_score,
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from pulearn import ElkanotoPuClassifier, BaggingPuClassifier
from sklearn.utils import shuffle
import seaborn as sns

models = ["popf-large", "popf-small"]
test_data = [
    "ghist_singlesweep",
    "ghist_singlesweep.growth_bg",
    "ghist_multisweep",
    "ghist_multisweep.growth_bg",
    "fasternn",
    "pan_4_snps",
    "pan_4_inorder",
    "genome_CEU",
    "ghist_const4",
    "ghist_const6",
    "len200_ghist_const1",
    "len200_ghist_const2",
]
MAX = None


def load_features(path: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as npz:
        feats = npz["features"]
    return feats


def load_labels(path: str, label_column: str = "label") -> np.ndarray:
    # Try HuggingFace dataset directory
    if os.path.isdir(path):
        ds = datasets.load_from_disk(path)
        if label_column not in ds.column_names:
            raise ValueError(
                f"Column '{label_column}' not found in dataset at {path}"
            )
        return np.asarray(ds[label_column])
    
    # csv file
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
        if label_column not in df.columns:
            raise ValueError(f"Column '{label_column}' not found in CSV: {path}")
        return df[label_column].to_numpy()
    
    raise ValueError(f"Unsupported labels file type: {path}")


def load_metadata(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported metadata file type: {path}")


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

    binary_preds = np.argmax(preds, axis=1)
    # convert -1s to 0s for accuracy calculation
    binary_preds = np.where(binary_preds == -1, 0, binary_preds)

    if has_label:
        overall_acc = accuracy_score(metadata[label_column].values, binary_preds)
        overall_aucroc = roc_auc_score(metadata[label_column].values, preds[:, 1])
        overall_precision = precision_score(metadata[label_column].values, binary_preds)
        overall_recall = recall_score(metadata[label_column].values, binary_preds)
        overall_aucprc = average_precision_score(
            metadata[label_column].values, preds[:, 1]
        )

        res = {
            "accuracy": overall_acc,
            "auc_roc": overall_aucroc,
            "precision": overall_precision,
            "recall": overall_recall,
            "auc_prc": overall_aucprc,
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


def pretty_print_results(res: dict, stds=None, header: str = "") -> None:
    print(f"\n{header}")
    print("-" * len(header))
    for k, v in res.items():
        if k in ["trainset", "testset", "model"]:
            continue
        if stds:
            std = stds.get(k, 0.0)
            print(f"- {k}: {v:.4f} ({std:.4f})")
        else:
            print(f"- {k}: {v:.4f}")


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


def save_predictions_npz(input_npz_path: str, preds: np.ndarray, out_path: str) -> None:
    # Pass through all arrays except 'features', add 'preds'
    with np.load(input_npz_path, allow_pickle=False) as npz:
        payload = {k: npz[k] for k in npz.files if k != "features"}
    payload["preds"] = preds
    np.savez(out_path, **payload)


def experiment():
    # Store results for summary
    results = []
    inorders = []

    train_set = "pureal-pan_4"

    for model in models:
        simulated_features = load_features(f"dataset/features/pan_4_snps_{model}.npz")
        simulated_labels = load_labels(
            "dataset/pan_4_snps/", label_column="label"
        )
        positive_mask = simulated_labels == 1
        real_features = load_features(f"dataset/features/genome_CEU_{model}.npz")
        real_labels = np.full(real_features.shape[0], -1, dtype=int)

        train_features = np.concatenate(
            [simulated_features[positive_mask], real_features], axis=0
        )
        train_labels = np.concatenate(
            [simulated_labels[positive_mask], real_labels], axis=0
        )

        # split training dataset
        train_features = shuffle(train_features, random_state=0, n_samples=MAX)
        train_labels = shuffle(train_labels, random_state=0, n_samples=MAX)
        X_tr, X_test, y_tr, y_test = train_test_split(
            train_features,
            train_labels,
            random_state=0,
            test_size=0.2,
            stratify=train_labels,
        )
        best_C = 1

        classifier = LogisticRegression(
            random_state=0, C=best_C, max_iter=1000
        )
        classifier = ElkanotoPuClassifier(classifier, hold_out_ratio=0.5, random_state=0)
        # classifier = BaggingPuClassifier(classifier, n_estimators=15, random_state=0)
        classifier.fit(X_tr, y_tr)

        # evaluate on test set
        y_test = np.where(y_test == -1, 0, y_test)
        res = evaluate_on_test(
            classifier.predict_proba(X_test),
            y_test,
        )
        pretty_print_results(res)

        for test_set in test_data:
            test_features = load_features(
                f"dataset/features/{test_set}_{model}.npz"
            )
            if "fasternn" in test_set:
                test_metadata = load_metadata(f"FASTER_NN/{test_set}_test_meta.csv")
            elif "const" in test_set:
                sels = pd.read_csv(f"1000g/regiontest/{test_set}.csv")
                t = np.load(f"dataset/features/{test_set}_{model}.npz")
                starts, ends = t["start_pos"], t["end_pos"]
                labels = []
                for start, end in zip(starts, ends):
                    # label is 1 if the window overlaps with a selected region
                    sel_region = sels[
                        (sels["start"] <= end) & (sels["start"] >= start)
                    ]
                    if len(sel_region) > 0:
                        sel_label = 1
                    else:
                        sel_label = 0
                    labels.append(sel_label)
                test_metadata = pd.DataFrame({"label": labels})
            else:
                try:
                    test_labels = load_labels(
                        f"dataset/{test_set}/", label_column="label"
                    )
                    test_metadata = pd.DataFrame({"label": test_labels})
                except ValueError:
                    test_metadata = None

            test_preds = classifier.predict_proba(test_features)

            if "inorder" in test_set:
                inorders.append(
                    {
                        "model": model,
                        "trainset": train_set,
                        "testset": test_set,
                        "preds": test_preds,
                    }
                )

            if test_metadata is not None:
                res = evaluate_on_test(
                    test_preds,
                    test_metadata,
                    label_column="label",
                    trainset=train_set,
                    model=model,
                    testset=test_set,
                )
                display = PrecisionRecallDisplay.from_predictions(
                    test_metadata["label"].values,
                    test_preds[:, 1],
                    name=f"{model}_{train_set}",
                    plot_chance_level=True,
                    despine=True,
                )
                res.update({"prc_curve": display})
                # res.update({"roc_curve": roc_display})
                res.update({"preds": test_preds})
                results.append(res)

            if "ghist" in test_set or "genome" in test_set:
                save_predictions_npz(
                    f"dataset/features/{test_set}_{model}.npz",
                    test_preds,
                    f"outs/pu_{test_set}_{model}_{train_set}.npz",
                )

    # Print summary tables
    df = pd.DataFrame(results)
    for test_set in test_data:
        dft = df[df["testset"] == test_set]
        if dft["accuracy"].notnull().any():
            print(f"\nSummary for test set: {test_set}")
            print("-" * 40)
            print(dft[[column for column in dft.columns if column != "preds" and column != "prc_curve" and column != "roc_curve"]].to_string(index=False))

        # plot prc curves
        if dft["prc_curve"].notnull().any():
            fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
            chance_level_plotted = False
            for row in results:
                if row["testset"] != test_set:
                    continue
                prc_curve = row["prc_curve"]
                if prc_curve is not None:
                    prc_curve.plot(ax=ax, plot_chance_level=not chance_level_plotted)
                    chance_level_plotted = True
            ax.set_title(f"Precision-Recall Curve: {test_set}")
            plt.savefig(f"figs/lppu_prc_{test_set}.png", dpi=300)
            plt.close()

        if test_set == "pan_4_snps":
            s = np.asarray(datasets.load_from_disk("dataset/pan_4_snps")["s"])
            lbls = load_labels("dataset/pan_4_snps/", label_column="label")

            # accuracy vs s bins per model
            bin_labels = ["s=0", "(0,0.02]", "(0.02,0.04]", "(0.04,0.06]", "(0.06,0.08]", "(0.08,0.1]"]
            bin_edges = [(0.0, 0.0), (0.0, 0.02), (0.02, 0.04), (0.04, 0.06), (0.06, 0.08), (0.08, 0.1)]

            rows = []
            for row in results:
                if row["testset"] != "pan_4_snps":
                    continue
                y_pred = np.argmax(row["preds"], axis=1)
                for label, (a, b) in zip(bin_labels, bin_edges):
                    if a == 0.0 and b == 0.0:
                        mask = np.isclose(s, 0.0)
                    else:
                        mask = (s > a) & (s <= b)
                    if np.any(mask):
                        acc = accuracy_score(lbls[mask], y_pred[mask])
                        rows.append({"s_bin": label, "acc": acc, 
                                     "model": row['model'],
                                     "train": row['trainset']})

            df_sacc = pd.DataFrame(rows)

            fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
            sns.lineplot(
                data=df_sacc,
                x="s_bin",
                y="acc",
                hue="model",
                style="train",
                markers=True,
                # order=bin_labels,
                ax=ax,
            )
            ax.set_xlabel("s bin")
            ax.set_ylabel("Accuracy")
            ax.grid(True, axis="y", alpha=0.3, linestyle="--")
            plt.savefig("figs/lppu_acc_vs_s.png", dpi=300)
            plt.close()

    # plot region predictions
    for test_set in test_data[-4:]:
        start_pos = np.load(f"dataset/features/{test_set}_popf-small.npz")["start_pos"]
        end_pos = np.load(f"dataset/features/{test_set}_popf-small.npz")["end_pos"]
        preds = [row["preds"][:, 1] for row in results if row["testset"] == test_set]
        
        labels = [f"{row['model']} {row['trainset']}" for row in results if row["testset"] == test_set]
        plot_region(
            preds=preds,
            labels=labels,
            start_pos=start_pos,
            end_pos=end_pos,
            outfig=f"figs/lppu_preds_{test_set}.png",
            window=1,
            window_type="mean",
            label_df=pd.read_csv(f"1000g/regiontest/{test_set}.csv")
        )
        
    df_inorder = pd.DataFrame()
    for inorder in inorders:
        # each inorder preds has 5 windows
        # plot all 1000 in one figure
        preds = inorder["preds"]
        n_series = preds.shape[0] // 5
        
        df_inorder = pd.concat([df_inorder, pd.DataFrame({
            "series": np.repeat(np.arange(n_series), 5),
            "window": np.tile(np.arange(1, 6), n_series),
            "pred": preds[:, 1],
            "model": inorder["model"],
        })], ignore_index=True)

    fig, ax = plt.subplots(figsize=(8, 8), layout="constrained")

    # add individual region trajectories (thin gray lines)
    for key, grp in df_inorder.groupby(["model", "series"]):
        ax.plot(grp["window"] - 1, grp["pred"], color="gray", alpha=0.1, linewidth=0.5)


    sns.pointplot(
        data=df_inorder,
        x="window",
        y="pred",
        hue="model",
        estimator=np.median,
        errorbar="sd",
        dodge=True,
        ax=ax,
        order=sorted(df_inorder["window"].unique()),
    )
    
    ax.set_xlabel("Window")
    ax.set_ylabel("Pr(selection)")
    ax.set_xlim(-0.05, 4.05)
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    plt.savefig("figs/lppu_inorder.png", dpi=300)
        

    return 0


if __name__ == "__main__":
    experiment()
