#!/usr/bin/env python3
import argparse
import os
import sys
import numpy as np
import pandas as pd
from typing import List, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle as sk_shuffle
from sklearn.metrics import accuracy_score

def load_features(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        with np.load(path, allow_pickle=False) as npz:
            if "features" not in npz:
                raise ValueError(f"'features' key not found in npz: {path}")
            feats = npz["features"]
    elif ext == ".npy":
        feats = np.load(path, allow_pickle=False)
    else:
        raise ValueError(f"Unsupported features file type: {path}")
    if not isinstance(feats, np.ndarray):
        raise ValueError(f"Loaded features are not a numpy array: {path}")
    return feats


def load_labels(path: str, label_column: str = "label") -> np.ndarray:
    # Try HuggingFace dataset directory
    if os.path.isdir(path):
        try:
            import datasets  # lazy import
            ds = datasets.load_from_disk(path)
            if label_column not in ds.column_names:
                raise ValueError(f"Column '{label_column}' not found in dataset at {path}")
            return np.asarray(ds[label_column])
        except Exception as e:
            raise ValueError(f"Failed to load labels from dataset dir '{path}': {e}") from e

    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
        if label_column not in df.columns:
            raise ValueError(f"Column '{label_column}' not found in CSV: {path}")
        return df[label_column].to_numpy()
    if ext == ".npz":
        with np.load(path, allow_pickle=False) as npz:
            if label_column not in npz and "labels" not in npz:
                raise ValueError(f"No '{label_column}' or 'labels' array in npz: {path}")
            key = label_column if label_column in npz else "labels"
            return npz[key]
    if ext == ".npy":
        return np.load(path, allow_pickle=False)

    raise ValueError(f"Unsupported labels file type: {path}")


def load_metadata(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported metadata file type: {path}")


def evaluate_on_test(preds: np.ndarray, metadata: pd.DataFrame, label_column: str = "label") -> None:
    has_label = label_column in metadata.columns and len(metadata) == len(preds)
    has_dataset = "dataset" in metadata.columns

    if has_label:
        overall_acc = accuracy_score(metadata[label_column].values, preds)
        print(f"\nOverall Test Accuracy: {overall_acc:.4f}")
        print("-" * 30)
    else:
        print("\nTest labels not provided or length mismatch; skipping accuracy.")
        print("-" * 30)

    if has_dataset and has_label:
        for dataset_name in metadata["dataset"].unique():
            idx = metadata["dataset"] == dataset_name
            ds_labels = metadata.loc[idx, label_column].values
            ds_preds = preds[idx]
            ds_acc = accuracy_score(ds_labels, ds_preds)
            print(f"Dataset: {dataset_name}, Accuracy: {ds_acc:.4f}")
    elif has_dataset:
        counts = metadata["dataset"].value_counts()
        print("Per-dataset counts (no labels):")
        for ds, cnt in counts.items():
            print(f"- {ds}: {cnt}")
    else:
        print("No 'dataset' column found in metadata; skipping per-dataset breakdown.")


def grid_search_C(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_ev: np.ndarray,
    y_ev: np.ndarray,
) -> float:
    C_grid = np.logspace(-4, 4, 16)
    best_C = None
    best_acc = -1.0
    print(f"Grid searching C in [{C_grid[0]}, {C_grid[-1]}] with {len(C_grid)} points")

    for C in C_grid:
        clf = LogisticRegression(
            random_state=0, C=C, max_iter=1000
        )
        clf.fit(X_tr, y_tr)
        ev_pred = clf.predict(X_ev)
        acc = accuracy_score(y_ev, ev_pred)
        print(f"C={C:.4g} eval accuracy={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_C = C

    print(f"Selected C={best_C} (eval accuracy={best_acc:.4f})")
    return float(best_C)


def save_predictions_npz(input_npz_path: str, preds: np.ndarray, out_path: str) -> None:
    # Pass through all arrays except 'features', add 'preds'
    with np.load(input_npz_path, allow_pickle=False) as npz:
        payload = {k: npz[k] for k in npz.files if k != "features"}
    payload["preds"] = preds
    np.savez(out_path, **payload)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Logistic Regression and predict on multiple datasets.")
    p.add_argument("--train-features", required=True, help="Path to training features (.npz with 'features' or .npy).")
    p.add_argument("--train-labels", required=True, help="Path to training labels (HF dataset dir, .csv, .npz, or .npy).")
    p.add_argument("--train-label-column", default="label", help="Label column name for CSV/HF dataset (default: label).")

    p.add_argument("--test-features", required=True, help="Path to test features (.npz or .npy).")
    p.add_argument("--test-metadata", required=True, help="Path to test metadata (.csv or .parquet).")
    p.add_argument("--test-label-column", default="label", help="Label column name in test metadata (default: label).")

    p.add_argument("--predict-features", nargs="*", default=[], help="Paths to additional feature .npz files to predict.")
    p.add_argument("--predict-outputs", nargs="*", default=[], help="Output .npz paths corresponding to --predict-features.")

    p.add_argument("--grid", action="store_true", help="Enable C hyperparameter sweep on a 5% eval split.")
    p.add_argument("--c", type=float, default=0.02, help="LogReg C value (ignored if --grid). Default: 0.02")

    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    # Load data
    train_features = load_features(args.train_features)
    train_labels = load_labels(args.train_labels, label_column=args.train_label_column)
    if train_features.shape[0] != train_labels.shape[0]:
        raise ValueError(f"Train features rows ({train_features.shape[0]}) != labels ({train_labels.shape[0]})")

    test_features = load_features(args.test_features)
    test_metadata = load_metadata(args.test_metadata)

    # Determine C (grid or fixed)
    best_C = args.c
    if args.grid:
        X_tr, X_ev, y_tr, y_ev = train_test_split(
            train_features, train_labels,
            test_size=0.05, stratify=train_labels, 
        )
        best_C = grid_search_C(
            X_tr, y_tr, X_ev, y_ev
        )

    # Fit on full training set with best C
    train_features, train_labels = sk_shuffle(
        train_features, train_labels, n_samples=None
    )

    classifier = LogisticRegression(
        random_state=0, C=best_C, max_iter=1000
    )
    classifier.fit(train_features, train_labels)

    # Evaluate on test
    test_preds = classifier.predict(test_features)
    evaluate_on_test(test_preds, test_metadata, label_column=args.test_label_column)

    # Predict for additional datasets
    if len(args.predict_features) != len(args.predict_outputs):
        if len(args.predict_features) == 0 and len(args.predict_outputs) == 0:
            return 0
        raise ValueError("--predict-features and --predict-outputs must have the same length")

    for in_path, out_path in zip(args.predict_features, args.predict_outputs):
        ext = os.path.splitext(in_path)[1].lower()
        if ext != ".npz":
            raise ValueError(f"Additional predictions require .npz feature files: {in_path}")
        with np.load(in_path, allow_pickle=False) as npz:
            if "features" not in npz:
                raise ValueError(f"'features' key not found in npz: {in_path}")
            feats = npz["features"]
        probs = classifier.predict_proba(feats)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        save_predictions_npz(in_path, probs, out_path)
        print(f"Saved predictions to {out_path}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)