import os

import datasets
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

models = ["popf-small"]
train_data = ["pan_4_train_with_low_s"]
# train_data = ["combined_train"]


def load_features(path: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as npz:
        feats = npz["features"]
    return feats


def load_labels(path: str, label_column: str = "label") -> np.ndarray:
    # Try HuggingFace dataset directory
    if os.path.isdir(path):
        ds = datasets.load_from_disk(path)
        if label_column not in ds.column_names:
            raise ValueError(f"Column '{label_column}' not found in dataset at {path}")
        return np.asarray(ds[label_column])

    # csv file
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
        if label_column not in df.columns:
            raise ValueError(f"Column '{label_column}' not found in CSV: {path}")
        return df[label_column].to_numpy()

    raise ValueError(f"Unsupported labels file type: {path}")


def grid_search_C(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_ev: np.ndarray,
    y_ev: np.ndarray,
) -> float:
    C_grid = np.logspace(-4, 4, 16)
    best_C = None
    best_acc = -1.0

    for C in C_grid:
        clf = LogisticRegression(random_state=0, C=C, max_iter=1000)
        clf.fit(X_tr, y_tr)
        ev_pred = clf.predict_proba(X_ev)[:, 1]
        acc = average_precision_score(y_ev, ev_pred)
        acc = (y_ev == (ev_pred >= 0.5)).mean()
        print(f"C={C:.4g} score={acc:.4f}")

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


def experiment():
    for model in models:
        for train_set in train_data:
            print(f"\n=== Model: {model}, Train set: {train_set} ===")
            train_features = load_features(
                f"data/dataset/features/{train_set}_{model}.npz"
            )
            train_labels = load_labels(
                f"data/dataset/{train_set}/", label_column="label"
            )

            # split training dataset
            train_features = shuffle(train_features, random_state=0)
            train_labels = shuffle(train_labels, random_state=0)
            X_tr, X_test, y_tr, y_test = train_test_split(
                train_features,
                train_labels,
                random_state=0,
                test_size=0.2,
                stratify=train_labels,
            )
            X_train, X_ev, y_train, y_ev = train_test_split(
                X_tr,
                y_tr,
                random_state=0,
                test_size=0.2,
                stratify=y_tr,
            )

            # best_C = grid_search_C(
            #     X_train, y_train, X_ev, y_ev
            # )
            best_C = 100

            classifier = LogisticRegression(random_state=0, C=best_C, max_iter=1000)
            classifier.fit(X_tr, y_tr)

            os.makedirs("models/lp", exist_ok=True)
            with open(f"models/lp/{train_set}_{model}_lp.pkl", "wb") as f:
                pickle.dump(classifier, f)

    return 0


if __name__ == "__main__":
    experiment()
