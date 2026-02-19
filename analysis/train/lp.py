import os

import datasets
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import sys

models = [sys.argv[1]]
train_data = [sys.argv[2]]

test_size = 0.05
if len(sys.argv) > 3:
    test_size = float(sys.argv[3])


def load_features(path: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as npz:
        feats = npz["features"]
    return feats


def load_labels(path: str, label_column: str = "label") -> np.ndarray:
    ds = datasets.load_from_disk(path)
    return np.asarray(ds[label_column])


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
        acc = accuracy_score(y_ev, ev_pred > 0.5)
        acc = (y_ev == (ev_pred >= 0.5)).mean()
        print(f"C={C:.4g} score={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_C = C

    print(f"Selected C={best_C} (eval accuracy={best_acc:.4f})")
    return float(best_C)


def experiment():
    for model in models:
        for train_set in train_data:
            print(f"\n=== Model: {model}, Train set: {train_set} ===")
            train_features = load_features(f"data/features/{train_set}_{model}.npz")
            train_labels = load_labels(
                f"data/dataset/{train_set}/", label_column="label"
            )

            # split training dataset
            X_tr, X_test, y_tr, y_test = train_test_split(
                train_features,
                train_labels,
                random_state=0,
                test_size=test_size,
                stratify=train_labels,
            )
            X_train, X_ev, y_train, y_ev = train_test_split(
                X_tr,
                y_tr,
                random_state=0,
                test_size=0.2,
                stratify=y_tr,
            )

            # best_C = grid_search_C(X_train, y_train, X_ev, y_ev)
            best_C = 1

            classifier = LogisticRegression(random_state=0, C=best_C, max_iter=1000)
            classifier.fit(X_tr, y_tr)

            test_pred = classifier.predict_proba(X_test)[:, 1]
            test_acc = (y_test == (test_pred >= 0.5)).mean()
            print(f"Test set accuracy: {test_acc:.4f}")

            os.makedirs("models/lp", exist_ok=True)
            with open(f"models/lp/{train_set}_{model}-{test_size}_lp.pkl", "wb") as f:
                pickle.dump(classifier, f)

    return 0


if __name__ == "__main__":
    experiment()
