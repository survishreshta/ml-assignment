"""Train and evaluate Dry Bean Logistic Regression and Decision Tree classifiers."""

from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "DryBeanDataset" / "Dry_Bean_Dataset.xlsx"
SEED = 42


def main():
    df = pd.read_excel(DATA)
    X = df.drop(columns="Class")
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["Class"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=3000, random_state=SEED)),
        ]),
        "Decision Tree": DecisionTreeClassifier(random_state=SEED),
    }
    filenames = {
        "Logistic Regression": "logistic_regression.joblib",
        "Decision Tree": "decision_tree.joblib",
    }

    model_dir = ROOT / "model"
    model_dir.mkdir(exist_ok=True)
    results = {}
    for name, estimator in models.items():
        print(f"Training {name}...", flush=True)
        estimator.fit(X_train, y_train)
        pred = estimator.predict(X_test)
        proba = estimator.predict_proba(X_test)
        results[name] = {
            "Accuracy": accuracy_score(y_test, pred),
            "AUC": roc_auc_score(y_test, proba, multi_class="ovr", average="macro"),
            "Precision": precision_score(y_test, pred, average="macro", zero_division=0),
            "Recall": recall_score(y_test, pred, average="macro", zero_division=0),
            "F1": f1_score(y_test, pred, average="macro", zero_division=0),
            "MCC": matthews_corrcoef(y_test, pred),
        }
        joblib.dump(estimator, model_dir / filenames[name], compress=3)

    test = X_test.copy()
    test["Class"] = encoder.inverse_transform(y_test)
    test.to_csv(ROOT / "test_data.csv", index=False)
    metadata = {
        "feature_names": list(X.columns),
        "class_names": list(encoder.classes_),
        "model_files": filenames,
        "metrics": results,
        "split": {"test_size": 0.20, "random_state": SEED, "stratified": True},
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    joblib.dump(encoder, model_dir / "label_encoder.joblib")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
