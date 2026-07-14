from pathlib import Path
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier


warnings.filterwarnings("ignore")


PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


DATASETS = {
    "phiusiil": PROCESSED_DIR / "phiusiil_features.csv",
    "malicious_urls": PROCESSED_DIR / "malicious_urls_features.csv",
    "combined": PROCESSED_DIR / "combined_features.csv",
}

USE_SAMPLE_FOR_SPEED = False
MAX_SAMPLE_SIZE = 200000


def load_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series, list[str]]:

    df = pd.read_csv(path, encoding_errors="replace")

    metadata_columns = ["url", "label", "source"]

    feature_columns = [
        col for col in df.columns
        if col not in metadata_columns
    ]

    X = df[feature_columns].copy()
    y = df["label"].copy()

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    return X, y, feature_columns


def maybe_sample(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:

    if not USE_SAMPLE_FOR_SPEED:
        return X, y

    if len(X) <= MAX_SAMPLE_SIZE:
        return X, y

    data = X.copy()
    data["label"] = y.values

    sampled = data.groupby("label", group_keys=False).apply(
        lambda x: x.sample(
            n=min(len(x), MAX_SAMPLE_SIZE // 2),
            random_state=42
        )
    )

    y_sampled = sampled["label"]
    X_sampled = sampled.drop(columns=["label"])

    return X_sampled, y_sampled


def build_models() -> dict:

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            ))
        ]),

        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.1,
            random_state=42
        )
    }

    return models


def evaluate_model(model, X_test, y_test) -> dict:

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = y_pred

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score),
        "pr_auc": average_precision_score(y_test, y_score)
    }

    return metrics


def save_confusion_matrix(model, X_test, y_test, title: str, filename: str) -> None:

    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["legitimate", "phishing"]
    )

    disp.plot(values_format="d")

    plt.title(title)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close()


def run_internal_experiment(dataset_name: str, path: Path) -> pd.DataFrame:

    print("\n" + "=" * 80)
    print(f"EKSPERYMENT WEWNĘTRZNY: {dataset_name}")
    print("=" * 80)

    X, y, feature_columns = load_dataset(path)
    X, y = maybe_sample(X, y)

    print("Liczba rekordów:", len(X))
    print("Liczba cech:", len(feature_columns))
    print("Rozkład klas:")
    print(y.value_counts().sort_index())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = build_models()

    rows = []
    best_model = None
    best_model_name = None
    best_f1 = -1

    for model_name, model in models.items():
        print(f"\nTrenowanie modelu: {model_name}")

        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)

        row = {
            "experiment": "internal",
            "dataset": dataset_name,
            "train_dataset": dataset_name,
            "test_dataset": dataset_name,
            "model": model_name,
            **metrics
        }

        rows.append(row)

        print(
            f"accuracy={metrics['accuracy']:.4f}, "
            f"precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, "
            f"f1={metrics['f1']:.4f}, "
            f"roc_auc={metrics['roc_auc']:.4f}, "
            f"pr_auc={metrics['pr_auc']:.4f}"
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_model = model
            best_model_name = model_name

    save_confusion_matrix(
        best_model,
        X_test,
        y_test,
        title=f"Confusion Matrix - {dataset_name} - {best_model_name}",
        filename=f"confusion_matrix_internal_{dataset_name}_{best_model_name.replace(' ', '_')}.png"
    )

    return pd.DataFrame(rows)


def run_cross_dataset_experiment(
    train_name: str,
    train_path: Path,
    test_name: str,
    test_path: Path
) -> pd.DataFrame:

    print("\n" + "=" * 80)
    print(f"EKSPERYMENT CROSS-DATASET: train={train_name}, test={test_name}")
    print("=" * 80)

    X_train, y_train, train_features = load_dataset(train_path)
    X_test, y_test, test_features = load_dataset(test_path)

    X_train, y_train = maybe_sample(X_train, y_train)
    X_test, y_test = maybe_sample(X_test, y_test)

    common_features = sorted(list(set(train_features).intersection(set(test_features))))

    X_train = X_train[common_features]
    X_test = X_test[common_features]

    print("Liczba rekordów treningowych:", len(X_train))
    print("Liczba rekordów testowych:", len(X_test))
    print("Liczba wspólnych cech:", len(common_features))

    models = build_models()

    rows = []
    best_model = None
    best_model_name = None
    best_f1 = -1

    for model_name, model in models.items():
        print(f"\nTrenowanie modelu: {model_name}")

        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)

        row = {
            "experiment": "cross_dataset",
            "dataset": f"{train_name}_to_{test_name}",
            "train_dataset": train_name,
            "test_dataset": test_name,
            "model": model_name,
            **metrics
        }

        rows.append(row)

        print(
            f"accuracy={metrics['accuracy']:.4f}, "
            f"precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, "
            f"f1={metrics['f1']:.4f}, "
            f"roc_auc={metrics['roc_auc']:.4f}, "
            f"pr_auc={metrics['pr_auc']:.4f}"
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_model = model
            best_model_name = model_name

    save_confusion_matrix(
        best_model,
        X_test,
        y_test,
        title=f"Confusion Matrix - train {train_name}, test {test_name} - {best_model_name}",
        filename=f"confusion_matrix_cross_{train_name}_to_{test_name}_{best_model_name.replace(' ', '_')}.png"
    )

    return pd.DataFrame(rows)


def plot_results(results: pd.DataFrame, metric: str, filename: str, title: str) -> None:

    plt.figure(figsize=(12, 6))

    labels = results["dataset"] + " | " + results["model"]

    plt.bar(labels, results[metric])

    plt.ylabel(metric)
    plt.title(title)
    plt.xticks(rotation=75, ha="right")
    plt.tight_layout()

    plt.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close()


def main() -> None:
    all_results = []

    for dataset_name, path in DATASETS.items():
        if not path.exists():
            raise FileNotFoundError(f"Brakuje pliku: {path}")

        result = run_internal_experiment(dataset_name, path)
        all_results.append(result)

    cross_1 = run_cross_dataset_experiment(
        "phiusiil",
        DATASETS["phiusiil"],
        "malicious_urls",
        DATASETS["malicious_urls"]
    )

    cross_2 = run_cross_dataset_experiment(
        "malicious_urls",
        DATASETS["malicious_urls"],
        "phiusiil",
        DATASETS["phiusiil"]
    )

    all_results.append(cross_1)
    all_results.append(cross_2)

    final_results = pd.concat(all_results, ignore_index=True)

    final_results = final_results.sort_values(
        by=["experiment", "dataset", "f1"],
        ascending=[True, True, False]
    )

    final_results.to_csv(RESULTS_DIR / "all_model_results.csv", index=False)

    internal_results = final_results[final_results["experiment"] == "internal"].copy()
    cross_results = final_results[final_results["experiment"] == "cross_dataset"].copy()

    internal_results.to_csv(RESULTS_DIR / "internal_model_results.csv", index=False)
    cross_results.to_csv(RESULTS_DIR / "cross_dataset_model_results.csv", index=False)

    plot_results(
        internal_results,
        metric="f1",
        filename="internal_f1_comparison.png",
        title="Internal evaluation: F1-score comparison"
    )

    plot_results(
        cross_results,
        metric="f1",
        filename="cross_dataset_f1_comparison.png",
        title="Cross-dataset evaluation: F1-score comparison"
    )

    print("\n" + "=" * 80)
    print("ZAKOŃCZONO EKSPERYMENTY")
    print("=" * 80)

    print("\nZapisano wyniki:")
    print(RESULTS_DIR / "all_model_results.csv")
    print(RESULTS_DIR / "internal_model_results.csv")
    print(RESULTS_DIR / "cross_dataset_model_results.csv")

    print("\nZapisano wykresy i macierze pomyłek w folderze:")
    print(FIGURES_DIR)

    print("\nNajlepsze wyniki według F1:")
    print(
        final_results.sort_values(by="f1", ascending=False)
        [["experiment", "dataset", "model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]]
        .head(20)
    )


if __name__ == "__main__":
    main()