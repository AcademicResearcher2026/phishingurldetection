from pathlib import Path
import warnings

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
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
    average_precision_score
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


METADATA_COLUMNS = ["url", "label", "source"]


ARTIFACT_PRONE_FEATURES = [
    "has_http",
    "has_https",
    "colon_count",
    "slash_count",
    "is_www"
]


def load_dataset(path: Path, remove_artifact_features: bool = False):
    df = pd.read_csv(path, encoding_errors="replace")

    feature_columns = [
        col for col in df.columns
        if col not in METADATA_COLUMNS
    ]

    if remove_artifact_features:
        feature_columns = [
            col for col in feature_columns
            if col not in ARTIFACT_PRONE_FEATURES
        ]

    X = df[feature_columns].copy()
    y = df["label"].copy()

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    return X, y, feature_columns


def build_models():
    return {
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
            n_jobs=1
        ),

        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.1,
            random_state=42
        )
    }


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = y_pred

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score),
        "pr_auc": average_precision_score(y_test, y_score),
    }


def run_internal_experiment(dataset_name, path, feature_variant, remove_artifact_features):
    print("\n" + "=" * 80)
    print(f"INTERNAL EXPERIMENT: {dataset_name} | {feature_variant}")
    print("=" * 80)

    X, y, feature_columns = load_dataset(
        path,
        remove_artifact_features=remove_artifact_features
    )

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

    rows = []

    for model_name, model in build_models().items():
        print(f"Trenowanie: {model_name}")

        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)

        rows.append({
            "experiment": "internal",
            "feature_variant": feature_variant,
            "dataset": dataset_name,
            "train_dataset": dataset_name,
            "test_dataset": dataset_name,
            "model": model_name,
            "n_features": len(feature_columns),
            **metrics
        })

        print(
            f"accuracy={metrics['accuracy']:.4f}, "
            f"precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, "
            f"f1={metrics['f1']:.4f}, "
            f"roc_auc={metrics['roc_auc']:.4f}, "
            f"pr_auc={metrics['pr_auc']:.4f}"
        )

    return pd.DataFrame(rows)


def run_cross_dataset_experiment(
    train_name,
    train_path,
    test_name,
    test_path,
    feature_variant,
    remove_artifact_features
):
    print("\n" + "=" * 80)
    print(f"CROSS-DATASET: train={train_name}, test={test_name} | {feature_variant}")
    print("=" * 80)

    X_train, y_train, train_features = load_dataset(
        train_path,
        remove_artifact_features=remove_artifact_features
    )

    X_test, y_test, test_features = load_dataset(
        test_path,
        remove_artifact_features=remove_artifact_features
    )

    common_features = sorted(list(set(train_features).intersection(set(test_features))))

    X_train = X_train[common_features]
    X_test = X_test[common_features]

    print("Liczba rekordów treningowych:", len(X_train))
    print("Liczba rekordów testowych:", len(X_test))
    print("Liczba wspólnych cech:", len(common_features))

    rows = []

    for model_name, model in build_models().items():
        print(f"Trenowanie: {model_name}")

        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)

        rows.append({
            "experiment": "cross_dataset",
            "feature_variant": feature_variant,
            "dataset": f"{train_name}_to_{test_name}",
            "train_dataset": train_name,
            "test_dataset": test_name,
            "model": model_name,
            "n_features": len(common_features),
            **metrics
        })

        print(
            f"accuracy={metrics['accuracy']:.4f}, "
            f"precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, "
            f"f1={metrics['f1']:.4f}, "
            f"roc_auc={metrics['roc_auc']:.4f}, "
            f"pr_auc={metrics['pr_auc']:.4f}"
        )

    return pd.DataFrame(rows)


def plot_ablation_results(results):
    selected = results[
        (results["experiment"] == "cross_dataset")
    ].copy()

    selected["label"] = (
        selected["dataset"] + " | " +
        selected["feature_variant"] + " | " +
        selected["model"]
    )

    selected = selected.sort_values(by="f1", ascending=False)

    plt.figure(figsize=(14, 7))
    plt.bar(selected["label"], selected["f1"])
    plt.ylabel("F1-score")
    plt.title("Cross-dataset ablation study: F1-score")
    plt.xticks(rotation=80, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ablation_cross_dataset_f1.png", dpi=300)
    plt.close()


def main():
    all_results = []

    variants = [
        ("all_features", False),
        ("without_artifact_prone_features", True)
    ]

    for feature_variant, remove_artifact_features in variants:
        print("\n" + "#" * 80)
        print(f"WARIANT CECH: {feature_variant}")
        print("#" * 80)

        for dataset_name, path in DATASETS.items():
            if not path.exists():
                raise FileNotFoundError(f"Brakuje pliku: {path}")

            result = run_internal_experiment(
                dataset_name=dataset_name,
                path=path,
                feature_variant=feature_variant,
                remove_artifact_features=remove_artifact_features
            )

            all_results.append(result)

        cross_1 = run_cross_dataset_experiment(
            train_name="phiusiil",
            train_path=DATASETS["phiusiil"],
            test_name="malicious_urls",
            test_path=DATASETS["malicious_urls"],
            feature_variant=feature_variant,
            remove_artifact_features=remove_artifact_features
        )

        cross_2 = run_cross_dataset_experiment(
            train_name="malicious_urls",
            train_path=DATASETS["malicious_urls"],
            test_name="phiusiil",
            test_path=DATASETS["phiusiil"],
            feature_variant=feature_variant,
            remove_artifact_features=remove_artifact_features
        )

        all_results.append(cross_1)
        all_results.append(cross_2)

    final_results = pd.concat(all_results, ignore_index=True)

    final_results = final_results.sort_values(
        by=["experiment", "dataset", "feature_variant", "f1"],
        ascending=[True, True, True, False]
    )

    final_results.to_csv(
        RESULTS_DIR / "ablation_study_results.csv",
        index=False
    )

    plot_ablation_results(final_results)

    print("\n" + "=" * 80)
    print("ZAKOŃCZONO ABLATION STUDY")
    print("=" * 80)

    print("\nZapisano:")
    print(RESULTS_DIR / "ablation_study_results.csv")
    print(FIGURES_DIR / "ablation_cross_dataset_f1.png")

    print("\nNajlepsze wyniki cross-dataset według F1:")
    cross = final_results[final_results["experiment"] == "cross_dataset"].copy()

    print(
        cross.sort_values(by="f1", ascending=False)
        [
            [
                "dataset",
                "feature_variant",
                "model",
                "n_features",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc"
            ]
        ]
        .head(20)
    )


if __name__ == "__main__":
    main()