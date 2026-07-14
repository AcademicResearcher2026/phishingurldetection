from pathlib import Path
import warnings

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.inspection import permutation_importance


warnings.filterwarnings("ignore")


PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

COMBINED_FEATURES_PATH = PROCESSED_DIR / "combined_features.csv"

RUN_PERMUTATION_IMPORTANCE = False

PERMUTATION_SAMPLE_SIZE = 20000

SOURCE_CLASSIFIER_SAMPLE_PER_DATASET = 100000


METADATA_COLUMNS = ["url", "label", "source"]


def load_combined_dataset() -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:

    if not COMBINED_FEATURES_PATH.exists():
        raise FileNotFoundError(f"Brakuje pliku: {COMBINED_FEATURES_PATH}")

    df = pd.read_csv(COMBINED_FEATURES_PATH, encoding_errors="replace")

    feature_columns = [
        col for col in df.columns
        if col not in METADATA_COLUMNS
    ]

    X = df[feature_columns].copy()
    y = df["label"].copy()
    source = df["source"].copy()

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    return X, y, source, feature_columns


def train_random_forest_for_importance(
    X: pd.DataFrame,
    y: pd.Series,
    feature_columns: list[str]
):

    print("\n" + "=" * 80)
    print("ANALIZA WAŻNOŚCI CECH — RANDOM FOREST")
    print("=" * 80)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=150,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    print("Trenowanie Random Forest na combined_features.csv...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("F1-score:", round(f1_score(y_test, y_pred), 4))

    importances = pd.DataFrame({
        "feature": feature_columns,
        "importance": model.feature_importances_
    })

    importances = importances.sort_values(
        by="importance",
        ascending=False
    )

    importances.to_csv(
        RESULTS_DIR / "feature_importance_random_forest.csv",
        index=False
    )

    print("\nTop 20 cech według Random Forest:")
    print(importances.head(20))

    plot_top_features(
        importances,
        value_column="importance",
        title="Top 20 features by Random Forest importance",
        filename="feature_importance_random_forest_top20.png"
    )

    if RUN_PERMUTATION_IMPORTANCE:
        run_permutation_importance(
            model,
            X_test,
            y_test,
            feature_columns
        )
    else:
        print("\nPermutation importance pominięto w tej wersji skryptu.")

    return model


def run_permutation_importance(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_columns: list[str]
) -> None:

    print("\n" + "=" * 80)
    print("PERMUTATION IMPORTANCE")
    print("=" * 80)

    if len(X_test) > PERMUTATION_SAMPLE_SIZE:
        sample = X_test.copy()
        sample["label"] = y_test.values

        sample = sample.sample(
            n=PERMUTATION_SAMPLE_SIZE,
            random_state=42
        )

        y_sample = sample["label"]
        X_sample = sample.drop(columns=["label"])
    else:
        X_sample = X_test
        y_sample = y_test

    print("Liczba rekordów użytych do permutation importance:", len(X_sample))

    result = permutation_importance(
        model,
        X_sample,
        y_sample,
        n_repeats=5,
        random_state=42,
        scoring="f1",
        n_jobs=1
    )

    permutation_df = pd.DataFrame({
        "feature": feature_columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std
    })

    permutation_df = permutation_df.sort_values(
        by="importance_mean",
        ascending=False
    )

    permutation_df.to_csv(
        RESULTS_DIR / "permutation_importance_random_forest.csv",
        index=False
    )

    print("\nTop 20 cech według permutation importance:")
    print(permutation_df.head(20))

    plot_top_features(
        permutation_df,
        value_column="importance_mean",
        title="Top 20 features by permutation importance",
        filename="permutation_importance_top20.png"
    )


def plot_top_features(
    df: pd.DataFrame,
    value_column: str,
    title: str,
    filename: str
) -> None:

    top = df.head(20).copy()
    top = top.sort_values(by=value_column, ascending=True)

    plt.figure(figsize=(10, 7))
    plt.barh(top["feature"], top[value_column])
    plt.xlabel(value_column)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close()


def compute_standardized_mean_difference(
    df: pd.DataFrame,
    feature_columns: list[str],
    group_column: str,
    group_a: str,
    group_b: str
) -> pd.DataFrame:

    rows = []

    df_a = df[df[group_column] == group_a]
    df_b = df[df[group_column] == group_b]

    for feature in feature_columns:
        mean_a = df_a[feature].mean()
        mean_b = df_b[feature].mean()

        std_a = df_a[feature].std()
        std_b = df_b[feature].std()

        pooled_std = np.sqrt((std_a ** 2 + std_b ** 2) / 2)

        if pooled_std == 0 or np.isnan(pooled_std):
            smd = 0
        else:
            smd = (mean_a - mean_b) / pooled_std

        rows.append({
            "feature": feature,
            f"mean_{group_a}": mean_a,
            f"mean_{group_b}": mean_b,
            "smd": smd,
            "abs_smd": abs(smd)
        })

    result = pd.DataFrame(rows)
    result = result.sort_values(by="abs_smd", ascending=False)

    return result


def analyze_dataset_shift(
    X: pd.DataFrame,
    y: pd.Series,
    source: pd.Series,
    feature_columns: list[str]
) -> None:

    print("\n" + "=" * 80)
    print("ANALIZA RÓŻNIC MIĘDZY DATASETAMI")
    print("=" * 80)

    df = X.copy()
    df["label"] = y.values
    df["source"] = source.values

    overall_shift = compute_standardized_mean_difference(
        df=df,
        feature_columns=feature_columns,
        group_column="source",
        group_a="phiusiil",
        group_b="malicious_urls"
    )

    overall_shift.to_csv(
        RESULTS_DIR / "dataset_shift_overall_smd.csv",
        index=False
    )

    print("\nTop 20 cech najbardziej różniących datasety — overall:")
    print(overall_shift.head(20))

    plot_shift(
        overall_shift,
        title="Top 20 dataset shift features: PhiUSIIL vs Malicious URLs",
        filename="dataset_shift_overall_top20.png"
    )

    for label_value, label_name in [(0, "legitimate"), (1, "phishing")]:
        subset = df[df["label"] == label_value].copy()

        shift = compute_standardized_mean_difference(
            df=subset,
            feature_columns=feature_columns,
            group_column="source",
            group_a="phiusiil",
            group_b="malicious_urls"
        )

        shift.to_csv(
            RESULTS_DIR / f"dataset_shift_{label_name}_smd.csv",
            index=False
        )

        print(f"\nTop 20 cech najbardziej różniących datasety — {label_name}:")
        print(shift.head(20))

        plot_shift(
            shift,
            title=f"Top 20 dataset shift features: {label_name}",
            filename=f"dataset_shift_{label_name}_top20.png"
        )


def plot_shift(df: pd.DataFrame, title: str, filename: str) -> None:

    top = df.head(20).copy()
    top = top.sort_values(by="abs_smd", ascending=True)

    plt.figure(figsize=(10, 7))
    plt.barh(top["feature"], top["abs_smd"])
    plt.xlabel("Absolute standardized mean difference")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close()


def run_source_classifier(
    X: pd.DataFrame,
    source: pd.Series
) -> None:

    print("\n" + "=" * 80)
    print("CZY DATASETY SĄ ŁATWE DO ODRÓŻNIENIA?")
    print("=" * 80)

    df = X.copy()
    df["source"] = source.values

    sampled_parts = []

    for src in df["source"].unique():
        part = df[df["source"] == src].copy()

        n = min(len(part), SOURCE_CLASSIFIER_SAMPLE_PER_DATASET)

        sampled_parts.append(
            part.sample(n=n, random_state=42)
        )

    sampled = pd.concat(sampled_parts, ignore_index=True)

    y_source = sampled["source"].map({
        "phiusiil": 0,
        "malicious_urls": 1
    })

    X_source = sampled.drop(columns=["source"])

    X_train, X_test, y_train, y_test = train_test_split(
        X_source,
        y_source,
        test_size=0.2,
        random_state=42,
        stratify=y_source
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Source classifier accuracy:", round(accuracy, 4))
    print("Source classifier F1:", round(f1, 4))

    pd.DataFrame([{
        "accuracy": accuracy,
        "f1": f1,
        "sample_per_dataset": SOURCE_CLASSIFIER_SAMPLE_PER_DATASET
    }]).to_csv(
        RESULTS_DIR / "source_classifier_results.csv",
        index=False
    )

    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["phiusiil", "malicious_urls"]
    )

    disp.plot(values_format="d")
    plt.title("Source classifier confusion matrix")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "source_classifier_confusion_matrix.png", dpi=300)
    plt.close()


def main() -> None:
    X, y, source, feature_columns = load_combined_dataset()

    print("Wczytano combined_features.csv")
    print("Liczba rekordów:", len(X))
    print("Liczba cech:", len(feature_columns))

    print("\nRozkład etykiet:")
    print(y.value_counts().sort_index())

    print("\nRozkład źródeł:")
    print(source.value_counts())

    train_random_forest_for_importance(X, y, feature_columns)

    analyze_dataset_shift(X, y, source, feature_columns)

    run_source_classifier(X, source)

    print("\n" + "=" * 80)
    print("ZAKOŃCZONO ANALIZĘ CECH I DATASET SHIFT")
    print("=" * 80)

    print("\nZapisano pliki wynikowe w folderze results:")
    print("- feature_importance_random_forest.csv")
    print("- dataset_shift_overall_smd.csv")
    print("- dataset_shift_legitimate_smd.csv")
    print("- dataset_shift_phishing_smd.csv")
    print("- source_classifier_results.csv")

    print("\nZapisano wykresy w folderze figures.")


if __name__ == "__main__":
    main()