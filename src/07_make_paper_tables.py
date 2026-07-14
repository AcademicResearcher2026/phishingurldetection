from pathlib import Path
import pandas as pd


RESULTS_DIR = Path("results")
PAPER_DIR = Path("paper")

PAPER_DIR.mkdir(parents=True, exist_ok=True)


def round_metrics(df: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"
    ]

    for col in metric_columns:
        if col in df.columns:
            df[col] = df[col].round(4)

    return df


def save_table(df: pd.DataFrame, filename_base: str) -> None:
    csv_path = PAPER_DIR / f"{filename_base}.csv"
    md_path = PAPER_DIR / f"{filename_base}.md"

    df.to_csv(csv_path, index=False)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(df.to_markdown(index=False))

    print(f"Zapisano: {csv_path}")
    print(f"Zapisano: {md_path}")


def make_dataset_summary_table() -> None:

    df = pd.DataFrame([
        {
            "Dataset": "PhiUSIIL",
            "Total URLs": 235370,
            "Legitimate URLs": 134850,
            "Phishing URLs": 100520,
            "Phishing ratio (%)": 42.71,
            "Use in study": "Internal evaluation and cross-dataset testing"
        },
        {
            "Dataset": "Malicious URLs",
            "Total URLs": 522166,
            "Legitimate URLs": 428080,
            "Phishing URLs": 94086,
            "Phishing ratio (%)": 18.02,
            "Use in study": "Internal evaluation and cross-dataset testing"
        },
        {
            "Dataset": "Combined",
            "Total URLs": 757483,
            "Legitimate URLs": 562930,
            "Phishing URLs": 194553,
            "Phishing ratio (%)": 25.68,
            "Use in study": "Internal evaluation"
        }
    ])

    save_table(df, "table_1_dataset_summary")


def make_internal_results_table() -> None:

    path = RESULTS_DIR / "internal_model_results.csv"

    if not path.exists():
        raise FileNotFoundError(f"Brakuje pliku: {path}")

    df = pd.read_csv(path)

    df = round_metrics(df)

    df = df[
        [
            "dataset",
            "model",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc"
        ]
    ].copy()

    df = df.rename(columns={
        "dataset": "Dataset",
        "model": "Model",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1-score",
        "roc_auc": "ROC-AUC",
        "pr_auc": "PR-AUC"
    })

    save_table(df, "table_2_internal_results")


def make_cross_dataset_results_table() -> None:

    path = RESULTS_DIR / "cross_dataset_model_results.csv"

    if not path.exists():
        raise FileNotFoundError(f"Brakuje pliku: {path}")

    df = pd.read_csv(path)

    df = round_metrics(df)

    df = df[
        [
            "train_dataset",
            "test_dataset",
            "model",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc"
        ]
    ].copy()

    df = df.rename(columns={
        "train_dataset": "Training dataset",
        "test_dataset": "Testing dataset",
        "model": "Model",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1-score",
        "roc_auc": "ROC-AUC",
        "pr_auc": "PR-AUC"
    })

    save_table(df, "table_3_cross_dataset_results")


def make_ablation_table() -> None:

    path = RESULTS_DIR / "ablation_study_results.csv"

    if not path.exists():
        raise FileNotFoundError(f"Brakuje pliku: {path}")

    df = pd.read_csv(path)

    df = df[df["experiment"] == "cross_dataset"].copy()

    df = round_metrics(df)

    df = df[
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
    ].copy()

    df = df.rename(columns={
        "dataset": "Dataset transfer",
        "feature_variant": "Feature variant",
        "model": "Model",
        "n_features": "Number of features",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1-score",
        "roc_auc": "ROC-AUC",
        "pr_auc": "PR-AUC"
    })

    df = df.sort_values(
        by=["Dataset transfer", "Feature variant", "F1-score"],
        ascending=[True, True, False]
    )

    save_table(df, "table_4_ablation_cross_dataset")


def make_feature_importance_table() -> None:

    path = RESULTS_DIR / "feature_importance_random_forest.csv"

    if not path.exists():
        raise FileNotFoundError(f"Brakuje pliku: {path}")

    df = pd.read_csv(path)

    df = df.head(20).copy()
    df["importance"] = df["importance"].round(4)

    df = df.rename(columns={
        "feature": "Feature",
        "importance": "Importance"
    })

    save_table(df, "table_5_feature_importance_top20")


def make_dataset_shift_table() -> None:

    path = RESULTS_DIR / "dataset_shift_overall_smd.csv"

    if not path.exists():
        raise FileNotFoundError(f"Brakuje pliku: {path}")

    df = pd.read_csv(path)

    df = df.head(20).copy()

    numeric_cols = [
        "mean_phiusiil",
        "mean_malicious_urls",
        "smd",
        "abs_smd"
    ]

    for col in numeric_cols:
        df[col] = df[col].round(4)

    df = df.rename(columns={
        "feature": "Feature",
        "mean_phiusiil": "Mean in PhiUSIIL",
        "mean_malicious_urls": "Mean in Malicious URLs",
        "smd": "SMD",
        "abs_smd": "Absolute SMD"
    })

    save_table(df, "table_6_dataset_shift_top20")


def make_source_classifier_table() -> None:

    path = RESULTS_DIR / "source_classifier_results.csv"

    if not path.exists():
        raise FileNotFoundError(f"Brakuje pliku: {path}")

    df = pd.read_csv(path)

    df = round_metrics(df)

    df = df.rename(columns={
        "accuracy": "Accuracy",
        "f1": "F1-score",
        "sample_per_dataset": "Sample per dataset"
    })

    save_table(df, "table_7_source_classifier")


def main() -> None:
    print("Tworzenie tabel do artykułu...\n")

    make_dataset_summary_table()
    make_internal_results_table()
    make_cross_dataset_results_table()
    make_ablation_table()
    make_feature_importance_table()
    make_dataset_shift_table()
    make_source_classifier_table()

    print("\nGotowe. Tabele zapisano w folderze paper.")


if __name__ == "__main__":
    main()