from pathlib import Path
import pandas as pd


RESULTS_DIR = Path("results")
PAPER_DIR = Path("paper")

PAPER_DIR.mkdir(parents=True, exist_ok=True)


METRIC_COLUMNS = [
    "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"
]


def round_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in METRIC_COLUMNS:
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


def make_normalized_internal_table() -> None:

    path = RESULTS_DIR / "normalized_url_control_results.csv"

    if not path.exists():
        raise FileNotFoundError(f"Brakuje pliku: {path}")

    df = pd.read_csv(path)

    df = df[df["experiment"] == "internal_normalized_url"].copy()
    df = round_metrics(df)

    df = df[
        [
            "dataset",
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
        "dataset": "Dataset",
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
        by=["Dataset", "F1-score"],
        ascending=[True, False]
    )

    save_table(df, "table_8_normalized_internal_results")


def make_normalized_cross_dataset_table() -> None:

    path = RESULTS_DIR / "normalized_url_control_results.csv"

    if not path.exists():
        raise FileNotFoundError(f"Brakuje pliku: {path}")

    df = pd.read_csv(path)

    df = df[df["experiment"] == "cross_dataset_normalized_url"].copy()
    df = round_metrics(df)

    df = df[
        [
            "dataset",
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
        by=["Dataset transfer", "F1-score"],
        ascending=[True, False]
    )

    save_table(df, "table_9_normalized_cross_dataset_results")


def make_source_classifier_comparison_table() -> None:

    original_path = RESULTS_DIR / "source_classifier_results.csv"
    normalized_path = RESULTS_DIR / "normalized_source_classifier_results.csv"

    if not original_path.exists():
        raise FileNotFoundError(f"Brakuje pliku: {original_path}")

    if not normalized_path.exists():
        raise FileNotFoundError(f"Brakuje pliku: {normalized_path}")

    original = pd.read_csv(original_path)
    normalized = pd.read_csv(normalized_path)

    original["Feature variant"] = "Original URL features"
    normalized["Feature variant"] = "Normalized URL features"

    original = original.rename(columns={
        "accuracy": "Accuracy",
        "f1": "F1-score",
        "sample_per_dataset": "Sample per dataset"
    })

    normalized = normalized.rename(columns={
        "accuracy": "Accuracy",
        "f1": "F1-score",
        "sample_per_dataset": "Sample per dataset"
    })

    df = pd.concat(
        [
            original[["Feature variant", "Accuracy", "F1-score", "Sample per dataset"]],
            normalized[["Feature variant", "Accuracy", "F1-score", "Sample per dataset"]]
        ],
        ignore_index=True
    )

    df["Accuracy"] = df["Accuracy"].round(4)
    df["F1-score"] = df["F1-score"].round(4)

    save_table(df, "table_10_source_classifier_comparison")


def make_best_results_summary_table() -> None:

    rows = []

    internal_path = RESULTS_DIR / "internal_model_results.csv"
    cross_path = RESULTS_DIR / "cross_dataset_model_results.csv"
    ablation_path = RESULTS_DIR / "ablation_study_results.csv"
    normalized_path = RESULTS_DIR / "normalized_url_control_results.csv"

    if not internal_path.exists():
        raise FileNotFoundError(f"Brakuje pliku: {internal_path}")

    if not cross_path.exists():
        raise FileNotFoundError(f"Brakuje pliku: {cross_path}")

    if not ablation_path.exists():
        raise FileNotFoundError(f"Brakuje pliku: {ablation_path}")

    if not normalized_path.exists():
        raise FileNotFoundError(f"Brakuje pliku: {normalized_path}")

    internal = pd.read_csv(internal_path)
    cross = pd.read_csv(cross_path)
    ablation = pd.read_csv(ablation_path)
    normalized = pd.read_csv(normalized_path)

    for dataset in internal["dataset"].unique():
        subset = internal[internal["dataset"] == dataset].copy()
        best = subset.sort_values(by="f1", ascending=False).iloc[0]

        rows.append({
            "Scenario": "Internal evaluation",
            "Variant": "Original URL features",
            "Dataset / transfer": dataset,
            "Best model": best["model"],
            "Accuracy": best["accuracy"],
            "Precision": best["precision"],
            "Recall": best["recall"],
            "F1-score": best["f1"],
            "ROC-AUC": best["roc_auc"],
            "PR-AUC": best["pr_auc"]
        })

    for transfer in cross["dataset"].unique():
        subset = cross[cross["dataset"] == transfer].copy()
        best = subset.sort_values(by="f1", ascending=False).iloc[0]

        rows.append({
            "Scenario": "Cross-dataset evaluation",
            "Variant": "Original URL features",
            "Dataset / transfer": transfer,
            "Best model": best["model"],
            "Accuracy": best["accuracy"],
            "Precision": best["precision"],
            "Recall": best["recall"],
            "F1-score": best["f1"],
            "ROC-AUC": best["roc_auc"],
            "PR-AUC": best["pr_auc"]
        })

    ablation_cross = ablation[
        (ablation["experiment"] == "cross_dataset") &
        (ablation["feature_variant"] == "without_artifact_prone_features")
    ].copy()

    for transfer in ablation_cross["dataset"].unique():
        subset = ablation_cross[ablation_cross["dataset"] == transfer].copy()
        best = subset.sort_values(by="f1", ascending=False).iloc[0]

        rows.append({
            "Scenario": "Cross-dataset evaluation",
            "Variant": "Without artifact-prone features",
            "Dataset / transfer": transfer,
            "Best model": best["model"],
            "Accuracy": best["accuracy"],
            "Precision": best["precision"],
            "Recall": best["recall"],
            "F1-score": best["f1"],
            "ROC-AUC": best["roc_auc"],
            "PR-AUC": best["pr_auc"]
        })

    normalized_cross = normalized[
        normalized["experiment"] == "cross_dataset_normalized_url"
    ].copy()

    for transfer in normalized_cross["dataset"].unique():
        subset = normalized_cross[normalized_cross["dataset"] == transfer].copy()
        best = subset.sort_values(by="f1", ascending=False).iloc[0]

        rows.append({
            "Scenario": "Cross-dataset evaluation",
            "Variant": "Normalized URL features",
            "Dataset / transfer": transfer,
            "Best model": best["model"],
            "Accuracy": best["accuracy"],
            "Precision": best["precision"],
            "Recall": best["recall"],
            "F1-score": best["f1"],
            "ROC-AUC": best["roc_auc"],
            "PR-AUC": best["pr_auc"]
        })

    df = pd.DataFrame(rows)

    numeric_cols = [
        "Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC", "PR-AUC"
    ]

    for col in numeric_cols:
        df[col] = df[col].round(4)

    save_table(df, "table_11_best_results_summary")


def make_final_results_master_file() -> None:

    all_parts = []

    original_path = RESULTS_DIR / "all_model_results.csv"
    ablation_path = RESULTS_DIR / "ablation_study_results.csv"
    normalized_path = RESULTS_DIR / "normalized_url_control_results.csv"

    if original_path.exists():
        original = pd.read_csv(original_path)
        original["result_family"] = "original"
        all_parts.append(original)

    if ablation_path.exists():
        ablation = pd.read_csv(ablation_path)
        ablation["result_family"] = "ablation"
        all_parts.append(ablation)

    if normalized_path.exists():
        normalized = pd.read_csv(normalized_path)
        normalized["result_family"] = "normalized_url"
        all_parts.append(normalized)

    if not all_parts:
        raise FileNotFoundError("Nie znaleziono żadnych plików wynikowych.")

    df = pd.concat(all_parts, ignore_index=True, sort=False)
    df = round_metrics(df)

    output_path = PAPER_DIR / "final_results_master.csv"
    df.to_csv(output_path, index=False)

    print(f"Zapisano: {output_path}")


def main() -> None:
    print("Tworzenie finalnych tabel...\n")

    make_normalized_internal_table()
    make_normalized_cross_dataset_table()
    make_source_classifier_comparison_table()
    make_best_results_summary_table()
    make_final_results_master_file()

    print("\nGotowe.")


if __name__ == "__main__":
    main()