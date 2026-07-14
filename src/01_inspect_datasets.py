from pathlib import Path
import pandas as pd


RAW_DIR = Path("data/raw")


def inspect_csv(path: Path) -> None:
    print("=" * 80)
    print(f"PLIK: {path}")
    print("=" * 80)

    try:
        df_sample = pd.read_csv(path, nrows=5, encoding_errors="replace")
    except Exception as e:
        print(f"Błąd podczas wczytywania próbki: {e}")
        return

    print("\nKOLUMNY:")
    for i, col in enumerate(df_sample.columns, start=1):
        print(f"{i}. {col}")

    print("\nPIERWSZE 5 WIERSZY:")
    print(df_sample.head())

    print("\nPRÓBA WYKRYCIA KOLUMN URL I ETYKIETY:")

    url_candidates = [
        "url", "URL", "Url", "website", "Website", "link", "Link"
    ]

    label_candidates = [
        "label", "Label", "class", "Class", "type", "Type",
        "status", "Status", "category", "Category"
    ]

    detected_url = [col for col in df_sample.columns if col in url_candidates]
    detected_label = [col for col in df_sample.columns if col in label_candidates]

    print("Możliwe kolumny URL:", detected_url if detected_url else "NIE WYKRYTO")
    print("Możliwe kolumny etykiety:", detected_label if detected_label else "NIE WYKRYTO")

    for label_col in detected_label:
        try:
            df_label = pd.read_csv(
                path,
                usecols=[label_col],
                encoding_errors="replace"
            )
            print(f"\nRozkład wartości w kolumnie '{label_col}':")
            print(df_label[label_col].value_counts(dropna=False).head(20))
        except Exception as e:
            print(f"Nie udało się policzyć wartości dla '{label_col}': {e}")


def main() -> None:
    if not RAW_DIR.exists():
        print("Brakuje folderu data/raw.")
        return

    csv_files = list(RAW_DIR.glob("*.csv"))

    if not csv_files:
        print("Nie znaleziono plików CSV w data/raw.")
        print("Dodaj pliki phiusiil.csv oraz malicious_urls.csv.")
        return

    print(f"Znaleziono {len(csv_files)} plików CSV.\n")

    for path in csv_files:
        inspect_csv(path)


if __name__ == "__main__":
    main()