from pathlib import Path
import pandas as pd


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

PHIUSIIL_PATH = RAW_DIR / "phiusiil.csv"
MALICIOUS_PATH = RAW_DIR / "malicious_urls.csv"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def prepare_phiusiil() -> pd.DataFrame:

    print("Wczytywanie PhiUSIIL...")

    df = pd.read_csv(PHIUSIIL_PATH, encoding_errors="replace")

    df = df[["URL", "label"]].copy()

    df = df.rename(columns={"URL": "url", "label": "original_label"})

    df["label"] = df["original_label"].map({
        1: 0,
        0: 1
    })

    df["source"] = "phiusiil"

    df = df[["url", "label", "source"]]

    return df


def prepare_malicious_urls() -> pd.DataFrame:

    print("Wczytywanie Malicious URLs Dataset...")

    df = pd.read_csv(MALICIOUS_PATH, encoding_errors="replace")

    df = df[["url", "type"]].copy()

    df = df[df["type"].isin(["benign", "phishing"])].copy()

    df["label"] = df["type"].map({
        "benign": 0,
        "phishing": 1
    })

    df["source"] = "malicious_urls"

    df = df[["url", "label", "source"]]

    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:

    print("Czyszczenie danych...")

    before = len(df)

    df = df.dropna(subset=["url", "label"]).copy()

    df["url"] = df["url"].astype(str)
    df["url"] = df["url"].str.strip()

    df = df[df["url"] != ""].copy()

    df = df.drop_duplicates(subset=["url"]).copy()

    after = len(df)

    print(f"Liczba rekordów przed czyszczeniem: {before}")
    print(f"Liczba rekordów po czyszczeniu:    {after}")
    print(f"Usunięto rekordów:                 {before - after}")

    return df


def print_summary(name: str, df: pd.DataFrame) -> None:

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    print("Liczba rekordów:", len(df))

    print("\nRozkład etykiet:")
    print(df["label"].value_counts().sort_index())

    print("\nRozkład etykiet procentowo:")
    print((df["label"].value_counts(normalize=True).sort_index() * 100).round(2))

    print("\nŹródła:")
    print(df["source"].value_counts())


def main() -> None:
    if not PHIUSIIL_PATH.exists():
        raise FileNotFoundError(f"Brakuje pliku: {PHIUSIIL_PATH}")

    if not MALICIOUS_PATH.exists():
        raise FileNotFoundError(f"Brakuje pliku: {MALICIOUS_PATH}")

    phiusiil = prepare_phiusiil()
    malicious = prepare_malicious_urls()

    phiusiil = clean_dataset(phiusiil)
    malicious = clean_dataset(malicious)

    combined = pd.concat([phiusiil, malicious], ignore_index=True)
    combined = clean_dataset(combined)

    phiusiil.to_csv(PROCESSED_DIR / "phiusiil_clean.csv", index=False)
    malicious.to_csv(PROCESSED_DIR / "malicious_urls_clean.csv", index=False)
    combined.to_csv(PROCESSED_DIR / "combined_clean.csv", index=False)

    print_summary("PHIUSIIL CLEAN", phiusiil)
    print_summary("MALICIOUS URLS CLEAN", malicious)
    print_summary("COMBINED CLEAN", combined)

    print("\nZapisano pliki:")
    print(PROCESSED_DIR / "phiusiil_clean.csv")
    print(PROCESSED_DIR / "malicious_urls_clean.csv")
    print(PROCESSED_DIR / "combined_clean.csv")


if __name__ == "__main__":
    main()