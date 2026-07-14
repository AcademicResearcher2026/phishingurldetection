from pathlib import Path
from urllib.parse import urlparse
import math
import re
import ipaddress
import warnings

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tldextract

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


CLEAN_FILES = {
    "phiusiil": PROCESSED_DIR / "phiusiil_clean.csv",
    "malicious_urls": PROCESSED_DIR / "malicious_urls_clean.csv",
    "combined": PROCESSED_DIR / "combined_clean.csv",
}

NORMALIZED_FEATURE_FILES = {
    "phiusiil": PROCESSED_DIR / "phiusiil_normalized_features.csv",
    "malicious_urls": PROCESSED_DIR / "malicious_urls_normalized_features.csv",
    "combined": PROCESSED_DIR / "combined_normalized_features.csv",
}


FORCE_REEXTRACT = False


TLD_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=())


SUSPICIOUS_KEYWORDS = [
    "login", "verify", "secure", "account", "update", "confirm",
    "bank", "paypal", "password", "signin", "wallet", "payment",
    "invoice", "support", "security", "webscr", "auth", "ebay",
    "amazon", "apple", "microsoft", "google", "facebook", "netflix"
]

BRAND_KEYWORDS = [
    "paypal", "amazon", "apple", "microsoft", "google",
    "facebook", "netflix", "ebay", "instagram", "whatsapp",
    "allegro", "pko", "mbank", "ing", "santander"
]

URL_SHORTENERS = [
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly",
    "is.gd", "buff.ly", "cutt.ly", "rebrand.ly", "rb.gy",
    "shorturl.at", "bitly.com"
]

COMMON_TLDS = [
    "com", "org", "net", "edu", "gov", "pl", "de", "uk", "fr",
    "it", "es", "nl", "ca", "au"
]


METADATA_COLUMNS = ["url", "label", "source"]


def normalize_url(url: str) -> str:

    url = str(url).strip().lower()

    url = re.sub(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", "", url)

    url = url.split("#")[0]

    if url.startswith("www."):
        url = url[4:]

    url = url.strip()
    url = url.rstrip("/")

    return url


def safe_divide(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return a / b


def add_scheme_for_parsing(url: str) -> str:

    return "http://" + str(url).strip()


def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0

    text_length = len(text)
    probabilities = []

    for char in set(text):
        probabilities.append(text.count(char) / text_length)

    entropy = -sum(p * math.log2(p) for p in probabilities)

    return entropy


def is_ip_address(hostname: str) -> int:
    if not hostname:
        return 0

    try:
        ipaddress.ip_address(hostname)
        return 1
    except ValueError:
        return 0


def count_query_parameters(query: str) -> int:
    if not query:
        return 0

    return query.count("&") + 1


def tokenize_url(url: str) -> list:
    tokens = re.split(r"[^A-Za-z0-9]+", url)
    tokens = [token for token in tokens if token]

    return tokens


def extract_features_from_normalized_url(url: str) -> dict:

    normalized_url = normalize_url(url)
    parsed_input = add_scheme_for_parsing(normalized_url)

    try:
        parsed = urlparse(parsed_input)
    except Exception:
        parsed = urlparse("http://")

    hostname = parsed.hostname.lower() if parsed.hostname else ""
    path = parsed.path if parsed.path else ""
    query = parsed.query if parsed.query else ""
    fragment = parsed.fragment if parsed.fragment else ""

    try:
        port = parsed.port
    except ValueError:
        port = None

    extracted = TLD_EXTRACTOR(hostname)

    subdomain = extracted.subdomain or ""
    domain = extracted.domain or ""
    suffix = extracted.suffix or ""

    registered_domain = ".".join(part for part in [domain, suffix] if part)

    url_lower = normalized_url.lower()
    hostname_lower = hostname.lower()
    registered_domain_lower = registered_domain.lower()

    tokens = tokenize_url(normalized_url)

    if tokens:
        token_lengths = [len(token) for token in tokens]
        max_token_length = max(token_lengths)
        avg_token_length = float(np.mean(token_lengths))
    else:
        max_token_length = 0
        avg_token_length = 0.0

    url_length = len(normalized_url)
    hostname_length = len(hostname)
    path_length = len(path)
    query_length = len(query)

    digit_count = sum(char.isdigit() for char in normalized_url)
    letter_count = sum(char.isalpha() for char in normalized_url)
    special_count = sum(not char.isalnum() for char in normalized_url)

    suspicious_keyword_count = sum(
        1 for word in SUSPICIOUS_KEYWORDS
        if word in url_lower
    )

    brand_keyword_count = sum(
        1 for word in BRAND_KEYWORDS
        if word in url_lower
    )

    has_shortener = int(
        any(shortener in registered_domain_lower for shortener in URL_SHORTENERS)
    )

    features = {
        "url_length": url_length,
        "hostname_length": hostname_length,
        "path_length": path_length,
        "query_length": query_length,
        "fragment_length": len(fragment),

        "registered_domain_length": len(registered_domain),
        "domain_length": len(domain),
        "subdomain_length": len(subdomain),
        "tld_length": len(suffix),
        "subdomain_count": 0 if not subdomain else subdomain.count(".") + 1,
        "hostname_dot_count": hostname.count("."),
        "is_common_tld": int(suffix in COMMON_TLDS),
        "is_com_tld": int(suffix == "com"),
        "is_org_tld": int(suffix == "org"),
        "is_net_tld": int(suffix == "net"),
        "is_pl_tld": int(suffix == "pl"),
        "is_www": 0,
        "dot_count": normalized_url.count("."),
        "hyphen_count": normalized_url.count("-"),
        "underscore_count": normalized_url.count("_"),
        "slash_count": normalized_url.count("/"),
        "question_count": normalized_url.count("?"),
        "equal_count": normalized_url.count("="),
        "ampersand_count": normalized_url.count("&"),
        "percent_count": normalized_url.count("%"),
        "at_count": normalized_url.count("@"),
        "colon_count": normalized_url.count(":"),
        "semicolon_count": normalized_url.count(";"),
        "digit_count": digit_count,
        "letter_count": letter_count,
        "special_char_count": special_count,
        "digit_ratio": safe_divide(digit_count, url_length),
        "letter_ratio": safe_divide(letter_count, url_length),
        "special_char_ratio": safe_divide(special_count, url_length),
        "path_depth": len([part for part in path.split("/") if part]),
        "query_param_count": count_query_parameters(query),
        "token_count": len(tokens),
        "max_token_length": max_token_length,
        "avg_token_length": avg_token_length,
        "url_entropy": shannon_entropy(normalized_url),
        "hostname_entropy": shannon_entropy(hostname),
        "has_https": 0,
        "has_http": 0,
        "has_ip_address": is_ip_address(hostname),
        "has_port": int(port is not None),
        "has_at_symbol": int("@" in normalized_url),
        "has_double_slash_in_path": int("//" in path),
        "has_punycode": int("xn--" in hostname_lower),
        "has_percent_encoding": int("%" in normalized_url),
        "suspicious_keyword_count": suspicious_keyword_count,
        "brand_keyword_count": brand_keyword_count,
        "has_login_keyword": int("login" in url_lower),
        "has_verify_keyword": int("verify" in url_lower),
        "has_secure_keyword": int("secure" in url_lower),
        "has_account_keyword": int("account" in url_lower),
        "has_update_keyword": int("update" in url_lower),
        "has_bank_keyword": int("bank" in url_lower),
        "has_paypal_keyword": int("paypal" in url_lower),
        "has_microsoft_keyword": int("microsoft" in url_lower),
        "has_apple_keyword": int("apple" in url_lower),
        "has_amazon_keyword": int("amazon" in url_lower),
        "has_url_shortener_domain": has_shortener,
        "has_hyphen_in_domain": int("-" in registered_domain),
        "has_digit_in_domain": int(any(char.isdigit() for char in registered_domain)),
    }

    return features


def extract_normalized_features_for_dataset(name: str, input_path: Path, output_path: Path) -> None:
    if output_path.exists() and not FORCE_REEXTRACT:
        print(f"Pominięto ekstrakcję, plik już istnieje: {output_path}")
        return

    print("\n" + "=" * 80)
    print(f"EKSTRAKCJA CECH PO NORMALIZACJI: {name}")
    print("=" * 80)

    df = pd.read_csv(input_path, encoding_errors="replace")

    feature_rows = []

    for i, url in enumerate(df["url"], start=1):
        feature_rows.append(extract_features_from_normalized_url(url))

        if i % 50000 == 0:
            print(f"Przetworzono {i} URL-i...")

    features_df = pd.DataFrame(feature_rows)

    result = pd.concat(
        [
            df[["url", "label", "source"]].reset_index(drop=True),
            features_df.reset_index(drop=True)
        ],
        axis=1
    )

    result.to_csv(output_path, index=False)

    print(f"Zapisano: {output_path}")
    print("Liczba rekordów:", result.shape[0])
    print("Liczba kolumn:", result.shape[1])


def load_feature_dataset(path: Path):
    df = pd.read_csv(path, encoding_errors="replace")

    feature_columns = [
        col for col in df.columns
        if col not in METADATA_COLUMNS
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
        "pr_auc": average_precision_score(y_test, y_score)
    }


def run_internal_experiment(dataset_name: str, path: Path) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print(f"NORMALIZACJA EKSPERYMENT WEWNĘTRZNY: {dataset_name}")
    print("=" * 80)

    X, y, feature_columns = load_feature_dataset(path)

    print("Liczba rekordów:", len(X))
    print("Liczba cech:", len(feature_columns))

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
            "experiment": "internal_normalized_url",
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
    train_name: str,
    train_path: Path,
    test_name: str,
    test_path: Path
) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print(f"NORMALIZED CROSS-DATASET: train={train_name}, test={test_name}")
    print("=" * 80)

    X_train, y_train, train_features = load_feature_dataset(train_path)
    X_test, y_test, test_features = load_feature_dataset(test_path)

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
            "experiment": "cross_dataset_normalized_url",
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


def run_source_classifier_on_normalized_features() -> None:
    print("\n" + "=" * 80)
    print("NORMALIZED SOURCE CLASSIFIER")
    print("=" * 80)

    df = pd.read_csv(
        NORMALIZED_FEATURE_FILES["combined"],
        encoding_errors="replace"
    )

    feature_columns = [
        col for col in df.columns
        if col not in METADATA_COLUMNS
    ]

    sampled_parts = []

    for src in df["source"].unique():
        part = df[df["source"] == src].copy()
        n = min(len(part), 100000)
        sampled_parts.append(part.sample(n=n, random_state=42))

    sampled = pd.concat(sampled_parts, ignore_index=True)

    X = sampled[feature_columns].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    y = sampled["source"].map({
        "phiusiil": 0,
        "malicious_urls": 1
    })

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=1
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Normalized source classifier accuracy:", round(accuracy, 4))
    print("Normalized source classifier F1:", round(f1, 4))

    pd.DataFrame([{
        "feature_variant": "normalized_url",
        "accuracy": accuracy,
        "f1": f1,
        "sample_per_dataset": 100000
    }]).to_csv(
        RESULTS_DIR / "normalized_source_classifier_results.csv",
        index=False
    )


def plot_normalized_cross_dataset_results(results: pd.DataFrame) -> None:
    cross = results[
        results["experiment"] == "cross_dataset_normalized_url"
    ].copy()

    cross["label"] = cross["dataset"] + " | " + cross["model"]
    cross = cross.sort_values(by="f1", ascending=False)

    plt.figure(figsize=(12, 6))
    plt.bar(cross["label"], cross["f1"])
    plt.ylabel("F1-score")
    plt.title("Normalized URL control experiment: cross-dataset F1-score")
    plt.xticks(rotation=75, ha="right")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "normalized_url_cross_dataset_f1.png",
        dpi=300
    )
    plt.close()


def main() -> None:
    print("START: normalized URL control experiment")

    for name in CLEAN_FILES:
        input_path = CLEAN_FILES[name]
        output_path = NORMALIZED_FEATURE_FILES[name]

        if not input_path.exists():
            raise FileNotFoundError(f"Brakuje pliku: {input_path}")

        extract_normalized_features_for_dataset(
            name=name,
            input_path=input_path,
            output_path=output_path
        )

    all_results = []

    for dataset_name, path in NORMALIZED_FEATURE_FILES.items():
        result = run_internal_experiment(dataset_name, path)
        all_results.append(result)

    cross_1 = run_cross_dataset_experiment(
        train_name="phiusiil",
        train_path=NORMALIZED_FEATURE_FILES["phiusiil"],
        test_name="malicious_urls",
        test_path=NORMALIZED_FEATURE_FILES["malicious_urls"]
    )

    cross_2 = run_cross_dataset_experiment(
        train_name="malicious_urls",
        train_path=NORMALIZED_FEATURE_FILES["malicious_urls"],
        test_name="phiusiil",
        test_path=NORMALIZED_FEATURE_FILES["phiusiil"]
    )

    all_results.append(cross_1)
    all_results.append(cross_2)

    final_results = pd.concat(all_results, ignore_index=True)

    final_results.to_csv(
        RESULTS_DIR / "normalized_url_control_results.csv",
        index=False
    )

    plot_normalized_cross_dataset_results(final_results)

    run_source_classifier_on_normalized_features()

    print("\n" + "=" * 80)
    print("ZAKOŃCZONO NORMALIZED URL CONTROL EXPERIMENT")
    print("=" * 80)

    print("\nZapisano:")
    print(RESULTS_DIR / "normalized_url_control_results.csv")
    print(RESULTS_DIR / "normalized_source_classifier_results.csv")
    print(FIGURES_DIR / "normalized_url_cross_dataset_f1.png")

    print("\nNajlepsze wyniki cross-dataset po normalizacji URL:")
    cross = final_results[
        final_results["experiment"] == "cross_dataset_normalized_url"
    ].copy()

    print(
        cross.sort_values(by="f1", ascending=False)
        [
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
        ]
        .head(20)
    )


if __name__ == "__main__":
    main()