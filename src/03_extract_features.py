from pathlib import Path
from urllib.parse import urlparse
import math
import re
import ipaddress

import pandas as pd
import numpy as np
import tldextract


PROCESSED_DIR = Path("data/processed")

INPUT_FILES = {
    "phiusiil": PROCESSED_DIR / "phiusiil_clean.csv",
    "malicious_urls": PROCESSED_DIR / "malicious_urls_clean.csv",
    "combined": PROCESSED_DIR / "combined_clean.csv",
}

OUTPUT_FILES = {
    "phiusiil": PROCESSED_DIR / "phiusiil_features.csv",
    "malicious_urls": PROCESSED_DIR / "malicious_urls_features.csv",
    "combined": PROCESSED_DIR / "combined_features.csv",
}

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


def safe_divide(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return a / b


def add_scheme_if_missing(url: str) -> str:

    url = str(url).strip()

    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        return url

    return "http://" + url


def shannon_entropy(text: str) -> float:

    if not text:
        return 0.0

    probabilities = []
    text_length = len(text)

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


def extract_features_from_url(url: str) -> dict:

    original_url = str(url).strip()
    parsed_url = add_scheme_if_missing(original_url)

    try:
        parsed = urlparse(parsed_url)
    except Exception:
        parsed = urlparse("http://")

    scheme = parsed.scheme.lower()
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

    url_lower = original_url.lower()
    hostname_lower = hostname.lower()
    registered_domain_lower = registered_domain.lower()

    tokens = tokenize_url(original_url)

    if tokens:
        token_lengths = [len(token) for token in tokens]
        max_token_length = max(token_lengths)
        avg_token_length = float(np.mean(token_lengths))
    else:
        max_token_length = 0
        avg_token_length = 0.0

    url_length = len(original_url)
    hostname_length = len(hostname)
    path_length = len(path)
    query_length = len(query)

    digit_count = sum(char.isdigit() for char in original_url)
    letter_count = sum(char.isalpha() for char in original_url)
    special_count = sum(not char.isalnum() for char in original_url)

    suspicious_keyword_count = sum(1 for word in SUSPICIOUS_KEYWORDS if word in url_lower)
    brand_keyword_count = sum(1 for word in BRAND_KEYWORDS if word in url_lower)

    has_shortener = int(any(shortener in registered_domain_lower for shortener in URL_SHORTENERS))

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
        "is_www": int(hostname.startswith("www.")),
        "dot_count": original_url.count("."),
        "hyphen_count": original_url.count("-"),
        "underscore_count": original_url.count("_"),
        "slash_count": original_url.count("/"),
        "question_count": original_url.count("?"),
        "equal_count": original_url.count("="),
        "ampersand_count": original_url.count("&"),
        "percent_count": original_url.count("%"),
        "at_count": original_url.count("@"),
        "colon_count": original_url.count(":"),
        "semicolon_count": original_url.count(";"),
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
        "url_entropy": shannon_entropy(original_url),
        "hostname_entropy": shannon_entropy(hostname),
        "has_https": int(scheme == "https"),
        "has_http": int(scheme == "http"),
        "has_ip_address": is_ip_address(hostname),
        "has_port": int(port is not None),
        "has_at_symbol": int("@" in original_url),
        "has_double_slash_in_path": int("//" in path),
        "has_punycode": int("xn--" in hostname_lower),
        "has_percent_encoding": int("%" in original_url),
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


def extract_features_for_dataframe(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:

    print("\n" + "=" * 80)
    print(f"Ekstrakcja cech: {dataset_name}")
    print("=" * 80)
    print("Liczba rekordów:", len(df))

    feature_rows = []

    for i, url in enumerate(df["url"], start=1):
        feature_rows.append(extract_features_from_url(url))

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

    return result


def main() -> None:
    for name, input_path in INPUT_FILES.items():
        if not input_path.exists():
            raise FileNotFoundError(f"Brakuje pliku: {input_path}")

        df = pd.read_csv(input_path, encoding_errors="replace")

        features_df = extract_features_for_dataframe(df, name)

        output_path = OUTPUT_FILES[name]
        features_df.to_csv(output_path, index=False)

        print(f"Zapisano: {output_path}")
        print(f"Liczba kolumn po ekstrakcji: {features_df.shape[1]}")
        print(f"Liczba rekordów: {features_df.shape[0]}")


if __name__ == "__main__":
    main()