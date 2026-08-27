"""
features.py -- Step 3 of the phishing URL detector.

A machine learning model cannot read a string. It only understands numbers.
So before we can train anything, we have to translate each URL into a small
set of numbers that describe it. Those numbers are called "features".

This file defines 5 features. Each one is a guess about how phishing URLs
differ structurally from legitimate ones. The model's job is to figure out
which guesses were actually useful.

Run it from the project root:
    python src/features.py
"""

import re
from pathlib import Path

import pandas as pd

# A regular expression that matches four numbers separated by dots,
# e.g. "192.168.1.1". We use it to spot URLs that use a raw IP address
# instead of a domain name.
IP_PATTERN = re.compile(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")

PROJECT_ROOT = Path(__file__).parent.parent
IN_PATH = PROJECT_ROOT / "data" / "raw" / "dataset.csv"
OUT_PATH = PROJECT_ROOT / "data" / "raw" / "features.csv"


def extract_features(url: str) -> dict:
    """Turn one URL string into a dictionary of 5 numbers."""

    return {
        # Phishing URLs are often long because attackers pad them with
        # reassuring words ("secure", "login", "verify") or bury the real
        # domain deep inside a long path to push it out of view.
        "url_length": len(url),

        # Each dot usually means another subdomain level. Attackers abuse
        # this: "paypal.com.security-check.evil.ru" looks like PayPal at a
        # glance, but the real domain is evil.ru -- the last two parts.
        "num_dots": url.count("."),

        # Legitimate brand domains rarely contain digits. Auto-generated
        # attacker domains and compromised-host paths often do.
        "num_digits": sum(1 for char in url if char.isdigit()),

        # Everything before an "@" in the authority section is ignored by
        # browsers. So "http://paypal.com@evil.ru" actually goes to evil.ru
        # while looking like PayPal. Almost no legitimate URL uses this.
        "has_at": int("@" in url),

        # A raw IP address instead of a hostname means no domain was
        # registered -- common for throwaway phishing infrastructure,
        # very rare for real businesses.
        "has_ip": int(bool(IP_PATTERN.search(url))),
    }


def main() -> None:
    df = pd.read_csv(IN_PATH)
    print(f"Loaded {len(df)} URLs from {IN_PATH.name}")

    # Apply extract_features to every URL. This produces a Series of dicts,
    # and pd.DataFrame(...) turns that list of dicts into a proper table
    # with one column per feature.
    feature_rows = df["url"].apply(extract_features)
    features_df = pd.DataFrame(feature_rows.tolist())

    # Glue the label column back on so everything stays aligned.
    features_df["label"] = df["label"]

    features_df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}\n")

    # Sanity check: compare the average value of each feature for phishing
    # vs benign URLs. If a feature is useful, the two rows will differ.
    # If they look identical, that feature is probably dead weight.
    print("Average feature values by class:")
    print(features_df.groupby("label").mean().rename(index={0: "benign", 1: "phishing"}))


if __name__ == "__main__":
    main()
