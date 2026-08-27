"""
get_data.py -- Step 2 of the phishing URL detector.

Downloads two lists of URLs and combines them into one labelled dataset:
  label 1 = phishing   (from the OpenPhish community feed)
  label 0 = benign     (from the Tranco top-sites list)

Output: data/raw/dataset.csv  with columns [url, label]

SAFETY NOTE: this script downloads a *list* of phishing URLs as text.
It never visits any of them. Do not add code that requests these URLs --
that would send traffic from your IP to live attacker infrastructure.

Run it from the project root:
    python src/get_data.py
"""

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# The free OpenPhish community feed. Plain text, one URL per line.
PHISH_FEED_URL = "https://openphish.com/feed.txt"

# Tranco: a research-grade ranking of the most popular sites on the internet.
# This URL always points at the latest list. It's a ~20 MB zip containing
# one CSV of "rank,domain".
TRANCO_URL = "https://tranco-list.eu/top-1m.csv.zip"

# How many benign domains to pull from the top of the Tranco list.
# We only need enough to balance against the phishing set.
TRANCO_TAKE = 20000

# Where everything lands. Path(__file__).parent.parent means "one folder up
# from the folder this script lives in", i.e. the project root.
OUT_DIR = Path(__file__).parent.parent / "data" / "raw"


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def fetch_phishing_urls() -> list[str]:
    """Download the OpenPhish feed and return it as a list of URL strings."""
    print(f"Downloading phishing feed from {PHISH_FEED_URL} ...")
    response = requests.get(PHISH_FEED_URL, timeout=30)
    response.raise_for_status()  # crash loudly if the server said no

    # .text is the whole file as one big string; splitlines() breaks it apart.
    urls = [line.strip() for line in response.text.splitlines() if line.strip()]
    print(f"  got {len(urls)} phishing URLs")
    return urls


def fetch_benign_urls() -> list[str]:
    """Download the Tranco list and return the top N domains as http:// URLs."""
    print(f"Downloading Tranco list from {TRANCO_URL} (this one is large) ...")
    response = requests.get(TRANCO_URL, timeout=120)
    response.raise_for_status()

    # The response body is a zip file. io.BytesIO wraps the raw bytes so that
    # zipfile can treat them like a file on disk without us saving it first.
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    inner_name = archive.namelist()[0]  # there's exactly one CSV inside

    with archive.open(inner_name) as csv_file:
        df = pd.read_csv(csv_file, names=["rank", "domain"], nrows=TRANCO_TAKE)

    # Tranco gives bare domains ("google.com"). Our phishing URLs are full
    # URLs ("http://evil.example/login"). Both classes must look structurally
    # similar or the model will just learn "has a scheme = benign".
    urls = ["http://" + domain for domain in df["domain"].tolist()]
    print(f"  got {len(urls)} benign URLs")
    return urls


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    phishing = fetch_phishing_urls()
    benign = fetch_benign_urls()

    # Balance the classes for this first pass. With equal numbers of each
    # label, a coin-flip model scores 50%, so any score above that means the
    # model learned *something*. Real-world traffic is nowhere near balanced --
    # we deal with that properly in week 3.
    n = min(len(phishing), len(benign))
    phishing = phishing[:n]
    benign = benign[:n]
    print(f"Balancing both classes to {n} rows each")

    # Build one table. Each row is a URL plus its label.
    df = pd.DataFrame(
        {
            "url": phishing + benign,
            "label": [1] * len(phishing) + [0] * len(benign),
        }
    )

    # Shuffle so the file isn't "all phishing, then all benign".
    # frac=1 means "sample 100% of the rows", which returns them in random
    # order. random_state=42 makes the shuffle reproducible.
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "dataset.csv"
    df.to_csv(out_path, index=False)

    print(f"\nWrote {len(df)} rows to {out_path}")
    print(df["label"].value_counts().rename({0: "benign", 1: "phishing"}))
    print("\nFirst few rows:")
    print(df.head())


if __name__ == "__main__":
    main()
