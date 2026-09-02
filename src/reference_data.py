"""
reference_data.py -- Week 2: a dataset without the collection artifact.

PROBLEM THIS SOLVES
Our week 1 data drew phishing URLs from a live feed (full URLs with paths,
~52 chars) and benign URLs from a domain ranking (bare hostnames, ~20 chars).
A model could score 92% by learning "has a path" -- an artifact of how the
two piles were collected, not a fact about phishing.

THE FIX
Take BOTH classes from a single source that collected them the same way, so
structural symmetry is guaranteed rather than hoped for. This downloads the
`pirocheto/phishing-url` dataset from HuggingFace: 11.4k URLs, both classes,
full URLs with paths on both sides.

NOTE ON THE OTHER COLUMNS
That dataset ships ~80 precomputed features, but many of them (page rank,
domain age, number of hyperlinks, presence of a login form) require actually
visiting the URL. We ignore all of them and keep only `url` and `status`,
then compute our own features from the string. Staying purely lexical is a
deliberate design constraint, not laziness -- see the safety note in README.

Output: data/raw/reference.csv with columns [url, label]

Run from the project root:
    python src/reference_data.py
"""

import io
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://huggingface.co/datasets/pirocheto/phishing-url/resolve/main/data"
SPLITS = ["train.parquet", "test.parquet"]

OUT_PATH = Path(__file__).parent.parent / "data" / "raw" / "reference.csv"


def download_split(filename: str, attempts: int = 4) -> pd.DataFrame:
    """Download one parquet split and return just the url + status columns.

    Retries on network errors. A single HTTP call failing occasionally is
    normal -- DNS blips, dropped connections -- and a script that dies on the
    first one is fragile for no reason. We wait a bit longer after each
    failure (2s, 4s, 8s) rather than hammering the server.
    """
    url = f"{BASE}/{filename}"

    for attempt in range(1, attempts + 1):
        try:
            print(f"Downloading {filename} (attempt {attempt}/{attempts}) ...")
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            break
        except requests.exceptions.RequestException as error:
            print(f"  failed: {type(error).__name__}")
            if attempt == attempts:
                print("  giving up -- check your connection and rerun")
                raise
            wait = 2**attempt
            print(f"  retrying in {wait}s")
            time.sleep(wait)

    # Parquet is a compressed columnar file format -- much smaller and faster
    # than CSV. io.BytesIO lets pandas read the downloaded bytes as if they
    # were a file on disk. Requires pyarrow (see requirements.txt).
    df = pd.read_parquet(io.BytesIO(response.content), columns=["url", "status"])
    print(f"  {len(df)} rows")
    return df


def main() -> None:
    frames = [download_split(name) for name in SPLITS]

    # We merge the provider's train/test split and make our own later. Their
    # split was made for their experiments, not ours, and train_test_split in
    # train.py needs to control the split to keep results comparable.
    df = pd.concat(frames, ignore_index=True)

    # Their labels are the strings "legitimate" and "phishing".
    # Ours are 0 and 1, matching the rest of the project.
    df["label"] = (df["status"] == "phishing").astype(int)
    df = df[["url", "label"]].drop_duplicates(subset="url")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {len(df)} rows to {OUT_PATH.name}")
    print(df["label"].value_counts().rename({0: "benign", 1: "phishing"}))

    # The whole point of this file. If these two numbers are close, the
    # structural giveaway is gone and the model has to work for its score.
    print("\nMean URL length by class (this is the artifact check):")
    print(df.groupby("label")["url"].apply(lambda s: s.str.len().mean()).round(1))


if __name__ == "__main__":
    main()
