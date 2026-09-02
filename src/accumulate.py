"""
accumulate.py -- daily dataset builder.

Unlike get_data.py (which overwrites everything each run), this script GROWS
your dataset over time:

  1. Downloads today's OpenPhish feed.
  2. Adds any URLs it hasn't seen before to a growing pool, with the date
     they first appeared.
  3. Caches the Tranco benign list so it is only re-downloaded monthly.
  4. Rebuilds data/raw/dataset.csv from the full pool.

The OpenPhish free feed is a rotating snapshot of currently-live phishing,
so running this daily is how a 300-URL sample becomes a few thousand.

SAFETY: downloads lists of URLs as text. Never visits them. Do not add code
that requests these URLs -- that sends traffic from your IP to live attacker
infrastructure.

Run from the project root:
    python src/accumulate.py
"""

import io
import sys
import zipfile
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests


class Tee:
    """Write to the console AND a log file at the same time.

    When this script runs from Task Scheduler there is no console to read,
    so everything it prints must also land in a file. Assigning an instance
    of this to sys.stdout makes every print() go to both places.
    """

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            if not stream.closed:
                stream.write(text)
                stream.flush()  # flush immediately so a crash still leaves a log

    def flush(self):
        # Python flushes sys.stdout again during interpreter shutdown, which
        # can happen after our log file has already been closed. Skip closed
        # streams rather than raising ValueError on the way out.
        for stream in self.streams:
            if not stream.closed:
                stream.flush()

PHISH_FEED_URL = "https://openphish.com/feed.txt"
TRANCO_URL = "https://tranco-list.eu/top-1m.csv.zip"

# Re-download the benign list only if the cache is older than this.
TRANCO_MAX_AGE_DAYS = 30

# How deep into the popularity ranking to draw benign URLs from.
# Deliberately much wider than the top 20k: if every benign example is a
# world-famous site, the model learns "famous = safe" instead of anything
# about phishing. Sampling across 500k ranks includes plenty of legitimate
# but obscure sites, which is a fairer test.
TRANCO_POOL_SIZE = 500_000

RAW = Path(__file__).parent.parent / "data" / "raw"
PHISH_POOL = RAW / "phish_pool.csv"      # grows over time
TRANCO_CACHE = RAW / "tranco_cache.csv"  # refreshed monthly
DATASET = RAW / "dataset.csv"            # rebuilt every run


def update_phish_pool() -> pd.DataFrame:
    """Fetch today's feed and merge any new URLs into the running pool."""
    print(f"[{datetime.now():%Y-%m-%d %H:%M}] Fetching {PHISH_FEED_URL}")
    response = requests.get(PHISH_FEED_URL, timeout=30)
    response.raise_for_status()

    todays_urls = [line.strip() for line in response.text.splitlines() if line.strip()]
    today = pd.DataFrame({"url": todays_urls, "first_seen": date.today().isoformat()})
    print(f"  feed returned {len(today)} URLs")

    if PHISH_POOL.exists():
        pool = pd.read_csv(PHISH_POOL)
        before = len(pool)
        # Stack old and new, then drop duplicate URLs. keep="first" means an
        # existing entry wins, so first_seen keeps the ORIGINAL sighting date.
        pool = pd.concat([pool, today]).drop_duplicates(subset="url", keep="first")
        print(f"  pool grew {before} -> {len(pool)} (+{len(pool) - before} new)")
    else:
        pool = today
        print(f"  created new pool with {len(pool)} URLs")

    pool.to_csv(PHISH_POOL, index=False)
    return pool


def load_benign_domains() -> list[str]:
    """Return benign domains, downloading the Tranco list only if stale."""
    if TRANCO_CACHE.exists():
        age_days = (date.today() - date.fromtimestamp(TRANCO_CACHE.stat().st_mtime)).days
        if age_days < TRANCO_MAX_AGE_DAYS:
            cached = pd.read_csv(TRANCO_CACHE)
            print(f"  using cached benign list ({len(cached)} domains, {age_days}d old)")
            return cached["domain"].tolist()

    print(f"  refreshing benign list from {TRANCO_URL} (large download)")
    response = requests.get(TRANCO_URL, timeout=180)
    response.raise_for_status()

    archive = zipfile.ZipFile(io.BytesIO(response.content))
    with archive.open(archive.namelist()[0]) as csv_file:
        df = pd.read_csv(csv_file, names=["rank", "domain"], nrows=TRANCO_POOL_SIZE)

    df.to_csv(TRANCO_CACHE, index=False)
    print(f"  cached {len(df)} benign domains")
    return df["domain"].tolist()


def rebuild_dataset(pool: pd.DataFrame, benign_domains: list[str]) -> None:
    """Write a fresh balanced dataset.csv from the accumulated pool."""
    phishing = pool["url"].tolist()
    n = min(len(phishing), len(benign_domains))

    # Sample benign domains randomly from the whole ranking rather than
    # taking the top N, so the benign class isn't only famous sites.
    benign = (
        pd.Series(benign_domains)
        .sample(n=n, random_state=42)
        .apply(lambda domain: "http://" + domain)
        .tolist()
    )
    phishing = phishing[:n]

    df = pd.DataFrame(
        {
            "url": phishing + benign,
            "label": [1] * len(phishing) + [0] * len(benign),
        }
    ).sample(frac=1, random_state=42).reset_index(drop=True)

    df.to_csv(DATASET, index=False)
    print(f"  wrote {len(df)} rows to {DATASET.name} ({n} per class)")


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    pool = update_phish_pool()
    benign_domains = load_benign_domains()
    rebuild_dataset(pool, benign_domains)
    print("Done.\n")


if __name__ == "__main__":
    # Everything printed from here on goes to the console and to
    # logs/accumulate.log, so scheduled runs leave a trace.
    LOG_DIR = Path(__file__).parent.parent / "logs"
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    with open(LOG_DIR / "accumulate.log", "a", encoding="utf-8") as log_file:
        original_stdout = sys.stdout
        sys.stdout = Tee(sys.__stdout__, log_file)
        print(f"\n===== run started {datetime.now():%Y-%m-%d %H:%M:%S} =====")
        try:
            main()
        except Exception as error:
            # Without this, a scheduled run that crashes leaves no explanation.
            print(f"FAILED: {type(error).__name__}: {error}")
            raise
        finally:
            # Put stdout back BEFORE the file closes, so nothing later in
            # shutdown tries to write through a Tee holding a dead handle.
            sys.stdout = original_stdout
