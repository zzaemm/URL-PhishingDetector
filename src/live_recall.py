"""
live_recall.py -- Week 5/7: how much CURRENTLY LIVE phishing does the model catch?

THE QUESTION

Every number in this project so far comes from pirocheto/phishing-url, a 2020
research dataset. That is fine for comparing models against each other, and it
says nothing about whether the thing works on attacks happening this week.

phish_pool.csv is the answer to that: ~3,500 URLs pulled from the OpenPhish
live feed by accumulate.py, one snapshot a day since August 2026. This script
runs the trained models over it and reports the fraction caught.

WHY RECALL ONLY, AND WHY THAT IS A FEATURE

The pool is phishing only. No benign URLs, so no precision, no AUC -- just
"what share of live phishing does this flag?". That sounds like a limitation
and is actually what makes the measurement clean: the week-1 collection
artifact lived in the BENIGN half (Tranco bare hostnames against OpenPhish
full URLs). With no benign half there is no artifact to leak.

WHAT THIS IS NOT

It is NOT a concept-drift study, and must never be written up as one. Drift
would require holding the source constant and varying time. Here BOTH change:
training data is pirocheto (2020), the pool is OpenPhish (2026). A low recall
could mean phishing has moved on, or simply that OpenPhish phishing looks
different from pirocheto phishing. The two explanations cannot be separated
and no amount of analysis here will separate them.

The honest framing is: "recall against contemporary live phishing, from a
different source than training."

THREE CONFOUNDS THIS SCRIPT MEASURES RATHER THAN IGNORES

  1. LENGTH. Pool URLs are visibly shorter than training phishing (median ~41
     vs ~55 characters). Shorter URLs carry less lexical signal, so a recall
     drop may be about length rather than about time or source.

  2. DOMAIN OVERLAP. If a pool URL sits on a domain that was in training, the
     model may recognise the domain rather than generalise -- the same
     leakage the domain-disjoint split was built to remove. Reported
     separately.

  3. TIME WITHIN THE POOL. The collection dates span ~a month from ONE source.
     Comparing the oldest batch against the newest holds source constant and
     varies only time, which is the one genuinely unconfounded drift signal
     available here. A month is far too short to expect real drift, so treat a
     flat line as the expected result and anything else as suspicious.

SECURITY: phish_pool.csv contains live malicious URLs. Nothing here requests
them; any example printed is defanged, and the per-URL dump goes to data/
(gitignored), never reports/ (committed).

Run from the project root:
    python src/live_recall.py
"""

from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate import (
    FALSE_ALARM_COST,
    GROUP_BY_DOMAIN,
    domain_groups,
    pick_threshold,
    registered_domain,
    split_three_ways,
)
from features import extract_features

ROOT = Path(__file__).parent.parent
RAW = ROOT / "data" / "raw"
MODEL_PATH = ROOT / "models" / "cnn.pt"
OUT_PATH = RAW / "live_recall_detail.csv"  # gitignored -- holds live URLs


def defang(url: str) -> str:
    return url.replace("http://", "hxxp://").replace("https://", "hxxps://", 1)


def load_cnn_scorer():
    """Return (score_fn, threshold) for the CNN, or None with a reason.

    Uses the NumPy export rather than torch. Same weights, same outputs --
    verified against PyTorch's own test-set probabilities by
    `python src/cnn_numpy.py --verify` -- but no deep-learning framework at
    inference, so this runs on a machine where torch is blocked.
    """
    try:
        from cnn_numpy import NumpyCharCNN
    except Exception as error:
        return None, f"cnn_numpy unavailable ({type(error).__name__})"

    npz = ROOT / "models" / "cnn_numpy.npz"
    if not npz.exists():
        return None, f"{npz.name} not found -- run `python src/export_numpy_model.py`"

    model = NumpyCharCNN(npz)
    return (model.predict_proba, model.threshold), None


def report(title, frame, models):
    """Print recall per model for one slice of the pool."""
    if frame.empty:
        print(f"  {title:<34} (no URLs)")
        return
    cells = "".join(f"{frame[f'{name}_caught'].mean():>14.1%}" for name in models)
    print(f"  {title:<34}{len(frame):>7}{cells}")


def main() -> None:
    pool = pd.read_csv(RAW / "phish_pool.csv")
    features = pd.read_csv(RAW / "features.csv")
    reference = pd.read_csv(RAW / "reference.csv")

    X = features.drop(columns=["label"])
    y = features["label"]
    groups = domain_groups(len(features)) if GROUP_BY_DOMAIN else None
    X_train, X_val, _, y_train, y_val, _ = split_three_ways(X, y, groups)

    print(f"live pool: {len(pool)} phishing URLs from OpenPhish")
    print(f"collected: {pool['first_seen'].min()} to {pool['first_seen'].max()}")
    print("trained on: pirocheto/phishing-url (2020) -- DIFFERENT SOURCE, see docstring\n")

    # --- feature models -----------------------------------------------------
    pool_features = pd.DataFrame([extract_features(url) for url in pool["url"]])

    models, thresholds = {}, {}
    for name, model in [
        (
            "LR",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
            ),
        ),
        (
            "GB",
            HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.1, max_depth=6, random_state=42
            ),
        ),
    ]:
        model.fit(X_train, y_train)
        # Thresholds come from validation, exactly as everywhere else. Tuning
        # them on the live pool would be choosing the cut-off that flatters
        # this result, which is the whole thing the three-way split prevents.
        thresholds[name] = pick_threshold(
            model.predict_proba(X_val)[:, 1], y_val, FALSE_ALARM_COST
        )
        probs = model.predict_proba(pool_features)[:, 1]
        pool[f"{name}_prob"] = probs
        pool[f"{name}_caught"] = probs >= thresholds[name]
        models[name] = True

    # --- CNN, if torch is available ----------------------------------------
    cnn, problem = load_cnn_scorer()
    if cnn is None:
        print(f"NOTE: CNN skipped -- {problem}\n")
    else:
        score, cnn_threshold = cnn
        thresholds["CNN"] = cnn_threshold
        probs = score(pool["url"])
        pool["CNN_prob"] = probs
        pool["CNN_caught"] = probs >= cnn_threshold
        models["CNN"] = True

    names = list(models)
    print("thresholds (chosen on validation, not tuned here): " +
          "  ".join(f"{n}={thresholds[n]:.2f}" for n in names))

    # --- headline -----------------------------------------------------------
    print(f"\n{'=' * 70}\nRECALL ON LIVE PHISHING\n{'=' * 70}")
    print(f"  {'slice':<34}{'n':>7}" + "".join(f"{n:>14}" for n in names))
    print("  " + "-" * 68)
    report("ALL live phishing", pool, names)

    # --- confound 1: domain overlap with training ---------------------------
    train_domains = set(groups.iloc[X_train.index]) if groups is not None else set()
    pool["domain_seen"] = pool["url"].map(
        lambda u: registered_domain(u) in train_domains
    )
    print()
    report("domain SEEN in training", pool[pool["domain_seen"]], names)
    report("domain unseen (the honest number)", pool[~pool["domain_seen"]], names)

    # --- confound 2: URL length --------------------------------------------
    pool["length"] = pool["url"].str.len()
    ref_median = int(reference[reference["label"] == 1]["url"].str.len().median())
    print()
    print(f"  (training phishing median length: {ref_median} chars; "
          f"pool median: {int(pool['length'].median())})")
    report("short URLs (<= 40 chars)", pool[pool["length"] <= 40], names)
    report("medium (41-80)", pool[pool["length"].between(41, 80)], names)
    report("long (> 80)", pool[pool["length"] > 80], names)

    # --- confound 3: time, source held constant -----------------------------
    print(f"\n{'=' * 70}\nBY COLLECTION DATE (same source, varying time)\n{'=' * 70}")
    print(f"  {'date':<34}{'n':>7}" + "".join(f"{n:>14}" for n in names))
    print("  " + "-" * 68)
    for date, batch in pool.groupby("first_seen"):
        report(str(date), batch, names)
    print(
        "\n  A month is far too short a window for real concept drift. A flat\n"
        "  line here is the expected result; a trend would need explaining."
    )

    # --- what gets missed ---------------------------------------------------
    primary = "CNN" if "CNN" in models else "GB"
    missed = pool[~pool[f"{primary}_caught"]]
    print(f"\n{'=' * 70}\nMISSED BY {primary} -- sample of {min(10, len(missed))} of {len(missed)}\n{'=' * 70}")
    for url in missed["url"].head(10):
        print(f"  {defang(url)[:110]}")

    pool_out = pool.copy()
    pool_out["url"] = pool_out["url"].map(defang)
    pool_out.to_csv(OUT_PATH, index=False)
    print(f"\nwrote {OUT_PATH} ({len(pool_out)} rows, defanged, gitignored)")

    print(
        "\nREMEMBER: this is recall against contemporary live phishing from a\n"
        "DIFFERENT SOURCE than training. It is not a drift measurement, and a\n"
        "low number has at least three available explanations -- time, source,\n"
        "and URL length -- which the tables above separate as far as possible."
    )


if __name__ == "__main__":
    main()
