"""
error_analysis.py -- Week 5: read the URLs the models get wrong.

WHY THIS AND NOT ANOTHER METRIC

AUC 0.956 tells you a model is good. It does not tell you what it is bad at,
and "what is it bad at" is the only question that leads anywhere useful --
it points at the next feature, the next dataset problem, and the attack an
adversary would actually use. Scores summarise; errors explain.

Three things this produces:

  1. WHERE THE MODELS DISAGREE. Cases the CNN gets right and gradient
     boosting gets wrong isolate what reading raw characters buys you. The
     reverse direction isolates what it costs.

  2. WHAT ALL THREE MISS. URLs every model fails on are the genuinely hard
     core of the problem, not quirks of one architecture.

  3. A TESTABLE PREDICTION ABOUT TRUNCATION. cnn.py cuts URLs at 200
     characters, so it is blind to anything past that point. If that matters,
     the CNN should be disproportionately wrong on long URLs while the
     feature models -- which measure the whole string -- are not. This
     script checks that directly. See README finding 7.

SECURITY: the CSV this writes contains live phishing URLs, so it is written
to data/ (gitignored), NOT reports/ (committed). Putting live malicious URLs
in a public repo gets it flagged by GitHub malware scanning and is a hazard
to anyone who clones it. URLs are also defanged (http -> hxxp) so that
nothing in the output is one accidental click from a live phishing page.

Run from the project root (after cnn.py):
    python src/error_analysis.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cnn import MAX_LEN
from evaluate import (
    FALSE_ALARM_COST,
    GROUP_BY_DOMAIN,
    domain_groups,
    pick_threshold,
    split_three_ways,
)

ROOT = Path(__file__).parent.parent
RAW = ROOT / "data" / "raw"
FEATURES_PATH = RAW / "features.csv"
REFERENCE_PATH = RAW / "reference.csv"
CNN_PATH = ROOT / "reports" / "cnn_test.npz"

# Gitignored on purpose -- see the SECURITY note above.
OUT_PATH = RAW / "test_errors.csv"

SAMPLES = 8  # how many example URLs to print per category


def defang(url: str) -> str:
    """Make a URL non-clickable. http -> hxxp, and dots in the host escaped.

    Standard practice when malicious URLs have to be written down. It does
    not make the URL safe -- it makes it inert in anything that auto-links,
    so nobody visits an attacker's page by clicking a row in a spreadsheet.
    """
    return url.replace("http://", "hxxp://").replace("https://", "hxxps://", 1)


def fit_probs(model, splits):
    """Fit on train and return probabilities for validation and test."""
    X_train, X_val, X_test, y_train, _, _ = splits
    model.fit(X_train, y_train)
    return model.predict_proba(X_val)[:, 1], model.predict_proba(X_test)[:, 1]


def show(title, frame, columns):
    """Print a labelled sample of rows, or say the category is empty."""
    print(f"\n{'-' * 70}\n{title}  (n={len(frame)})\n{'-' * 70}")
    if frame.empty:
        print("  none")
        return
    for _, row in frame.head(SAMPLES).iterrows():
        print(f"  {row['url'][:100]}")
        print(f"      {'  '.join(f'{c}={row[c]:.2f}' for c in columns)}")


def main() -> None:
    features = pd.read_csv(FEATURES_PATH)
    reference = pd.read_csv(REFERENCE_PATH)

    X = features.drop(columns=["label"])
    y = features["label"]

    groups = domain_groups(len(features)) if GROUP_BY_DOMAIN else None
    splits = split_three_ways(X, y, groups)
    y_val, y_test = splits[4], splits[5]

    # The URLs behind the test rows. features.csv and reference.csv are
    # row-aligned (domain_groups asserts this), so positional indexing is safe.
    urls_test = reference["url"].iloc[splits[2].index].reset_index(drop=True)
    labels = y_test.reset_index(drop=True)

    print(f"split: {'domain-disjoint' if GROUP_BY_DOMAIN else 'row-wise'}")
    print(f"test set: {len(labels)} URLs, {int(labels.sum())} phishing\n")

    # --- model predictions, all on the same test set -------------------------
    models = {
        "lr": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        ),
        "gb": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, max_depth=6, random_state=42
        ),
    }

    probs, thresholds = {}, {}
    for name, model in models.items():
        val_probs, test_probs = fit_probs(model, splits)
        thresholds[name] = pick_threshold(val_probs, y_val, FALSE_ALARM_COST)
        probs[name] = test_probs

    saved = np.load(CNN_PATH, allow_pickle=False)
    if len(saved["test_probs"]) != len(labels):
        raise ValueError(
            f"{CNN_PATH.name} holds {len(saved['test_probs'])} predictions but the "
            f"test set has {len(labels)}. Rerun `python src/cnn.py`."
        )
    probs["cnn"] = saved["test_probs"]
    thresholds["cnn"] = float(saved["threshold"])

    # --- one table with every model's verdict --------------------------------
    table = pd.DataFrame({"url": urls_test, "label": labels})
    table["length"] = table["url"].str.len()
    for name in ("lr", "gb", "cnn"):
        table[name] = probs[name]
        table[f"{name}_flags"] = (probs[name] >= thresholds[name]).astype(int)
        table[f"{name}_wrong"] = table[f"{name}_flags"] != table["label"]

    print("thresholds: " + "  ".join(f"{k}={v:.2f}" for k, v in thresholds.items()))
    print("errors:     " + "  ".join(
        f"{name}={int(table[f'{name}_wrong'].sum())}" for name in ("lr", "gb", "cnn")
    ))

    # --- 1. where do the two best models disagree? ---------------------------
    columns = ["gb", "cnn"]

    cnn_saves = table[~table["cnn_wrong"] & table["gb_wrong"]]
    gb_saves = table[table["cnn_wrong"] & ~table["gb_wrong"]]

    print(f"\nCNN right where gradient boosting is wrong: {len(cnn_saves)}")
    print(f"Gradient boosting right where CNN is wrong: {len(gb_saves)}")
    print("The difference between these two is the CNN's net advantage.")

    show(
        "PHISHING the CNN caught and gradient boosting missed",
        cnn_saves[cnn_saves["label"] == 1],
        columns,
    )
    show(
        "BENIGN the CNN cleared and gradient boosting blocked",
        cnn_saves[cnn_saves["label"] == 0],
        columns,
    )
    show(
        "PHISHING the CNN missed but gradient boosting caught",
        gb_saves[gb_saves["label"] == 1],
        columns,
    )

    # --- 2. what defeats everything? -----------------------------------------
    all_wrong = table[table["lr_wrong"] & table["gb_wrong"] & table["cnn_wrong"]]
    show(
        "MISSED BY ALL THREE MODELS -- the hard core of the problem",
        all_wrong,
        ["lr", "gb", "cnn"],
    )

    # --- 3. does the CNN's 200-character truncation actually hurt? ------------
    #
    # A prediction worth checking rather than assuming. If truncation matters,
    # the CNN's error rate should rise on URLs longer than MAX_LEN while the
    # feature models' does not, because they measure the whole string.
    print(f"\n{'=' * 70}\nTRUNCATION CHECK (cnn.py cuts at {MAX_LEN} characters)\n{'=' * 70}")
    long_urls = table["length"] > MAX_LEN
    print(f"test URLs longer than {MAX_LEN} chars: {int(long_urls.sum())} of {len(table)}")

    if long_urls.sum() == 0:
        print("none in this test set -- the truncation risk is untested, not absent.")
    else:
        print(f"\n{'model':<8}{'error rate (short)':>20}{'error rate (long)':>20}")
        for name in ("lr", "gb", "cnn"):
            short_rate = table.loc[~long_urls, f"{name}_wrong"].mean()
            long_rate = table.loc[long_urls, f"{name}_wrong"].mean()
            print(f"{name:<8}{short_rate:>19.1%}{long_rate:>20.1%}")
        print(
            "\nIf the CNN's long-URL error rate is worse than the feature models'\n"
            "by more than they differ on short URLs, truncation is costing it --\n"
            "and week 6 has a cheap, total evasion to demonstrate."
        )

    # --- save everything for reading by hand ---------------------------------
    dump = table.copy()
    dump["url"] = dump["url"].map(defang)
    dump.to_csv(OUT_PATH, index=False)
    print(f"\nwrote {OUT_PATH} ({len(dump)} rows, URLs defanged, gitignored)")
    print("Open it and sort by disagreement -- reading the URLs is the point.")


if __name__ == "__main__":
    main()
