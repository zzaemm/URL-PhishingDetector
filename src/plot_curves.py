"""
plot_curves.py -- Week 3/4: visualise the precision/recall trade-off.

WHY A CURVE AND NOT A NUMBER

evaluate.py reports precision and recall at ONE threshold. But the threshold
is a policy dial, not a property of the model: slide it down and you catch
more phishing while blocking more safe sites, slide it up and the reverse.
A single (precision, recall) pair is one point on a curve you never saw.

The precision-recall curve shows every point at once. It answers the question
a security team actually asks -- "if I insist on catching 95% of phishing,
how many safe sites do I break?" -- which no summary statistic can.

We plot PR rather than ROC as the headline because PR ignores true negatives.
On a live feed benign traffic vastly outnumbers phishing, and ROC flatters
models under that imbalance by rewarding correct benign predictions there is
no shortage of.

THREE MODELS, ONE SPLIT

Logistic regression and gradient boosting are retrained here -- they take
seconds. The CNN is not: it is read from reports/cnn_test.npz, written by
cnn.py, because retraining it inside a plotting script would be minutes of
work to reproduce numbers that already exist.

That introduces a way to be silently wrong. The saved probabilities belong to
ONE test set, and the row-wise and domain-disjoint splits produce test sets of
different sizes and different contents. Plotting stale probabilities against a
fresh split would draw a confident, meaningless curve. So the npz records
which split produced it, and this script refuses to use it unless the split
mode and the length both match. If they do not, it plots the two feature
models and tells you to rerun cnn.py rather than quietly drawing nonsense.

Run from the project root (after evaluate.py and cnn.py):
    python src/plot_curves.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files, never open a window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate import (
    FALSE_ALARM_COST,
    GROUP_BY_DOMAIN,
    domain_groups,
    pick_threshold,
    score_at,
    split_three_ways,
)

ROOT = Path(__file__).parent.parent
FEATURES_PATH = ROOT / "data" / "raw" / "features.csv"
REPORTS = ROOT / "reports"
CNN_PATH = REPORTS / "cnn_test.npz"

STYLE = {
    "Logistic regression": {"color": "#c44e52", "linestyle": "--"},
    "Gradient boosting": {"color": "#4c72b0", "linestyle": "-"},
    "Character-level CNN": {"color": "#55a868", "linestyle": "-"},
}


def fit_and_score(name, model, splits):
    """Fit, pick the threshold on validation, return test scores and probabilities."""
    X_train, X_val, X_test, y_train, y_val, y_test = splits

    model.fit(X_train, y_train)

    val_probs = model.predict_proba(X_val)[:, 1]
    threshold = pick_threshold(val_probs, y_val, FALSE_ALARM_COST)

    test_probs = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = score_at(test_probs, y_test, threshold, FALSE_ALARM_COST)

    return {
        "name": name,
        "probs": test_probs,
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "ap": average_precision_score(y_test, test_probs),
        "auc": roc_auc_score(y_test, test_probs),
    }


def load_cnn(y_test):
    """Load the CNN's saved predictions, or return None with a reason why.

    Refuses stale data rather than plotting it. Two checks: the saved split
    mode must match the split this run is using, and the number of saved
    probabilities must match the number of test URLs. Either mismatch means
    the npz was written against a different test set.
    """
    if not CNN_PATH.exists():
        return None, f"{CNN_PATH.name} not found -- run `python src/cnn.py` first"

    saved = np.load(CNN_PATH, allow_pickle=False)
    probs = saved["test_probs"]
    saved_mode = str(saved["split_mode"])
    current_mode = "domain-disjoint" if GROUP_BY_DOMAIN else "row-wise"

    if saved_mode != current_mode:
        return None, (
            f"{CNN_PATH.name} was written under the {saved_mode} split but this run "
            f"uses {current_mode} -- rerun `python src/cnn.py`"
        )

    if len(probs) != len(y_test):
        return None, (
            f"{CNN_PATH.name} holds {len(probs)} predictions but the test set has "
            f"{len(y_test)} URLs -- rerun `python src/cnn.py`"
        )

    threshold = float(saved["threshold"])
    precision, recall, _ = score_at(probs, y_test, threshold, FALSE_ALARM_COST)

    return {
        "name": "Character-level CNN",
        "probs": probs,
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "ap": average_precision_score(y_test, probs),
        "auc": roc_auc_score(y_test, probs),
    }, None


def plot_pr(results, y_test, path, subtitle):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    for r in results:
        precision, recall, _ = precision_recall_curve(y_test, r["probs"])
        ax.plot(
            recall,
            precision,
            label=f"{r['name']}  (AP {r['ap']:.3f})",
            linewidth=2,
            **STYLE[r["name"]],
        )
        # The operating point the project actually reports.
        ax.plot(
            r["recall"],
            r["precision"],
            marker="o",
            markersize=9,
            markerfacecolor="white",
            markeredgewidth=2,
            color=STYLE[r["name"]]["color"],
            zorder=5,
        )

    # A model that flags everything as phishing gets precision = base rate.
    base_rate = float(y_test.mean())
    ax.axhline(
        base_rate,
        color="#888888",
        linestyle=":",
        linewidth=1.5,
        label=f"flag everything (precision {base_rate:.2f})",
    )

    ax.set_xlabel("Recall  (fraction of phishing caught)")
    ax.set_ylabel("Precision  (fraction of flags that are real)")
    ax.set_title(
        "Precision-recall on the held-out test set\n"
        f"{subtitle}; circles mark the reported operating point"
    )
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(results, y_test, path, subtitle):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    for r in results:
        fpr, tpr, _ = roc_curve(y_test, r["probs"])
        ax.plot(
            fpr,
            tpr,
            label=f"{r['name']}  (AUC {r['auc']:.3f})",
            linewidth=2,
            **STYLE[r["name"]],
        )

    ax.plot([0, 1], [0, 1], color="#888888", linestyle=":", linewidth=1.5, label="random")
    ax.set_xlabel("False positive rate  (safe sites blocked)")
    ax.set_ylabel("True positive rate  (phishing caught)")
    ax.set_title(f"ROC on the held-out test set\n{subtitle}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(FEATURES_PATH)
    X = df.drop(columns=["label"])
    y = df["label"]

    # Same split as evaluate.py and cnn.py -- imported, not reimplemented.
    groups = domain_groups(len(df)) if GROUP_BY_DOMAIN else None
    splits = split_three_ways(X, y, groups)
    y_test = splits[5]

    subtitle = "domain-disjoint split" if GROUP_BY_DOMAIN else "row-wise split"
    print(f"split: {subtitle}")
    print(f"test set: {len(y_test)} URLs, {int(y_test.sum())} phishing")

    results = [
        fit_and_score(
            "Logistic regression",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
            ),
            splits,
        ),
        fit_and_score(
            "Gradient boosting",
            HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.1, max_depth=6, random_state=42
            ),
            splits,
        ),
    ]

    cnn, problem = load_cnn(y_test)
    if cnn is None:
        print(f"\nWARNING: plotting without the CNN -- {problem}\n")
    else:
        results.append(cnn)

    REPORTS.mkdir(exist_ok=True)
    plot_pr(results, y_test, REPORTS / "pr_curve.png", subtitle)
    plot_roc(results, y_test, REPORTS / "roc_curve.png", subtitle)

    for r in results:
        print(
            f"  {r['name']:<22} AP {r['ap']:.3f}  AUC {r['auc']:.3f}  "
            f"thr {r['threshold']:.2f}  precision {r['precision']:.3f}  recall {r['recall']:.3f}"
        )
    print(f"\nwrote {REPORTS / 'pr_curve.png'}")
    print(f"wrote {REPORTS / 'roc_curve.png'}")


if __name__ == "__main__":
    main()
