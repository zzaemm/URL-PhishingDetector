"""
plot_curves.py -- Week 3: visualise the precision/recall trade-off.

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
no shortage of. Our test set happens to be balanced 50/50, so both look fine
here, but the PR curve is the one that survives contact with deployment.

Reuses evaluate.py's split so the plotted numbers ARE the reported numbers --
same seed, same three-way split, same test set.

Run from the project root:
    python src/plot_curves.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files, never open a window

import matplotlib.pyplot as plt
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

from evaluate import pick_threshold, score_at, split_three_ways

ROOT = Path(__file__).parent.parent
FEATURES_PATH = ROOT / "data" / "raw" / "features.csv"
REPORTS = ROOT / "reports"

STYLE = {
    "Logistic regression": {"color": "#c44e52", "linestyle": "--"},
    "Gradient boosting": {"color": "#4c72b0", "linestyle": "-"},
}


def fit_and_score(name, model, splits):
    """Fit, pick the threshold on validation, return test scores and probabilities."""
    X_train, X_val, X_test, y_train, y_val, y_test = splits

    model.fit(X_train, y_train)

    val_probs = model.predict_proba(X_val)[:, 1]
    threshold = pick_threshold(val_probs, y_val)

    test_probs = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = score_at(test_probs, y_test, threshold)

    return {
        "name": name,
        "probs": test_probs,
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "ap": average_precision_score(y_test, test_probs),
        "auc": roc_auc_score(y_test, test_probs),
    }


def plot_pr(results, y_test, path):
    fig, ax = plt.subplots(figsize=(7, 5.5))

    for r in results:
        precision, recall, _ = precision_recall_curve(y_test, r["probs"])
        ax.plot(
            recall,
            precision,
            label=f"{r['name']}  (AP {r['ap']:.3f})",
            linewidth=2,
            **STYLE[r["name"]],
        )
        # The operating point evaluate.py actually reports.
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
        ax.annotate(
            f"thr {r['threshold']:.2f}",
            (r["recall"], r["precision"]),
            textcoords="offset points",
            xytext=(8, -14),
            fontsize=9,
            color=STYLE[r["name"]]["color"],
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
    ax.set_title("Precision-recall on the held-out test set\ncircles mark the reported operating point")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", frameon=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(results, y_test, path):
    fig, ax = plt.subplots(figsize=(7, 5.5))

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
    ax.set_title("ROC on the held-out test set")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(FEATURES_PATH)
    X = df.drop(columns=["label"])
    y = df["label"]

    splits = split_three_ways(X, y)
    y_test = splits[5]

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

    REPORTS.mkdir(exist_ok=True)
    plot_pr(results, y_test, REPORTS / "pr_curve.png")
    plot_roc(results, y_test, REPORTS / "roc_curve.png")

    print(f"test set: {len(y_test)} URLs, {int(y_test.sum())} phishing")
    for r in results:
        print(
            f"  {r['name']:<22} AP {r['ap']:.3f}  AUC {r['auc']:.3f}  "
            f"thr {r['threshold']:.2f}  precision {r['precision']:.3f}  recall {r['recall']:.3f}"
        )
    print(f"\nwrote {REPORTS / 'pr_curve.png'}")
    print(f"wrote {REPORTS / 'roc_curve.png'}")


if __name__ == "__main__":
    main()
