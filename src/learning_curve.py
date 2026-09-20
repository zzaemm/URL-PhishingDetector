"""
learning_curve.py -- Week 5: does more data help, and which model does it help?

THE CLAIM THIS TESTS

The CNN beats gradient boosting at this data scale (0.956 vs 0.917 AUC). The
natural next question is whether that gap is a property of the models or a
property of having 11,429 URLs. Two opposite stories fit the same result:

  Gradient boosting is capped by its FEATURES, not by its examples. It only
  ever sees 25 numbers, so past some point more URLs teach it nothing new --
  it runs out of information, not data. Its curve should flatten early.

  The CNN is capped by EXAMPLES. It reads the whole string, so there is no
  ceiling imposed by someone's feature list, but it must discover everything
  itself. Its curve should still be climbing at 100%.

If that is right, the gap widens with more data and the honest claim becomes
"more data would help the CNN specifically." If both curves are flat at 100%,
the opposite conclusion holds: URL text contains little more signal than the
25 features already capture, and handcrafted features win permanently at
every scale. Either answer is worth having; guessing is not.

WHY SUBSAMPLE BY DOMAIN, NOT BY ROW

There are two ways to build a smaller training set: keep 10% of the URLs, or
keep 10% of the WEBSITES and all their URLs. They are not equivalent, and
choosing the first would invert the conclusion.

The test set is domain-disjoint, so what is being measured is generalisation
to websites never seen before. A 10% ROW sample still contains a handful of
URLs from nearly every domain in the training set -- so the model retains
broad exposure while appearing small. Broad shallow exposure is exactly what
transfers to unseen domains, so that point on the curve scores far higher
than a genuinely 10%-sized dataset ever could. Every small point is lifted,
the curve looks flat, and the conclusion drawn is "more data would not help"
when the truth may be the opposite.

The lumpiness makes it worse: one host carries 237 URLs, so a 10% row sample
still keeps ~24 near-identical URLs from that single phishing kit.

Domain-wise sampling also makes the x-axis mean something actionable. This
dataset grows by acquiring new SITES -- that is what accumulate.py does. So
"6,000 URLs across 4,000 domains" corresponds to a collection effort that
could actually be undertaken. A row-wise x-axis corresponds to nothing.

WHY VALIDATION AND NOT TEST

This is a diagnostic, run many times over many subsets. Scoring it on test
would touch the test set dozens of times and quietly turn it into a second
validation set. The test set stays reserved for the numbers in the README.

NOTE: covers the two feature models only. The CNN needs torch, which
Windows Application Control is currently blocking on this machine; the CNN's
curve is the more interesting half and is outstanding.

Run from the project root:
    python src/learning_curve.py
"""

import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate import GROUP_BY_DOMAIN, domain_groups, split_three_ways

ROOT = Path(__file__).parent.parent
FEATURES_PATH = ROOT / "data" / "raw" / "features.csv"
REPORTS = ROOT / "reports"

FRACTIONS = (0.1, 0.25, 0.5, 0.75, 1.0)

# Small subsets are noisy -- one unlucky draw can move AUC by several points.
# Repeating each fraction with different draws and plotting the spread is the
# difference between a curve and a rumour.
REPEATS = 5
SEED = 42

STYLE = {
    "Logistic regression": {"color": "#c44e52", "linestyle": "--"},
    "Gradient boosting": {"color": "#4c72b0", "linestyle": "-"},
}


def fresh_models() -> dict:
    """A new unfitted model per run -- refitting a fitted model hides leakage."""
    return {
        "Logistic regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        ),
        "Gradient boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, max_depth=6, random_state=42
        ),
    }


def sample_domains(train_groups, fraction, rng):
    """Return a row mask keeping `fraction` of the training DOMAINS."""
    domains = list(pd.unique(train_groups))
    rng.shuffle(domains)
    keep = set(domains[: max(1, int(len(domains) * fraction))])
    return train_groups.isin(keep).to_numpy()


def main() -> None:
    features = pd.read_csv(FEATURES_PATH)
    X = features.drop(columns=["label"])
    y = features["label"]

    groups = domain_groups(len(features)) if GROUP_BY_DOMAIN else None
    X_train, X_val, _, y_train, y_val, _ = split_three_ways(X, y, groups)

    if groups is None:
        raise SystemExit(
            "learning_curve.py assumes GROUP_BY_DOMAIN = True -- subsampling by "
            "domain is the whole point. Set it in evaluate.py."
        )

    train_groups = groups.iloc[X_train.index].reset_index(drop=True)
    print(f"train {len(X_train)} URLs across {train_groups.nunique()} domains")
    print(f"scoring on validation ({len(y_val)} URLs); test untouched\n")

    # results[model][fraction] = list of AUCs, one per repeat
    results = {
        name: {fraction: [] for fraction in FRACTIONS} for name in fresh_models()
    }

    print(f"{'domains':>9}{'URLs':>8}   " + "".join(f"{n:>22}" for n in fresh_models()))
    print("-" * 62)

    for fraction in FRACTIONS:
        # At 100% every draw is identical, so one repeat is enough.
        repeats = 1 if fraction == 1.0 else REPEATS
        rng = random.Random(SEED)

        sizes = []
        for _ in range(repeats):
            mask = sample_domains(train_groups, fraction, rng)
            X_subset, y_subset = X_train[mask], y_train[mask]
            sizes.append((len(pd.unique(train_groups[mask])), mask.sum()))

            # A subset with one class present cannot be fitted or scored.
            if y_subset.nunique() < 2:
                continue

            for name, model in fresh_models().items():
                model.fit(X_subset, y_subset)
                auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
                results[name][fraction].append(auc)

        domains, urls = sizes[0]
        row = f"{domains:>9}{urls:>8}   "
        for name in fresh_models():
            scores = results[name][fraction]
            mean = float(np.mean(scores))
            spread = float(np.std(scores))
            row += f"{mean:>15.3f} ±{spread:.3f}"
        print(row)

    # --- plot ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    x_axis = [int(len(X_train) * f) for f in FRACTIONS]

    for name in fresh_models():
        means = np.array([np.mean(results[name][f]) for f in FRACTIONS])
        spreads = np.array([np.std(results[name][f]) for f in FRACTIONS])
        ax.plot(x_axis, means, marker="o", linewidth=2, label=name, **STYLE[name])
        ax.fill_between(
            x_axis, means - spreads, means + spreads,
            color=STYLE[name]["color"], alpha=0.15,
        )

    ax.set_xlabel("Training URLs (sampled by domain)")
    ax.set_ylabel("Validation ROC AUC")
    ax.set_title(
        "Learning curves under the domain-disjoint split\n"
        "a flattening curve means the model is out of INFORMATION, not data"
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()

    REPORTS.mkdir(exist_ok=True)
    fig.savefig(REPORTS / "learning_curve.png", dpi=150)
    plt.close(fig)
    print(f"\nwrote {REPORTS / 'learning_curve.png'}")

    print(
        "\nRead the right-hand end. A curve still rising at 100% means more data\n"
        "would still buy accuracy. A flat one means the model has extracted what\n"
        "its 25 features can express, and only better FEATURES -- or a model that\n"
        "reads the raw string -- can go further."
    )


if __name__ == "__main__":
    main()
