"""
ensemble.py -- Week 5: combine the CNN with gradient boosting.

WHY THIS SHOULD WORK, STATED BEFORE RUNNING IT

Error analysis showed the two models fail in opposite directions.

  Gradient boosting reads STRUCTURE. It catches
  lender.sandbox.natwest.poweredbydivido.com at 0.99 -- a bank name buried
  four subdomains deep -- because num_subdomains exists for exactly that.
  It also blocks stackoverflow.com/questions/44481051/relational-database-
  designing at 0.92, because deep path + digits + hyphens looks like an
  attack when all you can do is count.

  The CNN reads VOCABULARY. It clears Stack Overflow at 0.12 because
  "relational-database-designing" is English, and catches
  sites.google.com/view/serviceactivation at 0.52 where gradient boosting
  says 0.01, because it learned its own lexicon from characters instead of
  being limited to a hand-written phish_hints list. But it shrugs at the
  NatWest URL because structure is not what it looks at.

304 test URLs go one way, 168 the other. That is not noise -- the two sets
have mechanically different causes. Averaging should recover most of both.

Ensembling usually gets tried because it tends to help. Here there is a
reason to expect it to, formed before seeing the result, which is the
difference between a finding and a lottery ticket.

HONESTY CONSTRAINT

Every blending strategy below is scored on VALIDATION. One winner is chosen
there, and only that one is run on test. Picking the best blend by its test
score would be the same cheat the three-way split exists to prevent, moved
one level up -- neither model would have seen test, but the ENSEMBLE would
have been fitted to it.

Run from the project root (after cnn.py):
    python src/ensemble.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

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
CNN_PATH = ROOT / "reports" / "cnn_test.npz"


def to_ranks(probabilities) -> np.ndarray:
    """Convert probabilities to ranks scaled to [0, 1].

    Why bother: the two models are calibrated differently. Gradient boosting
    decides at 0.31, the CNN at 0.50 -- the same "suspicion" is a different
    number in each. A plain average is therefore slightly unfair to whichever
    model reports smaller numbers. Ranking throws away the values and keeps
    only the ordering, which is the part both models agree on the meaning of.
    """
    return rankdata(probabilities) / len(probabilities)


def blends(gb, cnn) -> dict:
    """Every combination strategy, as name -> combined score."""
    candidates = {
        "gradient boosting alone": gb,
        "CNN alone": cnn,
        "mean of probabilities": (gb + cnn) / 2,
        "mean of ranks": (to_ranks(gb) + to_ranks(cnn)) / 2,
        # max = "flag if EITHER model is suspicious". Should raise recall and
        # cost precision. Included because for a security tool that trade may
        # be the right one, not because it is expected to win on F-score.
        "max of probabilities": np.maximum(gb, cnn),
    }
    # Weighted averages. The CNN is the stronger model, so the best weight is
    # probably above 0.5 -- but that is a guess, and validation decides.
    for weight in (0.3, 0.4, 0.6, 0.7, 0.8):
        candidates[f"weighted {weight:.1f} CNN"] = weight * cnn + (1 - weight) * gb
    return candidates


def main() -> None:
    features = pd.read_csv(FEATURES_PATH)
    X = features.drop(columns=["label"])
    y = features["label"]

    groups = domain_groups(len(features)) if GROUP_BY_DOMAIN else None
    X_train, X_val, X_test, y_train, y_val, y_test = split_three_ways(X, y, groups)

    print(f"split: {'domain-disjoint' if GROUP_BY_DOMAIN else 'row-wise'}")
    print(f"validation {len(y_val)}   test {len(y_test)}\n")

    # --- gradient boosting: cheap to refit ----------------------------------
    gb_model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.1, max_depth=6, random_state=42
    )
    gb_model.fit(X_train, y_train)
    gb_val = gb_model.predict_proba(X_val)[:, 1]
    gb_test = gb_model.predict_proba(X_test)[:, 1]

    # --- CNN: loaded, not retrained -----------------------------------------
    saved = np.load(CNN_PATH, allow_pickle=False)
    if "val_probs" not in saved:
        raise ValueError(
            f"{CNN_PATH.name} has no val_probs -- it predates the ensemble work. "
            "Rerun `python src/cnn.py` to regenerate it."
        )
    saved_mode = str(saved["split_mode"])
    current_mode = "domain-disjoint" if GROUP_BY_DOMAIN else "row-wise"
    if saved_mode != current_mode:
        raise ValueError(
            f"{CNN_PATH.name} was written under the {saved_mode} split, this run uses "
            f"{current_mode}. Rerun `python src/cnn.py`."
        )
    cnn_val, cnn_test = saved["val_probs"], saved["test_probs"]
    if len(cnn_val) != len(y_val) or len(cnn_test) != len(y_test):
        raise ValueError(
            f"{CNN_PATH.name} holds {len(cnn_val)}/{len(cnn_test)} predictions but the "
            f"split gives {len(y_val)}/{len(y_test)}. Rerun `python src/cnn.py`."
        )

    # --- score every strategy ON VALIDATION ONLY ----------------------------
    print(f"{'strategy':<26}{'val AUC':>9}{'precision':>11}{'recall':>9}{'F-score':>9}")
    print("-" * 64)

    scored = []
    for name, combined in blends(gb_val, cnn_val).items():
        threshold = pick_threshold(combined, y_val, FALSE_ALARM_COST)
        precision, recall, f_score = score_at(
            combined, y_val, threshold, FALSE_ALARM_COST
        )
        auc = roc_auc_score(y_val, combined)
        scored.append((name, threshold, f_score, auc))
        print(f"{name:<26}{auc:>9.3f}{precision:>11.3f}{recall:>9.3f}{f_score:>9.3f}")

    # The F-score already encodes FALSE_ALARM_COST, so selecting on it means
    # selecting under the project's stated security policy rather than a
    # generic notion of "best".
    winner, threshold, _, _ = max(scored, key=lambda row: row[2])
    print(f"\nchosen on validation: {winner}  (threshold {threshold:.2f})")

    # --- the single trip to the test set ------------------------------------
    combined_test = blends(gb_test, cnn_test)[winner]
    precision, recall, _ = score_at(combined_test, y_test, threshold, FALSE_ALARM_COST)
    predictions = (combined_test >= threshold).astype(int)
    auc = roc_auc_score(y_test, combined_test)

    print(f"\n{'=' * 62}\nENSEMBLE ON TEST: {winner}\n{'=' * 62}")
    print(f"  TEST        precision {precision:.3f}   recall {recall:.3f}")
    print(f"  ROC AUC     {auc:.3f}")

    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
    print(f"\n  false alarms (blocked safe sites): {fp}")
    print(f"  missed attacks:                    {fn}\n")
    print(classification_report(y_test, predictions, target_names=["benign", "phishing"]))

    print("Single models on this same test set, for reference:")
    print("  gradient boosting  AUC 0.917   293 false alarms   167 missed")
    print("  character CNN      AUC 0.956   234 false alarms    90 missed")
    print(
        "\nIf the ensemble does not beat the CNN alone, that is a result too:\n"
        "it would mean the 168 URLs gradient boosting rescues are outweighed\n"
        "by the confidence it drags down elsewhere."
    )


if __name__ == "__main__":
    main()
