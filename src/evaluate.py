"""
evaluate.py -- Week 3: honest evaluation and a stronger model.

TWO PROBLEMS THIS FIXES

1. We previously chose a decision threshold by looking at the test set.
   That is cheating: once you make a decision based on the test set, it is
   no longer measuring performance on unseen data. Here we use THREE splits.

       train      (60%)  the model learns from this
       validation (20%)  we make all our decisions against this
       test       (20%)  touched exactly once, at the very end

   Validation is a practice exam you can retake. Test is the real one.

2. Logistic regression gives each feature one weight and adds them up --
   it draws a straight line. It cannot express "a numeric hostname matters
   ONLY WHEN a brand name also appears in the path". Gradient boosting
   builds many small decision trees in sequence, each one focusing on the
   examples earlier trees got wrong, so it can capture those conditional
   rules. On tabular data it usually beats both linear models and neural
   networks.

Run from the project root:
    python src/evaluate.py
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES_PATH = Path(__file__).parent.parent / "data" / "raw" / "features.csv"

# What a false alarm costs relative to a missed attack. At 1.0 both are
# equally bad and we optimise plain F1. Raise it if blocking a safe site is
# worse than letting phishing through; lower it if the reverse. This single
# number encodes the security policy -- make it explicit rather than
# accidental, which is what an unexamined 0.5 threshold does.
#
# WHY 0.5 AND NOT 1.0
#
# A false alarm costs a user five seconds of annoyance. A missed attack costs
# them their credentials. Treating those as equal -- which 1.0 does -- is a
# claim, not a neutral default, and it is the same blind spot as reporting
# accuracy: both count mistakes without asking what each one costs.
#
# 0.5 weights a missed attack twice as heavily as a false alarm. It is not
# pushed further because recall gets expensive fast: see the policy sweep
# printed at the end of this script. Buying recall from 1.0 down to 0.5 costs
# about 1.3 extra false alarms per attack caught; carrying on down to 0.1
# costs over 7, and ends up blocking 57% of safe sites. A detector that noisy
# gets switched off, and a switched-off detector has a real-world recall of
# zero. Alert fatigue is a security failure, not a UX complaint.
FALSE_ALARM_COST = 0.5


def split_three_ways(X, y):
    """Split into train / validation / test as 60 / 20 / 20.

    train_test_split only cuts in two, so we call it twice: first peel off
    the test set, then split what remains into train and validation.
    stratify keeps the phishing/benign ratio identical in every split.
    """
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    # 0.25 of the remaining 80% is 20% of the original.
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def score_at(probabilities, y_true, threshold, false_alarm_cost=FALSE_ALARM_COST):
    """Return (precision, recall, f-score) for one cut-off."""
    flagged = probabilities >= threshold

    true_pos = int((flagged & (y_true == 1)).sum())
    false_pos = int((flagged & (y_true == 0)).sum())
    false_neg = int((~flagged & (y_true == 1)).sum())

    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) else 0.0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) else 0.0

    # Weighted F-score. beta > 1 favours recall, beta < 1 favours precision.
    beta_sq = 1.0 / false_alarm_cost
    denominator = (beta_sq * precision) + recall
    f_score = (
        (1 + beta_sq) * precision * recall / denominator if denominator else 0.0
    )
    return precision, recall, f_score


def pick_threshold(probabilities, y_true, false_alarm_cost=FALSE_ALARM_COST):
    """Choose the cut-off that maximises our F-score ON VALIDATION DATA."""
    best = (0.5, 0.0)
    for step in range(5, 96):
        threshold = step / 100
        _, _, f_score = score_at(probabilities, y_true, threshold, false_alarm_cost)
        if f_score > best[1]:
            best = (threshold, f_score)
    return best[0]


def sweep_costs(val_probs, y_val, test_probs, y_test, costs) -> pd.DataFrame:
    """Show what the FALSE_ALARM_COST setting actually buys you.

    Nothing is retrained here. It is the same model and the same test set,
    read at different dial settings -- because the threshold is a POLICY
    choice, not a model property, and the policy is what this constant sets.

    The point is that "optimal threshold" is meaningless until someone says
    what they are optimising for. Leaving the cost at its 1.0 default is not
    a neutral choice: it is a claim that blocking a safe site and letting a
    credential-theft page through are equally bad. State it and defend it,
    or change it -- but do not inherit it silently.
    """
    rows = []
    for cost in costs:
        # Threshold is still chosen on VALIDATION only. Changing the policy
        # does not license us to peek at test.
        threshold = pick_threshold(val_probs, y_val, cost)
        precision, recall, _ = score_at(test_probs, y_test, threshold, cost)

        flagged = test_probs >= threshold
        false_alarms = int((flagged & (y_test == 0)).sum())
        missed = int((~flagged & (y_test == 1)).sum())

        rows.append(
            {
                "cost": cost,
                "threshold": threshold,
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "false_alarms": false_alarms,
                "missed_attacks": missed,
            }
        )
    return pd.DataFrame(rows)


def evaluate(name, model, splits) -> dict:
    """Fit, tune the threshold on validation, then report once on test."""
    X_train, X_val, X_test, y_train, y_val, y_test = splits

    model.fit(X_train, y_train)

    val_probs = model.predict_proba(X_val)[:, 1]
    threshold = pick_threshold(val_probs, y_val)
    val_precision, val_recall, _ = score_at(val_probs, y_val, threshold)

    # The only time we look at test.
    test_probs = model.predict_proba(X_test)[:, 1]
    test_precision, test_recall, _ = score_at(test_probs, y_test, threshold)
    predictions = (test_probs >= threshold).astype(int)

    # ROC AUC is threshold-independent: it measures how well the model RANKS
    # phishing above benign, regardless of where you cut. 0.5 is random,
    # 1.0 is perfect. Useful for comparing models without the threshold
    # choice muddying things.
    auc = roc_auc_score(y_test, test_probs)

    print(f"\n{'=' * 62}\n{name}\n{'=' * 62}")
    print(f"Threshold chosen on validation: {threshold:.2f}")
    print(f"  validation  precision {val_precision:.3f}   recall {val_recall:.3f}")
    print(f"  TEST        precision {test_precision:.3f}   recall {test_recall:.3f}")
    print(f"  ROC AUC     {auc:.3f}")

    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
    print(f"\n  false alarms (blocked safe sites): {fp}")
    print(f"  missed attacks:                    {fn}")
    print()
    print(classification_report(y_test, predictions, target_names=["benign", "phishing"]))

    return {
        "name": name,
        "auc": auc,
        "precision": test_precision,
        "recall": test_recall,
        "val_probs": val_probs,
        "test_probs": test_probs,
    }


def main() -> None:
    df = pd.read_csv(FEATURES_PATH)
    X = df.drop(columns=["label"])
    y = df["label"]

    splits = split_three_ways(X, y)
    X_train, X_val, X_test = splits[0], splits[1], splits[2]
    print(f"train {len(X_train)}   validation {len(X_val)}   test {len(X_test)}")
    print(f"{X.shape[1]} features")

    results = []

    results.append(
        evaluate(
            "Logistic regression (linear baseline)",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
            ),
            splits,
        )
    )

    gradient_boosting = HistGradientBoostingClassifier(
        max_iter=300,          # how many trees to build in sequence
        learning_rate=0.1,     # how much each tree is allowed to correct
        max_depth=6,           # deeper trees capture more interactions, overfit sooner
        random_state=42,
    )
    results.append(
        evaluate("Gradient boosting (trees, captures interactions)", gradient_boosting, splits)
    )

    print(f"\n{'=' * 62}\nSUMMARY\n{'=' * 62}")
    summary = pd.DataFrame(results).drop(columns=["val_probs", "test_probs"])
    print(summary.round(3).to_string(index=False))

    # What does the security policy setting cost us?
    print(f"\n{'=' * 62}\nTHRESHOLD POLICY SWEEP (gradient boosting)\n{'=' * 62}")
    print("cost < 1 favours catching phishing; cost > 1 favours not blocking safe sites\n")
    print(
        sweep_costs(
            results[1]["val_probs"],
            splits[4],
            results[1]["test_probs"],
            splits[5],
            costs=[0.1, 0.25, 0.5, 1.0, 2.0],
        ).to_string(index=False)
    )

    # Which features does the strong model actually rely on?
    #
    # Trees have no coefficients, so we measure importance by shuffling one
    # feature's values and seeing how much performance drops. If breaking a
    # feature barely hurts, the model wasn't using it. This is slower than
    # reading weights but far more trustworthy -- and unlike linear weights
    # it isn't distorted by features being correlated with each other.
    print("\nPermutation importance (drop in score when a feature is scrambled):")
    importance = permutation_importance(
        gradient_boosting, splits[1], splits[4], n_repeats=5, random_state=42, scoring="roc_auc"
    )
    ranked = (
        pd.Series(importance.importances_mean, index=X.columns)
        .sort_values(ascending=False)
        .head(12)
    )
    print(ranked.round(4).to_string())


if __name__ == "__main__":
    main()
