"""
train.py -- Step 4 of the phishing URL detector: the baseline model.

This is deliberately the simplest model that could possibly work. Its job is
not to be good. Its job is to give every later model a number to beat, and to
prove the pipeline runs end to end.

Run from the project root:
    python src/train.py
"""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES_PATH = Path(__file__).parent.parent / "data" / "raw" / "features.csv"


def main() -> None:
    df = pd.read_csv(FEATURES_PATH)

    # X = the inputs (everything except the answer).
    # y = the answer we want to predict.
    # Capital X is a maths convention: X is a table, y is a single column.
    X = df.drop(columns=["label"])
    y = df["label"]

    # Split into a training set and a held-out test set.
    #
    # This is the single most important line in the file. The model is only
    # allowed to see the training half. We judge it on the test half, which
    # it has never seen -- otherwise we would just be measuring memorisation.
    #
    #   test_size=0.2  -> keep 20% back for testing
    #   stratify=y     -> keep the phishing/benign ratio identical in both
    #                     halves, so the test set can't end up lopsided
    #   random_state=42-> makes the shuffle reproducible, so your numbers
    #                     don't change every time you run the script
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Training on {len(X_train)} URLs, testing on {len(X_test)}\n")

    # A Pipeline chains steps that run in order.
    #
    # StandardScaler rescales every feature to a comparable range. Without it,
    # url_length (values around 50) would dominate has_at (values 0 or 1)
    # simply because its numbers are bigger. Logistic regression is sensitive
    # to that; tree-based models are not.
    #
    # class_weight="balanced" tells the model to treat both classes as equally
    # important. It does nothing right now because our data is balanced, but
    # it matters the moment the data isn't.
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", random_state=42),
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # ---------------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------------
    print("Confusion matrix:")
    print("                  predicted benign   predicted phishing")
    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
    print(f"  actual benign     {tn:>10}         {fp:>10}   <- fp = blocked a safe site")
    print(f"  actual phishing   {fn:>10}         {tp:>10}   <- fn = missed an attack\n")

    # precision = of everything flagged as phishing, how much really was.
    #             low precision -> you annoy users by blocking safe sites
    # recall    = of all real phishing, how much did we catch.
    #             low recall -> attacks get through
    print(classification_report(y_test, predictions, target_names=["benign", "phishing"]))

    # ---------------------------------------------------------------------
    # What did it actually learn?
    # ---------------------------------------------------------------------
    # Logistic regression assigns a weight to each feature. Positive weight
    # pushes a URL towards "phishing", negative towards "benign". The size
    # tells you how much that feature mattered.
    weights = pd.Series(
        model.named_steps["logisticregression"].coef_[0], index=X.columns
    ).sort_values(ascending=False)

    print("Feature weights (positive = pushes towards phishing):")
    print(weights.to_string())


if __name__ == "__main__":
    main()
