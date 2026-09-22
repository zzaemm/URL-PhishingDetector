"""
adversarial.py -- Week 6: attack the models on purpose.

THE QUESTION

Every number so far assumes phishing URLs arrive as attackers happen to write
them today. They do not have to. An attacker who knows what a detector looks
at can write URLs that avoid it, cheaply, with no new infrastructure.

So: take the phishing URLs each model currently CATCHES, edit them the way an
adversary would, and measure how much recall survives. A detector that scores
0.956 AUC on unmodified data and collapses under ten minutes of edits is not
a 0.956 detector -- it is a 0.956 detector of attackers who are not trying.

TWO FAMILIES OF ATTACK, AND WHY THE SECOND MATTERS MORE

The original plan was obfuscation: shorten the URL, strip phishing vocabulary,
flatten the path, move off a suspicious TLD. These target the feature list
directly -- features.py is, read backwards, an evasion guide.

But error analysis pointed somewhere better. Findings 9 and 10: the models
cannot separate benign shared hosting from phishing on the same platform
(28 of the 112 universally-missed URLs are *.tumblr.com), and they clear
anything that reads like plain English (stackoverflow.com scored 0.12 by the
CNN). So the cheaper attack is not hiding -- it is LOOKING LEGITIMATE.

That distinction is the point of the week. Obfuscation makes a URL weird, and
weird is what the models were trained to notice. Camouflage makes it ordinary.

RULES OF THE EXPERIMENT

  1. Only mutate phishing URLs the model already catches. Recall on URLs it
     was missing anyway tells you nothing about evasion.
  2. Thresholds stay as chosen on validation. An attacker does not get to
     retune your detector, and moving the threshold to "fix" an attack just
     trades the loss for false alarms.
  3. Every mutation must leave a URL an attacker could actually deploy. An
     attacker controls their own path and can register elsewhere or host on
     a platform. They cannot make paypal.com serve their page -- that is why
     no mutation here simply swaps in a trusted registered domain.

SECURITY: constructs URL strings offline, never requests them. The mutated
strings are not registered domains and are printed defanged.

Run from the project root (after cnn.py):
    python src/adversarial.py
"""

import random
import re
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier

from cnn import RESERVED, CharCNN, encode
from evaluate import (
    FALSE_ALARM_COST,
    GROUP_BY_DOMAIN,
    domain_groups,
    pick_threshold,
    split_three_ways,
)
from features import PHISH_HINT_WORDS, extract_features

ROOT = Path(__file__).parent.parent
RAW = ROOT / "data" / "raw"
MODEL_PATH = ROOT / "models" / "cnn.pt"

SEED = 42

# Platforms that appear in the data as BENIGN shared hosting, and are also
# used by real phishing. That double use is exactly what makes them effective
# camouflage -- the model cannot learn "tumblr = bad" without breaking every
# legitimate blog on it.
PLATFORMS = [
    "sites.google.com/view/{slug}",
    "{slug}.tumblr.com/",
    "{slug}.pages.dev/",
    "{slug}.blogspot.com/",
    "{slug}.weebly.com/",
]

# Ordinary English path segments, the kind a blog or docs site produces.
# Finding 7: the CNN clears "relational-database-designing" because it reads
# as language. This is that observation weaponised.
ENGLISH_WORDS = [
    "how", "to", "make", "the", "best", "guide", "for", "beginners", "review",
    "of", "new", "home", "garden", "recipe", "with", "fresh", "summer",
    "tips", "and", "tricks", "about", "our", "team", "story", "notes",
]


def slug(rng, words=3) -> str:
    return "-".join(rng.choice(ENGLISH_WORDS) for _ in range(words))


# --------------------------------------------------------------------------
# Mutations. Each takes a URL and a random generator, returns a new URL.
# --------------------------------------------------------------------------


def flatten_path(url, rng):
    """evil.com/a/b/c/login.php -> evil.com/login.php

    Targets path_depth, the single most important feature by permutation
    importance (0.0801). Costs the attacker nothing: they choose their paths.
    """
    parsed = urlparse(url)
    tail = parsed.path.rstrip("/").split("/")[-1]
    return f"{parsed.scheme}://{parsed.netloc}/{tail}"


def strip_hints(url, rng):
    """Remove every word in the phish_hints vocabulary."""
    stripped = url
    for word in PHISH_HINT_WORDS:
        stripped = re.sub(word, "", stripped, flags=re.IGNORECASE)
    return re.sub(r"(?<!:)//+", "/", stripped)


def shorten(url, rng):
    """Drop the path and query entirely -- just scheme and host."""
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}/"


def clean_tld(url, rng):
    """Move off a suspicious TLD onto .com."""
    parsed = urlparse(url)
    host = re.sub(r"\.[a-z]{2,6}$", ".com", parsed.netloc)
    return url.replace(parsed.netloc, host, 1)


def english_path(url, rng):
    """Replace the path with a blog-style slug of ordinary English words."""
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}/{slug(rng, 4)}/"


def move_to_platform(url, rng):
    """Rehost on a shared platform the models see as benign.

    The attacker keeps their content and gives up their own domain. Entirely
    realistic -- OpenPhish's live feed is full of *.pages.dev and
    sites.google.com. This is camouflage, not obfuscation.
    """
    return "https://" + rng.choice(PLATFORMS).format(slug=slug(rng, 2))


def camouflage(url, rng):
    """Platform hosting AND an English path. The two cheapest wins combined."""
    return move_to_platform(url, rng).rstrip("/") + f"/{slug(rng, 4)}"


def gibberish(rng, length=12) -> str:
    """A random alphanumeric path segment."""
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    return "".join(rng.choice(alphabet) for _ in range(length))


def pad_with(url, rng, make_segment, target=210):
    """Insert `target` characters of filler between the host and the path.

    Pushes the real content past a fixed-size window anchored at the start
    of the string. Gradient boosting should be barely affected -- its
    features cover the whole URL -- so a CNN-only collapse isolates the
    truncation window as the cause.
    """
    parsed = urlparse(url)
    padding = ""
    while len(padding) < target:
        padding += "/" + make_segment(rng)
    return f"{parsed.scheme}://{parsed.netloc}{padding}{parsed.path}"


def pad_prefix_english(url, rng):
    """Pad with ordinary English words.

    NOT a clean truncation test. English filler is also CAMOUFLAGE -- it
    makes the whole URL read like a blog path, which is independently the
    most effective attack in this table. A low recall here could be caused
    by the window, by the camouflage, or by both.
    """
    return pad_with(url, rng, lambda r: slug(r, 3))


def pad_prefix_gibberish(url, rng):
    """Pad with random alphanumerics.

    Also not clean, but biased the OPPOSITE way. Random high-entropy strings
    are what host_entropy, digit_ratio and special_ratio were built to flag,
    so this filler makes the URL look MORE suspicious, inflating recall for
    reasons unrelated to the window.

    There is no neutral filler -- any text either resembles benign content
    or resembles attacker content, and both are signals. Running both bounds
    the answer instead of pretending one of them is clean:

        english padding   -> pessimistic bound (window + camouflage)
        gibberish padding -> optimistic bound (window - added suspicion)

    If the truncation fix works, the gap between these two should be large
    and gibberish should land near the unmodified 1.000. If BOTH stay low,
    the head+tail window is not doing its job.
    """
    return pad_with(url, rng, gibberish)


MUTATIONS = {
    "obfuscate: flatten path": flatten_path,
    "obfuscate: strip phish words": strip_hints,
    "obfuscate: host only": shorten,
    "obfuscate: clean TLD": clean_tld,
    "camouflage: english path": english_path,
    "camouflage: shared platform": move_to_platform,
    "camouflage: platform + english": camouflage,
    # Two paddings, biased in opposite directions. See their docstrings --
    # the truth about the truncation window lies between them.
    "truncation: pad w/ english": pad_prefix_english,
    "truncation: pad w/ gibberish": pad_prefix_gibberish,
}


# --------------------------------------------------------------------------


def cnn_probabilities(bundle, urls) -> np.ndarray:
    """Score arbitrary URLs with the saved CNN."""
    model = CharCNN(vocab_size=len(bundle["vocab"]) + RESERVED)
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    x = encode(list(urls), bundle["vocab"])
    with torch.no_grad():
        logits = torch.cat([model(x[i : i + 512]) for i in range(0, len(x), 512)])
    return torch.sigmoid(logits).numpy()


def gb_probabilities(model, urls) -> np.ndarray:
    """Re-extract the 25 features for mutated URLs and score them."""
    frame = pd.DataFrame([extract_features(url) for url in urls])
    return model.predict_proba(frame)[:, 1]


def main() -> None:
    rng = random.Random(SEED)

    features = pd.read_csv(RAW / "features.csv")
    reference = pd.read_csv(RAW / "reference.csv")
    X = features.drop(columns=["label"])
    y = features["label"]

    groups = domain_groups(len(features)) if GROUP_BY_DOMAIN else None
    X_train, X_val, X_test, y_train, y_val, y_test = split_three_ways(X, y, groups)

    # --- gradient boosting ---------------------------------------------------
    gb = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.1, max_depth=6, random_state=42
    ).fit(X_train, y_train)
    gb_threshold = pick_threshold(
        gb.predict_proba(X_val)[:, 1], y_val, FALSE_ALARM_COST
    )

    # --- CNN -----------------------------------------------------------------
    if not MODEL_PATH.exists():
        raise SystemExit(
            f"{MODEL_PATH} not found. Rerun `python src/cnn.py` -- it now saves "
            "the trained model, which this script needs to score URLs that did "
            "not exist at training time."
        )
    bundle = torch.load(MODEL_PATH, weights_only=False)
    cnn_threshold = float(bundle["threshold"])

    # --- the attack surface: phishing BOTH models currently catch ------------
    #
    # Restricting to URLs both models catch keeps one population across the
    # whole table, so a recall drop is attributable to the mutation rather
    # than to the two models starting from different sets.
    test_urls = reference["url"].iloc[X_test.index].reset_index(drop=True)
    labels = y_test.reset_index(drop=True)

    phishing = test_urls[labels == 1].reset_index(drop=True)
    caught_by_cnn = cnn_probabilities(bundle, phishing) >= cnn_threshold
    caught_by_gb = gb_probabilities(gb, phishing) >= gb_threshold
    caught = phishing[caught_by_cnn & caught_by_gb].reset_index(drop=True)

    print(f"split: {'domain-disjoint' if GROUP_BY_DOMAIN else 'row-wise'}")
    print(f"phishing URLs in test: {len(phishing)}")
    print(f"caught by BOTH models (the attack surface): {len(caught)}")
    print(f"thresholds: cnn={cnn_threshold:.2f}  gb={gb_threshold:.2f}\n")

    print(f"{'mutation':<34}{'CNN recall':>12}{'GB recall':>12}")
    print("-" * 58)
    print(f"{'(unmodified)':<34}{1.0:>12.3f}{1.0:>12.3f}")

    results = []
    for name, mutate in MUTATIONS.items():
        mutated = [mutate(url, rng) for url in caught]

        cnn_recall = float((cnn_probabilities(bundle, mutated) >= cnn_threshold).mean())
        gb_recall = float((gb_probabilities(gb, mutated) >= gb_threshold).mean())

        results.append((name, cnn_recall, gb_recall, mutated[0]))
        print(f"{name:<34}{cnn_recall:>12.3f}{gb_recall:>12.3f}")

    print("\nRecall here is the fraction of ALREADY-CAUGHT phishing still caught")
    print("after the edit. 1.000 means the attack failed; 0.000 means total evasion.\n")

    print("Example of each mutation (defanged, not registered):")
    for name, _, _, example in results:
        print(f"  {name}")
        print(f"      {example[:110].replace('http://', 'hxxp://').replace('https://', 'hxxps://')}")

    worst_cnn = min(results, key=lambda row: row[1])
    print(
        f"\nCheapest evasion against the shipped model: {worst_cnn[0]} "
        f"-- recall {worst_cnn[1]:.1%}"
    )


if __name__ == "__main__":
    main()
