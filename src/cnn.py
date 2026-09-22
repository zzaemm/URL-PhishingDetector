"""
cnn.py -- Week 4: a character-level convolutional network in PyTorch.

THE QUESTION THIS ASKS

Everything up to now fed the model 25 numbers that a human designed. The
model never saw a URL -- it saw a form describing one. That form is a
ceiling: if phishing has a giveaway nobody thought to measure, gradient
boosting cannot find it, because the information was discarded before
training began.

This model gets the raw string and must work out for itself what matters.
So the week 4 question is not "is deep learning better" but:

    does letting the model invent its own features beat handcrafting them,
    at THIS data scale?

Expect the answer to be no. 11,429 URLs is small for a network with a few
hundred thousand parameters that starts knowing nothing. Gradient boosting
begins with 25 expert hints. Losing here is the honest, expected result and
is worth reporting as a finding -- not something to tune away.

HOW A STRING BECOMES SOMETHING A NETWORK CAN EAT

  1. Each character maps to an integer ID (a -> 7, / -> 40, ...). Just a
     lookup, no meaning attached yet.
  2. Each ID maps to a short vector -- an "embedding". Think of a map where
     every character sits at some coordinate. The coordinates start random
     and are LEARNED during training: if digits behave alike for this task,
     training drifts them together without anyone saying digits are related.
  3. Convolution filters slide along the string like highlighters, each
     firing on the pattern it has learned to care about ("-login", "xn--",
     a run of digits in a host). A filter finds its pattern ANYWHERE in the
     string, which is what fixed features like num_hyphens_host cannot do.
  4. Global max pooling keeps only the strongest firing of each filter --
     "was this pattern present at all?" -- which turns a variable-length
     URL into a fixed-length summary.

FAIRNESS

Imports the split and scoring functions from evaluate.py rather than
reimplementing them. Same seed, same three-way split, same test set, same
FALSE_ALARM_COST policy. The only variable that changes is the model.

SECURITY: reads URL strings offline and never requests them. Note that
MAX_LEN truncates at 200 characters -- see the constant for why that is an
evasion surface worth testing in week 6.

Run from the project root:
    python src/cnn.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from evaluate import (
    FALSE_ALARM_COST,
    GROUP_BY_DOMAIN,
    pick_threshold,
    registered_domain,
    score_at,
    split_three_ways,
)

ROOT = Path(__file__).parent.parent
REFERENCE_PATH = ROOT / "data" / "raw" / "reference.csv"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

# HOW MUCH OF THE URL THE MODEL SEES, AND WHY IT IS BOTH ENDS
#
# Week 6 measured the cost of the old scheme -- a single 200-character window
# from the START of the URL. Padding the front with ~210 characters of
# innocuous path dropped CNN recall from 1.000 to 0.062 on phishing it had
# previously caught. A 94% evasion, caused entirely by this constant.
# Gradient boosting, whose features cover the whole string, stayed at 0.959.
#
# Raising the limit to 512 would not fix it: the attacker pads 520 instead.
# Any single window anchored at one end is defeated by padding the other.
#
# So the model now reads the first HEAD_LEN characters AND the last TAIL_LEN,
# joined by a separator token. The host always sits at the front; the payload
# usually sits at the end. Front-padding no longer blinds the model, because
# whatever it pushes rightwards lands in the tail window.
#
# HONEST LIMIT: this defeats front-padding and back-padding, not middle-
# padding. An attacker who inflates the MIDDLE of a long URL can still push
# content out of both windows. That is a more conspicuous URL and a more
# expensive attack, but it is not impossible -- the fix raises the price, it
# does not close the hole. A truly length-invariant model would be the real
# answer and is out of scope here.
HEAD_LEN = 150
TAIL_LEN = 150
MAX_LEN = HEAD_LEN + TAIL_LEN + 1  # +1 for the separator between the windows

EMBED_DIM = 32          # size of each character's learned coordinate
NUM_FILTERS = 128       # highlighters per width
KERNEL_SIZES = (3, 5, 7)  # how many characters each highlighter looks at
DROPOUT = 0.4

BATCH_SIZE = 64         # average the downhill direction over 64 URLs, not 1
LEARNING_RATE = 1e-3
MAX_EPOCHS = 25
PATIENCE = 4            # stop after this many epochs with no validation gain

SEED = 42

# Reserved IDs. SEP marks the join between the head and tail windows, so the
# model can tell "these two pieces are not adjacent" from a genuinely short
# URL where they are.
PAD, UNK, SEP = 0, 1, 2
RESERVED = 3


def build_vocab(train_urls) -> dict:
    """Map each character seen in TRAINING to an integer ID.

    Built from training data only. Using all the data would leak: the model
    would implicitly know which characters appear in the test set. Anything
    unseen at test time falls back to UNK, which is also what would happen
    to a genuinely novel character in deployment.
    """
    chars = sorted({character for url in train_urls for character in url})
    return {character: index + RESERVED for index, character in enumerate(chars)}


def windows(url: str):
    """Return the (head, tail) slices of a URL the model will actually read.

    Short URLs pass through whole, with no tail. Long ones are cut in the
    middle rather than at the end, so both the hostname and whatever sits at
    the end of the path survive. See the MAX_LEN comment for why both ends.
    """
    if len(url) <= HEAD_LEN + TAIL_LEN:
        return url, ""
    return url[:HEAD_LEN], url[-TAIL_LEN:]


def encode(urls, vocab) -> torch.Tensor:
    """Turn URLs into a (n_urls x MAX_LEN) tensor of character IDs.

    Layout: [head chars] SEP [tail chars] [padding]. A URL short enough to
    fit whole still gets a SEP after it, so the separator means the same
    thing everywhere -- "the readable part ends here" -- rather than only
    appearing on long URLs, which would let the model use its presence as a
    length signal instead of a boundary marker.
    """
    encoded = np.full((len(urls), MAX_LEN), PAD, dtype=np.int64)

    for row, url in enumerate(urls):
        head, tail = windows(url)

        for column, character in enumerate(head):
            encoded[row, column] = vocab.get(character, UNK)

        encoded[row, HEAD_LEN] = SEP

        for offset, character in enumerate(tail):
            encoded[row, HEAD_LEN + 1 + offset] = vocab.get(character, UNK)

    return torch.from_numpy(encoded)


class CharCNN(nn.Module):
    """Embedding -> parallel convolutions -> max pool -> one output."""

    def __init__(self, vocab_size: int):
        super().__init__()
        # padding_idx=PAD keeps the padding character's embedding pinned at
        # zero so the model never learns anything from empty space.
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=PAD)

        # One stack of filters per width. Width 3 catches short motifs, width
        # 7 catches longer ones like "-secure-". Running them in parallel and
        # concatenating means the model does not have to commit to one scale.
        self.convolutions = nn.ModuleList(
            nn.Conv1d(EMBED_DIM, NUM_FILTERS, kernel_size=width) for width in KERNEL_SIZES
        )

        self.dropout = nn.Dropout(DROPOUT)
        self.output = nn.Linear(NUM_FILTERS * len(KERNEL_SIZES), 1)

    def forward(self, x):
        # (batch, length) -> (batch, length, embed) -> (batch, embed, length)
        # Conv1d expects the channel dimension in the middle, hence transpose.
        embedded = self.embedding(x).transpose(1, 2)

        pooled = []
        for convolution in self.convolutions:
            activated = torch.relu(convolution(embedded))
            # Keep only the strongest firing of each filter across the whole
            # URL: "did this pattern appear anywhere?", discarding where.
            pooled.append(activated.max(dim=2).values)

        combined = self.dropout(torch.cat(pooled, dim=1))
        # Returns a raw score (a "logit"), not a probability. The loss
        # function applies the sigmoid itself, which is numerically safer.
        return self.output(combined).squeeze(1)


def predict_probabilities(model, x) -> np.ndarray:
    """Run the model in evaluation mode and return phishing probabilities."""
    model.eval()  # switches dropout off -- it is a training-time trick only
    with torch.no_grad():  # no gradient bookkeeping needed, saves memory
        logits = torch.cat([model(x[i : i + 512]) for i in range(0, len(x), 512)])
    return torch.sigmoid(logits).numpy()


def train(model, x_train, y_train, x_val, y_val) -> None:
    """The training loop: forward, loss, backward, step -- repeated.

    Each epoch walks the training set in batches of 64. For each batch:

        forward   push the batch through and get predictions
        loss      one number saying how wrong they were (your altitude)
        backward  work out, for every parameter, which way is downhill
        step      nudge every parameter a small distance that way

    After each epoch we score the VALIDATION set. Training loss will keep
    falling almost forever -- eventually by memorising training URLs rather
    than learning anything. Validation AUC is what tells us when that starts,
    and we keep the weights from the best epoch rather than the last one.
    """
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.BCEWithLogitsLoss()

    best_auc, best_state, epochs_without_gain = 0.0, None, 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()  # dropout on

        # Reshuffle each epoch so the model never sees the same batch twice.
        order = torch.randperm(len(x_train))
        running_loss = 0.0

        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start : start + BATCH_SIZE]

            optimiser.zero_grad()          # clear last batch's gradients
            logits = model(x_train[batch])  # forward
            loss = loss_function(logits, y_train[batch])  # how wrong
            loss.backward()                 # backward: blame every parameter
            optimiser.step()                # nudge downhill

            running_loss += loss.item() * len(batch)

        val_auc = roc_auc_score(y_val.numpy(), predict_probabilities(model, x_val))
        marker = ""

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_gain = 0
            marker = "  <- best"
        else:
            epochs_without_gain += 1

        print(
            f"  epoch {epoch:2d}  train loss {running_loss / len(order):.4f}"
            f"   val AUC {val_auc:.4f}{marker}"
        )

        if epochs_without_gain >= PATIENCE:
            print(f"  no validation gain for {PATIENCE} epochs -- stopping early")
            break

    model.load_state_dict(best_state)
    print(f"  restored best epoch (val AUC {best_auc:.4f})")


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    df = pd.read_csv(REFERENCE_PATH)

    # Same split function, same seed, same grouping setting as evaluate.py.
    # reference.csv and features.csv are row-aligned, so this is the
    # identical test set the other two models were scored on.
    groups = df["url"].map(registered_domain) if GROUP_BY_DOMAIN else None
    urls_train, urls_val, urls_test, y_train, y_val, y_test = split_three_ways(
        df["url"], df["label"], groups
    )

    vocab = build_vocab(urls_train)
    mode = "domain-disjoint" if GROUP_BY_DOMAIN else "row-wise"
    print(f"SPLIT: {mode}")
    print(
        f"train {len(urls_train)} ({y_train.mean():.1%} phishing)   "
        f"validation {len(urls_val)} ({y_val.mean():.1%})   "
        f"test {len(urls_test)} ({y_test.mean():.1%})"
    )
    print(f"vocabulary: {len(vocab)} characters seen in training")

    x_train = encode(urls_train, vocab)
    x_val = encode(urls_val, vocab)
    x_test = encode(urls_test, vocab)

    y_train_t = torch.tensor(y_train.values, dtype=torch.float32)
    y_val_t = torch.tensor(y_val.values, dtype=torch.float32)

    model = CharCNN(vocab_size=len(vocab) + RESERVED)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"{parameters:,} parameters to learn from {len(urls_train):,} URLs\n")

    train(model, x_train, y_train_t, x_val, y_val_t)

    # Same protocol as evaluate.py: threshold chosen on validation under the
    # project's cost policy, test set touched exactly once.
    val_probs = predict_probabilities(model, x_val)
    threshold = pick_threshold(val_probs, y_val, FALSE_ALARM_COST)

    test_probs = predict_probabilities(model, x_test)
    precision, recall, _ = score_at(test_probs, y_test, threshold, FALSE_ALARM_COST)
    predictions = (test_probs >= threshold).astype(int)
    auc = roc_auc_score(y_test, test_probs)

    print(f"\n{'=' * 62}\nCharacter-level CNN\n{'=' * 62}")
    print(f"Threshold chosen on validation: {threshold:.2f}")
    print(f"  TEST        precision {precision:.3f}   recall {recall:.3f}")
    print(f"  ROC AUC     {auc:.3f}")

    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
    print(f"\n  false alarms (blocked safe sites): {fp}")
    print(f"  missed attacks:                    {fn}\n")
    print(classification_report(y_test, predictions, target_names=["benign", "phishing"]))

    print("For comparison, run `python src/evaluate.py` -- same split, same")
    print("threshold policy, same test set. Under the ROW-WISE split gradient")
    print("boosting scored AUC 0.949 and this CNN 0.979; both are inflated by")
    print("domain overlap, so compare like with like.")

    # Saved so plot_curves.py and week 5 can use this model's predictions
    # without retraining it. The threshold goes in too -- without it the
    # operating point cannot be marked on a curve, and re-deriving it would
    # mean re-running the model on validation.
    #
    # split_mode is recorded because these probabilities are only meaningful
    # against the test set they came from: the row-wise and domain-disjoint
    # splits produce test sets of different sizes AND different contents.
    # Anything loading this file must check before trusting it.
    # val_probs is saved as well as test_probs so that anything combining this
    # model with another -- ensemble.py -- can tune its blend on VALIDATION.
    # Choosing a blend weight by looking at test scores would be the same
    # cheat the three-way split exists to prevent, just one level up: the
    # ensemble would be fitted to the test set even though neither model was.
    REPORTS.mkdir(exist_ok=True)
    np.savez(
        REPORTS / "cnn_test.npz",
        val_probs=val_probs,
        test_probs=test_probs,
        threshold=threshold,
        split_mode="domain-disjoint" if GROUP_BY_DOMAIN else "row-wise",
    )
    print(f"\nwrote {REPORTS / 'cnn_test.npz'}")

    # Save the trained model itself, not just its predictions.
    #
    # adversarial.py needs to score URLs that did not exist at training time
    # (mutated ones), which saved probabilities cannot answer. Week 7's API
    # needs the same thing. The vocabulary goes with it because the weights
    # are meaningless without the exact character->ID mapping they were
    # trained on -- a rebuilt vocab would silently permute every embedding.
    #
    # Gitignored (*.pt): models are regenerated, not versioned.
    MODELS.mkdir(exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocab": vocab,
            "threshold": threshold,
            "max_len": MAX_LEN,
            "split_mode": "domain-disjoint" if GROUP_BY_DOMAIN else "row-wise",
        },
        MODELS / "cnn.pt",
    )
    print(f"wrote {MODELS / 'cnn.pt'}")


if __name__ == "__main__":
    main()
