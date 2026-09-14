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

from evaluate import FALSE_ALARM_COST, pick_threshold, score_at, split_three_ways

ROOT = Path(__file__).parent.parent
REFERENCE_PATH = ROOT / "data" / "raw" / "reference.csv"
REPORTS = ROOT / "reports"

# Every URL is padded or cut to this many characters so they stack into a
# rectangular tensor. 200 covers about 98% of the dataset in full (95th
# percentile is 131 characters, 99th is 263).
#
# SECURITY NOTE: truncation is an evasion surface. An attacker who knows the
# cut-off can push the incriminating part of a URL past character 200 and the
# model will never see it. Week 6 should test exactly that -- it is a cheaper
# attack than any of the mutations currently on the list.
MAX_LEN = 200

EMBED_DIM = 32          # size of each character's learned coordinate
NUM_FILTERS = 128       # highlighters per width
KERNEL_SIZES = (3, 5, 7)  # how many characters each highlighter looks at
DROPOUT = 0.4

BATCH_SIZE = 64         # average the downhill direction over 64 URLs, not 1
LEARNING_RATE = 1e-3
MAX_EPOCHS = 25
PATIENCE = 4            # stop after this many epochs with no validation gain

SEED = 42

PAD, UNK = 0, 1  # reserved IDs: padding, and "character not seen in training"


def build_vocab(train_urls) -> dict:
    """Map each character seen in TRAINING to an integer ID.

    Built from training data only. Using all the data would leak: the model
    would implicitly know which characters appear in the test set. Anything
    unseen at test time falls back to UNK, which is also what would happen
    to a genuinely novel character in deployment.
    """
    chars = sorted({character for url in train_urls for character in url})
    return {character: index + 2 for index, character in enumerate(chars)}


def encode(urls, vocab) -> torch.Tensor:
    """Turn URLs into a (n_urls x MAX_LEN) tensor of character IDs."""
    encoded = np.full((len(urls), MAX_LEN), PAD, dtype=np.int64)
    for row, url in enumerate(urls):
        for column, character in enumerate(url[:MAX_LEN]):
            encoded[row, column] = vocab.get(character, UNK)
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

    # Same split function, same seed as evaluate.py. reference.csv and
    # features.csv are row-aligned, so this is the identical test set the
    # other two models were scored on.
    urls_train, urls_val, urls_test, y_train, y_val, y_test = split_three_ways(
        df["url"], df["label"]
    )

    vocab = build_vocab(urls_train)
    print(f"train {len(urls_train)}   validation {len(urls_val)}   test {len(urls_test)}")
    print(f"vocabulary: {len(vocab)} characters seen in training")

    x_train = encode(urls_train, vocab)
    x_val = encode(urls_val, vocab)
    x_test = encode(urls_test, vocab)

    y_train_t = torch.tensor(y_train.values, dtype=torch.float32)
    y_val_t = torch.tensor(y_val.values, dtype=torch.float32)

    model = CharCNN(vocab_size=len(vocab) + 2)
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

    print("For comparison, gradient boosting on the same test set:")
    print("  AUC 0.949   precision 0.844   recall 0.903   191 false alarms   111 missed")

    # Saved so week 5 can compare all three models without retraining.
    REPORTS.mkdir(exist_ok=True)
    np.save(REPORTS / "cnn_test_probs.npy", test_probs)
    print(f"\nwrote {REPORTS / 'cnn_test_probs.npy'}")


if __name__ == "__main__":
    main()
