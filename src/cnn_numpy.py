"""
cnn_numpy.py -- run the trained CNN with NumPy alone. No PyTorch.

WHAT THIS IS

The same forward pass as CharCNN in cnn.py, written out by hand:

    character ids -> embedding lookup
                  -> three 1-D convolutions (widths 3, 5, 7), ReLU
                  -> global max-pool over position
                  -> concatenate -> linear -> sigmoid

Dropout is absent because dropout only exists during training; at inference
it is the identity. That is the only difference from the training-time model,
and it is why the outputs match exactly rather than approximately.

WHY BOTHER

Inference does not need a deep-learning framework. Weights come from
export_numpy_model.py as a ~270 KB .npz; this module loads them and scores
URLs with NumPy. The CLI therefore has no torch dependency, starts instantly,
and runs on a machine where torch is blocked.

THE RISK, AND HOW IT IS CONTAINED

This file duplicates two things from cnn.py: the URL encoding (head/tail
windows, separator token) and the network architecture. Duplication drifts.
If cnn.py changes and this does not, scores silently diverge -- the worst
kind of bug, because nothing crashes.

So `--verify` is not optional politeness. It re-scores the exact test set the
trained model scored and compares against reports/cnn_test.npz, which holds
the probabilities PyTorch produced. Agreement to floating-point tolerance
proves the encoding, the weights and every layer are right. Run it after any
change to either file.

Usage:
    python src/cnn_numpy.py --verify
    python src/cnn_numpy.py "http://suspicious.example/login"
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
MODEL_NPZ = ROOT / "models" / "cnn_numpy.npz"

# Must mirror cnn.py. Changing either file means changing both -- see --verify.
HEAD_LEN = 150
TAIL_LEN = 150
PAD, UNK, SEP = 0, 1, 2


class NumpyCharCNN:
    """The trained character CNN, forward pass only, NumPy only."""

    def __init__(self, path: Path = MODEL_NPZ):
        if not path.exists():
            raise SystemExit(
                f"{path} not found -- run `python src/export_numpy_model.py` first"
            )
        data = np.load(path, allow_pickle=False)

        self.embedding = data["embedding.weight"]
        self.convs = [
            (data[f"convolutions.{i}.weight"], data[f"convolutions.{i}.bias"])
            for i in range(3)
        ]
        self.out_w = data["output.weight"]
        self.out_b = data["output.bias"]

        self.vocab = dict(zip(data["vocab_chars"].tolist(), data["vocab_ids"].tolist()))
        self.threshold = float(data["threshold"])
        self.max_len = int(data["max_len"])

    # -- encoding: must match cnn.py's windows()/encode() exactly -----------

    def encode(self, urls) -> np.ndarray:
        """[head chars] SEP [tail chars] [padding], as integer ids."""
        ids = np.full((len(urls), self.max_len), PAD, dtype=np.int64)

        for row, url in enumerate(urls):
            if len(url) <= HEAD_LEN + TAIL_LEN:
                head, tail = url, ""
            else:
                head, tail = url[:HEAD_LEN], url[-TAIL_LEN:]

            for column, character in enumerate(head):
                ids[row, column] = self.vocab.get(character, UNK)

            ids[row, HEAD_LEN] = SEP

            for offset, character in enumerate(tail):
                ids[row, HEAD_LEN + 1 + offset] = self.vocab.get(character, UNK)

        return ids

    # -- forward pass -------------------------------------------------------

    @staticmethod
    def _conv1d(x, weight, bias):
        """Valid 1-D convolution. x is (batch, in_channels, length).

        Torch's Conv1d computes
            out[b,o,t] = bias[o] + sum_c sum_j w[o,c,j] * x[b,c,t+j]
        which is a correlation, not a flipped convolution -- so no kernel
        reversal here. Summing one kernel offset at a time over contiguous
        slices avoids materialising a sliding-window view.
        """
        out_channels, _, kernel = weight.shape
        length = x.shape[2] - kernel + 1
        out = np.broadcast_to(bias[None, :, None], (x.shape[0], out_channels, length)).copy()
        for j in range(kernel):
            out += np.einsum("oc,bct->bot", weight[:, :, j], x[:, :, j : j + length])
        return out

    def predict_proba(self, urls, batch_size: int = 512) -> np.ndarray:
        """Phishing probability for each URL."""
        scores = []
        for start in range(0, len(urls), batch_size):
            ids = self.encode(list(urls)[start : start + batch_size])

            # (batch, length, embed) -> (batch, embed, length), as torch's
            # transpose(1, 2) does before the convolutions.
            embedded = self.embedding[ids].transpose(0, 2, 1)

            pooled = [
                np.maximum(self._conv1d(embedded, w, b), 0).max(axis=2)
                for w, b in self.convs
            ]
            features = np.concatenate(pooled, axis=1)

            logits = features @ self.out_w.T + self.out_b
            scores.append(1.0 / (1.0 + np.exp(-logits.squeeze(1))))

        return np.concatenate(scores)

    def predict(self, urls):
        probs = self.predict_proba(urls)
        return probs, probs >= self.threshold


def verify() -> int:
    """Re-score the test set and compare against what PyTorch produced."""
    import pandas as pd

    from evaluate import GROUP_BY_DOMAIN, domain_groups, split_three_ways

    saved_path = ROOT / "reports" / "cnn_test.npz"
    if not saved_path.exists():
        raise SystemExit(f"{saved_path} not found -- run cnn.py to produce it")

    features = pd.read_csv(ROOT / "data" / "raw" / "features.csv")
    reference = pd.read_csv(ROOT / "data" / "raw" / "reference.csv")

    groups = domain_groups(len(features)) if GROUP_BY_DOMAIN else None
    splits = split_three_ways(
        features.drop(columns=["label"]), features["label"], groups
    )
    test_urls = reference["url"].iloc[splits[2].index].tolist()

    saved = np.load(saved_path, allow_pickle=False)
    expected = saved["test_probs"]

    if len(expected) != len(test_urls):
        raise SystemExit(
            f"cnn_test.npz holds {len(expected)} predictions but the split gives "
            f"{len(test_urls)} URLs -- rerun cnn.py"
        )

    model = NumpyCharCNN()
    actual = model.predict_proba(test_urls)

    difference = np.abs(actual - expected)
    agree = int((( actual >= model.threshold) == (expected >= model.threshold)).sum())

    print(f"compared {len(expected)} test URLs against PyTorch's own output")
    print(f"  max absolute difference : {difference.max():.3e}")
    print(f"  mean absolute difference: {difference.mean():.3e}")
    print(f"  identical verdicts      : {agree}/{len(expected)}")

    # float32 accumulation order differs between torch and NumPy, so exact
    # equality is not expected; 1e-4 is far tighter than any decision needs.
    if difference.max() < 1e-4 and agree == len(expected):
        print("\nPASS -- the NumPy model reproduces PyTorch. Safe to ship.")
        return 0

    print(
        "\nFAIL -- outputs diverge. Most likely cnn.py's architecture or its URL\n"
        "encoding changed and this file was not updated to match."
    )
    return 1


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__.strip().split("Usage:")[-1].strip())
        raise SystemExit(1)

    if sys.argv[1] == "--verify":
        raise SystemExit(verify())

    urls = sys.argv[1:]
    model = NumpyCharCNN()
    probs, flagged = model.predict(urls)
    for url, prob, is_phishing in zip(urls, probs, flagged):
        verdict = "PHISHING" if is_phishing else "benign  "
        print(f"{verdict}  {prob:6.3f}  {url}")


if __name__ == "__main__":
    main()
