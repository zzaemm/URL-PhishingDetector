"""
export_numpy_model.py -- turn models/cnn.pt into a torch-free models/cnn_numpy.npz.

WHY

Training needs PyTorch. Inference does not. This CNN is an embedding lookup,
three 1-D convolutions, a ReLU, a max-pool and one linear layer -- roughly
forty lines of NumPy (see cnn_numpy.py). Extracting the weights into a plain
.npz means the CLI, the API and any live scoring run with NumPy alone.

Two reasons that matters here:

  1. PRACTICAL. Windows Application Control blocks torch's DLLs on the dev
     machine. Without this, every scoring run needs Colab.

  2. BETTER ENGINEERING ANYWAY. A CLI that pulls in a 300 MB deep-learning
     framework to classify a string is a bad deliverable. The exported model
     is ~270 KB and imports instantly. Even with torch working, this is the
     version worth shipping.

TWO EXTRACTION PATHS

If torch is importable, the weights are read the ordinary way. If it is not,
they are read straight out of the .pt file -- which is a ZIP archive of raw
little-endian float32 storages plus a pickle of the metadata. That fallback
depends on torch's serialization layout, which is an implementation detail
and could change between torch versions.

That fragility is acceptable ONLY because cnn_numpy.py verifies the result
against reports/cnn_test.npz -- the probabilities the real model produced on
the test set. If the extraction were wrong, the check fails loudly. An
unverifiable shortcut would not be worth taking; a verifiable one is.

Run from the project root:
    python src/export_numpy_model.py
    python src/cnn_numpy.py --verify      # always run this afterwards
"""

import io
import pickle
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
MODEL_PT = ROOT / "models" / "cnn.pt"
MODEL_NPZ = ROOT / "models" / "cnn_numpy.npz"

# Parameter tensors in the exact order torch writes their storages, with the
# shapes the architecture fixes. Used only by the no-torch fallback.
#
# MUST match CharCNN in cnn.py. If the architecture changes, this list and
# cnn_numpy.py's forward pass both need updating -- and --verify is what
# catches you forgetting.
TENSOR_LAYOUT = [
    ("embedding.weight", None),          # (vocab_size, EMBED_DIM), size inferred
    ("convolutions.0.weight", (128, 32, 3)),
    ("convolutions.0.bias", (128,)),
    ("convolutions.1.weight", (128, 32, 5)),
    ("convolutions.1.bias", (128,)),
    ("convolutions.2.weight", (128, 32, 7)),
    ("convolutions.2.bias", (128,)),
    ("output.weight", (1, 384)),
    ("output.bias", (1,)),
]


def _metadata_without_torch(archive: zipfile.ZipFile, root: str) -> dict:
    """Read vocab/threshold/max_len from the pickle, stubbing out torch."""

    def _ignore(*args, **kwargs):
        return None

    class TorchFreeUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith("torch"):
                return _ignore
            return super().find_class(module, name)

        def persistent_load(self, pid):
            return None  # storages are read separately, from the zip

    raw = archive.read(f"{root}/data.pkl")
    return TorchFreeUnpickler(io.BytesIO(raw)).load()


def _weights_without_torch(archive: zipfile.ZipFile, root: str, embed_dim=32) -> dict:
    """Read each float32 storage out of the archive in declaration order."""
    if archive.read(f"{root}/byteorder").decode().strip() != "little":
        raise ValueError("model was saved big-endian; this fallback assumes little")

    weights = {}
    for index, (name, shape) in enumerate(TENSOR_LAYOUT):
        raw = archive.read(f"{root}/data/{index}")
        flat = np.frombuffer(raw, dtype="<f4")
        if shape is None:  # embedding: infer vocab_size from the byte count
            shape = (flat.size // embed_dim, embed_dim)
        if flat.size != int(np.prod(shape)):
            raise ValueError(
                f"{name}: storage holds {flat.size} floats, expected {int(np.prod(shape))}. "
                "The architecture in cnn.py and TENSOR_LAYOUT have diverged."
            )
        weights[name] = flat.reshape(shape).copy()
    return weights


def main() -> None:
    if not MODEL_PT.exists():
        raise SystemExit(f"{MODEL_PT} not found -- run cnn.py (locally or on Colab) first")

    try:
        import torch

        bundle = torch.load(MODEL_PT, weights_only=False)
        weights = {k: v.detach().cpu().numpy() for k, v in bundle["state_dict"].items()}
        source = "torch"
    except ImportError:
        archive = zipfile.ZipFile(MODEL_PT)
        root = archive.namelist()[0].split("/")[0]
        bundle = _metadata_without_torch(archive, root)
        weights = _weights_without_torch(archive, root)
        source = "direct zip read (torch unavailable)"

    vocab = bundle["vocab"]
    payload = dict(weights)
    payload["vocab_chars"] = np.array(list(vocab.keys()), dtype="<U1")
    payload["vocab_ids"] = np.array(list(vocab.values()), dtype=np.int64)
    payload["threshold"] = np.array(float(bundle["threshold"]))
    payload["max_len"] = np.array(int(bundle["max_len"]))
    payload["split_mode"] = np.array(str(bundle["split_mode"]))

    MODEL_NPZ.parent.mkdir(exist_ok=True)
    np.savez_compressed(MODEL_NPZ, **payload)

    print(f"read {MODEL_PT.name} via {source}")
    print(f"  vocabulary      {len(vocab)} characters")
    print(f"  embedding       {weights['embedding.weight'].shape}")
    print(f"  threshold       {float(bundle['threshold']):.2f}")
    print(f"  parameters      {sum(w.size for w in weights.values()):,}")
    print(f"\nwrote {MODEL_NPZ} ({MODEL_NPZ.stat().st_size / 1024:.0f} KB)")
    print("\nNow run:  python src/cnn_numpy.py --verify")


if __name__ == "__main__":
    main()
