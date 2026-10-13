"""Tier 3: fine-tuned attribute classifier. A logistic head per attribute over SigLIP 2 image
embeddings, trained on a person-attribute dataset (see eval/attributes.py --train).
Fast and accurate, but only for attributes it was trained on."""
from pathlib import Path

import numpy as np

from .. import embed

MODELS_DIR = Path("models")


def _key(attribute):
    return "_".join(attribute.lower().replace("wearing", "").replace("a ", "").split())


class AttributeClassifier:
    def p_yes(self, crop_path: str, attribute: str) -> float:
        f = MODELS_DIR / f"attr_{_key(attribute)}.npz"
        if not f.exists():
            raise KeyError(f"No trained classifier for '{attribute}' (expected {f}). Use SigLIP or VLM instead.")
        w = np.load(f)
        z = float(embed.embed_images([crop_path])[0] @ w["w"] + w["b"])
        return 1.0 / (1.0 + np.exp(-z))


def train_head(X, y, l2=1.0, iters=300, lr=0.5):
    """Plain logistic regression by gradient descent. X: (n,d) unit vectors, y: (n,) 0/1."""
    w = np.zeros(X.shape[1])
    b = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        g = p - y
        w -= lr * (X.T @ g / len(y) + l2 * w / len(y))
        b -= lr * g.mean()
    return w, b
