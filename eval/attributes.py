"""Per-verifier attribute accuracy/F1, and --train for the tier-3 heads.
Expects a CSV: image_path,attribute,label (0/1) built from PA-100K (see README).
Usage: python -m eval.attributes labels.csv "SigLIP zero-shot" | --train labels.csv"""
import csv
import sys
from collections import defaultdict

import numpy as np

from crowdlens import pipeline
from crowdlens.verifiers.classifier import MODELS_DIR, _key, train_head
from eval.metrics import f1_acc

if __name__ == "__main__":
    if sys.argv[1] == "--train":
        from crowdlens.embed import embed_images
        rows = defaultdict(list)
        for p, a, y in csv.reader(open(sys.argv[2])):
            rows[a].append((p, int(y)))
        MODELS_DIR.mkdir(exist_ok=True)
        for a, items in rows.items():
            X = embed_images([p for p, _ in items]); y = np.array([l for _, l in items])
            w, b = train_head(X, y)
            np.savez(MODELS_DIR / f"attr_{_key(a)}.npz", w=w, b=b)
    else:
        ver = pipeline.make_verifier(sys.argv[2])
        rows = defaultdict(lambda: ([], []))
        for p, a, y in csv.reader(open(sys.argv[1])):
            rows[a][0].append(ver.p_yes(p, a) >= 0.5); rows[a][1].append(int(y))
        for a, (pred, y) in rows.items():
            acc, f1 = f1_acc(pred, y)
            print(f"{a},{acc:.3f},{f1:.3f}")
