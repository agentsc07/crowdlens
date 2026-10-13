import json
from pathlib import Path

import faiss
import numpy as np

from . import config

_model = None


def _load():
    global _model
    if _model is None:
        import torch
        from transformers import AutoModel, AutoProcessor

        dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        _model = (AutoModel.from_pretrained(config.EMBED_MODEL).to(dev).eval(),
                  AutoProcessor.from_pretrained(config.EMBED_MODEL), dev)
    return _model


def _norm(x):
    x = x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-9)
    return x.astype("float32")


def embed_images(paths, batch=32):
    import torch
    from PIL import Image

    model, proc, dev = _load()
    out = []
    for i in range(0, len(paths), batch):
        imgs = [Image.open(p).convert("RGB") for p in paths[i:i + batch]]
        with torch.no_grad():
            f = model.get_image_features(**proc(images=imgs, return_tensors="pt").to(dev))
        out.append(f.cpu().numpy())
    return _norm(np.concatenate(out)) if out else np.zeros((0, 1), "float32")


def embed_text(q):
    import torch

    model, proc, dev = _load()
    with torch.no_grad():
        t = proc(text=[q], padding="max_length", max_length=64, return_tensors="pt").to(dev)
        f = model.get_text_features(**t)
    return _norm(f.cpu().numpy())[0]


def save_index(out_dir, vecs, ids):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("index.faiss", "ids.json"):
        (out_dir / name).unlink(missing_ok=True)
    vecs = np.ascontiguousarray(vecs, dtype="float32")
    index = faiss.IndexFlatIP(vecs.shape[1] if vecs.ndim == 2 and vecs.shape[1] else 1)
    if len(ids):
        index.add(vecs)
    faiss.write_index(index, str(out_dir / "index.faiss"))
    (out_dir / "ids.json").write_text(json.dumps(list(map(int, ids))))


def build_index(people, out_dir, root=None):
    """One vector per person: mean of their crop vectors, renormalised."""
    root = Path(root or out_dir)
    vecs, ids = [], []
    for p in people:
        if not p.crops:
            continue
        v = embed_images([str(root / c.path) for c in p.crops]).mean(axis=0, keepdims=True)
        vecs.append(_norm(v)[0])
        ids.append(p.track_id)
    save_index(out_dir, np.stack(vecs) if vecs else np.zeros((0, 1), "float32"), ids)


def shortlist(out_dir, q_vec, k):
    out_dir = Path(out_dir)
    ids = json.loads((out_dir / "ids.json").read_text())
    if not ids:
        return []
    index = faiss.read_index(str(out_dir / "index.faiss"))
    k = min(k, index.ntotal)
    scores, rows = index.search(np.asarray(q_vec, "float32")[None], k)
    return [(ids[r], float(s)) for r, s in zip(rows[0], scores[0]) if r >= 0]
