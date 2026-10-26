"""Search stage: SigLIP 2 puts person crops and the question text in one space, so a new question becomes a
similarity lookup over embeddings made once per video. The shortlist is what the VLM then verifies."""
import threading
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import config

_state = None
_lock = threading.Lock()


def _load():
    global _state
    with _lock:
        if _state is None:
            from transformers import AutoModel, AutoProcessor

            dev = config.DEVICE or "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
            model = AutoModel.from_pretrained(config.EMBED_MODEL).to(dev).eval()
            _state = (model, AutoProcessor.from_pretrained(config.EMBED_MODEL), dev)
    return _state


def _norm(x):
    x = getattr(x, "pooler_output", x)  # transformers 5 returns an output object, 4 a tensor
    return x / x.norm(dim=-1, keepdim=True)


def embed_images(paths, batch=32):
    model, proc, dev = _load()
    out = []
    for s in range(0, len(paths), batch):
        imgs = [Image.open(p).convert("RGB") for p in paths[s:s + batch]]
        inp = proc(images=imgs, return_tensors="pt").to(dev)
        with torch.no_grad():
            out.append(_norm(model.get_image_features(**inp)).float().cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, 1), "float32")


def embed_text(texts):
    model, proc, dev = _load()
    # SigLIP was trained with fixed-length padding
    inp = proc(text=texts, padding="max_length", max_length=64, return_tensors="pt").to(dev)
    with torch.no_grad():
        return _norm(model.get_text_features(**inp)).float().cpu().numpy()


def index_embeddings(out_dir):
    """Embed each track's best crop once; cached in embeddings.npz next to the index. -> (track_ids, matrix)"""
    out_dir = Path(out_dir)
    f = out_dir / "embeddings.npz"
    if f.exists():
        z = np.load(f)
        return z["tracks"], z["emb"]
    files = sorted(p for p in (out_dir / "crops").glob("*.jpg") if "_s" not in p.stem)
    tracks = np.array([int(p.stem) for p in files], int)
    emb = embed_images([str(p) for p in files])
    np.savez_compressed(f, tracks=tracks, emb=emb)
    return tracks, emb


def similarity(out_dir, attribute, subject="person"):
    """-> {track: cosine similarity between the crop and 'a photo of a <subject> <attribute>'}"""
    tracks, emb = index_embeddings(out_dir)
    t = embed_text([f"a photo of a {subject} {attribute}"])[0]
    return dict(zip(tracks.tolist(), (emb @ t).tolist()))


def shortlist(sims, k):
    """Top-k tracks by similarity (k may be a count or a fraction of all tracks)."""
    n = max(int(round(k * len(sims))) if k < 1 else int(k), 1)
    return [t for t, _ in sorted(sims.items(), key=lambda kv: -kv[1])[:n]]
