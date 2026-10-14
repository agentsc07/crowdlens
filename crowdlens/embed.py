"""SigLIP 2: images and text in one space. Zero-shot yes/no = softmax(image.pos, image.neg)."""
import numpy as np

from . import config

_model = None


def _load():
    global _model
    if _model is None:
        import torch
        from transformers import AutoModel, AutoProcessor

        dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        _model = (AutoModel.from_pretrained(config.EMBED_MODEL).to(dev).eval(),
                  AutoProcessor.from_pretrained(config.EMBED_MODEL), dev)
    return _model


def _feat(f):
    # newer transformers return a ModelOutput (pooler_output); older ones a tensor
    return f if hasattr(f, "cpu") else f.pooler_output


def _norm(x):
    return (x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-9)).astype("float32")


def embed_images(imgs, batch=64):
    """imgs: list of PIL RGB images -> (n, d) unit vectors."""
    import torch

    model, proc, dev = _load()
    out = []
    for i in range(0, len(imgs), batch):
        with torch.no_grad():
            f = model.get_image_features(**proc(images=imgs[i:i + batch], return_tensors="pt").to(dev))
        out.append(_feat(f).float().cpu().numpy())
    return _norm(np.concatenate(out)) if out else np.zeros((0, 768), "float32")


def embed_text(texts):
    import torch

    model, proc, dev = _load()
    with torch.no_grad():
        t = proc(text=list(texts), padding="max_length", max_length=64, return_tensors="pt").to(dev)
        f = model.get_text_features(**t)
    return _norm(_feat(f).float().cpu().numpy())


def logit_scale():
    m = _load()[0]
    return float(m.logit_scale.detach().exp()) if hasattr(m, "logit_scale") else 100.0
