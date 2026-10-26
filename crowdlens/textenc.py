"""SigLIP 2 base text tower in plain NumPy, for the hosted demo (no torch, ~300 MB of RAM instead of ~2 GB).
Weights come from eval/export_text_encoder.py: int8 token rows (Latin vocab only) and fp16 layers, memory-mapped.
Same maths as Siglip2TextModel: token+position embeddings, 12 pre-LN bidirectional layers, final LN, linear head on
the last of 64 padded positions (that is how the model was trained). Output is L2-normalised like search.embed_text."""
from functools import lru_cache
from pathlib import Path

import numpy as np

ASSETS = Path(__file__).resolve().parent.parent / "demo_assets" / "siglip_text"
SEQ, HEADS, LAYERS = 64, 12, 12


@lru_cache(maxsize=1)
def _load():
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(ASSETS / "tokenizer_slim.json"))
    ld = lambda n: np.load(ASSETS / f"{n}.npy", mmap_mode="r")
    W = dict(tok_int8=ld("tok_int8"), tok_scale=ld("tok_scale"), pos=ld("pos"),
             fln_w=ld("final_layer_norm_weight"), fln_b=ld("final_layer_norm_bias"), head_w=ld("head_weight"), head_b=ld("head_bias"))
    W["layers"] = [{n: ld(f"l{i}_{n}") for n in (
        "layer_norm1_weight", "layer_norm1_bias", "layer_norm2_weight", "layer_norm2_bias",
        "q_proj_weight", "q_proj_bias", "k_proj_weight", "k_proj_bias", "v_proj_weight", "v_proj_bias",
        "out_proj_weight", "out_proj_bias", "fc1_weight", "fc1_bias", "fc2_weight", "fc2_bias")} for i in range(LAYERS)]
    return tok, W


def _ln(x, w, b, eps=1e-6):
    x = x - x.mean(-1, keepdims=True)
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w + b


def _gelu_tanh(x):
    return 0.5 * x * (1 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


def embed_text(texts):
    """-> (n, 768) float32, unit length."""
    tok, W = _load()
    out = []
    for text in texts:
        ids = tok.encode(text.lower()).ids[:SEQ]
        ids = np.array(ids + [0] * (SEQ - len(ids)))
        rows = ids  # the slim tokenizer already uses the compact row numbering
        x = W["tok_int8"][rows].astype(np.float32) * W["tok_scale"][rows].astype(np.float32) + np.asarray(W["pos"])
        for L in W["layers"]:
            f = lambda n: np.asarray(L[n], np.float32)
            h = _ln(x, f("layer_norm1_weight"), f("layer_norm1_bias"))
            q, k, v = (h @ f(n + "_weight").T + f(n + "_bias") for n in ("q_proj", "k_proj", "v_proj"))
            d = x.shape[1] // HEADS
            sp = lambda a: a.reshape(SEQ, HEADS, d).transpose(1, 0, 2)
            att = sp(q) @ sp(k).transpose(0, 2, 1) * d ** -0.5
            att = np.exp(att - att.max(-1, keepdims=True))
            att /= att.sum(-1, keepdims=True)
            x = x + (att @ sp(v)).transpose(1, 0, 2).reshape(SEQ, -1) @ f("out_proj_weight").T + f("out_proj_bias")
            h = _ln(x, f("layer_norm2_weight"), f("layer_norm2_bias"))
            x = x + _gelu_tanh(h @ f("fc1_weight").T + f("fc1_bias")) @ f("fc2_weight").T + f("fc2_bias")
        x = _ln(x, np.asarray(W["fln_w"], np.float32), np.asarray(W["fln_b"], np.float32))
        v = x[-1] @ np.asarray(W["head_w"], np.float32).T + np.asarray(W["head_b"], np.float32)
        out.append(v / np.linalg.norm(v))
    return np.stack(out).astype(np.float32)
