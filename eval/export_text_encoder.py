"""One-off: slim the SigLIP2-base TEXT tower for the hosted demo (no torch at runtime).
Keeps only Latin-lowercase vocab rows (103k of 256k), stores them as int8 + per-row fp16 scale, the rest fp16.
Output: demo_assets/siglip_text/ (~250 MB). Verified against the torch text tower by tests/test_textenc.py."""
import glob, json, os, re, shutil
from pathlib import Path

import numpy as np
from safetensors import safe_open

SNAP = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--google--siglip2-base-patch16-224/snapshots/*"))[0]
OUT = Path(__file__).resolve().parent.parent / "demo_assets" / "siglip_text"
OUT.mkdir(parents=True, exist_ok=True)
tok = json.load(open(f"{SNAP}/tokenizer.json"))
vocab = tok["model"]["vocab"]
rx = re.compile(r"^[▁a-z0-9 .,'\-!?&/:;()\"%$+#@=*_]+$")
keep = sorted({i for t, i in vocab.items() if rx.match(t)} | {0, 1, 2, 3})  # pad eos bos unk always
# slim tokenizer: only kept tokens (renumbered 0..n-1 in the same order as the embedding rows) and only the merges
# whose parts and result are all kept. Pure-ASCII lowercase text tokenizes identically; anything else becomes <unk>.
pos_of = {i: n for n, i in enumerate(keep)}
new = {t: pos_of[i] for t, i in vocab.items() if i in pos_of}
m = tok["model"]
merges = [mg for mg in m["merges"] if (lambda a, b: a in new and b in new and a + b in new)(*(mg if isinstance(mg, list) else mg.split(" ")))]
slim = dict(tok, model=dict(m, vocab=new, merges=merges, unk_token="<unk>", byte_fallback=False))
json.dump(slim, open(OUT / "tokenizer_slim.json", "w"), ensure_ascii=False)
f = safe_open(f"{SNAP}/model.safetensors", "pt")
g = lambda k: f.get_tensor(k).float().numpy()
E = g("text_model.embeddings.token_embedding.weight")[keep]
scale = np.abs(E).max(1, keepdims=True) / 127.0 + 1e-12
np.save(OUT / "tok_int8.npy", np.round(E / scale).astype(np.int8))
np.save(OUT / "tok_scale.npy", scale.astype(np.float16))
np.save(OUT / "pos.npy", g("text_model.embeddings.position_embedding.weight").astype(np.float32))
for i in range(12):
    p = f"text_model.encoder.layers.{i}."
    for name in ("layer_norm1", "layer_norm2"):
        for s in ("weight", "bias"):
            np.save(OUT / f"l{i}_{name}_{s}.npy", g(p + f"{name}.{s}").astype(np.float32))
    for name in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj", "mlp.fc1", "mlp.fc2"):
        for s in ("weight", "bias"):
            np.save(OUT / f"l{i}_{name.split('.')[-1]}_{s}.npy", g(p + f"{name}.{s}").astype(np.float16 if s == "weight" else np.float32))
for k in ("final_layer_norm.weight", "final_layer_norm.bias", "head.weight", "head.bias"):
    np.save(OUT / (k.replace(".", "_") + ".npy"), g("text_model." + k).astype(np.float16 if k == "head.weight" else np.float32))
print("kept", len(keep), "rows;", round(sum(p.stat().st_size for p in OUT.iterdir()) / 1e6), "MB")
