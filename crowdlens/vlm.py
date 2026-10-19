"""Small open VLM as a closed yes/no classifier: read P(Yes) vs P(No) from the next-token logits.
One forward pass per crop, no generated text."""
import threading
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import config

_state = None
_lock = threading.Lock()  # the app pre-warms in a thread; a question must wait for it, not import concurrently


def _load():
    with _lock:
        return _load_locked()


def _load_locked():
    global _state
    if _state is None:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if dev != "cpu" else torch.float32
        proc = AutoProcessor.from_pretrained(config.VLM_MODEL)
        proc.tokenizer.padding_side = "left"
        model = Qwen3VLForConditionalGeneration.from_pretrained(config.VLM_MODEL, dtype=dtype).to(dev).eval()
        tok = proc.tokenizer
        _state = (model, proc, dev, tok.encode("Yes", add_special_tokens=False)[0],
                  tok.encode("No", add_special_tokens=False)[0])
    return _state


def _small(path, side):
    """Shrink the crop so the VLM sees few image tokens: this is the main latency and memory knob."""
    img = Image.open(path).convert("RGB")
    img.thumbnail((side, side))
    return img


def prompt(attribute, subject, n):
    if n == 1:
        return f"Is this {subject} {attribute}? Answer Yes or No."
    return (f"These {n} images show the same {subject} at {n} different moments, in time order. "
            f"Is this {subject} {attribute}? Answer Yes or No.")


def p_yes(items, attribute, subject="person", batch=8, progress=None, on_batch=None):
    """items: one entry per object, a crop path or a list of paths (moments of the same object, in time order)
    -> P(yes) per object, in order."""
    model, proc, dev, yes_id, no_id = _load()
    items = [[x] if isinstance(x, (str, Path)) else list(x) for x in items]
    out = []
    for s in range(0, len(items), batch):
        chunk = items[s:s + batch]
        side = config.VLM_CROP_MAX if max(len(c) for c in chunk) == 1 else config.VLM_MULTI_CROP_MAX
        msgs = [[{"role": "user", "content": [{"type": "image", "image": _small(p, side)} for p in c]
                 + [{"type": "text", "text": prompt(attribute, subject, len(c))}]}] for c in chunk]
        inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                       return_tensors="pt", padding=True).to(dev)
        with torch.no_grad():
            lg = model(**inp).logits[:, -1].float()
        got = torch.softmax(lg[:, [yes_id, no_id]], dim=1)[:, 0].cpu().tolist()
        out += got
        if on_batch:
            on_batch(s, got)
        del inp, lg
        if dev == "mps":
            torch.mps.empty_cache()
        if progress:
            progress(min((s + batch) / len(items), 1.0))
    return out
