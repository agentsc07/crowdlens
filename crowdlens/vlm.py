"""Small open VLM as a closed yes/no classifier: read P(Yes) vs P(No) from the next-token logits.
One forward pass per crop, no generated text."""
import numpy as np
import torch
from PIL import Image

from . import config

_state = None


def _load():
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


def _small(path):
    """Shrink the crop so the VLM sees few image tokens: this is the main latency and memory knob."""
    img = Image.open(path).convert("RGB")
    img.thumbnail((config.VLM_CROP_MAX, config.VLM_CROP_MAX))
    return img


def p_yes(paths, attribute, batch=16, progress=None):
    """paths: crop image files -> P(yes) per crop, in order."""
    model, proc, dev, yes_id, no_id = _load()
    out = []
    for s in range(0, len(paths), batch):
        msgs = [[{"role": "user", "content": [
            {"type": "image", "image": _small(p)},
            {"type": "text", "text": f"Is this person {attribute}? Answer Yes or No."}]}]
            for p in paths[s:s + batch]]
        inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                       return_tensors="pt", padding=True).to(dev)
        with torch.no_grad():
            lg = model(**inp).logits[:, -1].float()
        out += torch.softmax(lg[:, [yes_id, no_id]], dim=1)[:, 0].cpu().tolist()
        del inp, lg
        if dev == "mps":
            torch.mps.empty_cache()
        if progress:
            progress(min((s + batch) / len(paths), 1.0))
    return out
