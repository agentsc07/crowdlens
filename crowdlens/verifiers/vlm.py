"""Tier 2 (default): small VLM, closed yes/no/unclear question, read next-token probabilities.

No free-text generation: one forward pass, probabilities of the three allowed first tokens.
"""
from .. import config
from .base import normalise

ANSWERS = ("yes", "no", "unclear")


class VLMVerifier:
    def __init__(self, model_name=config.VLM_MODEL):
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.proc = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="auto").to(self.dev).eval()
        tok = self.proc.tokenizer
        self.ids = [tok.encode(a, add_special_tokens=False)[0] for a in ANSWERS]

    def p_yes(self, crop_path: str, attribute: str) -> float:
        from PIL import Image

        img = Image.open(crop_path).convert("RGB")
        msg = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": f"Is this person {attribute}? Answer yes, no, or unclear."}]}]
        text = self.proc.apply_chat_template(msg, add_generation_prompt=True)
        inputs = self.proc(text=[text], images=[img], return_tensors="pt").to(self.dev)
        with self.torch.no_grad():
            logits = self.model(**inputs).logits[0, -1]
        return normalise(*[float(logits[i]) for i in self.ids])[0]
