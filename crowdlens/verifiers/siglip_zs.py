"""Tier 1: zero-shot SigLIP 2. Fast, weakest on fine detail."""
from .. import embed


class SigLIPZeroShot:
    def p_yes(self, crop_path: str, attribute: str) -> float:
        import torch

        img = embed.embed_images([crop_path])[0]
        pos = embed.embed_text(f"a person {attribute}")
        neg = embed.embed_text(f"a person not {attribute}")
        model = embed._load()[0]
        scale = float(model.logit_scale.detach().exp()) if hasattr(model, "logit_scale") else 100.0
        logits = torch.tensor([img @ pos, img @ neg]) * scale
        return float(torch.softmax(logits, 0)[0])
