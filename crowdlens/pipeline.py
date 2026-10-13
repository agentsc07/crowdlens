"""Glue: index a video once, answer many questions."""
import json
from pathlib import Path

from . import config
from .detect_track import index_people
from .embed import build_index


def video_dir(video_id):
    return config.DATA_DIR / video_id


def index_video(video_path, video_id, progress=None):
    d = video_dir(video_id)
    if d.exists():  # re-index overwrites cleanly
        import shutil
        shutil.rmtree(d)
    d.mkdir(parents=True)
    people = index_people(video_path, d, progress)
    build_index(people, d)
    return people


def fps(video_id):
    return json.loads((video_dir(video_id) / "meta.json").read_text())["fps"]


def make_verifier(name):
    if name == "SigLIP zero-shot":
        from .verifiers.siglip_zs import SigLIPZeroShot
        return SigLIPZeroShot()
    if name == "Fine-tuned classifier":
        from .verifiers.classifier import AttributeClassifier
        return AttributeClassifier()
    from .verifiers.vlm import VLMVerifier
    return VLMVerifier()
