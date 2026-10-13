import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class Crop:
    track_id: int
    frame_idx: int
    box: tuple
    path: str
    sharpness: float
    visibility: float

    def __post_init__(self):
        self.box = tuple(self.box)


@dataclass
class Person:
    track_id: int
    first_frame: int
    last_frame: int
    crops: list = field(default_factory=list)
    centroid_x: list = field(default_factory=list)
    boxes: dict = field(default_factory=dict)  # frame_idx -> (x1,y1,x2,y2), for overlay playback


@dataclass
class Verdict:
    track_id: int
    p_yes: float
    label: str  # "yes" | "no" | "unclear"


def save_people(path, people):
    Path(path).write_text(json.dumps([asdict(p) for p in people]))


def load_people(path):
    out = []
    for d in json.loads(Path(path).read_text()):
        crops = [Crop(**c) for c in d.pop("crops")]
        boxes = {int(k): tuple(v) for k, v in d.pop("boxes", {}).items()}
        out.append(Person(crops=crops, boxes=boxes, **d))
    return out
