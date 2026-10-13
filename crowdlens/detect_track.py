"""RF-DETR person detection + ByteTrack -> one Person per track, with best crops saved.

Privacy: track ids are per-video temporaries; nothing here identifies anyone.
"""
from pathlib import Path

import cv2
import numpy as np

from . import config
from .crops import select_best, sharpness
from .schema import Crop, Person, save_people

PERSON_CLASS = 1  # COCO id for "person" in RF-DETR (1-indexed COCO)


def _visibility(box, others):
    best = 0.0
    x1, y1, x2, y2 = box
    area = max((x2 - x1) * (y2 - y1), 1e-9)
    for o in others:
        ix = max(0, min(x2, o[2]) - max(x1, o[0]))
        iy = max(0, min(y2, o[3]) - max(y1, o[1]))
        best = max(best, ix * iy / area)
    return 1.0 - min(best, 1.0)


def index_people(video_path, out_dir, progress=None, stride=1):
    import supervision as sv
    from rfdetr import RFDETRBase

    out_dir = Path(out_dir)
    (out_dir / "crops").mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    model = RFDETRBase()
    tracker = sv.ByteTrack(frame_rate=int(fps))

    tracks = {}  # id -> dict(first,last,cx,boxes,crops[(Crop, img)])
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            det = model.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), threshold=0.5)
            det = det[det.class_id == PERSON_CLASS]
            det = tracker.update_with_detections(det)
            boxes = [tuple(map(float, b)) for b in det.xyxy]
            for i, (b, tid) in enumerate(zip(boxes, det.tracker_id)):
                tid = int(tid)
                t = tracks.setdefault(tid, dict(first=idx, last=idx, cx=[], boxes={}, crops=[]))
                t["last"] = idx
                t["cx"].append((b[0] + b[2]) / 2)
                t["boxes"][idx] = b
                x1, y1, x2, y2 = [int(v) for v in b]
                img = frame[max(y1, 0):y2, max(x1, 0):x2]
                if img.size == 0:
                    continue
                c = Crop(tid, idx, b, "", sharpness(img), _visibility(b, boxes[:i] + boxes[i + 1:]))
                t["crops"].append((c, img))
                # bound memory: keep only a rolling shortlist of candidates
                if len(t["crops"]) > 12:
                    keep = select_best([x[0] for x in t["crops"]], 6)
                    t["crops"] = [x for x in t["crops"] if x[0] in keep]
        idx += 1
        if progress:
            progress(min(idx / total, 1.0))
    cap.release()

    people = []
    for tid, t in sorted(tracks.items()):
        best = select_best([c for c, _ in t["crops"]])
        saved = []
        for n, c in enumerate(best):
            img = next(im for cc, im in t["crops"] if cc is c)
            rel = f"crops/{tid}_{n}.jpg"
            cv2.imwrite(str(out_dir / rel), img)
            saved.append(Crop(tid, c.frame_idx, c.box, rel, c.sharpness, c.visibility))
        people.append(Person(tid, t["first"], t["last"], saved, t["cx"], t["boxes"]))
    save_people(out_dir / "people.json", people)
    (out_dir / "meta.json").write_text(f'{{"fps": {fps}, "frames": {idx}}}')
    return people
