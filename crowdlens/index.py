"""Pass 1 (once per video): detect + track people on frames across the whole clip, embed every crop.
Saves data/<id>/dets.npz: one row per person-per-frame. Track ids are per-video temporaries."""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import config
from .embed import embed_images

PERSON = 1  # COCO "person" in RF-DETR


def index_video(video_path, out_dir, stride=config.DEFAULT_STRIDE, progress=None):
    import supervision as sv
    from rfdetr import RFDETRNano

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    model = RFDETRNano()
    tracker = sv.ByteTrack(frame_rate=max(int(fps / stride), 1))

    frames, tracks, boxes, crops = [], [], [], []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            det = model.predict(rgb, threshold=config.DETECT_CONF)
            det = det[det.class_id == PERSON]
            det = tracker.update_with_detections(det)
            for b, tid in zip(det.xyxy, det.tracker_id):
                x1, y1, x2, y2 = [int(v) for v in b]
                x1, y1 = max(x1, 0), max(y1, 0)
                if x2 - x1 < 8 or y2 - y1 < 8:
                    continue
                frames.append(i); tracks.append(int(tid)); boxes.append([x1, y1, x2, y2])
                crops.append(Image.fromarray(rgb[y1:y2, x1:x2]))
        i += 1
        if progress and i % 5 == 0:
            progress(min(i / total, 1.0) * 0.9, f"Detecting people… frame {i}/{total}")
    cap.release()

    if progress:
        progress(0.9, f"Embedding {len(crops)} person crops…")
    embs = embed_images(crops)
    np.savez_compressed(out_dir / "dets.npz", frame=np.array(frames, int), track=np.array(tracks, int),
                        box=np.array(boxes, int).reshape(-1, 4), emb=embs.astype("float16"))
    (out_dir / "meta.json").write_text(json.dumps(dict(fps=fps, w=w, h=h, frames=i, stride=stride)))
    if progress:
        progress(1.0, "Indexed")
    return len(set(tracks))


def load(out_dir):
    out_dir = Path(out_dir)
    z = np.load(out_dir / "dets.npz")
    return dict(frame=z["frame"], track=z["track"], box=z["box"], emb=z["emb"].astype("float32")), \
        json.loads((out_dir / "meta.json").read_text())
