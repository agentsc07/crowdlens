"""Pass 1 (once per video): detect + track people across the whole clip, keep each person's best crop.
Saves data/<id>/dets.npz (one row per person-per-frame), crops/<track>.jpg, meta.json.
Track ids are per-video temporaries."""
import json
from pathlib import Path

import cv2
import numpy as np

from . import config

PERSON = 1  # COCO "person" in RF-DETR


def crop_score(img):
    """Prefer large, sharp crops."""
    h, w = img.shape[:2]
    sharp = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    return float(np.sqrt(h * w) * np.log1p(sharp))


def detect_people(model, frame):
    """RF-DETR sees a ~600px image, so distant people vanish. Also scan overlapping 640px tiles of a
    1080px-wide copy and merge everything with NMS. Returns person boxes in `frame` coordinates."""
    import supervision as sv

    h, w = frame.shape[:2]
    k = min(1.0, config.DETECT_WIDTH / w)
    work = cv2.resize(frame, (int(w * k), int(h * k))) if k < 1 else frame
    wh, ww = work.shape[:2]
    t = config.TILE
    views = [(work, 0, 0)]
    for oy in range(0, max(wh - t, 0) + 1, t // 2):
        for ox in range(0, max(ww - t, 0) + 1, t // 2):
            views.append((work[oy:oy + t, ox:ox + t], ox, oy))
    xyxy, conf = [], []
    for img, ox, oy in views:
        d = model.predict(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), threshold=config.DETECT_CONF)
        d = d[d.class_id == PERSON]
        xyxy.append(d.xyxy + np.array([ox, oy, ox, oy]))
        conf.append(d.confidence)
    det = sv.Detections(xyxy=np.concatenate(xyxy).reshape(-1, 4).astype(float), confidence=np.concatenate(conf),
                        class_id=np.zeros(sum(len(c) for c in conf), int)).with_nms(threshold=0.5, class_agnostic=True)
    det.xyxy = det.xyxy / k
    return det


def index_video(video_path, out_dir, stride=config.DEFAULT_STRIDE, progress=None):
    import supervision as sv
    from rfdetr import RFDETRNano

    out_dir = Path(out_dir)
    (out_dir / "crops").mkdir(parents=True, exist_ok=True)
    for old in (out_dir / "crops").glob("*.jpg"):
        old.unlink()
    (out_dir / "answers.json").unlink(missing_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    model = RFDETRNano()
    tracker = sv.ByteTrack(frame_rate=max(int(fps / stride), 1))

    frames, tracks, boxes = [], [], []
    best = {}  # track -> (score, crop)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            det = detect_people(model, frame)
            det = tracker.update_with_detections(det)
            for b, tid in zip(det.xyxy, det.tracker_id):
                x1, y1, x2, y2 = [int(v) for v in b]
                x1, y1 = max(x1, 0), max(y1, 0)
                if x2 - x1 < 8 or y2 - y1 < 8:
                    continue
                tid = int(tid)
                frames.append(i); tracks.append(tid); boxes.append([x1, y1, x2, y2])
                img = frame[y1:y2, x1:x2]
                s = crop_score(img)
                if tid not in best or s > best[tid][0]:
                    best[tid] = (s, img.copy())
        i += 1
        if progress and i % 5 == 0:
            progress(min(i / total, 1.0), f"Detecting people… frame {i}/{total}")
    cap.release()

    # flicker tracks (seen in under MIN_TRACK_DETS detections) are mostly tracker noise: drop them so the
    # VLM isn't asked about the same person several times
    count = {}
    for t in tracks:
        count[t] = count.get(t, 0) + 1
    keep = {t for t, c in count.items() if c >= config.MIN_TRACK_DETS}
    sel = [j for j, t in enumerate(tracks) if t in keep]
    frames = [frames[j] for j in sel]; boxes = [boxes[j] for j in sel]; tracks = [tracks[j] for j in sel]
    best = {t: v for t, v in best.items() if t in keep}

    for tid, (_, img) in best.items():
        cv2.imwrite(str(out_dir / "crops" / f"{tid}.jpg"), img)
    np.savez_compressed(out_dir / "dets.npz", frame=np.array(frames, int), track=np.array(tracks, int),
                        box=np.array(boxes, int).reshape(-1, 4))
    (out_dir / "meta.json").write_text(json.dumps(dict(fps=fps, w=w, h=h, frames=i, stride=stride)))
    if progress:
        progress(1.0, "Indexed")
    return len(best)


def load(out_dir):
    out_dir = Path(out_dir)
    z = np.load(out_dir / "dets.npz")
    return dict(frame=z["frame"], track=z["track"], box=z["box"]), json.loads((out_dir / "meta.json").read_text())
