"""Movement questions ("who walked right", "who is standing still") answered from track geometry, no VLM.

A handheld camera would make "right" meaningless, so the camera's own motion is estimated per processed frame
from background features (people masked out) and removed: every box centre is mapped into the coordinates of
the first frame before displacement is measured."""
import cv2
import numpy as np

from . import config

CAM_WIDTH = 480  # camera motion is estimated on a downscaled grey frame


class CameraTracker:
    """Cumulative similarity transform M_t mapping frame-t pixels -> first-frame pixels (3x3, in proxy pixels)."""

    def __init__(self, frame_w, frame_h):
        self.s = CAM_WIDTH / frame_w
        self.size = (CAM_WIDTH, int(frame_h * self.s))
        self.M = np.eye(3)
        self.prev = None  # (grey, mask)

    def _grey_mask(self, frame, boxes):
        g = cv2.cvtColor(cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        mask = np.full(g.shape, 255, np.uint8)
        for x1, y1, x2, y2 in np.asarray(boxes).reshape(-1, 4):
            mask[max(int(y1 * self.s), 0):int(y2 * self.s) + 1, max(int(x1 * self.s), 0):int(x2 * self.s) + 1] = 0
        return g, mask

    def update(self, frame, boxes):
        """boxes: this frame's detections (excluded from the background features). Returns M_t (3x3)."""
        g, mask = self._grey_mask(frame, boxes)
        if self.prev is not None:
            pg, pm = self.prev
            pts = cv2.goodFeaturesToTrack(pg, 400, 0.01, 8, mask=pm)
            if pts is not None and len(pts) >= 12:
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(pg, g, pts, None)
                ok = st.ravel() == 1
                if ok.sum() >= 12:
                    A, _ = cv2.estimateAffinePartial2D(nxt[ok], pts[ok], method=cv2.RANSAC, ransacReprojThreshold=2.5)
                    if A is not None:
                        step = np.vstack([A, [0, 0, 1]])
                        self.M = self.M @ step
        self.prev = (g, mask)
        return self.M.copy()

    def to_ref(self, M, xyxy):
        """Map boxes (proxy pixels) of the frame whose transform is M into first-frame coordinates."""
        xyxy = np.asarray(xyxy, float).reshape(-1, 4)
        S = np.diag([self.s, self.s, 1.0])
        T = np.linalg.inv(S) @ M @ S  # proxy px -> proxy px of the reference frame
        c = np.stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2, np.ones(len(xyxy))], 1) @ T.T
        sc = float(np.sqrt(abs(np.linalg.det(T[:2, :2])))) or 1.0
        w, h = (xyxy[:, 2] - xyxy[:, 0]) * sc, (xyxy[:, 3] - xyxy[:, 1]) * sc
        return np.stack([c[:, 0] - w / 2, c[:, 1] - h / 2, c[:, 0] + w / 2, c[:, 1] + h / 2], 1)

    def from_ref(self, M, xyxy):
        """Inverse of to_ref: first-frame coordinates -> this frame's pixels."""
        xyxy = np.asarray(xyxy, float).reshape(-1, 4)
        S = np.diag([self.s, self.s, 1.0])
        T = np.linalg.inv(np.linalg.inv(S) @ M @ S)
        c = np.stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2, np.ones(len(xyxy))], 1) @ T.T
        sc = float(np.sqrt(abs(np.linalg.det(T[:2, :2])))) or 1.0
        w, h = (xyxy[:, 2] - xyxy[:, 0]) * sc, (xyxy[:, 3] - xyxy[:, 1]) * sc
        return np.stack([c[:, 0] - w / 2, c[:, 1] - h / 2, c[:, 0] + w / 2, c[:, 1] + h / 2], 1)


def camera_speed(cam_frames, cam_M, fps, frame_w):
    """Average camera drift as a fraction of frame width per second (0 = tripod)."""
    if len(cam_M) < 2:
        return 0.0
    step = np.linalg.norm(np.diff(cam_M[:, :2, 2], axis=0), axis=1)  # translation per processed step, proxy px*s? (scaled px)
    dt = np.diff(cam_frames) / fps
    return float(np.mean(step / np.maximum(dt, 1e-6)) / (CAM_WIDTH))


def track_stats(dets, fps):
    """Per track, motion in first-frame coordinates: displacement in body-heights, size change, duration.
    dets needs frame, track, box_ref (stabilised boxes); fps converts frames to seconds. Returns {track: dict or None (too short)}."""
    out = {}
    tracks = dets["track"]
    for t in np.unique(tracks):
        m = tracks == t
        order = np.argsort(dets["frame"][m])
        f = dets["frame"][m][order]
        b = dets["box_ref"][m][order]
        n = len(f)
        span = (f[-1] - f[0]) / fps
        if n < config.MIN_MOTION_DETS or span < config.MIN_MOTION_SECONDS:
            out[int(t)] = None
            continue
        cx, cy, h = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, b[:, 3] - b[:, 1]
        k = max(n // 3, 1)
        H = float(np.mean(h)) or 1.0
        out[int(t)] = dict(dx=float(np.median(cx[-k:]) - np.median(cx[:k])) / H,
                           dy=float(np.median(cy[-k:]) - np.median(cy[:k])) / H,
                           ratio=float(np.median(h[-k:]) / max(np.median(h[:k]), 1e-6)), span=float(span), n=n)
    return out


def classify(s, kind):
    """s: stats dict -> True/False for one motion kind. kinds: right left toward away still moving."""
    still = abs(s["dx"]) < 0.25 and abs(s["dy"]) < 0.25 and 0.85 < s["ratio"] < 1.18
    return {
        "right": s["dx"] >= 0.5 and s["dx"] >= 0.6 * abs(s["dy"]),
        "left": s["dx"] <= -0.5 and -s["dx"] >= 0.6 * abs(s["dy"]),
        # size change alone is noisy with partial bodies, so it must agree with perspective: toward the camera
        # = bigger and lower in the image, away = smaller and higher
        "toward": s["ratio"] >= 1.25 and s["dy"] > 0.1,
        "away": s["ratio"] <= 0.8 and s["dy"] < -0.1,
        "still": still,
        "moving": not still,
    }[kind]


def motion_scores(dets, kind, fps):
    """-> {track: 1.0 (matches) / 0.0 / nan (too short to judge)}, same shape the VLM path returns."""
    stats = track_stats(dets, fps)
    return {t: (float("nan") if s is None else float(classify(s, kind))) for t, s in stats.items()}
