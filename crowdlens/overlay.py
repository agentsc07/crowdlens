import cv2

from .schema import load_people


def render_overlay(video_path, people_json, matched_ids, out_path):
    """Replay the video with boxes only around matching people. No labels with identity, just boxes."""
    people = {p.track_id: p for p in load_people(people_json) if p.track_id in set(matched_ids)}
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for p in people.values():
            b = p.boxes.get(i)
            if b:
                cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 220, 0), 2)
        out.write(frame)
        i += 1
    cap.release()
    out.release()
