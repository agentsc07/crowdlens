def per_minute(people, verdicts, fps):
    yes = {v.track_id for v in verdicts if v.label == "yes"}
    out = {}
    per_min = fps * 60
    for p in people:
        if p.track_id not in yes:
            continue
        for m in range(int(p.first_frame // per_min), int(p.last_frame // per_min) + 1):
            out[m] = out.get(m, 0) + 1
    return out
