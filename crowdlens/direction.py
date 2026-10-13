def left_to_right(people, min_dx=50):
    return [p.track_id for p in people if len(p.centroid_x) >= 2 and p.centroid_x[-1] - p.centroid_x[0] >= min_dx]
