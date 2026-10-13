"""Counted vs true people on a tracked clip: python -m eval.tracking data/<id> <true_count>
(ID-switch counting on MOT17 needs py-motmetrics; run index_people on a MOT17 sequence and feed its output there.)"""
import sys

from crowdlens.schema import load_people

if __name__ == "__main__":
    n, true = len(load_people(f"{sys.argv[1]}/people.json")), int(sys.argv[2])
    print(f"counted={n} true={true} error={abs(n - true) / max(true, 1):.1%}")
