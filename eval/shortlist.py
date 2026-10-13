"""Headline experiment: shortlist recall vs verifier-call reduction.
Usage: python -m eval.shortlist data/<video_id> "<question>" truth.json   (truth.json = list of true track ids)"""
import json
import sys

from crowdlens.embed import embed_text, shortlist
from eval.metrics import recall_at_k

if __name__ == "__main__":
    d, q, truth = sys.argv[1], sys.argv[2], json.load(open(sys.argv[3]))
    n = len(json.load(open(f"{d}/ids.json")))
    ranked = [i for i, _ in shortlist(d, embed_text(q), n)]
    print("k,recall,verifier_calls_saved")
    for k in (10, 25, 50, 100, 200, 400):
        k = min(k, n)
        print(f"{k},{recall_at_k(ranked, truth, k):.3f},{n - k}")
