"""End-to-end counting: python -m eval.counting data/<id> labels.json [verifier]
labels.json = [{"question": "...", "true": 23}, ...]"""
import json
import sys

from crowdlens import pipeline
from crowdlens.query import answer
from eval.metrics import in_range_rate, mae

if __name__ == "__main__":
    d, labels = sys.argv[1], json.load(open(sys.argv[2]))
    ver = pipeline.make_verifier(sys.argv[3] if len(sys.argv) > 3 else "VLM")
    rs = [answer(d, l["question"], ver) for l in labels]
    true = [l["true"] for l in labels]
    print("MAE", mae([r.low for r in rs], true), "in-range", in_range_rate([(r.low, r.high) for r in rs], true),
          "sec/question", sum(r.seconds for r in rs) / len(rs))
