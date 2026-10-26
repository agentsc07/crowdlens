"""Labels assigned by Claude from the numbered contact sheets (eval/contact_sheets.py), NOT by a human.
Index = position in the judgeable-people list. Codes: R red top, W white shirt, B black jacket, K backpack,
H handbag/shoulder bag, S shorts, C hat/cap, P shopping bag. '?X' = unsure (excluded from scoring);
'X' alone = not judgeable (false detection, sliver, heavy occlusion): excluded for every question.
Anyone not listed is 'no' for every question. Writes ~/.crowdlens/eval/labels_claude.json in label_app's format."""
import json
from pathlib import Path

EVAL = Path.home() / ".crowdlens" / "eval"
Q = {"R": "wearing a red top", "W": "wearing a white shirt", "B": "wearing a black jacket", "K": "carrying a backpack",
     "H": "carrying a handbag or shoulder bag", "S": "wearing shorts", "C": "wearing a hat or cap", "P": "holding a shopping bag"}
L = {0: "R", 1: "R H", 2: "X", 3: "X", 4: "W", 5: "H ?P", 6: "X", 7: "?W H P", 9: "X", 10: "X", 11: "H", 12: "X", 13: "X", 14: "X",
     15: "H ?P ?B", 16: "P ?B", 17: "H", 18: "X", 19: "X", 20: "H", 21: "X", 23: "X", 24: "H ?P ?B", 27: "X", 29: "X", 30: "W",
     31: "K S", 32: "H", 33: "X", 34: "K S", 35: "K S W", 36: "X", 37: "X", 38: "R K S", 39: "?W", 40: "R H ?P", 41: "C", 42: "C",
     43: "?C", 45: "?K", 46: "?K", 47: "B", 48: "W", 49: "W", 50: "W H", 51: "B W H", 52: "C ?H", 53: "?B", 54: "H", 55: "H ?S",
     56: "B", 57: "?K", 58: "S", 59: "?W", 60: "K", 62: "W H", 63: "X", 64: "C H", 65: "B", 66: "?B", 69: "K", 70: "B", 71: "?H",
     73: "W", 75: "?B", 76: "W", 77: "W", 78: "B", 79: "X", 80: "B H", 81: "?W", 82: "W H", 83: "B", 84: "X", 85: "X", 86: "X",
     87: "B H", 88: "X", 89: "K H S", 90: "W", 91: "W", 93: "H ?P", 94: "H", 95: "W ?H", 97: "S", 99: "C K", 100: "C ?H", 101: "S",
     103: "W ?H", 104: "W", 107: "?W", 108: "P", 109: "?K", 110: "P", 112: "P", 113: "S", 114: "B", 115: "X", 116: "B P",
     117: "B H", 118: "X", 119: "X", 123: "X", 124: "S", 126: "R S P", 127: "P", 128: "R S P", 129: "X", 130: "R S ?P",
     131: "R S", 132: "X", 134: "H", 135: "S", 136: "K", 138: "R S P", 140: "X", 141: "X", 142: "?H", 143: "C", 144: "?H",
     145: "C", 146: "X", 147: "W", 149: "X", 150: "P", 151: "H", 152: "C", 153: "X", 154: "?B C", 155: "X", 157: "W ?S ?H",
     158: "X", 159: "X", 160: "?B", 162: "R", 163: "W"}
people = [p for p in json.loads((EVAL / "people.json").read_text()) if p["judge"]]
out = {q: {} for q in Q.values()}
for i, p in enumerate(people):
    k, codes = f"{p['clip']}:{p['track']}", L.get(i, "").split()
    for c, q in Q.items():
        out[q][k] = -1 if "X" in codes or f"?{c}" in codes else int(c in codes)
(EVAL / "labels_claude.json").write_text(json.dumps(out))
for q, d in out.items():
    v = list(d.values())
    print(f"{q:38s} yes {v.count(1):3d}  no {v.count(0):3d}  unsure/skip {v.count(-1):3d}")
