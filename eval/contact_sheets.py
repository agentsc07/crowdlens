"""Numbered contact sheets of the judgeable people, for labelling by eye. -> ~/.crowdlens/eval/sheets/sheet_<n>.jpg"""
import json
from pathlib import Path

import cv2
import numpy as np

EVAL = Path.home() / ".crowdlens" / "eval"
people = [p for p in json.loads((EVAL / "people.json").read_text()) if p["judge"]]
(EVAL / "sheets").mkdir(exist_ok=True)
W, H, COLS, ROWS = 200, 330, 6, 3
per = COLS * ROWS
for s in range(0, len(people), per):
    sheet = np.full((ROWS * H, COLS * W, 3), 255, np.uint8)
    for j, p in enumerate(people[s:s + per]):
        im = cv2.imread(str(EVAL / "idx" / p["clip"] / "crops" / f"{p['track']}.jpg"))
        k = min((W - 6) / im.shape[1], (H - 34) / im.shape[0])
        im = cv2.resize(im, (int(im.shape[1] * k), int(im.shape[0] * k)), interpolation=cv2.INTER_CUBIC)
        r, c = divmod(j, COLS)
        y, x = r * H + 30, c * W + 3
        sheet[y:y + im.shape[0], x:x + im.shape[1]] = im
        cv2.putText(sheet, f"#{s + j}", (c * W + 6, r * H + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)
    cv2.imwrite(str(EVAL / "sheets" / f"sheet_{s // per}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
print(len(people), "people,", -(-len(people) // per), "sheets")
