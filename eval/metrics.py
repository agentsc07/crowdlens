"""Step 2 + 3 metrics from ~/.crowdlens/eval/scores.json and labels_claude.json (Claude-labelled, human audit pending).
Writes eval/results.json and eval/results.md."""
import json
from pathlib import Path

import numpy as np

EVAL = Path.home() / ".crowdlens" / "eval"
HERE = Path(__file__).parent
S = json.loads((EVAL / "scores.json").read_text())
L = json.loads((EVAL / "labels_claude.json").read_text())
Q = [q for q in L if q in S]
HELD = "11"


def prf(pred, true):
    tp = int((pred & true).sum()); fp = int((pred & ~true).sum()); fn = int((~pred & true).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return dict(p=p, r=r, f1=2 * p * r / (p + r) if p + r else 0.0, tp=tp, fp=fp, fn=fn)


def arrays(q, clips=None):
    keys = [k for k, v in L[q].items() if v != -1 and (clips is None or k.split(":")[0] in clips)]
    return (keys, np.array([S[q]["vlm"][k] for k in keys]), np.array([S[q]["sim"][k] for k in keys]),
            np.array([L[q][k] == 1 for k in keys]))


def pooled_f1(th, clips):
    pr = tr = None
    pred, true = [], []
    for q in Q:
        _, v, _, t = arrays(q, clips)
        pred.append(v >= th); true.append(t)
    return prf(np.concatenate(pred), np.concatenate(true))["f1"]


ths = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]
TRAIN = {"02", "04", "09"}
curve = {th: (pooled_f1(th, TRAIN), pooled_f1(th, {HELD})) for th in ths}
best = max(ths, key=lambda th: curve[th][0])
R = dict(chosen_threshold=best, threshold_curve={str(k): dict(train_f1=v[0], heldout_f1=v[1]) for k, v in curve.items()})

per_q = {}
for q in Q:
    _, v, _, t = arrays(q)
    d = {"@0.5": prf(v >= 0.5, t), f"@{best}": prf(v >= best, t)}
    for name, th in (("@0.5", 0.5), (f"@{best}", best)):
        d[name]["true"] = int(t.sum()); d[name]["pred"] = int((v >= th).sum())
        d[name]["count_err_pct"] = 100 * abs(d[name]["pred"] - d[name]["true"]) / max(d[name]["true"], 1)
    d["vlm_seconds"] = S[q]["vlm_s"]; d["n_judged"] = len(t)
    per_q[q] = d
R["per_question"] = per_q
pool = {}
for name, th in (("@0.5", 0.5), (f"@{best}", best)):
    pred = np.concatenate([arrays(q)[1] >= th for q in Q]); true = np.concatenate([arrays(q)[3] for q in Q])
    pool[name] = prf(pred, true)
    pool[name]["mean_count_err_pct"] = float(np.mean([per_q[q][name]["count_err_pct"] for q in Q]))
R["pooled"] = pool
R["heldout_clip11_at_chosen"] = {"f1": curve[best][1], "f1_at_0.5": curve[0.5][1]}

# --- search-then-verify: shortlist the top fraction by SigLIP2 similarity, verify only those with the VLM
per_person = float(np.mean([S[q]["vlm_s"] / len(S[q]["vlm"]) for q in Q]))
sim_s = float(np.mean([S[q]["sim_s"] for q in Q]))
stv = {}
for frac in (0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0):
    pred, true, rec_hit, rec_tot, verified, n = [], [], 0, 0, 0, 0
    for q in Q:
        _, v, s, t = arrays(q)
        k = max(int(round(frac * len(s))), 1)
        top = np.zeros(len(s), bool); top[np.argsort(-s)[:k]] = True
        pred.append((v >= best) & top); true.append(t)
        rec_hit += int((top & t).sum()); rec_tot += int(t.sum()); verified += k; n += len(s)
    m = prf(np.concatenate(pred), np.concatenate(true))
    stv[str(frac)] = dict(shortlist_recall=rec_hit / rec_tot, f1=m["f1"], precision=m["p"], recall=m["r"],
                          vlm_calls_per_question=verified / len(Q), calls_vs_all=verified / n,
                          est_seconds_per_question=sim_s + verified / len(Q) * per_person)
R["search_then_verify"] = dict(per_person_vlm_seconds=per_person, sim_seconds_per_question=sim_s,
                               embed_seconds_once=S["_embed_seconds"], people=len(S[Q[0]]["vlm"]), by_fraction=stv)
# SigLIP2 alone as the classifier (no VLM): best-threshold-free ranking quality
from itertools import chain
aps = {}
for q in Q:
    _, v, s, t = arrays(q)
    def ap(score):
        o = np.argsort(-score); hit = t[o]; prec = np.cumsum(hit) / (np.arange(len(hit)) + 1)
        return float((prec * hit).sum() / max(hit.sum(), 1))
    aps[q] = dict(siglip_ap=ap(s), vlm_ap=ap(v), prevalence=float(t.mean()))
R["average_precision"] = aps
(HERE / "results.json").write_text(json.dumps(R, indent=1))

md = ["# Results (Claude-labelled, human audit pending)", "",
      f"{len(S[Q[0]]['vlm'])} people from 4 MOT17 clips x {len(Q)} questions; people marked unsure/unjudgeable are excluded per question.", "",
      f"Threshold chosen on clips 02/04/09 (max pooled F1): **{best}**. Held-out clip 11 pooled F1: {curve[best][1]:.2f} (at 0.5: {curve[0.5][1]:.2f}).", "",
      "| Question | true | pred@0.5 | P | R | F1 | count err % | " + f"pred@{best} | F1 | count err % |", "|---|---|---|---|---|---|---|---|---|---|"]
for q in Q:
    a, b = per_q[q]["@0.5"], per_q[q][f"@{best}"]
    md.append(f"| {q} | {a['true']} | {a['pred']} | {a['p']:.2f} | {a['r']:.2f} | {a['f1']:.2f} | {a['count_err_pct']:.0f} | {b['pred']} | {b['f1']:.2f} | {b['count_err_pct']:.0f} |")
for name in pool:
    md.append(f"| **pooled {name}** | | | {pool[name]['p']:.2f} | {pool[name]['r']:.2f} | {pool[name]['f1']:.2f} | {pool[name]['mean_count_err_pct']:.0f} (mean) | | | |")
md += ["", "## Search-then-verify (SigLIP2 shortlist, then Qwen3-VL on the shortlist only)", "",
       f"VLM cost {per_person:.2f} s/person on this machine; SigLIP2 query {sim_s:.2f} s; embedding done once ({S['_embed_seconds']:.1f} s).", "",
       "| shortlist | shortlist recall | F1 | VLM calls/question | est. s/question |", "|---|---|---|---|---|"]
for f, d in stv.items():
    md.append(f"| top {float(f):.0%} | {d['shortlist_recall']:.2f} | {d['f1']:.2f} | {d['vlm_calls_per_question']:.0f} ({d['calls_vs_all']:.0%}) | {d['est_seconds_per_question']:.1f} |")
md += ["", "## Ranking quality (average precision)", "", "| Question | prevalence | SigLIP2 AP | Qwen3-VL AP |", "|---|---|---|---|"]
for q, d in aps.items():
    md.append(f"| {q} | {d['prevalence']:.2f} | {d['siglip_ap']:.2f} | {d['vlm_ap']:.2f} |")
(HERE / "results.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
