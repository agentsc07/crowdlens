"""Look and feel: frosted-glass panels floating over a soft blue field (spatial-UI depth), plus small HTML
helpers. Everything degrades gracefully: no backdrop-filter -> more opaque panels, reduced-motion -> no animation."""
import base64
import html
from functools import lru_cache
from pathlib import Path

import cv2

CSS = """
<style>
:root{
  --ink:#0b1437; --muted:#5b6690; --blue:#0a3cff; --blue-soft:#b9c8ff; --glass:rgba(255,255,255,.58);
  --edge:rgba(255,255,255,.75); --yes:#12b76a; --no:#f04438; --unc:#98a2b3;
}
html,body,.stApp{background:#eef2ff !important;color:var(--ink);}
.stApp{
  background:
    radial-gradient(900px 600px at 8% -10%, rgba(10,60,255,.22), transparent 60%),
    radial-gradient(800px 600px at 105% 10%, rgba(120,150,255,.35), transparent 60%),
    radial-gradient(700px 700px at 50% 115%, rgba(10,60,255,.14), transparent 60%),
    #eef2ff !important;
}
/* floating depth blobs */
.stApp::before,.stApp::after{content:"";position:fixed;z-index:0;border-radius:50%;filter:blur(70px);pointer-events:none;opacity:.55}
.stApp::before{width:380px;height:380px;background:#0a3cff;top:-120px;left:-80px;animation:drift 18s ease-in-out infinite alternate}
.stApp::after{width:320px;height:320px;background:#9db4ff;bottom:-100px;right:-60px;animation:drift 22s ease-in-out infinite alternate-reverse}
@keyframes drift{to{transform:translate(60px,40px) scale(1.12)}}
[data-testid="stHeader"],[data-testid="stSidebar"],[data-testid="collapsedControl"],footer{display:none !important}
.block-container{position:relative;z-index:1;max-width:1180px;padding:2rem 1rem 4rem}

.glass{
  background:var(--glass);border:1px solid var(--edge);border-radius:28px;padding:20px 22px;margin:0 0 18px;
  box-shadow:0 24px 60px rgba(10,60,255,.12),0 2px 6px rgba(11,20,55,.05),inset 0 1px 0 rgba(255,255,255,.9);
  transition:transform .35s cubic-bezier(.2,.8,.2,1),box-shadow .35s;
}
@supports ((backdrop-filter:blur(1px)) or (-webkit-backdrop-filter:blur(1px))){
  .glass{-webkit-backdrop-filter:blur(22px) saturate(170%);backdrop-filter:blur(22px) saturate(170%)}
}
@supports not ((backdrop-filter:blur(1px)) or (-webkit-backdrop-filter:blur(1px))){.glass{background:rgba(255,255,255,.9)}}
.glass.lift:hover{transform:translateY(-4px);box-shadow:0 34px 70px rgba(10,60,255,.18),inset 0 1px 0 rgba(255,255,255,.9)}

.hero{display:flex;align-items:center;gap:16px;margin-bottom:6px}
.logo{width:54px;height:54px;border-radius:18px;position:relative;flex:none}
.logo i{position:absolute;border-radius:16px}
.logo i:nth-child(1){inset:0 10px 10px 0;background:var(--blue)}
.logo i:nth-child(2){inset:10px 0 0 10px;background:rgba(185,200,255,.8);border:1px solid #fff;
  -webkit-backdrop-filter:blur(6px);backdrop-filter:blur(6px)}
.hero h1{margin:0;font-size:2rem;letter-spacing:-.03em;color:var(--ink)}
.hero p{margin:2px 0 0;color:var(--muted);font-size:.95rem}
.privacy{display:inline-block;margin:10px 0 22px;padding:5px 12px;border-radius:999px;font-size:.78rem;color:var(--blue);
  background:rgba(10,60,255,.08);border:1px solid rgba(10,60,255,.15)}

/* stepper */
.steps{display:flex;gap:10px;flex-wrap:wrap;margin:0 0 18px}
.step{display:flex;align-items:center;gap:8px;padding:8px 14px;border-radius:999px;font-size:.85rem;color:var(--muted);
  background:rgba(255,255,255,.45);border:1px solid var(--edge)}
.step b{display:grid;place-items:center;width:22px;height:22px;border-radius:50%;background:#dfe5fb;color:var(--muted);font-size:.75rem}
.step.on{color:var(--ink);background:#fff;box-shadow:0 8px 22px rgba(10,60,255,.15)}
.step.on b{background:var(--blue);color:#fff}
.step.done b{background:var(--yes);color:#fff}
.step.done{color:var(--ink)}

/* stat cards */
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px;margin-bottom:18px}
.stat{padding:16px 18px;border-radius:24px;background:var(--glass);border:1px solid var(--edge);
  box-shadow:0 18px 40px rgba(10,60,255,.10),inset 0 1px 0 rgba(255,255,255,.9);
  -webkit-backdrop-filter:blur(20px);backdrop-filter:blur(20px);transition:transform .3s}
.stat:hover{transform:translateY(-3px) perspective(700px) rotateX(2deg)}
.stat small{display:block;color:var(--muted);font-size:.78rem;margin-bottom:4px}
.stat span{font-size:2.3rem;font-weight:700;letter-spacing:-.03em;line-height:1.05;font-variant-numeric:tabular-nums}
.stat.yes span{color:var(--yes)} .stat.no span{color:var(--no)} .stat.unc span{color:var(--unc)} .stat.blue span{color:var(--blue)}

/* live status line: shimmering text = "something is happening" */
.live{display:flex;align-items:center;gap:12px;font-weight:600}
.live .dot{width:10px;height:10px;border-radius:50%;background:var(--blue);flex:none;animation:pulse 1.2s ease-in-out infinite}
.live .txt{background:linear-gradient(90deg,var(--ink) 20%,var(--blue) 50%,var(--ink) 80%);background-size:200% 100%;
  -webkit-background-clip:text;background-clip:text;color:transparent;animation:sheen 2.2s linear infinite}
.live .eta{margin-left:auto;color:var(--muted);font-weight:500;font-size:.85rem;font-variant-numeric:tabular-nums}
@keyframes pulse{50%{transform:scale(1.7);opacity:.35}}
@keyframes sheen{to{background-position:-200% 0}}
.bar{height:8px;border-radius:99px;background:rgba(10,60,255,.10);overflow:hidden;margin-top:12px}
.bar i{display:block;height:100%;border-radius:99px;background:linear-gradient(90deg,#0a3cff,#6f8cff);
  transition:width .35s ease;position:relative;overflow:hidden}
.bar i::after{content:"";position:absolute;inset:0;background:linear-gradient(90deg,transparent,rgba(255,255,255,.65),transparent);
  animation:slide 1.3s linear infinite}
@keyframes slide{from{transform:translateX(-100%)}to{transform:translateX(100%)}}

/* skeleton */
.skel{border-radius:22px;background:linear-gradient(100deg,rgba(255,255,255,.35) 30%,rgba(255,255,255,.85) 50%,rgba(255,255,255,.35) 70%);
  background-size:300% 100%;animation:shim 1.4s ease-in-out infinite;border:1px solid var(--edge)}
@keyframes shim{to{background-position:-300% 0}}

/* evidence grid */
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(84px,1fr));gap:10px}
.tile{position:relative;border-radius:18px;overflow:hidden;aspect-ratio:3/4;background:#dfe5fb;
  box-shadow:0 8px 20px rgba(11,20,55,.14);transition:transform .25s;animation:pop .35s cubic-bezier(.2,1.4,.4,1) both}
.tile:hover{transform:translateY(-4px) scale(1.05) perspective(500px) rotateX(3deg);z-index:2}
.tile img{width:100%;height:100%;object-fit:cover;display:block}
.tile::after{content:"";position:absolute;inset:0;border-radius:18px;border:3px solid var(--c);pointer-events:none}
.tile em{position:absolute;left:6px;bottom:6px;font-style:normal;font-size:.7rem;font-weight:700;color:#fff;padding:2px 8px;
  border-radius:99px;background:var(--c);-webkit-backdrop-filter:blur(6px);backdrop-filter:blur(6px)}
.tile.yes{--c:var(--yes)} .tile.no{--c:var(--no)} .tile.unc{--c:var(--unc)} .tile.wait{--c:#c9d3f7}
@keyframes pop{from{opacity:0;transform:scale(.7)}}

/* streamlit widgets in glass */
[data-testid="stFileUploader"] section{background:rgba(255,255,255,.55);border:1.5px dashed rgba(10,60,255,.35);border-radius:22px}
.stTextInput input,.stSelectbox div[data-baseweb="select"]>div{background:rgba(255,255,255,.8) !important;border-radius:18px !important;
  border:1px solid var(--edge) !important;padding:.8rem 1rem !important;font-size:1.05rem;box-shadow:0 10px 30px rgba(10,60,255,.10)}
.stTextInput input:focus{box-shadow:0 0 0 3px rgba(10,60,255,.25),0 10px 30px rgba(10,60,255,.15) !important}
.stButton>button,.stFormSubmitButton>button{border-radius:999px;border:1px solid var(--edge);background:rgba(255,255,255,.75);
  color:var(--ink);padding:.45rem 1.1rem;transition:transform .2s,box-shadow .2s;box-shadow:0 6px 16px rgba(10,60,255,.10)}
.stButton>button:hover,.stFormSubmitButton>button:hover{transform:translateY(-2px);box-shadow:0 12px 26px rgba(10,60,255,.2);color:var(--blue)}
.stButton>button[kind="primary"],.stFormSubmitButton>button[kind="primary"]{background:linear-gradient(135deg,#0a3cff,#4f74ff);color:#fff;border:none}
.stButton>button[kind="primary"]:hover,.stFormSubmitButton>button[kind="primary"]:hover{color:#fff}
[data-testid="stVideo"] video,[data-testid="stImage"] img{border-radius:22px}
[data-testid="stExpander"]{background:var(--glass);border:1px solid var(--edge);border-radius:22px}
h2,h3{letter-spacing:-.02em;color:var(--ink)}
.sec{font-size:1.05rem;font-weight:700;margin:6px 0 10px;color:var(--ink)}
.sec small{font-weight:500;color:var(--muted);margin-left:8px}

@media (max-width:640px){
  .block-container{padding:1rem .75rem 3rem}.hero h1{font-size:1.5rem}.glass{padding:16px;border-radius:22px}
  .stat span{font-size:1.9rem}
}
@media (prefers-reduced-motion:reduce){*{animation:none !important;transition:none !important}}
[data-testid="stVerticalBlockBorderWrapper"]{background:var(--glass);border:1px solid var(--edge) !important;border-radius:28px;
  padding:6px 10px;box-shadow:0 24px 60px rgba(10,60,255,.12),inset 0 1px 0 rgba(255,255,255,.9);
  -webkit-backdrop-filter:blur(22px) saturate(170%);backdrop-filter:blur(22px) saturate(170%)}
.hint{margin:0 0 12px;color:var(--muted);font-size:.9rem;line-height:1.5}.hint b{color:var(--ink)}
[data-testid="stButtonGroup"] button{border-radius:999px !important;font-weight:600}
[data-testid="stButtonGroup"] button[aria-checked="true"],[data-testid="stButtonGroup"] button[data-testid*="Active"]{
  background:#fff !important;color:var(--blue) !important;box-shadow:0 8px 22px rgba(10,60,255,.18)}
[data-baseweb="tab-list"]{gap:6px}[data-baseweb="tab"]{border-radius:999px !important}
</style>
"""


def hero():
    return ('<div class="hero"><div class="logo"><i></i><i></i></div><div><h1>CrowdLens</h1>'
            '<p>Ask a question of any video.</p></div></div>'
            '<span class="privacy">Counts attributes and movement, never identities. Runs fully on this machine.</span>')


def stepper(active, done=()):
    names = ["Index video", "Ask a question", "Explore answers"]
    out = []
    for i, n in enumerate(names, 1):
        cls = "done" if i in done else "on" if i == active else ""
        out.append(f'<div class="step {cls}"><b>{"✓" if i in done else i}</b>{n}</div>')
    return '<div class="steps">' + "".join(out) + "</div>"


def stat(label, value, tone=""):
    return f'<div class="stat {tone}"><small>{html.escape(label)}</small><span>{value}</span></div>'


def stats(items):
    return '<div class="stats">' + "".join(stat(*i) for i in items) + "</div>"


def live(text, frac, eta=""):
    return (f'<div class="glass"><div class="live"><span class="dot"></span><span class="txt">{html.escape(text)}</span>'
            f'<span class="eta">{html.escape(eta)}</span></div><div class="bar"><i style="width:{frac * 100:.1f}%"></i></div></div>')


def skeleton(height=320):
    return f'<div class="skel" style="height:{height}px"></div>'


@lru_cache(maxsize=4096)
def _thumb(path, mtime):
    img = cv2.imread(path)
    if img is None:
        return ""
    h, w = img.shape[:2]
    k = 160 / max(h, w)
    img = cv2.resize(img, (max(int(w * k), 1), max(int(h * k), 1)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode() if ok else ""


def thumb(path):
    p = Path(path)
    return _thumb(str(p), p.stat().st_mtime) if p.exists() else ""


def tile(src, kind, label):
    return f'<div class="tile {kind}"><img src="{src}" alt=""><em>{html.escape(label)}</em></div>'


def grid(tiles):
    return '<div class="glass"><div class="grid">' + "".join(tiles) + "</div></div>"
