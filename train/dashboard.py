"""DiabloGym training monitor dashboard (no dependencies, stdlib HTTP).

Usage (from the repo root):  .venv/bin/python train/dashboard.py [--port 8787] [--run-dir runs/xxx]
By default it follows the newest run under runs/ (re-detected on every refresh, so it can stay open across runs).
Open:  http://127.0.0.1:8787
"""

from __future__ import annotations

import argparse
import json
import pathlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

RUNS_DIR = pathlib.Path(__file__).resolve().parent / "runs"
FORCED_RUN: pathlib.Path | None = None
MAX_EPISODES = 800  # maximum number of (most recent) episodes returned to the front end


def tail_lines(path: pathlib.Path, limit: int) -> list[str]:
    """Read only the tail of the JSONL so that polling every two seconds never scans the full training log and slows sampling."""
    if limit <= 0:
        return []
    with open(path, "rb") as f:
        f.seek(0, 2)
        pos = f.tell()
        chunks = []
        newlines = 0
        while pos > 0 and newlines <= limit:
            size = min(64 * 1024, pos)
            pos -= size
            f.seek(pos)
            chunk = f.read(size)
            chunks.append(chunk)
            newlines += chunk.count(b"\n")
    data = b"".join(reversed(chunks))
    return data.decode("utf-8", errors="replace").splitlines()[-limit:]


def latest_run() -> pathlib.Path | None:
    if FORCED_RUN is not None:
        return FORCED_RUN
    if not RUNS_DIR.is_dir():
        return None
    runs = [d for d in RUNS_DIR.iterdir() if d.is_dir() and (d / "status.json").exists()]
    return max(runs, key=lambda d: (d / "status.json").stat().st_mtime, default=None)


def collect() -> dict:
    run = latest_run()
    if run is None:
        return {"status": None, "episodes": []}
    try:
        status = json.loads((run / "status.json").read_text())
    except Exception:
        status = None
    episodes = []
    progress = run / "progress.jsonl"
    if progress.exists():
        lines = tail_lines(progress, MAX_EPISODES)
        for line in lines:
            try:
                episodes.append(json.loads(line))
            except Exception:
                pass
    return {"status": status, "episodes": episodes}


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>DiabloGym Training Monitor</title>
<style>
  :root { --bg:#12100e; --card:#1c1917; --line:#d97706; --line2:#78716c;
          --text:#e7e5e4; --dim:#a8a29e; --good:#4ade80; --bad:#f87171; }
  * { box-sizing:border-box; margin:0; }
  body { background:var(--bg); color:var(--text);
         font:14px/1.5 -apple-system,'PingFang SC',sans-serif; padding:20px; }
  h1 { font-size:19px; margin-bottom:4px; } h1 em { color:var(--line); font-style:normal; }
  .sub { color:var(--dim); font-size:12px; margin-bottom:14px; }
  .stats { display:flex; gap:10px; flex-wrap:wrap; margin-bottom:14px; }
  .stat { background:var(--card); border-radius:8px; padding:9px 15px; min-width:105px; }
  .stat b { display:block; font-size:19px; font-variant-numeric:tabular-nums; }
  .stat span { color:var(--dim); font-size:11px; }
  .badge { padding:2px 9px; border-radius:99px; font-size:12px; }
  .run  { background:#14532d; color:var(--good); }
  .idle { background:#450a0a; color:var(--bad); }
  .bar { height:6px; background:#292524; border-radius:3px; overflow:hidden; margin:10px 0 16px; }
  .bar i { display:block; height:100%; background:var(--line); transition:width .5s; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(330px,1fr)); gap:12px; }
  .card { background:var(--card); border-radius:10px; padding:12px 14px; }
  .card h3 { font-size:12px; color:var(--dim); font-weight:500; margin-bottom:6px; }
  svg { width:100%; height:130px; }
  table { width:100%; border-collapse:collapse; font-size:12.5px;
          font-variant-numeric:tabular-nums; }
  th { color:var(--dim); font-weight:500; text-align:right; padding:3px 8px; }
  td { text-align:right; padding:3px 8px; border-top:1px solid #292524; }
  th:first-child, td:first-child { text-align:left; }
</style></head><body>
<h1>⚔️ DiabloGym <em>Training Monitor</em></h1>
<div class="sub" id="runline">Waiting for training data…</div>
<div class="stats" id="stats"></div>
<div class="bar"><i id="bar" style="width:0%"></i></div>
<div class="grid">
  <div class="card"><h3>Reward per episode (orange = 20-episode moving mean)</h3><svg id="c-reward"></svg></div>
  <div class="card"><h3>Kills per episode</h3><svg id="c-kills"></svg></div>
  <div class="card"><h3>XP gained per episode</h3><svg id="c-xp"></svg></div>
  <div class="card"><h3>Steps per episode (survival time)</h3><svg id="c-len"></svg></div>
</div>
<div class="card" style="margin-top:12px"><h3>Recent episodes</h3>
  <table><thead><tr><th>Ep</th><th>Reward</th><th>Steps</th><th>Kills</th><th>Clear %</th><th>XP</th>
  <th>Level</th><th>Depth</th><th>Gold</th><th>Outcome</th></tr></thead>
  <tbody id="tbody"></tbody></table>
</div>
<script>
function roll(vs, k) { const out=[]; let s=0;
  for (let i=0;i<vs.length;i++){ s+=vs[i]; if(i>=k) s-=vs[i-k];
    out.push(s/Math.min(i+1,k)); } return out; }
function chart(id, vs, opts={}) {
  const svg=document.getElementById(id); const W=600,H=130,P=6;
  svg.setAttribute('viewBox',`0 0 ${W} ${H}`);
  if (vs.length<2) { svg.innerHTML=''; return; }
  const lo=Math.min(...vs), hi=Math.max(...vs), span=(hi-lo)||1;
  const pt=a=>a.map((v,i)=>`${P+i*(W-2*P)/(a.length-1)},${H-P-(v-lo)/span*(H-2*P)}`).join(' ');
  let s=`<polyline points="${pt(vs)}" fill="none" stroke="#78716c" stroke-width="1" opacity=".65"/>`;
  s+=`<polyline points="${pt(roll(vs,20))}" fill="none" stroke="#d97706" stroke-width="2"/>`;
  s+=`<text x="${W-P}" y="12" fill="#a8a29e" font-size="10" text-anchor="end">max ${hi.toFixed(1)}</text>`;
  s+=`<text x="${W-P}" y="${H-2}" fill="#a8a29e" font-size="10" text-anchor="end">min ${lo.toFixed(1)}</text>`;
  svg.innerHTML=s;
}
function fmt(n){ return n>=1e6?(n/1e6).toFixed(2)+'M':n>=1e3?(n/1e3).toFixed(1)+'k':''+n; }
function hms(s){ const h=~~(s/3600),m=~~(s%3600/60); return (h?h+'h':'')+m+'m'+(s%60)+'s'; }
async function tick() {
  try {
    const d = await (await fetch('/data')).json();
    const st = d.status, eps = d.episodes;
    if (!st) { document.getElementById('runline').textContent='No training data yet (runs/ is empty)'; return; }
    const fresh = (Date.now()/1000 - st.updated_at) < 6;
    document.getElementById('runline').innerHTML =
      `run <b>${st.run}</b> · ${st.config.algo} · ${st.config.num_envs} envs · ` +
      `<span class="badge ${fresh?'run':'idle'}">${fresh?'training':'stopped/idle'}</span>`;
    const pct = (100*st.total_steps/st.target_steps).toFixed(1);
    document.getElementById('bar').style.width = pct+'%';
    const last20 = eps.slice(-20);
    const mean = a=>a.length? (a.reduce((x,y)=>x+y,0)/a.length) : 0;
    document.getElementById('stats').innerHTML = [
      [fmt(st.total_steps)+' / '+fmt(st.target_steps), 'steps ('+pct+'%)'],
      [fmt(st.sps)+'/s', 'sampling speed'],
      [st.episodes, 'total episodes'],
      [hms(st.elapsed_sec), 'elapsed'],
      [mean(last20.map(e=>e.reward)).toFixed(2), 'mean reward, last 20'],
      [mean(last20.map(e=>e.kills||0)).toFixed(1), 'mean kills, last 20'],
      [(100*mean(last20.map(e=>e.died?1:0))).toFixed(0)+'%', 'death rate, last 20'],
      [Math.max(0,...eps.map(e=>e.depth||1)), 'deepest level reached'],
    ].map(([v,l])=>`<div class="stat"><b>${v}</b><span>${l}</span></div>`).join('');
    chart('c-reward', eps.map(e=>e.reward));
    chart('c-kills',  eps.map(e=>e.kills||0));
    chart('c-xp',     eps.map(e=>e.xp||0));
    chart('c-len',    eps.map(e=>e.len));
    document.getElementById('tbody').innerHTML = eps.slice(-12).reverse().map(e=>
      `<tr><td>#${e.ep}</td><td>${e.reward.toFixed(2)}</td><td>${e.len}</td>`+
      `<td>${e.kills??'-'}</td><td>${e.clear_pct!=null?e.clear_pct+'%':'-'}</td>`+
      `<td>${e.xp??'-'}</td><td>${e.char_level??'-'}</td>`+
      `<td>${e.depth??'-'}</td><td>${e.gold??'-'}</td>`+
      `<td>${e.died?'💀':'⏳'}</td></tr>`).join('');
  } catch (err) { /* a read can catch a half-written line while training writes the file; the next tick recovers */ }
}
tick(); setInterval(tick, 2000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/data":
            body = json.dumps(collect(), ensure_ascii=False).encode()
            ctype = "application/json; charset=utf-8"
        elif self.path == "/":
            body = PAGE.encode()
            ctype = "text/html; charset=utf-8"
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass  # silence the access log


def main():
    global FORCED_RUN
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--run-dir", default=None, help="monitor one fixed run (default: follow the newest)")
    args = ap.parse_args()
    if args.run_dir:
        p = pathlib.Path(args.run_dir).expanduser()
        if not p.is_absolute():
            cwd_path = p.resolve()
            if cwd_path.exists():
                p = cwd_path       # e.g. train/runs/foo passed from the repo root
            elif p.parts[:1] == ("runs",):
                p = RUNS_DIR.parent / p   # runs/foo in the docs is relative to train/
            else:
                p = RUNS_DIR / p          # bare run name
        FORCED_RUN = p.resolve()
        if not FORCED_RUN.is_dir():
            ap.error(f"run directory does not exist: {FORCED_RUN}")

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Dashboard: http://127.0.0.1:{args.port}  (following the newest run in {RUNS_DIR})")
    server.serve_forever()


if __name__ == "__main__":
    main()
