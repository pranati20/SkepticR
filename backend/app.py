"""FastAPI server: REST + WebSocket live event stream. Serves the built frontend if present.

    uvicorn skeptic.backend.app:app --port 8000
Works fully offline with no API key (MockLLM, pre-downloaded data, recorded demo runs).
"""
from __future__ import annotations

import asyncio
import io
import json
import threading
import uuid
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .runner import HONESTY, SCENARIOS, TITLES, run_scenario

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo"
DIST = ROOT / "frontend" / "dist"

app = FastAPI(title="Skeptic", description="Continuous integration for claims.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

RUNS: dict[str, dict] = {}


def _clean(o):
    """JSON-safe: numpy scalars/arrays, NaN."""
    import numpy as np
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if f != f or f in (float("inf"), float("-inf")) else f
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


@app.get("/api/scenarios")
def scenarios():
    return {"scenarios": [{"id": k, "title": TITLES[k][0], "blurb": TITLES[k][1]} for k in SCENARIOS], "honesty": HONESTY,
            "llm": "mock (offline, deterministic)"}


@app.post("/api/run")
def start_run(body: dict):
    name = body.get("scenario", "market")
    if name not in SCENARIOS:
        raise HTTPException(404, f"unknown scenario {name}")
    rid = uuid.uuid4().hex[:10]
    st = {"events": [], "done": False, "scenario": name}
    RUNS[rid] = st

    def emit(ev):
        st["events"].append(_clean(ev))

    def work():
        try:
            run_scenario(name, body.get("params") or {}, emit, int(body.get("seed", 0)))
        except Exception:
            pass
        finally:
            st["done"] = True

    threading.Thread(target=work, daemon=True).start()
    return {"run_id": rid}


@app.websocket("/ws/{run_id}")
async def ws(websocket: WebSocket, run_id: str):
    await websocket.accept()
    st = RUNS.get(run_id)
    if st is None:
        await websocket.close(code=4404)
        return
    i = 0
    try:
        while True:
            while i < len(st["events"]):
                await websocket.send_text(json.dumps(st["events"][i]))
                i += 1
            if st["done"] and i >= len(st["events"]):
                break
            await asyncio.sleep(0.03)
        await websocket.close()
    except WebSocketDisconnect:
        return


@app.get("/api/runs/{run_id}/events")
def run_events(run_id: str, after: int = 0):
    st = RUNS.get(run_id)
    if st is None:
        raise HTTPException(404, "no such run")
    return {"events": st["events"][after:], "done": st["done"], "next": len(st["events"])}


@app.get("/api/benchmark")
def benchmark():
    f = DEMO / "benchmark.json"
    if not f.exists():
        raise HTTPException(404, "run: python -m skeptic.backend.benchmark.run_benchmark")
    return JSONResponse(json.loads(f.read_text(encoding="utf-8")))


@app.get("/api/demo")
def demo_list():
    d = DEMO / "runs"
    return {"runs": sorted(p.stem for p in d.glob("*.json"))} if d.exists() else {"runs": []}


@app.get("/api/demo/{name}")
def demo_run(name: str):
    f = DEMO / "runs" / f"{Path(name).name}.json"
    if not f.exists():
        raise HTTPException(404, "not recorded: python -m skeptic.demo --record")
    return JSONResponse(json.loads(f.read_text(encoding="utf-8")))


@app.post("/api/upload")
def upload(body: dict):
    """Profile an uploaded CSV (sent as text) and suggest a mapping. Nothing is stored."""
    df = pd.read_csv(io.StringIO(body["csv"]))
    if len(df) > 200_000:
        raise HTTPException(413, "file too large for the demo (200k rows max)")
    cols = []
    for c in df.columns:
        cols.append({"name": c, "dtype": str(df[c].dtype), "n_unique": int(df[c].nunique()), "missing": int(df[c].isna().sum()),
                     "binary": bool(df[c].nunique() == 2), "numeric": bool(df[c].dtype.kind in "if")})
    binary = [c["name"] for c in cols if c["binary"]]
    numeric = [c["name"] for c in cols if c["numeric"] and not c["binary"]]
    claim = (body.get("claim") or "").lower()
    treat = next((c for c in binary if c.lower() in claim), binary[0] if binary else None)
    outcome = next((c for c in numeric if c.lower() in claim), numeric[0] if numeric else None)
    return {"rows": len(df), "columns": cols, "suggest": {"treat": treat, "outcome": outcome,
            "subgroups": [c["name"] for c in cols if c["name"] not in (treat, outcome) and (not c["numeric"] or c["n_unique"] <= 8)][:6]},
            "preview": json.loads(df.head(5).to_json(orient="records"))}


_NULLLAB: dict = {}


@app.post("/api/nullrerun")
def null_rerun(body: dict):
    """Re-run the null-world audit on the synthetic benchmark with a chosen number of worlds (the UI's live slider)."""
    import time
    import numpy as np
    from .adapters.synthetic import SynthConfig, generate
    from .court import null_worlds as NW, stats as S
    n = int(max(50, min(2000, body.get("n_worlds", 300))))
    if "data" not in _NULLLAB:
        _NULLLAB["data"] = generate(SynthConfig(seed=int(body.get("seed", 0))))
    d = _NULLLAB["data"]
    t0 = time.time()
    r = NW.run_null_worlds(d.pnl, d.factors, S.raw_t, "stationary_bootstrap", n, seed=7)
    obs = np.abs(r.observed)
    planted = [i for i in range(len(d.meta)) if d.meta[i]["kind"] == "planted"]
    hi = float(max(r.null_max.max(), obs[planted].max())) * 1.05
    counts, edges = np.histogram(r.null_max, bins=30, range=(0, hi))
    return _clean({"n_worlds": n, "seconds": time.time() - t0,
                   "histogram": {"counts": counts.tolist(), "edges": edges.tolist(), "p95": float(np.quantile(r.null_max, 0.95))},
                   "planted": [{"id": d.meta[i]["id"], "stat": float(obs[i]), "p_adj": float(r.p_stepdown[i])} for i in planted]})


@app.get("/api/sample/health")
def sample_health():
    from fastapi.responses import PlainTextResponse
    from .adapters.claims_data import health_supplement
    df = health_supplement(1, 4000).df.drop(columns=["enrolled"])
    return PlainTextResponse(df.to_csv(index=False))


@app.get("/api/health")
def health():
    return {"ok": True, "frontend": DIST.exists()}


if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        f = DIST / path
        return FileResponse(f if f.is_file() else DIST / "index.html")
