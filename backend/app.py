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
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:8000", "http://127.0.0.1:8000", "http://localhost:5173", "http://127.0.0.1:5173"], allow_methods=["*"], allow_headers=["*"])

RUNS: dict[str, dict] = {}
MAX_RUNS = 40                     # oldest runs are evicted
MAX_WORLDS = 1000                 # clamp on caller-supplied null-world counts (CPU protection)
MAX_CSV_CHARS = 20_000_000


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
    while len(RUNS) > MAX_RUNS:
        RUNS.pop(next(iter(RUNS)), None)

    def emit(ev):
        st["events"].append(_clean(ev))

    def work():
        try:
            params = body.get("params") or {}
            if not isinstance(params, dict):
                raise ValueError("params must be an object")
            params = dict(params)
            if "n_worlds" in params:
                params["n_worlds"] = int(max(50, min(MAX_WORLDS, int(params["n_worlds"]))))
            run_scenario(name, params, emit, int(body.get("seed", 0)))
        except Exception as e:  # surfaced to the UI instead of a silent empty run
            emit({"type": "error", "message": f"{type(e).__name__}: {e}"})
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


@app.get("/api/library")
def library_list():
    """Papers/datasets looked at so far. All come from live searches; there is no built-in list."""
    from . import library
    return {"looked_at": library.entries()}


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
    csv = body.get("csv") if isinstance(body, dict) else None
    if not isinstance(csv, str) or not csv.strip():
        raise HTTPException(400, "body must be {\"csv\": \"<csv text>\"}")
    if len(csv) > MAX_CSV_CHARS:
        raise HTTPException(413, "file too large for the demo")
    try:
        df = pd.read_csv(io.StringIO(csv))
    except Exception as e:
        raise HTTPException(400, f"could not parse CSV: {e}")
    if len(df) == 0 or df.shape[1] == 0:
        raise HTTPException(400, "CSV has no data rows")
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


@app.post("/api/chat")
def chat(body: dict):
    """Research assistant: finds papers and datasets live and explains them. Sources come only from real API results."""
    from .chat import run_chat
    msgs = body.get("messages") if isinstance(body, dict) else None
    if not isinstance(msgs, list) or not msgs or not all(isinstance(m, dict) and isinstance(m.get("content"), str) for m in msgs):
        raise HTTPException(400, "body must be {\"messages\": [{\"role\": \"user\", \"content\": \"...\"}]}")
    try:
        return _clean(run_chat(msgs))
    except ValueError as e:
        raise HTTPException(400, str(e))


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
        root = DIST.resolve()
        try:
            f = (root / path).resolve()
            inside = f.is_file() and (f == root or root in f.parents)
        except (OSError, ValueError):
            inside = False
        return FileResponse(f if inside else root / "index.html")
