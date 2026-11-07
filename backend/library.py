"""Research library: every paper/dataset the Librarian or the chat agent has actually looked at, and what it was used for.

Persisted as JSON (git-ignored) so the UI can show 'papers it has looked at / is referencing'. Entries are only ever
added from real live search results; nothing is generated or hardcoded.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

_LOCK = threading.Lock()


def _path() -> Path:
    return Path(os.environ.get("SKEPTIC_LIBRARY_PATH", Path(__file__).resolve().parents[1] / ".library" / "papers.json"))


def _load() -> dict:
    try:
        return json.loads(_path().read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001  missing/corrupt file -> empty library
        return {}


def record(items: list[dict], context: str) -> None:
    """Add/refresh entries. `items` need at least title and (url or id); `context` says why it was looked up."""
    now = time.time()
    with _LOCK:
        lib = _load()
        for it in items:
            key = it.get("url") or it.get("id") or it.get("title")
            if not key or not it.get("title"):
                continue
            e = lib.get(key) or {"first_seen": now, "uses": 0, "contexts": []}
            e.update({"title": it["title"], "authors": it.get("authors") or it.get("by") or "", "year": it.get("year"),
                      "venue": it.get("venue") or "", "url": it.get("url"), "source": it.get("source") or "",
                      "kind": it.get("kind", "paper"), "snippet": (it.get("snippet") or it.get("abstract") or it.get("claim") or "")[:300],
                      "last_seen": now})
            e["uses"] += 1
            ctx = context[:120]
            if it.get("used"):
                e["used_list"] = [u for u in e.get("used_list", []) if u["context"] != ctx][-5:] + [{"context": ctx, "text": it["used"]}]
            if ctx not in e["contexts"]:
                e["contexts"] = (e["contexts"] + [ctx])[-8:]
            lib[key] = e
        try:
            p = _path()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(lib), encoding="utf-8")
        except OSError:
            pass


def entries() -> list[dict]:
    """Newest first. Each entry carries `used`: paragraphs saying what was taken from the paper and for what. Entries saved before
    this was tracked get one derived from their stored abstract snippet and the question they were looked up for."""
    from .literature import enrich_abstract, used_paragraph
    allv = _load()
    missing = [e for e in allv.values() if not e.get("snippet") and e.get("kind", "paper") == "paper" and (e.get("url") or e.get("title")) and not e.get("abs_tried")]
    if missing:                                           # one-time back-fill of real abstracts for papers saved without one
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(6) as ex:
            got = list(ex.map(lambda e: enrich_abstract({"title": e["title"], "url": e.get("url"), "abstract": ""}, 6.0), missing[:20]))
        with _LOCK:
            lib = _load()
            for e, a in zip(missing[:20], got):
                for k, v in lib.items():
                    if v["title"] == e["title"]:
                        v["abs_tried"] = True
                        if a:
                            v["snippet"] = a[:300]
            try:
                _path().write_text(json.dumps(lib), encoding="utf-8")
            except OSError:
                pass
    out = []
    for e in sorted(_load().values(), key=lambda e: -e["last_seen"]):
        used = [u["text"] for u in e.get("used_list", []) if "What the paper says" in u["text"] or not e.get("snippet")]
        if not used:
            ctx = (e.get("contexts") or [""])[-1]
            used = [used_paragraph({"title": e["title"], "abstract": e.get("snippet", "")},
                                   (f"{ctx or 'a question you asked'}" if e.get("kind") == "paper" else f"finding data for: {ctx}"),
                                   [w for w in ctx.lower().split() if len(w) > 3])]
        out.append({**e, "used": used[-3:]})
    return out
