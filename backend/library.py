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
    return sorted(_load().values(), key=lambda e: -e["last_seen"])
