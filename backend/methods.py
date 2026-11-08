"""Method research: find papers whose MATHEMATICAL ALGORITHMS could make Skeptic's own statistics stronger.

Design:
  * The user cannot choose what is searched. The topics below are fixed by the system (they are the weak points of our own
    pipeline). The only user input is a free-text style for how the CONCLUSION is displayed.
  * Every paper comes from a live search (OpenAlex / Crossref / arXiv). Nothing is hardcoded or written by a model.
  * Equations are copied verbatim from the paper's own LaTeX (arXiv HTML `alttext`). For papers without a readable full text we
    show the abstract and the link, and say so: we never reconstruct a formula from memory.
  * The LLM (if any) only writes the conclusion, in the requested style, citing [S#]; unknown citations and URLs are stripped.
"""
from __future__ import annotations

import html
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

from . import literature as LIT
from .llm import LLM

# What we want to strengthen in OUR pipeline -> a keyword query. These are system decisions, not user input.
TOPICS = [
    {"id": "fdr", "title": "Controlling false discoveries when many ideas are tested", "component": "Storey q-values / evidence score",
     "query": "false discovery rate multiple hypothesis testing", "must": ["discovery", "multiple"]},
    {"id": "alpha_investing", "title": "Online testing: spending an error budget over time", "component": "Alpha-investing (credibility meter)",
     "query": "online false discovery rate alpha-investing", "must": ["online", "alpha-invest", "alpha invest"]},
    {"id": "selection_bias", "title": "Sharpe ratio inflated by selection among many backtests", "component": "Search-size correction / evidence score",
     "query": "backtest overfitting Sharpe ratio selection bias", "must": ["backtest", "sharpe"]},
    {"id": "reality_check", "title": "Bootstrap tests for the best of many strategies", "component": "Null-world audit",
     "query": "reality check superior predictive ability bootstrap data snooping", "must": ["bootstrap", "snooping", "predictive ability", "reality check"]},
    {"id": "dependent_bootstrap", "title": "Resampling dependent time series", "component": "Null generators (block / stationary bootstrap)",
     "query": "stationary bootstrap dependent time series resampling", "must": ["bootstrap"]},
    {"id": "sequential", "title": "Valid inference when you peek at results", "component": "Peeking Skeptic (A/B tests)",
     "query": "always valid p-values sequential testing continuous monitoring", "must": ["sequential", "always valid", "always-valid", "monitoring"]},
]

_CACHE: dict = {}
_TTL = 3600


def _arxiv_id(url: str | None, doi: str | None = None) -> str | None:
    d = re.search(r"10\.48550/arxiv\.([\w.\-/]+)", (doi or url or "").lower())
    if d:
        return d.group(1)
    m = re.search(r"arxiv\.org/(?:abs|pdf)/([\w.\-/]+?)(?:v\d+)?(?:\.pdf)?$", url or "")
    return m.group(1) if m else None


def equations(arxiv_id: str, limit: int = 5) -> list[dict]:
    """Display equations copied verbatim from the paper's own LaTeX (arXiv HTML `alttext`). [] if the paper has no HTML version."""
    def fetch():
        raw = LIT._get(f"https://arxiv.org/html/{arxiv_id}", 15).decode("utf-8", "ignore")
        out, seen = [], set()
        for m in re.finditer(r"<math[^>]*>", raw):
            tag = m.group(0)
            a = re.search(r'alttext="([^"]+)"', tag)
            if 'display="block"' not in tag or not a:
                continue
            tex = html.unescape(a.group(1)).strip()
            if 14 <= len(tex) <= 420 and tex not in seen:
                seen.add(tex)
                head = re.findall(r"<h[23][^>]*>(.*?)</h[23]>", raw[:m.start()], flags=re.S)
                sec = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", head[-1]))).strip()[:80] if head else ""
                out.append({"latex": tex, "section": sec})
        if not out:
            raise ValueError("no display equations")      # never cache an empty extraction
        return out
    try:
        found = LIT._cached("eq1", arxiv_id, fetch)
    except Exception:  # noqa: BLE001  no HTML version / network down: honest empty list
        return []
    def score(e):                                   # prefer definitions of the method over proofs, examples and prose-heavy lines
        t, sec = e["latex"], e["section"].lower()
        return ((2 if any(w in sec for w in ("method", "procedure", "algorithm", "setup", "definition", "problem", "proposed", "estimator")) else 0)
                + (1 if "=" in t or "\\leq" in t else 0) - 0.7 * t.count("\\text") -(1 if len(t) > 220 else 0) + (0.5 if len(t) >= 30 else 0))
    return sorted(found, key=lambda e: -score(e))[:limit]


def _topic(t: dict) -> dict:
    papers = []
    try:
        live, notes = LIT.live_search(t["query"], 8)
    except Exception as e:  # noqa: BLE001
        live, notes = [], [f"search failed ({type(e).__name__})"]
    terms = LIT._terms(t["query"])
    scored = []
    for p in live:
        blob = (p["title"] + " " + (p.get("abstract") or "")).lower()
        if any(m in blob for m in t["must"]):
            scored.append((LIT._live_relevance(p, terms), p))
    scored.sort(key=lambda x: (-x[0], -(x[1].get("cited_by") or 0)))
    for rel, p in [x for x in scored if x[0] >= 2][:3]:
        p = {**p, "abstract": LIT.enrich_abstract(p)}
        aid = _arxiv_id(p.get("url"), p.get("doi"))
        eq = equations(aid) if aid else []
        note = "" if eq else ("No readable full text with equations was available: open the paper." if aid
                              else "Not on arXiv, so equations are not extracted: open the paper.")
        papers.append({"title": re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", p["title"])).strip(), "authors": p.get("authors", ""), "year": p.get("year"), "venue": p.get("venue", ""),
                       "url": p.get("url"), "source": p.get("source", ""), "abstract": (p.get("abstract") or "")[:500],
                       "cited_by": p.get("cited_by"), "equations": eq, "equation_note": note})
    from .literature import used_paragraph
    for p in papers:
        p["used"] = used_paragraph(p, f"strengthening Skeptic's {t['component']}", t["title"].lower().split())
    try:
        from . import library
        library.record([{**p, "kind": "paper"} for p in papers], f"Improve the algorithm: {t['component']}")
    except Exception:  # noqa: BLE001  bookkeeping only
        pass
    return {"id": t["id"], "title": t["title"], "component": t["component"], "papers": papers, "status": notes}


def research() -> dict:
    now = time.time()
    if _CACHE.get("t", 0) > now - _TTL and _CACHE.get("data"):
        return _CACHE["data"]
    if os.environ.get("SKEPTIC_OFFLINE") == "1":
        return {"topics": [{**{k: t[k] for k in ("id", "title", "component")}, "papers": [], "status": ["offline (SKEPTIC_OFFLINE=1): nothing searched"]}
                           for t in TOPICS]}
    with ThreadPoolExecutor(max_workers=3) as ex:
        topics = list(ex.map(_topic, TOPICS))
    data = {"topics": topics}
    if any(t["papers"] for t in topics):
        _CACHE.update(t=now, data=data)

    return data


def _sources(data: dict) -> list[dict]:
    return [p for t in data["topics"] for p in t["papers"]]


SYSTEM = ("You write the CONCLUSION of a literature scan for Skeptic's developers. The scan found papers whose mathematical methods might "
          "strengthen Skeptic's own statistics. You get a numbered SOURCES list (real search results) and the DISPLAY STYLE the reader wants. "
          "Rules: use ONLY the sources; cite them as [S1], [S2]; never invent a paper, result, number, equation or link; do not write URLs or LaTeX; "
          "say which Skeptic component (the 'improves=' field) each source could strengthen and what its abstract says the method does. "
          "The style only changes presentation, never the facts. Reply with JSON only: {\"text\": \"...\"}.")
SCHEMA = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}


def conclusion(data: dict, style: str, llm: LLM | None = None) -> dict:
    style = (style or "short bullet points, one per Skeptic component").strip()[:300]
    srcs = _sources(data)
    if not srcs:
        return {"text": "No relevant papers were found by the live search (offline, or the sources were unreachable), so there is nothing to conclude.",
                "llm": "none", "style": style}
    llm = llm or LLM()
    if llm.is_mock:
        lines, n = [], 0
        for t in data["topics"]:
            idx = list(range(n + 1, n + len(t["papers"]) + 1))
            n += len(t["papers"])
            if idx:
                lines.append(f"• {t['component']}: {len(idx)} candidate paper(s) on “{t['title'].lower()}” " + " ".join(f"[S{i}]" for i in idx))
        return {"text": "\n".join(lines) + f"\n\n(No AI model is configured, so this is a plain list, not an analysis. Requested style: {style})",
                "llm": "offline (no model)", "style": style}
    digest, n = [], 0
    for t in data["topics"]:
        for p in t["papers"]:
            n += 1
            digest.append(f"[S{n}] improves={t['component']} | {p['title']} ({p['year']}) | equations_extracted={len(p['equations'])} | {p['abstract'][:240]}")
    try:
        obj, _ = llm.complete_json("conclusion", SYSTEM, f"DISPLAY STYLE: {style}\n\nSOURCES:\n" + "\n".join(digest), SCHEMA, max_tokens=900)
        txt = re.sub(r"\[S(\d+)\]", lambda m: m.group(0) if 1 <= int(m.group(1)) <= n else "", str(obj.get("text", "")))
        txt = re.sub(r"https?://\S+", "", txt).strip()
        return {"text": txt, "llm": llm.provider.name, "style": style}
    except Exception as e:  # noqa: BLE001
        return {"text": f"The AI model could not be reached ({type(e).__name__}). The papers below are real search results that I have not analysed.",
                "llm": "error", "style": style}
