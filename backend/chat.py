"""Chat agent: answers questions by searching for research papers and datasets ON ITS OWN.

Design (anti-hallucination):
  * The LLM never writes a paper/dataset/link. It can only ask for two tools (search_papers, search_datasets); every source
    shown to the user comes straight from a live API response (Crossref, arXiv, OpenAlex, Hugging Face, Zenodo).
  * The model may cite sources only as [S1], [S2]... which refer to that list. Unknown citations and any URL it typed itself
    are stripped from the answer before it is returned.
  * With no LLM configured (or if the LLM call fails) the agent still runs both searches and returns the raw results,
    saying plainly that no model analysed them.
"""
from __future__ import annotations

import json
import re
import urllib.parse

from . import literature as LIT
from .llm import LLM

MAX_STEPS = 3
MAX_MSG_CHARS = 2000

SYSTEM = """You are SkepticR's research assistant. SkepticR audits whether a pattern in data is real or just luck.
You help users find research papers and datasets and explain them in plain, simple English (no jargon without a short explanation).
You work in steps. Reply with JSON only: {"action": "search_papers" | "search_datasets" | "answer", "query": "...", "answer": "..."}.
- Use search_papers / search_datasets to look things up (query = a short keyword search, not a sentence). Search before answering any
  question about papers, evidence or data. You may search several times, at most 3 steps in total.
- When you have enough, use action "answer". Cite sources ONLY as [S1], [S2] using the numbered SOURCES list you are given.
- NEVER invent a paper, dataset, author, number or link. Do not write URLs. If SOURCES has nothing relevant, say so honestly and suggest a better search.
- Be clear that finding a paper or dataset is not proof that a claim is true; SkepticR's "Find a conclusion" tab can test a claim on a CSV.
Keep answers short: a few sentences, then which sources to look at first and why."""

SCHEMA = {"type": "object", "properties": {"action": {"enum": ["search_papers", "search_datasets", "answer"]},
                                           "query": {"type": "string"}, "answer": {"type": "string"}}, "required": ["action"]}


# ----------------------------------------------------------------------------------------------- tools
def search_papers(query: str, rows: int = 5) -> tuple[list[dict], list[str]]:
    papers, notes = LIT.live_search(query, rows)
    terms = LIT._terms(query, True)
    scored = [(LIT._live_relevance(p, terms), p) for p in papers]
    # must mention at least half of the content words (title counts double), or it is just sharing filler words
    need = max(1, (len(terms) + 1) // 2)
    hit = lambda p: sum(t in (p["title"] + " " + (p.get("abstract") or "")).lower() for t in terms)
    keep = [p for r, p in sorted(scored, key=lambda x: -x[0]) if r > 0 and hit(p) >= need][:rows]
    return [{"kind": "paper", "title": p["title"], "by": p["authors"], "year": p["year"], "venue": p["venue"], "url": p["url"],
             "snippet": (p.get("abstract") or "")[:300], "source": p["source"]} for p in keep], notes


def _hf(query: str, rows: int, timeout: float = 8.0) -> list[dict]:
    def fetch():
        out, seen = [], set()
        words = LIT._terms(query, True)
        for q in [" ".join(words[:2]), words[0] if words else query, words[1] if len(words) > 1 else ""]:   # HF search wants short queries
            if not q:
                continue
            url = "https://huggingface.co/api/datasets?" + urllib.parse.urlencode({"search": q, "limit": rows, "sort": "downloads", "direction": -1})
            for d in json.loads(LIT._get(url, timeout).decode()):
                if d["id"] not in seen:
                    seen.add(d["id"])
                    out.append(d)
            if len(out) >= rows:
                break
        out = [{"kind": "dataset", "title": d["id"], "by": d.get("author") or "", "year": (d.get("lastModified") or "")[:4] or None,
                        "venue": f"{d.get('downloads', 0):,} downloads", "url": f"https://huggingface.co/datasets/{d['id']}",
                        "snippet": re.sub(r"\s+", " ", d.get("description") or "")[:300], "source": "Hugging Face (live)"} for d in out[:rows]]
        return out
    return LIT._cached("hf2", f"{query}|{rows}", fetch)


def _zenodo(query: str, rows: int, timeout: float = 8.0) -> list[dict]:
    def fetch():
        url = "https://zenodo.org/api/records?" + urllib.parse.urlencode({"q": " AND ".join(LIT._terms(query, True)[:4]) or query, "type": "dataset", "size": rows, "sort": "bestmatch"})
        out = []
        for h in json.loads(LIT._get(url, timeout).decode())["hits"]["hits"]:
            m = h.get("metadata", {})
            out.append({"kind": "dataset", "title": m.get("title", ""), "by": ", ".join(c.get("name", "") for c in m.get("creators", [])[:2]),
                        "year": (m.get("publication_date") or "")[:4] or None, "venue": "Zenodo", "url": h.get("doi_url") or h.get("links", {}).get("html"),
                        "snippet": re.sub(r"<[^>]+>|\s+", " ", m.get("description", "") or "").strip()[:300], "source": "Zenodo (live)"})
        return out
    return LIT._cached("zenodo2", f"{query}|{rows}", fetch)


def search_datasets(query: str, rows: int = 5) -> tuple[list[dict], list[str]]:
    from concurrent.futures import ThreadPoolExecutor
    out, notes = [], []
    with ThreadPoolExecutor(2) as ex:
        futs = {"Hugging Face": ex.submit(_hf, query, rows), "Zenodo": ex.submit(_zenodo, query, rows)}
    for n, f in futs.items():
        try:
            r = f.result()
            out += r
            notes.append(f"{n}: {len(r)} results")
        except Exception as e:  # noqa: BLE001  one source failing must not stop the other
            notes.append(f"{n}: unavailable ({type(e).__name__})")
    terms = LIT._terms(query, True)
    rel = lambda d: sum(t in (d["title"] + " " + d["snippet"]).lower() for t in terms)
    need = max(1, (len(terms) + 1) // 2)
    out = [d for d in out if d["url"] and d["title"] and rel(d) >= need]        # must match at least half the content words
    out.sort(key=lambda d: -rel(d))
    return out[:rows * 2], notes


TOOLS = {"search_papers": search_papers, "search_datasets": search_datasets}


# ----------------------------------------------------------------------------------------------- mock (no LLM configured)
def _mock(hint: dict) -> dict:
    step = hint["step"]
    if step == 0:
        return {"action": "search_papers", "query": " ".join(LIT._terms(hint["question"], True)[:6]) or hint["question"]}
    if step == 1:
        return {"action": "search_datasets", "query": " ".join(LIT._terms(hint["question"], True)[:6]) or hint["question"]}
    return {"action": "answer", "answer": ""}          # empty -> the caller writes the plain "raw results" message


def register_mock(llm: LLM):
    llm.register_mock("chat", _mock)


# ----------------------------------------------------------------------------------------------- agent loop
def _sources_text(sources: list[dict]) -> str:
    if not sources:
        return "SOURCES: (none yet)"
    return "SOURCES:\n" + "\n".join(f"[S{i + 1}] ({s['kind']}) {s['title']} - {s['by']} {s['year'] or ''} - {s['snippet'][:160]}" for i, s in enumerate(sources))


def _sanitize(text: str, n_sources: int) -> str:
    text = re.sub(r"\[S(\d+)\]", lambda m: m.group(0) if 1 <= int(m.group(1)) <= n_sources else "", text)
    return re.sub(r"https?://\S+", "(link removed: use the source list below)", text).strip()


def run_chat(messages: list[dict], llm: LLM | None = None) -> dict:
    messages = [{"role": m["role"], "content": str(m["content"])[:MAX_MSG_CHARS]} for m in messages[-10:] if m.get("role") in ("user", "assistant")]
    if not messages or messages[-1]["role"] != "user":
        raise ValueError("last message must be from the user")
    question = messages[-1]["content"]
    llm = llm or LLM()
    register_mock(llm)
    history = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages[:-1])[-3000:]
    sources: list[dict] = []
    seen: set[str] = set()
    steps, notes_all, answer, llm_error = [], [], "", None

    def add(results):
        for r in results:
            if r["url"] not in seen:
                seen.add(r["url"])
                sources.append(r)

    for step in range(MAX_STEPS + 1):
        prompt = f"{history}\nUSER QUESTION: {question}\n\n{_sources_text(sources)}\n\nStep {step + 1} of {MAX_STEPS + 1}."
        if step >= MAX_STEPS:
            prompt += " You must answer now."
        try:
            obj, _ = llm.complete_json("chat", SYSTEM, prompt, SCHEMA, mock_hint={"step": step, "question": question}, max_tokens=1200)
        except Exception as e:  # noqa: BLE001  bad key / endpoint: still give the user real search results
            llm_error = f"{type(e).__name__}: {e}"[:300]
            break
        act = obj.get("action")
        if act in TOOLS and step < MAX_STEPS:
            q = (obj.get("query") or question)[:200]
            try:
                res, notes = TOOLS[act](q)
            except Exception as e:  # noqa: BLE001
                res, notes = [], [f"{act} failed ({type(e).__name__})"]
            add(res)
            notes_all += notes
            steps.append({"tool": act, "query": q, "found": len(res), "status": notes})
            continue
        answer = _sanitize(obj.get("answer") or "", len(sources))
        break

    if llm_error:
        for act in TOOLS:                       # make sure the user still gets real results
            if not any(st["tool"] == act for st in steps):
                try:
                    res, notes = TOOLS[act](question)
                    add(res)
                    steps.append({"tool": act, "query": question, "found": len(res), "status": notes})
                except Exception:  # noqa: BLE001
                    pass
        answer = f"The AI model could not be reached ({llm_error}). Here are the raw search results instead; I have not analysed them."
    elif not answer:
        n_p, n_d = sum(s["kind"] == "paper" for s in sources), sum(s["kind"] == "dataset" for s in sources)
        answer = (f"I searched live and found {n_p} paper(s) and {n_d} dataset(s) for \"{question[:80]}\". " if sources else
                  f"I searched live but found nothing relevant for \"{question[:80]}\". Try different or fewer keywords. ") + \
                 ("No AI model is configured (set SKEPTIC_USE_LLM=1 and a key in .env), so this is the raw list, not an analysis." if llm.is_mock else "")
    try:
        from . import library
        library.record(sources, f"chat: {question[:80]}")
    except Exception:  # noqa: BLE001
        pass
    return {"answer": answer, "sources": sources, "steps": steps, "llm": "offline (no model)" if llm.is_mock else llm.provider.name,
            "error": llm_error}
