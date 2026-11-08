"""Librarian: finds research papers for a hypothesis ON THE INTERNET, checks whether they APPLY to the user's current scenario,
and feeds that evidence to the Prosecutors.

There are NO hardcoded papers. Every paper is a real result from a live search of OpenAlex (needs OPENALEX_API_KEY),
Crossref and arXiv, with a DOI/arXiv link. A source that is down contributes nothing and the status is reported. With
SKEPTIC_OFFLINE=1 (or no network) the Librarian simply finds nothing and says so.

Applicability (zero tokens) uses only what the search APIs return (publication year, venue, citations): whether all of the
user's data post-dates the paper, and that its sample period/universe are NOT known and must be read in the paper itself.
The decay haircuts are heuristics, not citations.
"""
from __future__ import annotations

import hashlib
import html
import json
import os
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

CACHE = Path(__file__).resolve().parents[1] / ".lit_cache"



@dataclass
class Scenario:
    """The user's CURRENT case: what data, which period, which universe."""
    asset_class: str = "US equities"
    start_year: int = 2018
    end_year: int = 2025
    universe: str = "simulated stand-in"
    n_assets: int = 150
    survivor_universe: bool = False
    frequency: str = "daily"
    cost_bps: float = 5.0

    @classmethod
    def from_meta(cls, meta: dict, dates=None, n_assets: int = 150):
        sy, ey = (int(dates[0].year), int(dates[-1].year)) if dates is not None and len(dates) else (2018, 2025)
        return cls(start_year=sy, end_year=ey, n_assets=n_assets, universe=meta.get("source", "unknown"),
                   survivor_universe=bool(meta.get("survivorship_bias")))


_STOP = set("the a an of and or to in on for with by from that this these those is are was were be been it its as at stocks stock "
            "tend returns return next past over their than more less which when what who how not no can will would should "
            "does do did still work works working find found papers paper dataset datasets data evidence use used using about there any some "
            "i me my we you your show give tell please want need could may might much many very also just like".split())
FAMILY_QUERY = {"momentum": "momentum stock returns winners losers", "mean_reversion": "short-term reversal stock returns",
                "volatility": "low volatility anomaly stock returns", "fundamentals": "value book-to-market stock returns",
                "volume_sentiment": "trading volume stock returns predictability", "subgroup": "peeking A/B testing false positives"}
_FIN = ("stock", "equit", "return", "asset pric", "portfolio", "trading", "investor", "anomal", "factor", "securities", "market",
        "a/b", "experiment", "multiple testing", "backtest", "sharpe", "volatil", "momentum", "reversal")
UA = "skeptic-research-demo/1.0 (academic literature check)"


_DOMAIN = {"stock", "stocks", "returns", "return"}      # stop-words for finance-anchored claim queries, but real content in a free question


def _terms(q: str, keep_domain: bool = False) -> list[str]:
    out = []
    for w in re.findall(r"[a-z][a-z\-]{2,}", q.lower()):
        if (w not in _STOP or (keep_domain and w in _DOMAIN)) and w not in out:
            out.append(w)
    return out


def _get(url: str, timeout: float) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _cached(kind: str, query: str, fn):
    """Disk cache of REAL API responses only (a failed call is never cached)."""
    key = hashlib.sha1(f"{kind}|{query}".encode()).hexdigest()[:16]
    f = CACHE / f"{kind}_{key}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    out = fn()
    CACHE.mkdir(exist_ok=True)
    f.write_text(json.dumps(out), encoding="utf-8")
    return out


def crossref(query: str, rows: int = 5, timeout: float = 8.0) -> list[dict]:
    """Live Crossref search. Every field comes from the API response; raises on network failure."""
    def fetch():
        url = "https://api.crossref.org/works?" + urllib.parse.urlencode(
            {"query.bibliographic": query, "rows": rows, "select": "title,author,issued,DOI,container-title,abstract,URL",
             "filter": "type:journal-article"})
        items = json.loads(_get(url, timeout).decode())["message"]["items"]
        out = []
        for it in items:
            yr = (it.get("issued", {}).get("date-parts") or [[None]])[0][0]
            au = ", ".join(re.sub(r"<[^>]+>", "", a.get("family", "")) for a in it.get("author", [])[:3]) + (" et al." if len(it.get("author", [])) > 3 else "")
            out.append({"title": html.unescape(html.unescape((it.get("title") or [""])[0])), "authors": au, "year": yr, "doi": it.get("DOI"),
                        "url": f"https://doi.org/{it['DOI']}" if it.get("DOI") else it.get("URL"),
                        "venue": (it.get("container-title") or [""])[0],
                        "abstract": re.sub(r"<[^>]+>", " ", it.get("abstract", "") or "").strip()[:600], "source": "Crossref (live)"})
        return out
    return _cached("crossref", f"{query}|{rows}", fetch)


def arxiv(query: str, rows: int = 5, timeout: float = 8.0) -> list[dict]:
    """Live arXiv search restricted to quantitative finance + statistics. Raises on network failure."""
    def fetch():
        import xml.etree.ElementTree as ET
        terms = " AND ".join(f"all:{t}" for t in _terms(query)[:6])
        url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(
            {"search_query": f"({terms}) AND (cat:q-fin.* OR cat:stat.*)", "max_results": rows, "sortBy": "relevance"})
        ns = {"a": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(_get(url, timeout))
        out = []
        for e in root.findall("a:entry", ns):
            g = lambda tag: (e.findtext(f"a:{tag}", default="", namespaces=ns) or "").strip()
            names = [n.findtext("a:name", default="", namespaces=ns) for n in e.findall("a:author", ns)]
            au = ", ".join(n.split()[-1] for n in names[:3]) + (" et al." if len(names) > 3 else "")
            out.append({"title": re.sub(r"\s+", " ", g("title")), "authors": au, "year": int(g("published")[:4] or 0) or None, "doi": None,
                        "url": g("id"), "venue": "arXiv preprint (not peer reviewed)", "abstract": re.sub(r"\s+", " ", g("summary"))[:600],
                        "source": "arXiv (live)"})
        return out
    return _cached("arxiv", f"{query}|{rows}", fetch)


def _abstract(inv: dict | None) -> str:
    """OpenAlex stores abstracts as an inverted index {word: [positions]}; rebuild the text."""
    if not inv:
        return ""
    pos = {}
    for w, idxs in inv.items():
        for i in idxs:
            pos[i] = w
    return " ".join(pos[i] for i in sorted(pos))


def openalex(query: str, rows: int = 5, timeout: float = 10.0) -> list[dict]:
    """Live OpenAlex search (title, authors, year, DOI, venue, abstract, citation count). Needs OPENALEX_API_KEY."""
    key = os.environ.get("OPENALEX_API_KEY")           # optional: OpenAlex also answers keyless (lower rate limit)

    def fetch():
        url = "https://api.openalex.org/works?" + urllib.parse.urlencode(
            {"search": query, "per-page": rows, **({"api_key": key} if key else {}),
             "select": "title,authorships,publication_year,doi,primary_location,abstract_inverted_index,cited_by_count,id,type"})
        out = []
        for it in json.loads(_get(url, timeout).decode())["results"]:
            names = [a["author"]["display_name"] for a in it.get("authorships", [])]
            au = ", ".join(n.split()[-1] for n in names[:3]) + (" et al." if len(names) > 3 else "")
            out.append({"title": html.unescape(it.get("title") or ""), "authors": au, "year": it.get("publication_year"), "doi": it.get("doi"),
                        "url": it.get("doi") or it.get("id"), "venue": ((it.get("primary_location") or {}).get("source") or {}).get("display_name") or "",
                        "abstract": _abstract(it.get("abstract_inverted_index"))[:600], "cited_by": it.get("cited_by_count"),
                        "type": it.get("type"), "source": "OpenAlex (live)"})
        return out
    return _cached("openalex2", f"{query}|{rows}", fetch)


def live_search(query: str, rows: int = 5) -> tuple[list[dict], list[str]]:
    """Query all live sources in parallel. Returns (papers, per-source status notes). Never fabricates: a source that
    fails contributes nothing and is reported in the notes."""
    from concurrent.futures import ThreadPoolExecutor
    srcs = {"Crossref": crossref, "arXiv": arxiv, "OpenAlex": openalex}
    papers, notes = [], []
    with ThreadPoolExecutor(3) as ex:
        futs = {n: ex.submit(fn, query, rows) for n, fn in srcs.items()}
    for n, f in futs.items():
        try:
            res = f.result()
            papers += res
            notes.append(f"{n}: {len(res)} results")
        except Exception as e:  # noqa: BLE001  (one source failing must not stop the others)
            code = getattr(e, "code", None)
            notes.append(f"{n}: unavailable ({'HTTP ' + str(code) if code else str(e)[:60] or type(e).__name__})")
    seen, uniq = set(), []
    for p in papers:
        k = re.sub(r"\W+", "", p["title"].lower())[:60]
        if p["title"] and k not in seen:
            seen.add(k)
            uniq.append(p)
    return uniq, notes


def _live_relevance(p: dict, terms: list[str]) -> float:
    title, abst = p["title"].lower(), (p.get("abstract") or "").lower()
    return sum(2.0 for t in terms if t in title) + sum(1.0 for t in terms if t in abst and t not in title)


def assess_live(p: dict, sc: Scenario) -> dict:
    """Applicability of a LIVE paper from what the API actually tells us (publication year). Its sample period is NOT known."""
    cav = ["Sample period and universe not verified: read the paper (" + (p.get("url") or "no link") + ")."]
    haircut = 0.0
    if p.get("source", "").startswith("arXiv"):
        cav.append("Preprint, not peer reviewed.")
    yr = p.get("year")
    if yr:
        if sc.start_year > yr:
            cav.append(f"All of your data ({sc.start_year}-{sc.end_year}) is after this paper appeared ({yr}): published effects often weaken "
                       "after publication (assumed ~50% here: a heuristic, not a citation).")
            haircut = 0.5
        elif sc.start_year < yr < sc.end_year:
            cav.append(f"Published in {yr}, inside your sample: compare before vs after.")
            haircut = 0.30
    return {"applicability": None, "expected_haircut": haircut, "caveats": cav}


@dataclass
class Librarian:
    scenario: Scenario
    online: bool = field(default_factory=lambda: os.environ.get("SKEPTIC_OFFLINE") != "1")
    cache: dict = field(default_factory=dict)
    status: list = field(default_factory=list)       # per-source live-search notes, surfaced in the UI
    extra: list = field(default_factory=list)        # papers the user's chat already found (real search results), offered to every claim

    def find(self, text: str, family: str | None = None, top: int = 3) -> list[dict]:
        key = (text, family)
        if key in self.cache:
            return self.cache[key]
        found = []
        if self.online:
            query = FAMILY_QUERY.get(family or "", "") or " ".join(_terms(text)[:8])
            if not any(f in query.lower() for f in _FIN):
                query += " stock returns"                      # anchor to finance so common words ("rebound", "fell") don't match medicine
            terms = _terms(query)
            live, notes = live_search(query, 8)
            self.status = notes
            on_topic = [p for p in live if any(f in (p["title"] + " " + (p.get("abstract") or "")).lower() for f in _FIN)]
            ranked = sorted(((_live_relevance(p, terms), p) for p in on_topic), key=lambda x: (-x[0], -(x[1].get("cited_by") or 0)))   # ties: more-cited first
            for rel, p in [x for x in ranked if x[0] >= 2][:top]:
                p = {**p, "abstract": enrich_abstract(p)}
                found.append({**p, "id": p.get("doi") or p["url"], "claim": (p.get("abstract") or "")[:240] or "(no abstract returned: open the link)",
                              "sample": None, "universe": None, "relevance": round(rel, 1), **assess_live(p, self.scenario)})
                found[-1]["used"] = used_paragraph(p, self._use_text(found[-1]), terms)
        if self.extra:                                       # papers from the user's chat: judged by the same relevance rule, no finance filter
            query = FAMILY_QUERY.get(family or "", "") or " ".join(_terms(text)[:8])
            terms = _terms(query)
            have = {(p.get("url") or p["title"]).lower() for p in found}
            for p in self.extra:
                if (p.get("url") or p["title"]).lower() in have:
                    continue
                rel = _live_relevance(p, terms)
                if rel >= 2:
                    found.append({**p, "id": p.get("url") or p["title"], "from_chat": True,
                                  "claim": (p.get("abstract") or "")[:240] or "(no abstract: open the link)", "sample": None, "universe": None,
                                  "relevance": round(rel, 1), **assess_live(p, self.scenario)})
                    found[-1]["used"] = used_paragraph(p, self._use_text(found[-1]), terms)
            found.sort(key=lambda p: -p["relevance"])
            found = found[:top + 2]
            self.status = self.status + [f"{len(self.extra)} paper(s) carried over from your chat"]
        if not self.online and not self.extra:
            self.status = ["offline (SKEPTIC_OFFLINE=1): no papers searched"]
        self.cache[key] = found
        try:
            from . import library
            library.record([{**p, "kind": "paper"} for p in found], f"{family or 'claim'}: {text.strip()[:80]}")
        except Exception:  # noqa: BLE001  the library is bookkeeping; never break a run
            pass
        return found

    def _use_text(self, f: dict) -> str:
        sc = self.scenario
        hc = f.get("expected_haircut") or 0
        return (f"the Literature Skeptic: to check whether this idea is known and whether its published result still applies to your data "
                f"({sc.start_year}-{sc.end_year}, {sc.universe})" + (f"; it implies an expected {hc:.0%} weaker effect than published" if hc else ""))

    def evidence(self, hyp: dict) -> dict:
        """Literature evidence for one hypothesis: supporting papers, applicability, implied haircut, and a prior suggestion."""
        text = f"{hyp.get('description', '')} {hyp.get('spec', {}).get('formula', '') or ''} {hyp.get('spec', {}).get('rule', '') or ''}"
        papers = self.find(text, hyp.get("family"))
        direct = [p for p in papers if p["relevance"] >= 4.0]
        return {"papers": papers, "has_direct_support": bool(direct),
                "expected_haircut": max([p["expected_haircut"] for p in direct], default=0.0),
                "prior_suggestion": 0.8 if direct else 0.3,
                "caveats": sorted({c for p in direct for c in p["caveats"]})[:5]}


def literature_attack_factory(evidence: dict[str, dict]):
    """Prosecutor input: does the literature (a) support this idea, (b) make the observed edge plausible for TODAY's scenario?
    Advisory (medium): the literature can raise suspicion but cannot kill a finding on its own."""
    def literature(search, k, cfg):
        import numpy as np
        h = search.hyps[k]
        ev = evidence.get(h["id"], {})
        sr = float(search.pnl[k].mean() / (search.pnl[k].std(ddof=1) + 1e-12) * np.sqrt(252)) if search.pnl is not None else 0.0
        notes, ok = [], True
        if not ev.get("has_direct_support"):
            notes.append("No published support found for this construction: treat as data-mined unless there is an economic rationale.")
        hc = ev.get("expected_haircut", 0.0)
        # heuristic (not a citation): published anomaly long-short Sharpes are typically below ~1 and fall after publication
        plausible_max = 1.0 * (1 - hc) * 2.5 if ev.get("has_direct_support") else 1.0
        if sr > plausible_max * 2.0 and sr > 1.5:
            ok = False
            notes.append(f"Observed Sharpe {sr:.1f} is far above what the literature makes plausible for this idea today (~{plausible_max:.1f}). "
                         "Too good to be true: look for leakage or a data artefact.")
        return {"attack": "literature", "severity": "medium", "survived": ok, "evidence": {
            "observed_sharpe": sr, "papers": [{k2: p[k2] for k2 in ("authors", "year", "title", "applicability", "url", "source", "used") if k2 in p} for p in ev.get("papers", [])[:3]],
            "expected_post_publication_haircut": hc, "caveats": ev.get("caveats", []), "notes": notes}}
    return literature


# ---------------------------------------------------------------- Methods Scout: papers that improve the CRITIQUING itself
METHOD_QUERIES = {
    "multiple testing / data snooping": ("data snooping reality check multiple testing backtest", ("snooping", "multiple testing", "false discovery", "reality check", "backtest")),
    "leakage / lookahead": ("look-ahead bias data leakage backtest overfitting", ("look-ahead", "lookahead", "leakage", "overfitting", "backtest")),
    "survivorship": ("survivorship bias stock returns database", ("survivorship", "survivor")),
    "costs / capacity": ("transaction costs anomalies trading profitability", ("transaction cost", "trading cost", "anomal")),
    "factors / beta": ("factor model alpha anomalies cross-section of returns", ("factor", "anomal", "cross-section")),
    "post-publication decay": ("anomaly returns after publication decay", ("publication", "anomal", "replicat")),
    "peeking / sequential testing": ("optional stopping sequential testing false positives", ("sequential", "stopping", "peeking", "false positive")),
}


def validate_method_paper(p: dict, must: tuple[str, ...], min_cites: int = 30) -> dict | None:
    """A methods paper counts as VALID only if it passes checks computed from the API record (we cannot verify its maths):
    peer-reviewed article (not a preprint), has an abstract, on-topic, and is established (citations). Returns the checks, or None."""
    text = (p["title"] + " " + (p.get("abstract") or "")).lower()
    checks = {"peer_reviewed": (p.get("type") == "article") and bool(p.get("venue")), "has_abstract": len(p.get("abstract") or "") > 80,
              "on_topic": any(m in text for m in must), "established": (p.get("cited_by") or 0) >= min_cites}
    return checks if all(checks.values()) else None


def methods_scout(online: bool = True, per_attack: int = 2) -> dict:
    """OpenAlex search for papers that could sharpen each Skeptic. Returns {"papers": {area: [paper+checks]}, "status": [...]}.
    Never fabricates: offline or on failure it returns nothing and says so."""
    out, status = {}, []
    if not online:
        return {"papers": {}, "status": ["offline: methods scout not run"]}
    for area, (q, must) in METHOD_QUERIES.items():
        try:
            found = []
            for p in openalex(q, 10):
                p = {**p, "abstract": enrich_abstract(p)}
                chk = validate_method_paper(p, must)
                if chk:
                    found.append({k: p.get(k) for k in ("title", "authors", "year", "venue", "url", "doi", "cited_by", "source")} | {"checks": chk}
                             | {"used": used_paragraph(p, METHOD_USE.get(area, "a Skeptic check"), list(must))})
            found.sort(key=lambda x: -(x["cited_by"] or 0))
            out[area] = found[:per_attack]
            try:
                from . import library
                library.record([{**p, "kind": "paper"} for p in out[area]], f"Methods scout: {area}")
            except Exception:  # noqa: BLE001  bookkeeping only
                pass
            status.append(f"{area}: {len(out[area])} validated of 10 returned")
        except Exception as e:  # noqa: BLE001
            status.append(f"{area}: OpenAlex unavailable ({str(e)[:50] or type(e).__name__})")
    return {"papers": out, "status": status}


# ---------------------------------------------------------------- "what did we use from this paper?"
METHOD_USE = {
    "multiple testing / data snooping": "the null-world audit: the whole search is re-run in worlds where nothing is real, and each result is compared with the best that luck alone produced",
    "leakage / lookahead": "the Leakage Skeptic: it scans a feature for lookahead and checks whether next-day predictive power is implausibly high",
    "survivorship": "the Survivorship Skeptic: it re-runs an idea on the universe that existed at the time, including names that later disappeared",
    "costs / capacity": "the Costs Skeptic: it re-prices a strategy after realistic trading costs",
    "factors / beta": "the Confounder Skeptic: it checks whether an apparent edge is just exposure to known factors",
    "post-publication decay": "the Literature Skeptic: it applies a haircut when an effect was published before or inside your sample",
    "peeking / sequential testing": "the Peeking Skeptic: it judges an early look against the best result across all looks",
}
_STOP = {"the", "and", "for", "with", "that", "this", "from", "are", "was", "were", "its", "into", "our", "than", "have", "has"}


def enrich_abstract(p: dict, timeout: float = 8.0) -> str:
    """Crossref often returns no abstract. Look the paper up in OpenAlex (by DOI, else by exact title) and return its REAL abstract, or ''."""
    if (p.get("abstract") or "").strip():
        return p["abstract"]
    doi = (p.get("doi") or p.get("url") or "")
    m = re.search(r"10\.\d{4,9}/\S+", doi)
    title = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", p.get("title") or "")).strip()
    key = os.environ.get("OPENALEX_API_KEY")
    kq = {"api_key": key} if key else {}

    def fetch():
        if m:
            url = f"https://api.openalex.org/works/doi:{urllib.parse.quote(m.group(0), safe='/')}?" + urllib.parse.urlencode({**kq, "select": "abstract_inverted_index"})
            return _abstract(json.loads(_get(url, timeout).decode()).get("abstract_inverted_index"))[:1500]
        url = "https://api.openalex.org/works?" + urllib.parse.urlencode({**kq, "search": title, "per-page": 1, "select": "title,abstract_inverted_index"})
        rs = json.loads(_get(url, timeout).decode())["results"]
        same = rs and re.sub(r"\W+", "", (rs[0].get("title") or "").lower()) == re.sub(r"\W+", "", title.lower())
        return _abstract(rs[0].get("abstract_inverted_index"))[:1500] if same else ""
    try:
        return _cached("abstract", (m.group(0) if m else title), fetch)
    except Exception:  # noqa: BLE001  network down / not found: the paragraph will say no abstract was available
        return ""


def _quote(p: dict, terms: list[str]) -> str:
    """The sentence of the paper's own abstract that best matches what we used it for (verbatim, trimmed)."""
    abst = (p.get("abstract") or "").strip()
    if not abst:
        return ""
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", abst) if len(s.strip()) > 30]
    if not sents:
        return abst[:220]
    terms = [t for t in terms if t not in _STOP and len(t) > 3]
    ranked = sorted(range(len(sents)), key=lambda i: (-sum(t in sents[i].lower() for t in terms), i))[:2]
    text = " ".join(sents[i] for i in sorted(ranked))                  # the two best-matching sentences, in the paper's own order
    return text[:420] + ("..." if len(text) > 420 else "")


def used_paragraph(p: dict, use: str, terms: list[str] | None = None) -> str:
    """A short, honest paragraph: what part of the paper was used, and for what. We only read the abstract and metadata, never the full text."""
    q = _quote(p, terms or _terms(use))
    src = f'What the paper says (its abstract): "{q}" ' if q else "No abstract could be retrieved for this paper, so only its title, year and venue were used. "
    return f"{src}How we use it: {use}. (We read the abstract only, not the full text.)"
