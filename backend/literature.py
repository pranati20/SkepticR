"""Librarian: finds research papers for a hypothesis, checks whether they APPLY to the user's current scenario, and
feeds that evidence to the Prosecutors.

Sources
  * a bundled index of well-known papers (works offline; deterministic). Entries carry only facts I am confident of:
    authors, year, venue, the claim, sample period/universe. Verify before citing: this is an index, not a literature review.
  * LIVE search (default on; SKEPTIC_OFFLINE=1 disables): Crossref + arXiv (+ OpenAlex if OPENALEX_API_KEY is set). Every live
    paper is a real API result with a DOI/arXiv link; nothing is generated. A source that is down contributes nothing and is
    reported; the curated index is then the only fallback and is labelled 'curated index, verify'.
LLM role 'librarian' (cheap model) is only used to summarise relevance when a real provider is configured.

Applicability checks (zero tokens) compare a paper with the CURRENT scenario: sample overlap (is our data out-of-sample?),
years since publication (McLean & Pontiff 2016: anomaly returns are ~26% lower out-of-sample and ~58% lower post-publication),
universe/size mismatch, asset class and frequency. The result is a literature-implied haircut and a list of caveats.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

CACHE = Path(__file__).resolve().parents[1] / ".lit_cache"

# year published, sample period of the original evidence (start, end), universe, plus what later work found
CORPUS = [
    {"id": "jt93", "authors": "Jegadeesh & Titman", "year": 1993, "title": "Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency",
     "venue": "Journal of Finance", "families": ["momentum"], "keys": ["momentum", "winners", "12-1", "trend"], "sample": (1965, 1989),
     "universe": "all NYSE/AMEX stocks", "claim": "Stocks that did well over the past 3-12 months keep outperforming over the next 3-12 months.",
     "later": "Momentum survives out of sample in many markets but suffers sharp crashes (e.g. 2009) and weaker returns in large caps."},
    {"id": "j90", "authors": "Jegadeesh", "year": 1990, "title": "Evidence of Predictable Behavior of Security Returns", "venue": "Journal of Finance",
     "families": ["mean_reversion"], "keys": ["reversal", "short-term", "short term", "losers", "mean reversion"], "sample": (1934, 1987),
     "universe": "NYSE/AMEX stocks", "claim": "Last month's losers outperform last month's winners the next month (short-term reversal).",
     "later": "Reversal profits are concentrated in small, illiquid stocks and are largely consumed by trading costs."},
    {"id": "l90", "authors": "Lehmann", "year": 1990, "title": "Fads, Martingales, and Market Efficiency", "venue": "Quarterly Journal of Economics",
     "families": ["mean_reversion"], "keys": ["reversal", "weekly", "fads"], "sample": (1962, 1986), "universe": "CRSP stocks",
     "claim": "Weekly returns reverse: last week's losers beat last week's winners.", "later": "Much of the profit disappears after transaction costs and liquidity effects."},
    {"id": "ahxz06", "authors": "Ang, Hodrick, Xing & Zhang", "year": 2006, "title": "The Cross-Section of Volatility and Expected Returns", "venue": "Journal of Finance",
     "families": ["volatility"], "keys": ["volatility", "low risk", "low-volatility", "idiosyncratic"], "sample": (1963, 2000), "universe": "US stocks",
     "claim": "Stocks with high idiosyncratic volatility have abysmally low subsequent returns.", "later": "The low-volatility effect is strong in small stocks and weaker in large caps; related to the lottery/leverage-constraint explanations."},
    {"id": "ff92", "authors": "Fama & French", "year": 1992, "title": "The Cross-Section of Expected Stock Returns", "venue": "Journal of Finance",
     "families": ["fundamentals"], "keys": ["value", "book-to-market", "book to market", "size"], "sample": (1963, 1990), "universe": "NYSE/AMEX/Nasdaq stocks",
     "claim": "Size and book-to-market explain the cross-section of average returns.", "later": "The value premium was weak for much of the 2010s; value is itself a standard risk factor, so a 'value' signal is beta, not new alpha."},
    {"id": "gkm01", "authors": "Gervais, Kaniel & Mingelgrin", "year": 2001, "title": "The High-Volume Return Premium", "venue": "Journal of Finance",
     "families": ["volume_sentiment"], "keys": ["volume", "high-volume"], "sample": (1963, 1996), "universe": "NYSE/AMEX stocks",
     "claim": "Stocks with unusually high trading volume tend to appreciate over the following month.", "later": "Evidence is mixed outside the original sample."},
    {"id": "fp14", "authors": "Frazzini & Pedersen", "year": 2014, "title": "Betting Against Beta", "venue": "Journal of Financial Economics",
     "families": ["volatility"], "keys": ["low beta", "low risk", "volatility"], "sample": (1926, 2012), "universe": "US and 20 international equity markets",
     "claim": "Low-beta assets earn higher risk-adjusted returns than high-beta assets.", "later": "Sensitive to leverage and funding assumptions."},
    {"id": "mp16", "authors": "McLean & Pontiff", "year": 2016, "title": "Does Academic Research Destroy Stock Return Predictability?", "venue": "Journal of Finance",
     "families": ["*"], "keys": ["decay", "publication", "replication", "anomaly", "out-of-sample"], "sample": (1926, 2013), "universe": "97 published anomalies",
     "claim": "Anomaly returns are about 26% lower out of sample and about 58% lower after publication.", "later": "Standard reference for post-publication decay."},
    {"id": "hlz16", "authors": "Harvey, Liu & Zhu", "year": 2016, "title": "...and the Cross-Section of Expected Returns", "venue": "Review of Financial Studies",
     "families": ["*"], "keys": ["multiple testing", "data mining", "t-statistic", "factor zoo", "luck"], "sample": (1967, 2014), "universe": "316 published factors",
     "claim": "After accounting for the many factors tried, a new factor needs a t-statistic above about 3.0.", "later": "The motivating result for auditing the whole search, not only the winner."},
    {"id": "hxz20", "authors": "Hou, Xue & Zhang", "year": 2020, "title": "Replicating Anomalies", "venue": "Review of Financial Studies",
     "families": ["*"], "keys": ["replication", "anomaly", "microcap", "failed"], "sample": (1967, 2014), "universe": "452 anomalies",
     "claim": "Most published anomalies fail to replicate once microcaps are controlled and stricter significance thresholds are used.", "later": ""},
    {"id": "bl14", "authors": "Bailey & Lopez de Prado", "year": 2014, "title": "The Deflated Sharpe Ratio", "venue": "Journal of Portfolio Management",
     "families": ["*"], "keys": ["sharpe", "backtest", "overfitting", "selection bias"], "sample": (0, 0), "universe": "method",
     "claim": "A Sharpe ratio must be deflated for the number of strategies tried and for non-normal returns.", "later": ""},
    {"id": "w00", "authors": "White", "year": 2000, "title": "A Reality Check for Data Snooping", "venue": "Econometrica", "families": ["*"],
     "keys": ["data snooping", "reality check", "bootstrap"], "sample": (0, 0), "universe": "method",
     "claim": "Bootstrap the best performance among all models tried to test whether the best one beats a benchmark.", "later": ""},
    {"id": "bickel75", "authors": "Bickel, Hammel & O'Connell", "year": 1975, "title": "Sex Bias in Graduate Admissions: Data from Berkeley", "venue": "Science",
     "families": ["subgroup"], "keys": ["simpson", "confound", "pooled", "stratif"], "sample": (0, 0), "universe": "admissions data",
     "claim": "A pooled association can reverse sign within every subgroup (Simpson's paradox).", "later": ""},
    {"id": "jo17", "authors": "Johari, Koomen, Pekelis & Walsh", "year": 2017, "title": "Peeking at A/B Tests: Why It Matters, and What to Do About It", "venue": "KDD",
     "families": ["subgroup"], "keys": ["peeking", "a/b", "experiment", "sequential", "optional stopping"], "sample": (0, 0), "universe": "online experiments",
     "claim": "Repeatedly checking significance and stopping early inflates false positives far above the nominal rate.", "later": ""},
]


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


def _score(text: str, entry: dict, family: str | None) -> float:
    t = text.lower()
    s = sum(2.0 for k in entry["keys"] if k in t)
    if family and (family in entry["families"]):
        s += 3.0
    if "*" in entry["families"]:
        s += 0.5 if s else 0.0
    return s


def assess(entry: dict, sc: Scenario) -> dict:
    """Does this paper's evidence apply to the current scenario? (deterministic, zero tokens)"""
    cav, haircut = [], 0.0
    s0, s1 = entry["sample"]
    if s1:
        if sc.start_year > s1:
            cav.append(f"Your data ({sc.start_year}-{sc.end_year}) is entirely after the paper's sample ({s0}-{s1}): a genuine out-of-sample test.")
            haircut = max(haircut, 0.26)
        if sc.start_year > entry["year"]:
            cav.append(f"All of your data is post-publication ({entry['year']}): expect a decay of roughly 58% in anomaly returns (McLean & Pontiff 2016).")
            haircut = max(haircut, 0.58)
        elif sc.end_year > entry["year"] > sc.start_year:
            cav.append(f"The paper appeared in {entry['year']}, inside your sample: compare before vs after.")
            haircut = max(haircut, 0.30)
    if "NYSE" in entry["universe"] or "CRSP" in entry["universe"] or "all" in entry["universe"]:
        if sc.n_assets < 500:
            cav.append(f"Original universe is broad ({entry['universe']}); yours has {sc.n_assets} names, so small-stock effects are mostly absent.")
    if sc.survivor_universe and entry["year"]:
        cav.append("Your universe is survivors only, which flatters anything that buys past losers or holds long.")
    if entry.get("later"):
        cav.append("Later work: " + entry["later"])
    applicability = 1.0 - min(0.8, 0.15 * len([c for c in cav if "Original universe" in c or "survivors" in c]) + haircut * 0.5)
    return {"applicability": round(applicability, 2), "expected_haircut": haircut, "caveats": cav}


_STOP = set("the a an of and or to in on for with by from that this these those is are was were be been it its as at stocks stock "
            "tend returns return next past over their than more less which when what who how not no can will would should".split())
FAMILY_QUERY = {"momentum": "momentum stock returns winners losers", "mean_reversion": "short-term reversal stock returns",
                "volatility": "low volatility anomaly stock returns", "fundamentals": "value book-to-market stock returns",
                "volume_sentiment": "trading volume stock returns predictability", "subgroup": "peeking A/B testing false positives"}
_FIN = ("stock", "equit", "return", "asset pric", "portfolio", "trading", "investor", "anomal", "factor", "securities", "market",
        "a/b", "experiment", "multiple testing", "backtest", "sharpe", "volatil", "momentum", "reversal")
UA = "skeptic-research-demo/1.0 (academic literature check)"


def _terms(q: str) -> list[str]:
    out = []
    for w in re.findall(r"[a-z][a-z\-]{2,}", q.lower()):
        if w not in _STOP and w not in out:
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
            au = ", ".join(a.get("family", "") for a in it.get("author", [])[:3]) + (" et al." if len(it.get("author", [])) > 3 else "")
            out.append({"title": (it.get("title") or [""])[0], "authors": au, "year": yr, "doi": it.get("DOI"),
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


def openalex(query: str, rows: int = 5, timeout: float = 8.0) -> list[dict]:
    """Live OpenAlex search. Needs OPENALEX_API_KEY (free) for more than the tiny shared daily budget; skipped otherwise."""
    key = os.environ.get("OPENALEX_API_KEY")
    if not key:
        raise RuntimeError("OPENALEX_API_KEY not set (optional)")

    def fetch():
        url = "https://api.openalex.org/works?" + urllib.parse.urlencode(
            {"search": query, "per-page": rows, "api_key": key, "select": "title,authorships,publication_year,doi,primary_location"})
        out = []
        for it in json.loads(_get(url, timeout).decode())["results"]:
            names = [a["author"]["display_name"] for a in it.get("authorships", [])]
            au = ", ".join(n.split()[-1] for n in names[:3]) + (" et al." if len(names) > 3 else "")
            out.append({"title": it.get("title") or "", "authors": au, "year": it.get("publication_year"), "doi": it.get("doi"),
                        "url": it.get("doi"), "venue": ((it.get("primary_location") or {}).get("source") or {}).get("display_name") or "",
                        "abstract": "", "source": "OpenAlex (live)"})
        return out
    return _cached("openalex", f"{query}|{rows}", fetch)


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
            notes.append(f"{n}: unavailable ({type(e).__name__})")
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
            cav.append(f"All of your data ({sc.start_year}-{sc.end_year}) is after this paper appeared ({yr}): expect post-publication decay "
                       "(McLean & Pontiff 2016: about 58% lower anomaly returns).")
            haircut = 0.58
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
            live, notes = live_search(query, 6)
            self.status = notes
            on_topic = [p for p in live if any(f in (p["title"] + " " + (p.get("abstract") or "")).lower() for f in _FIN)]
            ranked = sorted(((_live_relevance(p, terms), p) for p in on_topic), key=lambda x: -x[0])
            for rel, p in [x for x in ranked if x[0] >= 2][:top]:
                found.append({**p, "id": p.get("doi") or p["url"], "claim": (p.get("abstract") or "")[:240] or "(no abstract returned: open the link)",
                              "sample": None, "universe": None, "relevance": round(rel, 1), **assess_live(p, self.scenario)})
        ranked = sorted(((_score(text, e, family), e) for e in CORPUS), key=lambda x: -x[0])
        for sc_, e in ranked[:top]:
            if sc_ <= 0:
                continue
            found.append({**{k: e[k] for k in ("id", "authors", "year", "title", "venue", "claim", "sample", "universe")},
                          "relevance": round(sc_, 1), "source": "curated index (not live: verify before citing)", "url": None,
                          **assess(e, self.scenario)})
        self.cache[key] = found
        return found

    def evidence(self, hyp: dict) -> dict:
        """Literature evidence for one hypothesis: supporting papers, applicability, implied haircut, and a prior suggestion."""
        text = f"{hyp.get('description', '')} {hyp.get('spec', {}).get('formula', '') or ''} {hyp.get('spec', {}).get('rule', '') or ''}"
        papers = self.find(text, hyp.get("family"))
        direct = [p for p in papers if p["relevance"] >= 4.0 and p["id"] not in ("mp16", "hlz16", "hxz20", "bl14", "w00")]
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
            "observed_sharpe": sr, "papers": [{k2: p[k2] for k2 in ("authors", "year", "title", "applicability", "url", "source") if k2 in p} for p in ev.get("papers", [])[:3]],
            "expected_post_publication_haircut": hc, "caveats": ev.get("caveats", []), "notes": notes}}
    return literature
