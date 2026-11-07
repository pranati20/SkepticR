"""Librarian: finds research papers for a hypothesis, checks whether they APPLY to the user's current scenario, and
feeds that evidence to the Prosecutors.

Sources
  * a bundled index of well-known papers (works offline; deterministic). Entries carry only facts I am confident of:
    authors, year, venue, the claim, sample period/universe. Verify before citing: this is an index, not a literature review.
  * optional live search of Crossref (title/authors/year/DOI), cached on disk. If the network is down it silently falls back.
LLM role 'librarian' (cheap model) is only used to summarise relevance when a real provider is configured.

Applicability checks (zero tokens) compare a paper with the CURRENT scenario: sample overlap (is our data out-of-sample?),
years since publication (McLean & Pontiff 2016: anomaly returns are ~26% lower out-of-sample and ~58% lower post-publication),
universe/size mismatch, asset class and frequency. The result is a literature-implied haircut and a list of caveats.
"""
from __future__ import annotations

import hashlib
import json
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


def crossref(query: str, rows: int = 3, timeout: float = 6.0) -> list[dict]:
    """Live search (title/authors/year/DOI only). Cached. Returns [] when offline."""
    key = hashlib.sha1(query.encode()).hexdigest()[:16]
    f = CACHE / f"{key}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    try:
        url = "https://api.crossref.org/works?" + urllib.parse.urlencode({"query": query, "rows": rows, "select": "title,author,issued,DOI,container-title",
                                                                           "filter": "type:journal-article"})
        req = urllib.request.Request(url, headers={"User-Agent": "skeptic-research-demo/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            items = json.loads(r.read().decode())["message"]["items"]
    except Exception:
        return []
    out = []
    for it in items:
        yr = (it.get("issued", {}).get("date-parts") or [[None]])[0][0]
        au = ", ".join(a.get("family", "") for a in it.get("author", [])[:3])
        out.append({"title": (it.get("title") or [""])[0], "authors": au, "year": yr, "doi": it.get("DOI"),
                    "venue": (it.get("container-title") or [""])[0], "source": "crossref (live)"})
    CACHE.mkdir(exist_ok=True)
    f.write_text(json.dumps(out), encoding="utf-8")
    return out


@dataclass
class Librarian:
    scenario: Scenario
    online: bool = False
    cache: dict = field(default_factory=dict)

    def find(self, text: str, family: str | None = None, top: int = 3) -> list[dict]:
        key = (text, family)
        if key in self.cache:
            return self.cache[key]
        ranked = sorted(((_score(text, e, family), e) for e in CORPUS), key=lambda x: -x[0])
        found = []
        for sc_, e in ranked[:top]:
            if sc_ <= 0:
                continue
            found.append({**{k: e[k] for k in ("id", "authors", "year", "title", "venue", "claim", "sample", "universe")},
                          "relevance": round(sc_, 1), "source": "bundled index", **assess(e, self.scenario)})
        if self.online and found:
            q = f"{found[0]['title']} {text}"[:160]
            for p in crossref(q, 2):
                found.append({**p, "claim": "(live result: read the abstract)", "relevance": 0.0, "applicability": None, "caveats": [], "expected_haircut": 0.0})
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
            "observed_sharpe": sr, "papers": [{k2: p[k2] for k2 in ("authors", "year", "title", "applicability") if k2 in p} for p in ev.get("papers", [])[:3]],
            "expected_post_publication_haircut": hc, "caveats": ev.get("caveats", []), "notes": notes}}
    return literature
