"""Hunter swarm. Each Hunter specialises in one hypothesis family and emits STRUCTURED hypotheses (JSON), never code.

Real LLM: the Hunter prompts the model with the DSL reference and gets back a JSON list. MockLLM: deterministic family
templates with seeded parameter draws, so the swarm works offline with identical structure.
The Hunter's `prior` only scales how much alpha-wealth a round may stake; it never affects the null test.
"""
from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass

import numpy as np

from ..features_dsl import DSLError, parse
from ..llm import LLM

FAMILIES = ["momentum", "mean_reversion", "calendar", "volatility", "fundamentals", "volume_sentiment"]

HYP_SCHEMA = {
    "type": "object",
    "properties": {"hypotheses": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "description": {"type": "string"},
            "feature_formula": {"type": "string"},
            "params": {"type": "object"},
            "rationale": {"type": "string"},
            "source_paper": {"type": ["string", "null"]},
            "prior": {"type": "number", "minimum": 0, "maximum": 1},
            "universe": {"enum": ["point_in_time", "current_constituents"]},
            "hold": {"type": "integer", "minimum": 1, "maximum": 60},
        },
        "required": ["description", "feature_formula", "rationale", "prior"],
    }}},
    "required": ["hypotheses"],
}

DSL_DOC = """Fields: close, ret, volume, btm (book-to-market), sentiment. Time x asset arrays.
Functions: ts_return(x,n) delta(x,n) lag(x,n) rolling_mean/std/sum/max/min(x,n) ts_zscore(x,n) rank(x) zscore(x) abs sign log sqrt clip(x,lo,hi) dow() month().
Operators: + - * / and unary minus. The feature is ranked cross-sectionally each day: long top quintile, short bottom.
Windows must be integer constants. Do NOT use lead(): it looks into the future."""

SYSTEM = ("You are a quantitative Researcher. Propose testable hypotheses as JSON only. "
          "Use only the documented feature DSL. Be diverse. Higher `prior` only for ideas with published support.\n" + DSL_DOC)


@dataclass
class Hunter:
    family: str
    llm: LLM
    seed: int = 0
    fields: set | None = None            # data fields that actually exist; formulas using others are rejected

    @property
    def name(self) -> str:
        return f"{self.family}-researcher"

    def propose(self, n: int, seen: set[str], round_idx: int = 0, min_prior: float = 0.0) -> tuple[list[dict], int]:
        """Returns (valid hypotheses, number rejected by the DSL validator). `seen` dedupes formula+universe+hold."""
        prompt = (f"Family: {self.family}. Propose {n} NEW hypotheses (round {round_idx}). "
                  f"Only ideas with prior >= {min_prior:.2f}. Avoid these formulas: {sorted(seen)[:40]}.")
        obj, usage = self.llm.complete_json("hunter", SYSTEM, prompt, HYP_SCHEMA,
                                            mock_hint={"family": self.family, "n": n, "seed": self.seed,
                                                       "round": round_idx, "min_prior": min_prior}, max_tokens=3000)
        out, bad = [], 0
        for h in obj["hypotheses"]:
            try:
                pf = parse(h["feature_formula"])
            except DSLError:
                bad += 1
                continue
            if self.fields is not None and not pf.fields <= self.fields:
                bad += 1
                continue
            uni = h.get("universe", "point_in_time")
            hold = int(h.get("hold", 1))
            key = f"{ast.unparse(pf.tree)}|{uni}|{hold}"
            if key in seen or h["prior"] < min_prior:
                continue
            seen.add(key)
            hid = "h_" + hashlib.sha1(key.encode()).hexdigest()[:8]
            out.append({
                "id": hid, "family": self.family, "agent": self.name, "description": h["description"],
                "prior": float(h["prior"]), "rationale": h.get("rationale", ""), "source_paper": h.get("source_paper"),
                "params": h.get("params", {}),
                "spec": {"formula": pf.formula, "universe": uni, "hold": hold, "uses_future_bar": pf.uses_lead,
                         "windows": pf.windows, "fields": sorted(pf.fields)},
                "tokens": {"in": usage.tokens_in, "out": usage.tokens_out},
            })
        return out, bad


# ------------------------------------------------------------------ deterministic offline hunters (MockLLM)
def _templates(family: str):
    """(formula template, param space, description template, prior, source). Literature entries get high prior."""
    L = {
        "momentum": [
            ("ts_return(lag(close, {skip}), {n})", {"n": [20, 40, 60, 90, 120, 180, 231, 250], "skip": [1, 5, 10, 21]}, "{n}-day momentum skipping {skip}d", 0.35),
            ("rank(ts_return(close, {n})) - rank(rolling_std(ret, {m}))", {"n": [30, 60, 120], "m": [20, 60]}, "risk-adjusted {n}d momentum", 0.3),
            ("ts_zscore(close, {n})", {"n": [20, 60, 120, 250]}, "price z-score trend over {n}d", 0.3),
            ("ts_return(close, {n}) / rolling_std(ret, {n})", {"n": [20, 60, 120]}, "Sharpe-style momentum {n}d", 0.3),
        ],
        "mean_reversion": [
            ("-ts_return(close, {n})", {"n": [1, 2, 3, 5, 7, 10, 15, 20, 30]}, "{n}-day reversal", 0.3),
            ("-ts_zscore(close, {n})", {"n": [5, 10, 20, 40]}, "reversion of {n}d price z-score", 0.3),
            ("-(close / rolling_mean(close, {n}) - 1)", {"n": [5, 10, 20, 50]}, "distance to {n}d moving average", 0.3),
            ("-delta(ret, {n})", {"n": [1, 2, 5]}, "fade change in daily returns over {n}d", 0.25),
        ],
        "calendar": [
            ("(dow() - {d}) * ts_return(close, {n})", {"d": [0, 1, 2, 3, 4], "n": [1, 5, 10]}, "weekday-{d} tilted {n}d return", 0.15),
            ("(month() - {d}) * ts_return(close, {n})", {"d": [1, 3, 6, 9, 12], "n": [5, 20]}, "month-{d} tilted {n}d return", 0.15),
            ("dow() * rolling_mean(ret, {n})", {"n": [5, 10, 20]}, "day-of-week x {n}d mean return", 0.15),
        ],
        "volatility": [
            ("-rolling_std(ret, {n})", {"n": [10, 20, 40, 60, 90, 120]}, "low {n}d volatility premium", 0.3),
            ("rolling_std(ret, {n})", {"n": [10, 20, 60]}, "high {n}d volatility", 0.2),
            ("-rolling_std(ret, {a}) / rolling_std(ret, {b})", {"a": [5, 10, 20], "b": [60, 120]}, "vol term structure {a}/{b}", 0.25),
            ("-(rolling_max(close, {n}) - rolling_min(close, {n})) / close", {"n": [10, 20, 60]}, "low {n}d price range", 0.25),
        ],
        "fundamentals": [
            ("btm", {}, "book-to-market (value)", 0.8),
            ("zscore(btm) + {w} * zscore(ts_return(lag(close, 21), 231))", {"w": [0.25, 0.5, 1.0]}, "value plus momentum blend w={w}", 0.35),
            ("rank(btm) - rank(rolling_std(ret, {n}))", {"n": [20, 60]}, "cheap and calm ({n}d)", 0.3),
            ("btm * rank(volume)", {}, "value in liquid names", 0.25),
        ],
        "volume_sentiment": [
            ("-(volume / rolling_mean(volume, {n}))", {"n": [5, 10, 20, 40]}, "fade volume spikes ({n}d)", 0.3),
            ("volume / rolling_mean(volume, {n})", {"n": [5, 10, 20]}, "follow volume surges ({n}d)", 0.25),
            ("sentiment", {}, "news sentiment level", 0.3),
            ("rolling_mean(sentiment, {n})", {"n": [5, 10, 20]}, "smoothed {n}d sentiment", 0.25),
            ("delta(sentiment, {n})", {"n": [1, 3, 5]}, "sentiment change over {n}d", 0.25),
        ],
    }
    return L[family]


LITERATURE = {  # canonical, published ideas -> high prior
    "momentum": [("ts_return(lag(close, 21), 231)", "12-1 month momentum", "Jegadeesh & Titman (1993)")],
    "mean_reversion": [("-ts_return(close, 5)", "short-term reversal", "Jegadeesh (1990)"),
                        ("-ts_return(close, 10)", "10-day reversal", "Lehmann (1990)")],
    "volatility": [("-rolling_std(ret, 60)", "low-volatility anomaly", "Ang, Hodrick, Xing & Zhang (2006)")],
    "fundamentals": [("btm", "value (book-to-market)", "Fama & French (1992)")],
    "volume_sentiment": [("-(volume / rolling_mean(volume, 20))", "high-volume return premium (fade)", "Gervais, Kaniel & Mingelgrin (2001)")],
    "calendar": [],
}


def mock_hunter(hint: dict) -> dict:
    fam, n = hint["family"], hint["n"]
    rng = np.random.default_rng(int(hashlib.sha1(f"{hint['seed']}|{fam}|{hint['round']}".encode()).hexdigest()[:8], 16))
    out = []
    if hint["round"] == 0:                                      # round 0 starts with the canonical published ideas
        for f, d, src in LITERATURE[fam]:
            out.append({"description": d, "feature_formula": f, "rationale": f"Published anomaly: {src}",
                        "source_paper": src, "prior": 0.8, "params": {}})
    tpls = _templates(fam)
    guard = 0
    while len(out) < n and guard < n * 20:
        guard += 1
        f, space, d, prior = tpls[int(rng.integers(0, len(tpls)))]
        params = {k: v[int(rng.integers(0, len(v)))] for k, v in space.items()}
        formula, desc = f.format(**params), d.format(**params)
        if rng.random() < 0.5 and not formula.startswith("-") and fam != "fundamentals":
            formula, desc = f"-({formula})", f"fade: {desc}"            # sign variants are separate forking paths
        h = {"description": desc, "feature_formula": formula, "params": params, "prior": prior,
             "rationale": f"Template variant in family {fam}.", "source_paper": None}
        r = rng.random()
        if r < 0.06:
            h["universe"] = "current_constituents"
        elif r < 0.10:
            h["hold"] = int(rng.choice([5, 21]))
        if rng.random() < 0.03:                                     # sloppy hunter: accidentally peeks into the future
            h["feature_formula"] = f"lead(ret, 1) + 0.0 * ({formula})"
        out.append(h)
    return {"hypotheses": out[:n]}


def register_mock(llm: LLM):
    llm.register_mock("hunter", mock_hunter)
