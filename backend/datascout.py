"""Data Scout: the user gives only a question; this finds a real, public dataset for it on the internet.

Flow:  question -> search terms (cause / effect split) -> live search of OpenML, Hugging Face and Zenodo
       -> download the most relevant CSVs (size-capped) -> check each is a usable table
       -> map the question onto columns: a CAUSE (treatment) column and an EFFECT (outcome) column
       -> hand the table to the normal Court (null worlds, Skeptics, Judge) and the conclusion writer.

Honesty rules
  * Every dataset shown is a real API result with its link; nothing is invented. Rejected candidates are listed with the reason.
  * Column mapping: with a real LLM configured it may map synonyms (e.g. smoking -> tobacco) but may only name columns that
    exist; otherwise a deterministic word match is used. If no column matches the question's cause, we say so and stop
    rather than test an unrelated column.
  * How each column was turned into a yes/no group (e.g. "above its median") is reported, because that choice matters.
  * Searching the internet is disabled with SKEPTIC_OFFLINE=1.
"""
from __future__ import annotations

import io
import json
import os
import re
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import literature as LIT

CACHE = Path(__file__).resolve().parents[1] / ".lit_cache"
UA = "SkepticR/1.0 (research demo; dataset search)"
MAX_BYTES = 12_000_000            # download cap per file
FILE_SECONDS = 20                 # per-file download time limit: from a slow server we use what arrived by then
MAX_ROWS = 200_000
MIN_ROWS = 150
MAX_TRIES = 10                    # candidates downloaded per question
TIME_BUDGET = 75                  # seconds spent downloading candidates before settling for the best so far
OPENML_TTL = 7 * 86400            # the OpenML catalogue is cached on disk for a week

# words that state the relation in a question, not what it is about ("does X increase Y")
_REL = (r"increases?|raises?|reduces?|lowers?|decreases?|affects?|improves?|causes?|predicts?|impacts?|influences?|boosts?|hurts?|"
        r"worsens?|drives?|determines?|leads? to|relates? to|related to|associated with|linked to|correlated with|changes?")
_COMPARE = r"(?:have|has|get|gets|show|shows|earn|earns|live|lives|score|scores)\s+(?:a\s+)?(?:higher|lower|better|worse|more|less|longer|shorter|greater|bigger|smaller)"
_FILLER = set("does do did is are was were has have had can could would will should really actually likely truly generally "
              "people person someone those whether what which who why how".split()) | {
    w for w in re.findall(r"[a-z]+", _REL.replace("?", "").replace("|", " "))}
# A comparative or everyday word rarely appears as a column name; these say which attribute it refers to.
# Used only by the offline word match (a real LLM maps synonyms itself) and only ever ADDS candidate words.
ATTRIBUTE_WORDS = {
    "older": ["age"], "younger": ["age"], "elderly": ["age"], "aged": ["age"],
    "richer": ["income", "wealth"], "poorer": ["income", "wealth"], "wealthier": ["income", "wealth"],
    "heavier": ["weight", "bmi"], "lighter": ["weight", "bmi"], "obese": ["bmi", "obesity"], "overweight": ["bmi", "weight"],
    "taller": ["height"], "shorter": ["height"],
    "smoking": ["smoker", "tobacco", "cigarette"], "smokers": ["smoker", "tobacco", "cigarette"], "smoke": ["smoker", "tobacco"],
    "drinking": ["alcohol", "drinker"], "drinkers": ["alcohol", "drinker"],
    "men": ["sex", "gender"], "women": ["sex", "gender"], "male": ["sex", "gender"], "female": ["sex", "gender"],
    "survive": ["survived", "survival"], "die": ["death", "died", "mortality"], "death": ["died", "mortality"],
    "educated": ["education"], "salary": ["income", "wage"], "pay": ["income", "wage", "salary"],
}


def _expand(words: list[str]) -> list[str]:
    out = list(words)
    for w in words:
        out += [x for x in ATTRIBUTE_WORDS.get(w, []) if x not in out]
    return out


_TARGET_NAMES = ("class", "target", "label", "outcome", "y", "diagnosis", "result", "status")
_POSITIVE = ("yes", "true", "1", "pos", "positive", "present", "smoker", "treated", "treatment", "success", "high")


# ----------------------------------------------------------------------------------------------- question -> terms
def _words(text: str) -> list[str]:
    return [w for w in LIT._terms(text, True) if w not in _FILLER]


# "X earn more than Y": the verb itself names the outcome, and "than Y" belongs to the comparison (the cause side)
VERB_OUTCOME = {"earn": ["income", "earnings", "wage", "salary"], "live": ["lifespan", "survival"], "score": ["score", "grade"],
                "spend": ["spending", "expenditure"], "survive": ["survived", "survival"], "weigh": ["weight"], "sleep": ["sleep"]}
_VERB_COMPARE = rf"\b({'|'.join(VERB_OUTCOME)})s?\s+(?:higher|lower|better|worse|more|less|longer|shorter|greater)\b"


def parse_question(q: str) -> dict:
    """Split 'Does X increase Y?' into cause terms (X) and effect terms (Y). No split -> all terms, effect unknown."""
    low = q.lower()
    v = re.search(_VERB_COMPARE, low)
    if v:
        than = re.search(r"\bthan\b", low[v.end():])
        between = low[v.end(): v.end() + than.start()] if than else low[v.end():]
        cause = _words(low[:v.start()]) + (_words(low[v.end() + than.end():]) if than else [])
        effect = VERB_OUTCOME[v.group(1)] + _words(between)
        if cause:
            return {"cause": _expand(cause), "effect": _expand(effect), "all": _words(low[:v.start()]) + effect[:2], "relation": v.group(0)}
    m = re.search(rf"\b({_REL}|{_COMPARE})\b", low)
    if m:
        cause, effect = _words(low[:m.start()]), _words(low[m.end():])
        if effect and not cause:                                            # "What predicts Y?": no cause named
            return {"cause": [], "effect": _expand(effect), "all": effect, "relation": m.group(0), "open": True}
        if cause and effect:
            return {"cause": _expand(cause), "effect": _expand(effect), "all": cause + [w for w in effect if w not in cause], "relation": m.group(0)}
    w = _words(low)
    return {"cause": _expand(w), "effect": [], "all": w, "relation": None}


def _stem_match(term: str, token: str) -> bool:
    if term == token:
        return True
    if len(term) >= 4 and len(token) >= 4:
        return term[:5] == token[:5] if min(len(term), len(token)) >= 5 else term[:4] == token[:4]
    return False


def _tokens(name: str) -> list[str]:
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(name))
    name = re.sub(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])", " ", name)          # T2Diabetic -> T 2 Diabetic
    return [t for t in re.split(r"[^a-z0-9]+", name.lower()) if t]


def _match(terms: list[str], text: str) -> int:
    toks = _tokens(text)
    return sum(any(_stem_match(t, k) for k in toks) for t in terms)


# ----------------------------------------------------------------------------------------------- HTTP
def _get(url: str, timeout: float = 20, cap: int | None = None, seconds: float | None = None) -> tuple[bytes, bool]:
    """GET with an optional byte cap and wall-clock limit. Returns (data, truncated)."""
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        if cap is None and seconds is None:
            return r.read(), False
        buf, n, t0 = [], 0, time.time()
        while True:
            chunk = r.read(1 << 16)
            if not chunk:
                return b"".join(buf), False
            buf.append(chunk)
            n += len(chunk)
            if (cap and n > cap) or (seconds and time.time() - t0 > seconds):
                return b"".join(buf)[: cap or n], True


# ----------------------------------------------------------------------------------------------- sources
_openml_lock = threading.Lock()


def _openml_catalogue() -> list[dict]:
    """All active OpenML datasets (name, size), slimmed and cached on disk; first fetch takes ~30 s."""
    f = CACHE / "openml_catalogue.json"
    with _openml_lock:
        if f.exists() and time.time() - f.stat().st_mtime < OPENML_TTL:
            return json.loads(f.read_text(encoding="utf-8"))
        raw, _ = _get("https://www.openml.org/api/v1/json/data/list/status/active/limit/10000", timeout=90)
        out, seen = [], set()
        for d in json.loads(raw)["data"]["dataset"]:
            q = {x["name"]: x["value"] for x in d.get("quality", [])}
            n = float(q.get("NumberOfInstances") or 0)
            # skip OpenML's generated variants (BNG, *_seed_N_nrows_*): copies of a dataset, not new data
            if d.get("format", "").upper() != "ARFF" or re.search(r"bng|_seed_\d|nrows_\d", d["name"].lower()) or d["name"].lower() in seen:
                continue
            seen.add(d["name"].lower())
            out.append({"did": d["did"], "name": d["name"], "file_id": d["file_id"], "rows": int(n), "cols": int(float(q.get("NumberOfFeatures") or 0))})
        CACHE.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(out), encoding="utf-8")
        return out


def warm_up():
    """Fetch the OpenML catalogue in the background at server start, so the first question does not wait for it."""
    if os.environ.get("SKEPTIC_OFFLINE") == "1":
        return
    threading.Thread(target=lambda: _safe(_openml_catalogue), daemon=True).start()


def _safe(fn):
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return None


def _search_openml(terms: list[str], rows: int) -> list[dict]:
    hits = []
    for d in _openml_catalogue():
        if not (MIN_ROWS <= d["rows"] <= MAX_ROWS * 5):
            continue
        rel = _match(terms, d["name"])
        if rel:
            hits.append((rel, d))
    hits.sort(key=lambda x: (-x[0], -min(x[1]["rows"], 50_000)))
    return [{"title": d["name"], "source": "OpenML", "url": f"https://www.openml.org/d/{d['did']}", "rel": rel,
             "download": f"https://www.openml.org/data/v1/download/{d['file_id']}", "format": "arff", "note": f"{d['rows']:,} rows, {d['cols']} columns"}
            for rel, d in hits[:rows]]


def _search_hf(terms: list[str], rows: int) -> list[dict]:
    ids, out = [], []
    for q in [" ".join(terms[:2]), *terms[:3]]:
        if not q:
            continue
        url = "https://huggingface.co/api/datasets?" + urllib.parse.urlencode({"search": q, "limit": 8, "sort": "downloads", "direction": -1})
        for d in json.loads(_get(url, 12)[0]):
            if d["id"] not in ids:
                ids.append(d["id"])
    scored = sorted(((_match(terms, i.split("/")[-1]), i) for i in ids), key=lambda x: -x[0])
    for rel, i in [x for x in scored if x[0] > 0][:rows]:
        info = json.loads(_get(f"https://huggingface.co/api/datasets/{i}", 12)[0])
        csvs = [s["rfilename"] for s in info.get("siblings", []) if s["rfilename"].lower().endswith(".csv")]
        if csvs:
            f = sorted(csvs, key=lambda n: (0 if "train" in n.lower() else 1, n.count("/"), len(n)))[0]
            out.append({"title": i, "source": "Hugging Face", "url": f"https://huggingface.co/datasets/{i}", "rel": rel,
                        "download": f"https://huggingface.co/datasets/{i}/resolve/main/{urllib.parse.quote(f)}", "note": f"file {f}"})
    return out


def _search_zenodo(terms: list[str], rows: int) -> list[dict]:
    url = "https://zenodo.org/api/records?" + urllib.parse.urlencode({"q": " AND ".join(terms[:3]), "type": "dataset", "size": 15, "sort": "bestmatch"})
    out = []
    for h in json.loads(_get(url, 15)[0])["hits"]["hits"]:
        m = h.get("metadata", {})
        csvs = [f for f in h.get("files", []) if f["key"].lower().endswith(".csv") and f.get("size", 0) <= MAX_BYTES]
        rel = _match(terms, m.get("title", "") + " " + re.sub(r"<[^>]+>", " ", m.get("description", "") or "")[:600])
        if csvs and rel:
            f = csvs[0]
            out.append({"title": m.get("title", "")[:140], "source": "Zenodo", "url": h.get("doi_url") or h.get("links", {}).get("html"), "rel": rel,
                        "download": f["links"]["self"], "note": f"file {f['key']}"})
    return sorted(out, key=lambda d: -d["rel"])[:rows]


SOURCES = {"OpenML": _search_openml, "Hugging Face": _search_hf, "Zenodo": _search_zenodo}


def search(terms: list[str], rows: int = 5) -> tuple[list[dict], list[str]]:
    out, notes = [], []
    with ThreadPoolExecutor(len(SOURCES)) as ex:
        futs = {n: ex.submit(fn, terms, rows) for n, fn in SOURCES.items()}
    for n, f in futs.items():
        try:
            r = f.result()
            out += r
            notes.append(f"{n}: {len(r)} with a downloadable table")
        except Exception as e:  # noqa: BLE001  one source failing must not stop the others
            notes.append(f"{n}: unavailable ({type(e).__name__})")
    order = list(SOURCES)
    out.sort(key=lambda d: (-d["rel"], order.index(d["source"])))
    return out, notes


# ----------------------------------------------------------------------------------------------- load + map columns
def _read_arff(raw: bytes) -> pd.DataFrame:
    """Minimal dense-ARFF reader (OpenML's format): @attribute names, then CSV rows ('?' = missing)."""
    lines = raw.decode("utf-8", errors="replace").splitlines()
    names, start = [], None
    for i, line in enumerate(lines):
        s = line.strip()
        if s.lower().startswith("@attribute"):
            m = re.match(r"@attribute\s+(?:'([^']+)'|\"([^\"]+)\"|(\S+))", s, re.I)
            names.append(m.group(1) or m.group(2) or m.group(3))
        elif s.lower().startswith("@data"):
            start = i + 1
            break
    body = "\n".join(lines[start:]) if start is not None else ""
    if not names or not body.strip() or body.lstrip().startswith("{"):
        raise ValueError("unsupported ARFF file")
    return pd.read_csv(io.StringIO(body), names=names, header=None, quotechar="'", na_values=["?"], skipinitialspace=True,
                       comment="%", low_memory=False, on_bad_lines="skip", escapechar="\\")


def load_table(url: str, fmt: str = "csv") -> tuple[pd.DataFrame, str]:
    # downloaded files are kept on disk (git-ignored), so asking again, or re-running a demo, is instant
    import hashlib
    f = CACHE / "data" / (hashlib.sha1(url.encode()).hexdigest()[:20] + ".bin")
    if f.exists():
        raw = f.read_bytes()
        cut = raw.startswith(b"#TRUNCATED\n")
        raw = raw[len(b"#TRUNCATED\n"):] if cut else raw
    else:
        raw, cut = _get(url, 30, MAX_BYTES, FILE_SECONDS)
        if cut:
            raw = raw[: raw.rfind(b"\n")]
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes((b"#TRUNCATED\n" if cut else b"") + raw)
    df = _read_arff(raw) if fmt == "arff" else None
    for sep in (() if df is not None else (",", ";", "\t")):
        try:
            d = pd.read_csv(io.BytesIO(raw), sep=sep, low_memory=False, on_bad_lines="skip")
        except Exception:  # noqa: BLE001
            continue
        if d.shape[1] >= 2:
            df = d
            break
    if df is None:
        raise ValueError("not a readable table")
    df = df.loc[:, [c for c in df.columns if not str(c).lower().startswith("unnamed")]]
    df.columns = [str(c).strip() for c in df.columns]
    for c in df.columns:                                               # ' Male' -> 'Male'
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip().where(df[c].notna())
    note = f"first {len(raw) / 1e6:.1f} MB of the file used (download limit)" if cut else ""
    if len(df) > MAX_ROWS:
        df, note = df.sample(MAX_ROWS, random_state=0), f"random {MAX_ROWS:,} of {len(df):,} rows used"
    return df, note


def _usable_outcome(s: pd.Series) -> bool:
    k = s.dropna().nunique()
    return k >= 2 and not _id_like(s) and (s.dtype.kind in "if" or k == 2)


def _id_like(s: pd.Series) -> bool:
    """Row identifiers (PassengerId, patient_no...) are unique per row: never a cause or an effect."""
    v = s.dropna()
    return len(v) > 20 and v.nunique() >= 0.95 * len(v) and (s.dtype.kind in "iu" or s.dtype == object or bool(re.search(r"(^|_)id$|id$|_no$", str(s.name), re.I)))


def _usable_treat(s: pd.Series) -> bool:
    k = s.dropna().nunique()
    return k >= 2 and not _id_like(s) and (s.dtype.kind in "if" or k <= 20)


def map_columns(df: pd.DataFrame, qp: dict, title: str = "", target: str | None = None) -> dict:
    """Deterministic mapping. Outcome: a column named for the question's effect (yes/no columns preferred), or the dataset's
    declared target when the dataset itself is about that effect. Treatment: a column named for the cause, preferring an
    ordered (numeric) version over a list of categories. Raises with the reason if either is missing: we never test a column
    the question did not ask about."""
    cols = list(df.columns)
    is_target = lambda c: c == target or c.lower() in _TARGET_NAMES
    outcome, how_o = None, ""
    if qp["effect"]:
        about_effect = _match(qp["effect"], title) > 0
        scored = []
        for c in cols:
            if not _usable_outcome(df[c]) or _match(qp["cause"], c):
                continue
            words = _match(qp["effect"], c)
            s = 2 * words + (1 if df[c].dropna().nunique() == 2 else 0) + (2 if is_target(c) and about_effect else 0)
            if s >= 2:
                scored.append((s, words, c))
        if scored:
            _, words, outcome = max(scored)
            how_o = "matches the effect in your question" if words else f"is the target column of a dataset about {', '.join(qp['effect'])}"
    else:
        named = [c for c in cols if is_target(c) and _usable_outcome(df[c])]
        outcome = named[0] if named else (cols[-1] if _usable_outcome(df[cols[-1]]) else None)
        how_o = "is the dataset's target column"
    if outcome is None:
        raise ValueError(f"no column for the effect ({', '.join(qp['effect'][:3]) or 'target'})")
    # cause: word match first; then an ordered (numeric) column over a yes/no one over a list of categories,
    # because "more X" is only meaningful when X has an order (e.g. education-num rather than education)
    tscore = []
    for c in cols:
        if c == outcome or not _usable_treat(df[c]):
            continue
        words = _match(qp["cause"], c)
        if words:
            k = df[c].dropna().nunique()
            tscore.append((2 * words + (1.0 if df[c].dtype.kind in "if" and k > 2 else 0.5 if k == 2 else 0.0), c))
    if not tscore:
        raise ValueError(f"no column for the cause ({', '.join(qp['cause'][:3])})")
    treat = max(tscore)[1]
    return {"treat": treat, "outcome": outcome, "why": f"'{treat}' matches the cause in your question; '{outcome}' {how_o}."}


SCHEMA = {"type": "object", "properties": {"treat": {"type": "string"}, "outcome": {"type": "string"}, "why": {"type": "string"}},
          "required": ["treat", "outcome"]}


def map_columns_llm(llm, question: str, df: pd.DataFrame) -> dict | None:
    """Real model only: map the question onto columns (synonyms allowed). It may only name columns that exist."""
    desc = "\n".join(f"- {c} ({df[c].dtype}, {df[c].nunique()} distinct, e.g. {', '.join(map(str, df[c].dropna().unique()[:4]))})" for c in df.columns[:60])
    prompt = (f"QUESTION: {question}\n\nCOLUMNS:\n{desc}\n\nPick the column that is the question's CAUSE (treatment / exposure) and the column that is its "
              "EFFECT (outcome). Use only the column names listed. If no column fits the cause, use \"\" for treat.")
    obj, _ = llm.complete_json("extractor", "You map a research question onto a table's columns. JSON only: {\"treat\":..., \"outcome\":..., \"why\":...}.",
                               prompt, SCHEMA, max_tokens=300)
    t, o = obj.get("treat"), obj.get("outcome")
    if t in df.columns and o in df.columns and t != o and _usable_treat(df[t]) and _usable_outcome(df[o]):
        return {"treat": t, "outcome": o, "why": str(obj.get("why") or "")[:300]}
    return None


def _positive_level(levels) -> object:
    levels = list(levels)
    try:
        if sorted(float(x) for x in levels) == [0.0, 1.0]:
            return next(x for x in levels if float(x) == 1.0)
    except (TypeError, ValueError):
        pass
    for lv in levels:
        if str(lv).strip().lower() in _POSITIVE or any(str(lv).strip().lower().startswith(p) for p in ("tested_pos", "pos", "yes")):
            return lv
    return None


def prepare(df: pd.DataFrame, treat: str, outcome: str, seed: int) -> tuple[pd.DataFrame, dict]:
    """Turn the treatment into yes/no and the outcome into a number, and say exactly how."""
    d = df.copy()
    t, o = d[treat], d[outcome]
    # the yes/no and number columns are NAMED after their rule, so every finding and the conclusion say what was compared
    if t.dtype.kind in "if" and t.nunique() > 2:
        med = float(t.median())
        t_name, t_rule = f"{treat} above {med:g}", f"{treat} above its median ({med:g}) vs at or below it"
        t_val = (t > med).astype(float).where(t.notna())
    else:
        counts = t.value_counts()
        pos = _positive_level(counts.index) if len(counts) == 2 else None
        pos = pos if pos is not None else (counts.index[-1] if len(counts) == 2 else counts.index[0])
        t_name, t_rule = f"{treat} = {pos}", f"{treat} = {pos} vs " + ("the other value" if len(counts) == 2 else "all other values")
        t_val = (t == pos).astype(float).where(t.notna())
    if o.dtype.kind in "if" and o.nunique() > 2:
        o_name, o_rule, o_val = outcome, f"{outcome} as a number", o.astype(float)
    else:
        counts = o.value_counts()
        pos = _positive_level(counts.index)
        pos = pos if pos is not None else counts.index[-1]                 # the rarer level counts as "the event"
        o_name, o_rule = f"{outcome} = {pos}", f"share with {outcome} = {pos}"
        o_val = (o == pos).astype(float).where(o.notna())
    d = d.drop(columns=[treat, outcome]).assign(**{t_name: t_val, o_name: o_val}).dropna(subset=[t_name, o_name])
    if d[t_name].nunique() < 2:
        raise ValueError(f"'{treat}' has only one group after cleaning")
    # public tables are often sorted (by class, by date): shuffle so the held-out slice is a fair sample
    d = d.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    sub = [c for c in d.columns if c not in (t_name, o_name) and 2 <= d[c].nunique() <= 8][:5]
    # few categorical columns: add numeric ones too (the Court splits them into thirds) so "like with like" can be checked
    sub += [c for c in d.columns if c not in (t_name, o_name, *sub) and d[c].dtype.kind in "if" and d[c].nunique() > 8
            and not _id_like(d[c])][: max(0, 4 - len(sub))]
    return d, {"treat_rule": t_rule, "outcome_rule": o_rule, "subgroups": sub, "treat_col": t_name, "outcome_col": o_name}


def _declared_target(c: dict) -> str | None:
    """OpenML publishes each dataset's target column; other sources do not."""
    if c["source"] != "OpenML":
        return None
    try:
        did = c["url"].rstrip("/").rsplit("/", 1)[-1]
        d = json.loads(_get(f"https://www.openml.org/api/v1/json/data/{did}", 15)[0])["data_set_description"]
        return (d.get("default_target_attribute") or "").split(",")[0] or None
    except Exception:  # noqa: BLE001
        return None


# ----------------------------------------------------------------------------------------------- the whole step
@dataclass
class Found:
    df: pd.DataFrame
    treat: str
    outcome: str
    info: dict = field(default_factory=dict)


def find_dataset(question: str, emit, llm=None, seed: int = 0) -> Found:
    """Search, download, check and map. `emit(type, **kw)` reports progress. Raises with a plain reason if nothing fits."""
    qp = parse_question(question)
    if not qp["all"]:
        raise RuntimeError("The question has no searchable words. Try e.g. 'Does smoking increase heart disease?'")
    if qp.get("open"):
        raise RuntimeError(f"SkepticR tests one specific cause at a time, and this question does not name one. Try e.g. "
                           f"'Does <something> increase {' '.join(qp['effect'][:2])}?'")
    if os.environ.get("SKEPTIC_OFFLINE") == "1":
        raise RuntimeError("Searching the internet is turned off (SKEPTIC_OFFLINE=1). Upload a CSV or pick an example dataset instead.")
    say = lambda text: emit("agent", agent="Data Scout", text=text)
    say(f"Searching OpenML, Hugging Face and Zenodo for: {', '.join(qp['all'])}"
        + (f" (cause: {', '.join(qp['cause'])}; effect: {', '.join(qp['effect'])})" if qp["effect"] else ""))
    cands, notes = search(qp["all"], rows=8)
    say("; ".join(notes))
    if not cands:
        raise RuntimeError("No public dataset with a downloadable table matched the question. Try different words, or upload a CSV.")
    use_llm = llm is not None and not getattr(llm, "is_mock", True)
    tried, best, t0 = [], None, time.time()
    for c in cands[:MAX_TRIES]:
        if best and time.time() - t0 > TIME_BUDGET:
            break
        row = {k: c[k] for k in ("title", "source", "url", "note")}
        try:
            df, note = load_table(c["download"], c.get("format", "csv"))
            if len(df) < MIN_ROWS:
                raise ValueError(f"only {len(df)} rows")
            m = (map_columns_llm(llm, question, df) if use_llm else None) or map_columns(df, qp, c["title"], _declared_target(c))
            fit = c["rel"] + _match(qp["cause"], m["treat"]) + (_match(qp["effect"], m["outcome"]) if qp["effect"] else 0)
            row.update(status="usable", rows=len(df), fit=fit, mapping=f"{m['treat']} → {m['outcome']}")
            say(f"{c['source']} · {c['title']}: usable ({len(df):,} rows; {m['treat']} → {m['outcome']})")
            if best is None or fit > best[0]:
                best = (fit, c, df, m, note)
        except Exception as e:  # noqa: BLE001  a bad candidate is reported, never fatal
            row.update(status="rejected", reason=str(e)[:160] or type(e).__name__)
            say(f"{c['source']} · {c['title']}: skipped ({row['reason']})")
        tried.append(row)
        if best and best[0] >= len(qp["all"]) + 1:                       # a strong fit: stop downloading
            break
    if best is None:
        emit("dataset", status="none", question=question, tried=tried, sources=notes)
        raise RuntimeError("Found candidate datasets, but none had columns for both the cause and the effect in your question. See the list of what was tried.")
    _, c, df, m, note = best
    d, rules = prepare(df, m["treat"], m["outcome"], seed)
    for r in tried:
        r["chosen"] = r["url"] == c["url"] and r.get("status") == "usable"
    info = {"status": "found", "question": question, "title": c["title"], "source": c["source"], "url": c["url"], "download": c["download"],
            "rows": len(d), "columns": list(df.columns)[:40], "n_columns": df.shape[1], "treat": m["treat"], "outcome": m["outcome"],
            "why": m["why"], "mapped_by": "AI model" if use_llm else "word match", "note": note, **rules, "tried": tried, "sources": notes}
    emit("dataset", **info)
    try:
        from . import library
        library.record([{"kind": "dataset", "title": c["title"], "url": c["url"], "source": f"{c['source']} (live)", "venue": c["source"]}],
                       f"data for: {question[:80]}")
    except Exception:  # noqa: BLE001  bookkeeping only
        pass
    return Found(d, rules["treat_col"], rules["outcome_col"], info)
