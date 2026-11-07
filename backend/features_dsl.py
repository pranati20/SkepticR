"""Safe feature expression DSL. LLM-written formulas are parsed with `ast` and interpreted by a whitelist
evaluator. There is NO eval/exec anywhere: only whitelisted node types, functions and data fields are allowed.

Example formulas:   -ts_return(close, 5)      zscore(btm)      -rolling_std(ret, 60)      rank(volume) - rank(ts_return(close, 20))

Time axis is 0, assets axis 1. Every function is causal (uses data at or before t) EXCEPT `lead`, which looks into
the future. `lead` exists so lookahead bugs are expressible, detectable from the AST, and killable by the Prosecutor.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

FIELDS = {"close", "ret", "volume", "btm", "sentiment"}
MAX_WINDOW = 500
MAX_NODES = 60
MAX_DEPTH = 10


class DSLError(ValueError):
    pass


def _df(x):
    return pd.DataFrame(x)


def _win(n) -> int:
    if not isinstance(n, (int, float)) or isinstance(n, bool) or n != int(n) or not (1 <= int(n) <= MAX_WINDOW):
        raise DSLError(f"window must be an integer in [1,{MAX_WINDOW}], got {n!r}")
    return int(n)


def _lag(x, n):
    n = int(n)
    if n < 0 or n > MAX_WINDOW:
        raise DSLError(f"lag must be in [0,{MAX_WINDOW}]")
    return np.vstack([np.full((n, x.shape[1]), np.nan), x[: x.shape[0] - n]]) if n else x.copy()


def _lead(x, n):
    n = _win(n)
    return np.vstack([x[n:], np.full((n, x.shape[1]), np.nan)])


def _roll(x, n, how):
    n = _win(n)
    r = _df(x).rolling(n, min_periods=max(2, n // 2))
    return getattr(r, how)().to_numpy()


def _rank(x):
    return _df(x).rank(axis=1, pct=True).to_numpy()


def _zscore(x):
    m = np.nanmean(x, axis=1, keepdims=True)
    s = np.nanstd(x, axis=1, keepdims=True)
    return (x - m) / np.where(s == 0, np.nan, s)


def _ts_zscore(x, n):
    n = _win(n)
    d = _df(x).rolling(n, min_periods=max(2, n // 2))
    return ((_df(x) - d.mean()) / d.std().replace(0, np.nan)).to_numpy()


FUNCS = {
    "ts_return": lambda x, n: x / _lag(x, _win(n)) - 1.0,
    "delta": lambda x, n: x - _lag(x, _win(n)),
    "lag": lambda x, n: _lag(x, n),
    "lead": lambda x, n: _lead(x, n),                       # LOOKAHEAD: flagged by the parser
    "rolling_mean": lambda x, n: _roll(x, n, "mean"),
    "rolling_std": lambda x, n: _roll(x, n, "std"),
    "rolling_sum": lambda x, n: _roll(x, n, "sum"),
    "rolling_max": lambda x, n: _roll(x, n, "max"),
    "rolling_min": lambda x, n: _roll(x, n, "min"),
    "ts_zscore": _ts_zscore,
    "rank": _rank,
    "zscore": _zscore,
    "abs": np.abs,
    "sign": np.sign,
    "log": lambda x: np.log(np.where(x > 0, x, np.nan)),
    "sqrt": lambda x: np.sqrt(np.where(x >= 0, x, np.nan)),
    "clip": lambda x, lo, hi: np.clip(x, lo, hi),
}
ARITY = {"ts_return": 2, "delta": 2, "lag": 2, "lead": 2, "rolling_mean": 2, "rolling_std": 2, "rolling_sum": 2,
         "rolling_max": 2, "rolling_min": 2, "ts_zscore": 2, "rank": 1, "zscore": 1, "abs": 1, "sign": 1, "log": 1,
         "sqrt": 1, "clip": 3}
WINDOW_FUNCS = {"ts_return", "delta", "lag", "lead", "rolling_mean", "rolling_std", "rolling_sum", "rolling_max",
                "rolling_min", "ts_zscore"}
CALENDAR = {"dow", "month"}                                  # zero-arg calendar features


@dataclass
class ParsedFormula:
    formula: str
    tree: ast.Expression
    uses_lead: bool = False
    fields: set = field(default_factory=set)
    windows: list = field(default_factory=list)
    funcs: set = field(default_factory=set)
    n_nodes: int = 0


def parse(formula: str) -> ParsedFormula:
    if not isinstance(formula, str) or len(formula) > 300:
        raise DSLError("formula must be a string up to 300 chars")
    try:
        tree = ast.parse(formula.strip(), mode="eval")
    except SyntaxError as e:
        raise DSLError(f"syntax error: {e.msg}")
    pf = ParsedFormula(formula=formula.strip(), tree=tree)

    def walk(node, depth=0):
        pf.n_nodes += 1
        if pf.n_nodes > MAX_NODES or depth > MAX_DEPTH:
            raise DSLError("formula too large")
        if isinstance(node, ast.Expression):
            return walk(node.body, depth + 1)
        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                raise DSLError("operator not allowed")
            walk(node.left, depth + 1); walk(node.right, depth + 1); return
        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, (ast.USub, ast.UAdd)):
                raise DSLError("unary operator not allowed")
            walk(node.operand, depth + 1); return
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise DSLError("only numeric constants allowed")
            return
        if isinstance(node, ast.Name):
            if node.id not in FIELDS:
                raise DSLError(f"unknown field '{node.id}'")
            pf.fields.add(node.id); return
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.keywords:
                raise DSLError("only plain whitelisted calls allowed")
            fn = node.func.id
            if fn in CALENDAR:
                if node.args:
                    raise DSLError(f"{fn}() takes no arguments")
                pf.funcs.add(fn); return
            if fn not in FUNCS:
                raise DSLError(f"function '{fn}' not allowed")
            if len(node.args) != ARITY[fn]:
                raise DSLError(f"{fn} expects {ARITY[fn]} args")
            pf.funcs.add(fn)
            if fn == "lead":
                pf.uses_lead = True
            if fn in WINDOW_FUNCS:
                w = node.args[1]
                if not (isinstance(w, ast.Constant) and isinstance(w.value, (int, float))):
                    raise DSLError("window argument must be a numeric constant")
                _win(w.value)
                pf.windows.append(int(w.value))
            for a in node.args:
                walk(a, depth + 1)
            return
        raise DSLError(f"node type {type(node).__name__} not allowed")

    walk(tree)
    return pf


def evaluate(formula: str | ParsedFormula, panel: dict) -> np.ndarray:
    """Evaluate on a panel dict {'close','ret','volume','btm','sentiment' : (T,N), 'dates': DatetimeIndex}."""
    pf = parse(formula) if isinstance(formula, str) else formula
    T, N = panel["close"].shape

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.Name):
            return panel[node.id]
        if isinstance(node, ast.UnaryOp):
            v = ev(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.BinOp):
            a, b = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Add): return a + b
            if isinstance(node.op, ast.Sub): return a - b
            if isinstance(node.op, ast.Mult): return a * b
            with np.errstate(divide="ignore", invalid="ignore"):
                return a / np.where(b == 0, np.nan, b) if isinstance(b, np.ndarray) else a / (b if b else np.nan)
        if isinstance(node, ast.Call):
            fn = node.func.id
            if fn in CALENDAR:
                v = panel["dates"].dayofweek.to_numpy() if fn == "dow" else panel["dates"].month.to_numpy()
                return np.repeat(v[:, None].astype(float), N, axis=1)
            args = [ev(a) for a in node.args]
            with np.errstate(all="ignore"):
                return FUNCS[fn](*args)
        raise DSLError("unreachable")

    out = ev(pf.tree)
    if not isinstance(out, np.ndarray):
        raise DSLError("formula must produce a (time x asset) array")
    out = np.asarray(out, float)
    out[~np.isfinite(out)] = np.nan
    return out


def scale_windows(formula: str, factor: float) -> str:
    """Return the formula with every window constant scaled by `factor` (parameter-fragility probes)."""
    pf = parse(formula)

    class T(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id in WINDOW_FUNCS:
                w = node.args[1]
                node.args[1] = ast.Constant(value=int(max(1, round(w.value * factor))))
            return node

    tree = T().visit(ast.parse(pf.formula, mode="eval"))
    return ast.unparse(ast.fix_missing_locations(tree))
