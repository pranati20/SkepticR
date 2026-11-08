"""Conclusion from VALID findings only.

Flow:  prompt (what do we want to conclude?) -> dataset(s) -> Skeptic audit -> keep ONLY findings the Judge ruled DISCOVERY
       -> write the conclusion from those findings alone.

Guarantees:
  * The writer (LLM or template) never sees ideas ruled LUCK or CAN'T TELL YET: it is told only how many there were.
  * If nothing survived, no model is called and the answer says plainly that no valid conclusion can be drawn.
  * The model may cite findings only as [F1], [F2]...; unknown citations and URLs are stripped; the findings table is
    returned next to the text so the reader can verify every number.
"""
from __future__ import annotations

import re
from collections import Counter

from .llm import LLM

MAX_FINDINGS = 12
NL = chr(10) * 2

SYSTEM = ("You write the conclusion of a data analysis that has ALREADY been audited for luck, leakage and bias. You are given the user's QUESTION "
          "and a numbered list of VALID FINDINGS: the only results that survived every statistical check. "
          "Rules: answer the question using ONLY these findings and cite them as [F1], [F2]. Never mention, guess or imply any result that is not listed. "
          "Do not invent numbers, causes or links. Say whether the findings fully, partly or do not answer the question. "
          "These are statistical findings, not proof of causation and not investment or medical advice; say so briefly. Be concise and plain. Begin with \"Yes.\", \"No.\", \"It depends.\" or \"Can't say.\" answering the QUESTION directly, then \"Why:\" in plain English. "
          "Reply with JSON only: {\"text\": \"...\"}.")
SCHEMA = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}


def facts(events: list[dict]) -> dict:
    """Valid findings (DISCOVERY only) plus bare counts of everything else."""
    verdicts: dict = {}
    for e in events:
        if e.get("type") == "verdict":
            verdicts[e["hypothesis_id"]] = e
    signed = {}
    for e in events:
        if e.get("type") == "naive":
            for c in e.get("claims") or []:
                signed[c["id"]] = c.get("t")
    meta = next((e for e in events if e.get("type") == "meta"), {})
    pan = meta.get("panel") or {}
    pros = {}
    for e in events:
        if e.get("type") == "prosecution":
            pros[e["hypothesis_id"]] = e.get("attacks") or []
    valid = sorted((v for v in verdicts.values() if v["verdict"] == "DISCOVERY"), key=lambda v: (-(v.get("confidence") or 0), -(v.get("statistic") or 0)))
    out = []
    for i, v in enumerate(valid[:MAX_FINDINGS], 1):
        att = v.get("attacks") or pros.get(v["hypothesis_id"]) or []
        failed = [a["attack"] for a in att if not a.get("survived")]
        t = signed.get(v["hypothesis_id"])
        out.append({"ref": f"F{i}", "id": v["hypothesis_id"], "claim": v.get("description") or v["hypothesis_id"], "formula": v.get("formula"),
                    "direction": (("treated group higher" if t > 0 else "treated group lower") if (t is not None and pan.get("treat")) else None),
                    "evidence": round(float(v.get("confidence") or 0), 3), "p_adj": v.get("p_adj"), "power": v.get("power_at_observed"),
                    "checks_passed": f"{len(att) - len(failed)} of {len(att)}", "flags": failed})
    c = Counter(v["verdict"] for v in verdicts.values())
    causes = Counter(v["cause_of_death"] for v in verdicts.values() if v.get("cause_of_death"))
    return {"findings": out, "n_valid_total": len(valid), "n_tested": len(verdicts), "n_luck": c["LUCK"], "n_unclear": c["CAN'T TELL YET"],
            "top_causes": [k for k, _ in causes.most_common(3)], "data": {"treat": pan.get("treat"), "outcomes": pan.get("outcomes")}}


EFFECT_RE = re.compile(r"increas|decreas|effect|improv|help|reduc|rais|lower|higher|better|worse|impact|caus|boost|harm|benefit|work|affect|change|differ|compar|less|more|safe|risk", re.I)
OUTLIER_RE = re.compile(r"outlier|anomal|unusual|extreme|abnormal|weird|strange|exceptional|suspicious", re.I)


def classify(question: str) -> str:
    """What is the user asking? 'outliers' | 'effect' (does X change Y) | 'other'. Plain keyword rules, so it is inspectable."""
    q = question or ""
    if OUTLIER_RE.search(q):
        return "outliers"
    return "effect" if EFFECT_RE.search(q) else "other"


def outlier_answer(question: str, table) -> dict | None:
    """Answer 'what are the outliers' from the DATA itself: robust z-score (median/MAD > 3.5), how many, where they concentrate, extremes.
    Descriptive: it is not run through the luck audit, and an outlier is not necessarily an error."""
    import pandas as pd
    from scipy import stats as st
    if table is None:
        return None
    df, treat = table.df, table.treat
    cols = [c for c in table.outcomes if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    cats = [c for c in table.subgroup_cols if c in df.columns]
    if not cols:
        return None
    N = len(df)
    out, tops = [], []
    for c in cols:
        x = df[c].astype(float)
        med = x.median()
        mad = (x - med).abs().median() or 1e-9
        z = 0.6745 * (x - med) / mad
        mask = z.abs() > 3.5
        n = int(mask.sum()); hi = int((z > 3.5).sum()); lo = int((z < -3.5).sum())
        exp = N * 2 * st.norm.sf(3.5)
        if n <= max(3, 2 * exp):
            verdict = "about what a normal bell curve would produce"
        else:
            verdict = f"about {n / max(exp, 1e-9):.0f}x more than a normal bell curve would produce, so the data has unusually heavy tails"
        out.append(f"In {c}, {n} of {N:,} values ({100 * n / N:.2f}%) are outliers: {hi} unusually high and {lo} unusually low. That is {verdict} (expected about {exp:.0f}).")
        if n == 0:
            r0 = df.assign(_z=z).sort_values("_z", key=lambda s: -s.abs()).iloc[0]
            tops.append(f"The most extreme value is {r0[c]:.2f} ({abs(r0['_z']):.1f} robust standard deviations from typical, below the 3.5 cut-off), so none is a real outlier.")
        if n:
            ex = df.loc[mask].assign(_z=z[mask]).sort_values("_z", key=lambda s: -s.abs()).head(3)
            tops.append(f"Most extreme {c} values: " + "; ".join(
                f"{r[c]:.2f} (" + ", ".join(f"{k}={r[k]}" for k in (cats + [treat])[:3] if k in df.columns) + ")" for _, r in ex.iterrows()) + ".")
            tests = []
            for g in cats + [treat]:
                if g not in df.columns or df[g].nunique() > 12:
                    continue
                tab = pd.crosstab(df[g], mask)
                if tab.shape[1] < 2 or tab.shape[0] < 2:
                    continue
                pv = st.chi2_contingency(tab)[1]
                tests.append((pv * (len(cats) + 1), g, tab))      # Bonferroni across the columns looked at
            tests.sort(key=lambda r: r[0])
            if tests and tests[0][0] < 0.05:
                pv, g, tab = tests[0]
                share = tab[True] / tab.sum(axis=1)
                top = share.idxmax()
                out.append(f"They are not spread evenly: they cluster in {g} = {top} ({100 * share[top]:.1f}% of that group are outliers versus {100 * n / N:.1f}% overall; "
                           f"adjusted p = {min(pv, 1):.1g}, so this is unlikely to be chance).")
            elif tests:
                out.append("They are spread roughly evenly across the groups, with no group standing out after correcting for the number of groups checked.")
    if not out:
        return None
    note = ("How this was found: a value is an outlier if it is more than 3.5 robust standard deviations (median and MAD) from the typical value. "
            "This describes the data; it is not a luck-audited discovery, and an outlier is not necessarily an error, so check it before dropping it.")
    return {"answer": " ".join(out + tops), "note": note}


def _segment(fid: str) -> tuple[str, str]:
    """'conversion|device=mobile' -> ('conversion', 'device=mobile'); 'bet|all predicted bets' -> ('bet', 'all predicted bets')."""
    o, _, s = fid.partition("|")
    return o, s


def _words(seg: str) -> str:
    return seg.replace("&", "and").replace("=", " = ")


def plain_answer(question: str, f: dict) -> str:
    """A plain-English Yes / No / Can't say answer to the user's own question, with the reason, built only from the valid findings."""
    fs = f["findings"]
    if not fs:
        return ("Can't say. The data does not give a reliable answer to this question. Everything we found either looked like luck or bias "
                "or had too little data. More data, or a more specific question, may help.")
    treat = (f.get("data") or {}).get("treat")
    outs = sorted({_segment(x["id"])[0] for x in fs})
    outcome = ", ".join(outs[:2]) if outs else "the outcome"
    dirs = {x["direction"] for x in fs if x["direction"]}
    segs = [s for s in (_segment(x["id"])[1] for x in fs) if s not in ("overall", "all predicted bets")]
    overall = any(_segment(x["id"])[1] in ("overall", "all predicted bets") for x in fs)
    who = treat or "the tested factor"
    if dirs == {"treated group higher"}:
        head = f"Yes. {who.capitalize()} goes with higher {outcome} in this data."
    elif dirs == {"treated group lower"}:
        head = f"No, the opposite. {who.capitalize()} goes with lower {outcome} in this data."
    elif len(dirs) > 1:
        head = f"It depends. {who.capitalize()} raises {outcome} for some groups and lowers it for others."
    else:
        head = "Yes, a real pattern was found."
    shown = ", ".join(_words(s) for s in segs[:4]) + (f" and {len(segs) - 4} more groups" if len(segs) > 4 else "")
    why = [f"Why: of {f['n_tested']} comparisons tried, {f['n_valid_total']} held up after every check (luck, bias and cost tests), "
           f"with up to {max(x['evidence'] for x in fs) * 100:.0f}% confidence it is not chance."]
    if segs:
        why.append(("It holds overall and " if overall else "It holds ") + f"when comparing like with like, e.g. within: {shown}.")
    causes = " ".join(f.get("top_causes") or []).lower()
    if "simpson" in causes or "confound" in causes or "beta" in causes:
        why.append("A simple raw comparison of the whole dataset was misleading (the groups being compared differ in other ways, such as age), "
                   "so it was not used; only comparisons that account for that are counted.")
    why.append(f"The other {f['n_luck'] + f['n_unclear']} comparisons did not hold up and are not claimed. "
               "This shows an association in this data, not proof of cause, and it is not medical, financial or other advice.")
    return head + " " + " ".join(why)


def _sanitize(text: str, n: int) -> str:
    text = re.sub(r"\[F(\d+)\]", lambda m: m.group(0) if 1 <= int(m.group(1)) <= n else "", text)
    return re.sub(r"https?://\S+", "", text).strip()


def conclude(question: str, events: list[dict], llm: LLM | None = None, table=None) -> dict:
    question = (question or "").strip()[:500]
    kind = classify(question)
    if kind == "outliers":
        o = outlier_answer(question, table)
        if o:
            return {"question": question, "kind": "outliers", "status": "outliers", "findings": [], "n_valid": 0, "n_tested": 0, "n_luck": 0, "n_unclear": 0,
                    "top_causes": [], "answer": o["answer"], "llm": "none", "text": o["answer"] + "\n\n" + o["note"]}
        kind = "other"
    if kind == "other":
        f0 = facts(events)
        d0 = f0["data"]
        tested = (f"This run tested whether {d0['treat']} changes {', '.join(d0['outcomes'] or [])} in different groups. " if d0.get("treat") else "")
        txt = ("Can't say. This audit can answer two kinds of question: does X change Y (checked for luck and bias), and what are the outliers. "
               f"Your question looks like neither, so I will not force an answer out of unrelated results. {tested}"
               "Try rephrasing, for example: Does <factor> increase <outcome>? or What are the outliers in <outcome>?")
        return {"question": question, "kind": "other", "status": "none", "findings": [], "n_valid": 0, "n_tested": f0["n_tested"], "n_luck": 0, "n_unclear": 0,
                "top_causes": [], "answer": txt, "llm": "none", "text": txt}
    f = facts(events)
    fs = f["findings"]
    base = {"question": question, "findings": fs, "n_valid": f["n_valid_total"], "n_tested": f["n_tested"], "n_luck": f["n_luck"],
            "n_unclear": f["n_unclear"], "top_causes": f["top_causes"], "answer": plain_answer(question, f), "kind": "effect"}
    def with_answer(r: dict) -> dict:
        """Put the plain-English answer at the top of the write-up the UI already displays."""
        body = r["text"] if r["text"].startswith(("Yes", "No", "It depends", "Can't say")) else base["answer"] + NL + r["text"]
        return {**r, "text": body}
    if not fs:
        why = f"{f['n_luck']} looked like luck or bias and {f['n_unclear']} had too little data to judge"
        return {**base, "status": "none", "llm": "none", "text": base["answer"] + NL + f"No valid conclusion can be drawn. SkepticR examined {f['n_tested']} possible findings and none survived every check ({why}). "
                        f"So the data, as it stands, does not support an answer to: \"{question}\". More data or a sharper question may help."}
    llm = llm or LLM()
    if llm.is_mock:
        # the findings themselves are rendered by the UI from `findings`; don't repeat them in the text
        refs = ", ".join(f"[{x['ref']}]" for x in fs)
        text = (f"{len(fs)} finding(s) survived every check ({refs}); they are listed below.\n"
                "(No AI model is configured, so this is a plain summary, not a written analysis.)")
        return with_answer({**base, "status": "valid", "llm": "offline (no model)", "text": text})
    digest = "\n".join(f"[{x['ref']}] {x['claim']} | direction: {x['direction'] or 'n/a'} | evidence {x['evidence']:.2f} | adjusted p {x['p_adj']:.3g} | "
                       f"power {('%.2f' % x['power']) if x['power'] is not None else 'n/a'} | checks passed {x['checks_passed']}"
                       + (f" | caution flags: {', '.join(x['flags'])}" if x["flags"] else "") for x in fs)
    prompt = (f"QUESTION: {question}\n\nVALID FINDINGS ({len(fs)} shown; {f['n_luck']} other candidates were rejected as luck/bias and "
              f"{f['n_unclear']} lacked data: do not discuss them):\n{digest}")
    try:
        obj, _ = llm.complete_json("conclusion", SYSTEM, prompt, SCHEMA, max_tokens=900)
        return with_answer({**base, "status": "valid", "llm": llm.provider.name, "text": _sanitize(str(obj.get("text", "")), len(fs))})
    except Exception as e:  # noqa: BLE001
        return {**base, "status": "valid", "llm": "error",
                "text": base["answer"] + NL + f"The AI model could not be reached ({type(e).__name__}). The valid findings are listed below; I have not written an analysis."}
