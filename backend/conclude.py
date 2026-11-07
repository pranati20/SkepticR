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
          "These are statistical findings, not proof of causation and not investment or medical advice; say so briefly. Be concise and plain. Begin with \"Short answer:\" and one plain-English sentence that answers the QUESTION directly. "
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


def _segment(fid: str) -> tuple[str, str]:
    """'conversion|device=mobile' -> ('conversion', 'device=mobile'); 'bet|all predicted bets' -> ('bet', 'all predicted bets')."""
    o, _, s = fid.partition("|")
    return o, s


def _words(seg: str) -> str:
    return seg.replace("&", "and").replace("=", " = ")


def plain_answer(question: str, f: dict) -> str:
    """A short plain-English answer to the user's own question, built only from the valid findings (no model, fully checkable)."""
    fs = f["findings"]
    q = f'"{question}"' if question else "your question"
    if not fs:
        return (f"Short answer: we cannot say. The data does not give a reliable answer to {q}. Everything we found either looked like luck or bias "
                f"or had too little data. More data, or a more specific question, may help.")
    treat = (f.get("data") or {}).get("treat")
    by_outcome: dict = {}
    for x in fs:
        o, s = _segment(x["id"])
        by_outcome.setdefault(o, []).append((s, x))
    parts = []
    for o, items in list(by_outcome.items())[:3]:
        overall = [x for s, x in items if s in ("overall", "all predicted bets")]
        segs = [s for s, _ in items if s not in ("overall", "all predicted bets")]
        dirs = {x["direction"] for _, x in items if x["direction"]}
        d = (" (" + ("higher" if dirs == {"treated group higher"} else "lower" if dirs == {"treated group lower"} else "in different directions") +
             (f" for the {treat} group" if treat else "") + ")") if dirs else ""
        s_txt = ""
        if overall:
            s_txt = f"a real effect on {o}{d} that holds overall"
        if segs:
            shown = ", ".join(_words(s) for s in segs[:4]) + (f" and {len(segs) - 4} more groups" if len(segs) > 4 else "")
            s_txt += (" and " if s_txt else f"a real effect on {o}{d} ") + f"within: {shown}"
        parts.append(s_txt)
    more = f" ({len(by_outcome) - 3} more outcomes also held up)" if len(by_outcome) > 3 else ""
    strongest = max(x["evidence"] for x in fs)
    skip = f["n_luck"] + f["n_unclear"]
    return (f"Short answer: yes, in part. Of {f['n_tested']} things tried to answer {q}, {len(fs) if f['n_valid_total'] <= len(fs) else f['n_valid_total']} held up after "
            f"every check. The data supports " + "; and ".join(parts) + more + f". Strongest evidence is about {strongest * 100:.0f}% that it is not luck. "
            f"The other {skip} things tried (including any headline that looked impressive at first) did not hold up, so we do not claim them. "
            "This shows an association in this data, not proof of cause, and it is not advice.")


def _sanitize(text: str, n: int) -> str:
    text = re.sub(r"\[F(\d+)\]", lambda m: m.group(0) if 1 <= int(m.group(1)) <= n else "", text)
    return re.sub(r"https?://\S+", "", text).strip()


def conclude(question: str, events: list[dict], llm: LLM | None = None) -> dict:
    question = (question or "").strip()[:500]
    f = facts(events)
    fs = f["findings"]
    base = {"question": question, "findings": fs, "n_valid": f["n_valid_total"], "n_tested": f["n_tested"], "n_luck": f["n_luck"],
            "n_unclear": f["n_unclear"], "top_causes": f["top_causes"], "answer": plain_answer(question, f)}
    def with_answer(r: dict) -> dict:
        """Put the plain-English answer at the top of the write-up the UI already displays."""
        body = r["text"] if r["text"].startswith("Short answer:") else base["answer"] + NL + r["text"]
        return {**r, "text": body}
    if not fs:
        why = f"{f['n_luck']} looked like luck or bias and {f['n_unclear']} had too little data to judge"
        return {**base, "status": "none", "llm": "none", "text": base["answer"] + NL + f"No valid conclusion can be drawn. Skeptic examined {f['n_tested']} possible findings and none survived every check ({why}). "
                        f"So the data, as it stands, does not support an answer to: \"{question}\". More data or a sharper question may help."}
    llm = llm or LLM()
    if llm.is_mock:
        lines = [f"Question: {question}", "", "Valid findings (they survived every check):"]
        for x in fs:
            lines.append(f"- [{x['ref']}] {x['claim']}" + (f" ({x['direction']})" if x["direction"] else "") + f"; evidence {x['evidence'] * 100:.0f}%, passed {x['checks_passed']} attacks")
        lines += ["", f"{f['n_luck']} other candidate findings were judged luck or bias and {f['n_unclear']} lacked data; none of them are used here.",
                  "(No AI model is configured, so this is a plain summary, not a written analysis.)"]
        return with_answer({**base, "status": "valid", "llm": "offline (no model)", "text": "\n".join(lines)})
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
