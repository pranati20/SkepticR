"""Institutional 'tear sheet': a compact PDF of one finished Skeptic run.

Built only from the run's own event stream (nothing is re-computed and nothing is written by a model):
the Hunters' claim, the Prosecutors' verdicts and the reasons ideas died, the Cost of Illusion, the papers consulted, the limits.
"""
from __future__ import annotations

import io
import time
from collections import Counter
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

INK, MUTE, GOOD, BAD, MAYBE = colors.HexColor("#0f172a"), colors.HexColor("#64748b"), colors.HexColor("#047857"), colors.HexColor("#be123c"), colors.HexColor("#b45309")
VERDICT_PLAIN = {"DISCOVERY": "Looks real", "LUCK": "Probably luck", "CAN'T TELL YET": "Not enough data"}


def _t(x, n=110) -> str:
    """Helvetica is latin-1 only: keep the text safe and short."""
    s = str(x if x is not None else "")
    s = s.replace("−", "-").replace("–", "-").replace("—", "-").replace("’", "'").replace("“", '"').replace("”", '"')
    s = s.encode("latin-1", "replace").decode("latin-1")
    return escape(s if len(s) <= n else s[: n - 1] + "...")


def _usd(x: float) -> str:
    return f"{'-' if x < 0 else ''}${abs(round(x)):,}"


def build_pdf(events: list[dict]) -> bytes:
    ev = [e for e in events if isinstance(e, dict)]
    by = lambda t: [e for e in ev if e.get("type") == t]
    meta = (by("meta") or [{}])[0]
    desc = {e["id"]: e.get("description") for e in by("proposal")}
    verdicts = {}
    for v in by("verdict"):
        verdicts[v["hypothesis_id"]] = v
    vs = list(verdicts.values())
    n = Counter(v["verdict"] for v in vs)
    naive = (by("naive") or [{}])[-1]
    ledger = ((by("done_summary") or [{}])[-1]).get("ledger") or {}

    ss = getSampleStyleSheet()
    H1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=18, leading=22, textColor=INK, alignment=0, spaceAfter=2)
    H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=10.5, leading=13, textColor=INK, spaceBefore=6, spaceAfter=2)
    B = ParagraphStyle("b", parent=ss["BodyText"], fontSize=8.6, leading=11.5, textColor=INK)
    S = ParagraphStyle("s", parent=B, fontSize=7.4, leading=9.5, textColor=MUTE)
    C = ParagraphStyle("c", parent=B, fontSize=7.8, leading=9.6)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=10 * mm, bottomMargin=10 * mm,
                            title=f"SkepticR tear sheet: {meta.get('title', 'run')}", author="SkepticR")
    f = []
    f.append(Paragraph("SkepticR - Strategy Tear Sheet", H1))
    f.append(Paragraph(f"{_t(meta.get('title', 'Run'), 90)} &nbsp;|&nbsp; generated {time.strftime('%Y-%m-%d %H:%M')} &nbsp;|&nbsp; "
                       f"{_t(meta.get('blurb', ''), 140)}", S))

    concl = (by("conclusion") or [None])[-1]
    if concl:
        f.append(Paragraph("Question and conclusion (from valid findings only)", H2))
        f.append(Paragraph("<b>Question:</b> " + _t(concl.get("question"), 300), B))
        for line in str(concl.get("text", "")).splitlines()[:14]:
            if line.strip():
                f.append(Paragraph(_t(line, 260), B))

    # --- headline
    total = len(vs)
    f.append(Paragraph("Conclusion", H2))
    cant = n["CAN'T TELL YET"]
    tested = ledger.get("tests") or total
    f.append(Paragraph(
        f"The AI tested <b>{tested}</b> ideas"
        + (f" ({ledger.get('cleaning_choices', 0)} data-cleaning choices counted as extra forking paths)" if ledger.get("cleaning_choices") else "")
        + f". A naive analyst would have reported <b>{naive.get('count', 'n/a')}</b> as 'significant'. After the court: "
        f"<font color='#047857'><b>{n['DISCOVERY']}</b> look real</font>, <font color='#be123c'><b>{n['LUCK']}</b> are probably luck</font>, "
        f"<font color='#b45309'><b>{cant}</b> cannot be judged yet</font>. "
        "'Looks real' means it survived every test; it is not proof and not investment advice.", B))

    # --- Hunter's claim
    claims = (naive.get("claims") or [])[:3]
    if claims:
        f.append(Paragraph("The Researchers' claim (before the court)", H2))
        for c in claims:
            f.append(Paragraph(f"- {_t(c.get('text'), 80)} (t = {c.get('t', 0):.1f}, naive p = {c.get('p', 0):.1e})", B))

    # --- verdict table
    rank = {"DISCOVERY": 0, "CAN'T TELL YET": 1, "LUCK": 2}
    rows_src = sorted(vs, key=lambda v: (rank.get(v["verdict"], 3), -(v.get("statistic") or 0)))
    shown = [v for v in rows_src if v["verdict"] == "DISCOVERY"][:5] + [v for v in rows_src if v["verdict"] != "DISCOVERY"][:4]
    if shown:
        f.append(Paragraph("The Skeptics' verdicts (most notable ideas)", H2))
        data = [["Idea", "Verdict", "Evidence", "Why"]]
        for v in shown:
            att = v.get("attacks") or []
            failed = [a["attack"] for a in att if not a.get("survived")]
            why = (v.get("cause_of_death") or "") if v["verdict"] != "DISCOVERY" else f"survived {len(att) - len(failed)} of {len(att)} attacks" + (f" (flags: {', '.join(failed)})" if failed else "")
            data.append([Paragraph(_t(v.get("description") or desc.get(v["hypothesis_id"]) or v["hypothesis_id"], 60), C), VERDICT_PLAIN.get(v["verdict"], v["verdict"]),
                         "n/a" if v["verdict"] == "CAN'T TELL YET" else f"{(v.get('confidence') or 0) * 100:.0f}%", Paragraph(_t(why or ("needs more data to judge" if v["verdict"] == "CAN'T TELL YET" else ""), 70), C)])
        t = Table(data, colWidths=[62 * mm, 28 * mm, 18 * mm, 72 * mm], repeatRows=1)
        st = [("FONTSIZE", (0, 0), (-1, -1), 7.8), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("BACKGROUND", (0, 0), (-1, 0), INK),
              ("VALIGN", (0, 0), (-1, -1), "TOP"), ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
              ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")])]
        for i, v in enumerate(shown, 1):
            st.append(("TEXTCOLOR", (1, i), (1, i), GOOD if v["verdict"] == "DISCOVERY" else BAD if v["verdict"] == "LUCK" else MAYBE))
        t.setStyle(TableStyle(st))
        f.append(t)
        f.append(Paragraph("Evidence = chance the idea is not luck, capped at 99%. It is an audit score, not a probability of future profit.", S))

    # --- why ideas died
    causes = Counter(v["cause_of_death"] for v in vs if v.get("cause_of_death"))
    if causes:
        f.append(Paragraph("What killed the ideas", H2))
        f.append(Paragraph("; ".join(f"{_t(c, 60)}: <b>{k}</b>" for c, k in causes.most_common(6)), B))

    # --- Cost of illusion
    ill = by("illusion")
    if ill:
        f.append(Paragraph("The Cost of Illusion (lucky-looking strategies the Judge threw out)", H2))
        data = [["Strategy", "Backtest said", "Reality check", "Believing it would cost"]]
        for e in ill[:3]:
            stake = 100 if e.get("kind") == "bets" else 10000
            a, r = e["in_total"] * stake, e["real_total"] * stake
            data.append([Paragraph(_t(e.get("description"), 60), C), _usd(a), Paragraph(f"{_usd(r)} ({_t(e.get('basis'), 60)})", C), _usd(max(0, a - r))])
        t = Table(data, colWidths=[58 * mm, 26 * mm, 58 * mm, 38 * mm])
        t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 7.8), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                               ("VALIGN", (0, 0), (-1, -1), "TOP"), ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1"))]))
        f.append(t)
        f.append(Paragraph("Illustrative: $10,000 invested (or $100 per bet) over the backtest window, before fees unless stated. Past results, not a forecast.", S))

    # --- papers
    seen, papers = set(), []
    for e in by("literature"):
        for p in e.get("papers") or []:
            k = p.get("url") or p.get("title")
            if k and k not in seen:
                seen.add(k)
                papers.append(p)
    if papers:
        f.append(Paragraph("Papers the Librarian consulted (live search results)", H2))
        for p in papers[:4]:
            f.append(Paragraph(f"- {_t(p.get('title'), 100)} - {_t(p.get('authors'), 40)} ({p.get('year') or 'n.d.'}) - {_t(p.get('url'), 60)}", C))

    f.append(Paragraph("Limits", H2))
    for h in (meta.get("honesty") or [])[:3]:
        f.append(Paragraph("- " + _t(h, 220), S))
    doc.build(f)
    return buf.getvalue()
