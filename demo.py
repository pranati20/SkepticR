"""Offline demo.   python -m skeptic.demo            replay the 3-minute story in the terminal (records first if needed)
                   python -m skeptic.demo --record   (re)record the runs used by the UI's Demo mode
No API key, no network. Replays deterministically from recorded event streams in under 3 minutes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .backend.runner import run_scenario
from .backend.app import _clean

DEMO = Path(__file__).resolve().parent / "demo" / "runs"
PLAN = [("market", {}), ("equities", {}), ("synthetic", {"n_worlds": 200}), ("betting", {}), ("health", {}), ("ab_test", {}),
        ("replication", {"dataset": "market"})]


def record(only: list[str] | None = None):
    DEMO.mkdir(parents=True, exist_ok=True)
    for name, params in PLAN:
        if only and name not in only:
            continue
        t = time.time()
        events, _ = run_scenario(name, params, None, seed=0)
        (DEMO / f"{name}.json").write_text(json.dumps(_clean(events)), encoding="utf-8")
        print(f"recorded {name:12s} {len(events):4d} events in {time.time()-t:4.1f}s", flush=True)


def load(name: str) -> list[dict]:
    f = DEMO / f"{name}.json"
    if not f.exists():
        record([name])
    return json.loads(f.read_text(encoding="utf-8"))


def say(s: str, pause: float = 0.0):
    print(s, flush=True)
    time.sleep(pause)


def replay(speed: float = 1.0, budget: float = 165.0):
    """Narrated replay of the 3-minute script; pacing is scaled so the total stays under `budget` seconds."""
    steps = []

    def wait(x): steps.append(x)
    t0 = time.time()
    B = budget / 165.0
    sleep = lambda s: time.sleep(max(0.0, s * B / speed))

    say("=" * 78); say(" SKEPTIC  -  continuous integration for claims"); say(" An AI researcher that audits its own luck."); say("=" * 78); sleep(1.2)

    ev = load("market")
    naive = next(e for e in ev if e["type"] == "naive")
    say(f"\n[1] A NAIVE AI analyst scans {naive['total']} ideas on market data and announces {naive['count']} 'discoveries' (p<0.05):")
    for c in naive["claims"][:5]:
        say(f"      * {c['text']:<58s} t={c['t']:+.1f}"); sleep(0.25)
    say(f"      ... and {naive['count']-5} more."); sleep(1.2)

    say("\n[2] SKEPTIC runs the same data. Every idea goes in the LEDGER; the Court re-runs the WHOLE search in null worlds:")
    last_led = None
    for e in ev:
        if e["type"] == "agent" and e.get("agent") in ("Cleaner", "Court"):
            say(f"      [{e['agent']}] {e['text']}"); sleep(0.35)
        if e["type"] == "ledger" and e.get("tests") != last_led and e["tests"] % 24 == 0:
            last_led = e["tests"]; say(f"      ledger: {e['tests']} ideas tried, {e['cleaning_choices']} cleaning choices"); sleep(0.2)
        if e["type"] == "null_audit":
            say(f"      null audit: search size {e['search_size']}, luck's best result ~{e['histogram']['p95']:.1f} (95th pct), level {e['level']:.3f}"); sleep(0.5)
        if e["type"] == "credibility":
            say(f"      Credibility Meter: {e['wealth']:.3f} / {e['wealth0']:.3f}"); sleep(0.35)
    grave = [e for e in ev if e["type"] == "graveyard"]
    from collections import Counter
    say(f"\n      THE GRAVEYARD now holds {len(grave)} tombstones:")
    for cause, n in Counter(g["cause"] for g in grave).most_common():
        say(f"         {n:3d} x {cause}"); sleep(0.3)

    say("\n[3] What survives:")
    vs = [e for e in ev if e["type"] == "verdict" and e["verdict"] == "DISCOVERY"]
    for v in vs[:6]:
        say(f"      DISCOVERY  {v['formula'][:56]:<56s} conf {v['confidence']:.2f}"); sleep(0.3)
    tc = next(e for e in ev if e["type"] == "truth_check")
    sc = tc["score"]
    say(f"      vs the planted truth: {sc['tp']} of {sc['claimed']} claims are real, false discoveries = {sc['fp']}  (naive claimed {naive['count']})")
    say(f"      \"{vs[0]['confession']}\""); sleep(1.2)
    pr = [e for e in ev if e["type"] == "prereg"]
    if pr:
        say(f"      pre-registered {pr[0]['hash'][:16]}... then scored on a held-out slice: OOS Sharpe {pr[0]['oos']['oos_sharpe']:.2f}, confirmed={pr[0]['oos']['confirmed']}"); sleep(0.8)

    bm = Path(__file__).resolve().parent / "demo" / "benchmark.json"
    if bm.exists():
        b = json.loads(bm.read_text(encoding="utf-8"))
        say(f"\n    BENCHMARK ({b['n_seeds']} seeds, ground truth known):  method / false-discovery rate / recall")
        for m, v in b["table"].items():
            say(f"      {m:26s} FDR {v['pooled_fdr']:.2f}   recall {v['recall']['mean']:.2f}"); sleep(0.18)
        sleep(0.6)

    ev = load("equities")
    say("\n[4] REAL EQUITIES (117 large caps, 2010-2025). Universe = today's survivors -> survivorship bias is built in:")
    n = next(e for e in ev if e["type"] == "naive")
    say(f"      naive analyst: {n['count']} 'discoveries'.  Skeptic: {sum(e['verdict']=='DISCOVERY' for e in ev if e['type']=='verdict')} survive."); sleep(1.0)
    ev_m = load("market")
    sv = [e for e in ev_m if e["type"] == "prosecution"]
    killed = next((e for e in ev_m if e["type"] == "graveyard" and "survivorship" in e["cause"]), None)
    if killed:
        a = next(x for x in killed["attacks"] if x["attack"] == "survivorship")["evidence"]
        say(f"      PROSECUTOR caught live: '{killed['formula']}' used today's constituents.")
        say(f"        reported Sharpe {a.get('reported_sharpe',0):.2f} -> {a.get('point_in_time_sharpe',0):.2f} on the true point-in-time universe  => {killed['cause']}"); sleep(1.5)

    ev = load("replication")
    d = next(e for e in ev if e["type"] == "replication" and e.get("stage") == "decay")
    ex = next(e for e in ev if e["type"] == "replication" and e.get("stage") == "extracted")["extracted"]
    say(f"\n[5] REPLICATION MODE: paper claim -> '{ex['name']}' ({ex.get('source')}) -> rebuilt on our data")
    if "sharpe_pre" in d:
        say(f"      Sharpe before publication {d['sharpe_pre']:.2f}, after {d['sharpe_post']:.2f}  =>  {d['verdict']}")
    say(f"      Judge: {d['primary_verdict']}.  \"{d['confession']}\""); sleep(1.5)

    ev = load("health")
    say("\n[6] SECOND DOMAIN: health headline 'supplement users are LESS healthy'")
    ov = next(e for e in ev if e["type"] == "verdict" and e["hypothesis_id"].endswith("|overall"))
    sim = next(a for a in ov["attacks"] if a["attack"] == "simpson")["evidence"]
    say(f"      pooled effect {sim['pooled_effect']:+.2f} SD  ->  within-age-band effect {sim['stratum_adjusted_effect']:+.2f} SD  (reversed)  => {ov['cause_of_death']}")
    say(f"      \"{ov['confession']}\"")
    bet = load("betting")
    bv = next(e for e in bet if e["type"] == "verdict" and e["hypothesis_id"] == "bet|all predicted bets")
    pl = next(a for a in bv["attacks"] if a["attack"] == "placebo")["evidence"]
    say(f"      and the betting 'edge': ROI {pl['roi_predicted_pct']:+.0f}% - but a random side earns {pl['roi_random_side_pct']:+.0f}%  => {bv['cause_of_death']}"); sleep(1.5)

    say("\n" + "-" * 78)
    say(' "Every field that searches for patterns has this problem.')
    say('  We built the AI that checks its own luck."')
    say("-" * 78)
    say(f"(replayed offline in {time.time()-t0:.0f}s, no API key, no network; honesty: reduces false discoveries, does not prove truth; not trading advice)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--speed", type=float, default=1.0)
    a = ap.parse_args()
    if a.record:
        record()
    else:
        replay(a.speed)
