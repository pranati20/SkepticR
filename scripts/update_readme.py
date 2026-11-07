"""Fill the README results block from demo/benchmark.json.   python skeptic/scripts/update_readme.py"""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
b = json.loads((root / "demo" / "benchmark.json").read_text(encoding="utf-8"))
rows = []
for m, v in b["table"].items():
    rows.append(f"| {'**' + m + '**' if m.startswith('SKEPTIC') else m} | {v['claimed']['mean']:.1f} | {v['tp']['mean']:.2f} | {v['fp']['mean']:.1f} | "
                f"{v['precision']['mean']:.2f} ± {v['precision']['ci95']:.2f} | {v['recall']['mean']:.2f} ± {v['recall']['ci95']:.2f} | {100*v['pooled_fdr']:.0f}% | "
                f"{v['fp_decoy']['mean']:.1f} / {v['fp_trap']['mean']:.1f} / {v['fp_null']['mean']:.1f} |")
cal = "; ".join(f"stated {c['mean_stated']:.2f} → actual {c['actual']:.2f} (n={c['n']})" for c in b["calibration"] if c["n"] >= 5)
pc = b["power_curve"]["SKEPTIC (full court)"]
pw = ", ".join(f"SR {p['sr']:.1f}: {100*p['detect']:.0f}%" for p in pc)
block = f"""## Results: synthetic benchmark ({b['n_seeds']} seeds, ground truth known)
{b['setup']['signals']} signals over {b['setup']['assets']} assets, {b['setup']['search_days']} search days + {b['setup']['holdout_days']} held out; {b['setup']['planted']} planted real signals (Sharpe {', '.join(map(str, b['setup']['planted_sr']))}), {b['setup']['decoys']} confounded decoys, {b['setup']['traps']} traps. Mean per run, ± 95% CI over seeds; FDR pooled over runs.

| Method | claimed | true | false | precision | recall | FDR | false = decoy / trap / noise |
|---|---|---|---|---|---|---|---|
""" + "\n".join(rows) + f"""

* **Calibration** (stated confidence vs actual hit rate, all hypotheses, all seeds): {cal}. Confidence is modestly optimistic in the top bin.
* **Power** (Skeptic, detection rate by true Sharpe): {pw}. Safety is paid for with lower power on weak signals.
* Per run, {b['real_signals_called_luck_per_run']:.2f} of the 5 real signals were called LUCK and {b['real_signals_cant_tell_per_run']:.2f} CAN'T TELL YET.
* Measured failure modes: """ + "; ".join(f"**{f['title']}** ({f['metric']})" for f in b["failure_modes"][:1] + b["failure_modes"][3:4]) + """. See the Benchmark tab for all of them.
"""
p = root / "README.md"
t = p.read_text(encoding="utf-8")
a, z = t.index("<!--BENCH-->") + len("<!--BENCH-->"), t.index("<!--/BENCH-->")
p.write_text(t[:a] + "\n" + block + t[z:], encoding="utf-8")
print("README updated")
