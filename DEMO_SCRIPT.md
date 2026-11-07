# The 3-minute demo script

Everything below runs **offline, with no API key** (MockLLM, pre-downloaded data, recorded runs).
Start the app: `python -m uvicorn skeptic.backend.app:app --port 8000`, open http://localhost:8000, keep **Demo mode (replay)** selected.
Terminal-only version of the same story: `python -m skeptic.demo` (about 25 seconds, scaled to stay under 3 minutes).

| Time | Screen | Say | What the audience sees |
|---|---|---|---|
| 0:00 | **Naive AI vs Skeptic** tab, scenario *Market swarm*, press **Replay** | "Finding patterns is easy; knowing which to trust is hard. Here a naive AI analyst scans 141 ideas on market data." | Left column fills with ~59 "discoveries!" (anything with p < 0.05). |
| 0:25 | same tab, right column | "Skeptic gets the same data. It does not ask whether the winner looks good. It asks whether luck could have produced the best result of the whole search." | Right column strikes claims through as the Court judges them; survivor counter settles near 10. |
| 0:50 | **Court** tab | "Every idea any agent tried goes in a ledger. The Court re-runs the whole search in 300 worlds where nothing is real." | Swarm graph lights up (Cleaner, Librarian, Hunters, null worlds, Prosecutors, Judge). Ledger ticker climbs. Credibility Meter moves each round. Zero-token ops dwarf LLM calls. |
| 1:15 | Graveyard | "Ideas that survive the null still have to get past the Prosecutors. Each death has a cause." | Tombstones: beta (it was never alpha), after costs, leakage, survivorship, multiple comparisons. Click one: null-world histogram + prosecutor evidence + the Librarian's papers. |
| 1:30 | Verdicts + ground truth | "Ten survive. The simulator knows the truth: all ten contain a planted effect. The naive analyst claimed 59." | Discoveries with the confession line ("I tried 141 ideas... luck alone would produce this 0.3% of the time"), pre-registration hash, out-of-sample confirmation. |
| 1:45 | **Benchmark** tab | "Across 20 seeds with known truth: false-discovery rate of naive, Bonferroni and BH is 64-90%; Skeptic's is 0% at 80% recall." | Methods table, calibration plot, power curve, the live null-world slider, and the failure-mode section. |
| 2:05 | **Court** tab, scenario *Real equities* | "Real prices, 117 large caps. The universe is today's survivors, so survivorship bias is in the data. Skeptic says so, and zero of 12 naive discoveries survive." | Red survivorship banner; stress-test evidence in the drawer. Contrast with the market swarm where the Prosecutor re-runs on the true point-in-time universe and kills survivor-only ideas live. |
| 2:25 | **Replication** tab, *Replay example* | "Paste a paper's claim. The Librarian finds the source and checks it applies to *this* data. The replicator tries every variant a researcher could, and counts them all." | Extracted hypothesis, papers with caveats (post-publication decay), rolling Sharpe with a "published" line, verdict + confession. |
| 2:40 | scenario *Health claim* | "Second domain. 'Supplement users are less healthy.' Pooled, that is true: t = -13. Hold age fixed and it reverses." | Simpson's paradox tombstone: pooled -0.36 SD vs within-age-band +0.25 SD. (Or: A/B test where peeking kills early wins; or betting where a random side earns the same ROI.) |
| 2:55 | Close | "Every field that searches for patterns has this problem. We built the AI that checks its own luck." | Honesty guardrails footer. |

## Q&A cheat sheet
* **Is the LLM doing the statistics?** No. Null worlds, ledger, multiple-testing math, backtests and most attacks are plain numpy/scipy (see the Cost & tokens panel). LLMs only propose, review specs, read papers and write explanations.
* **Why not just Bonferroni/BH?** They correct p-values but do not know about decoys (beta), leakage, survivorship, costs or regimes. The benchmark shows the cost.
* **Where does it fail?** Hidden confounders it was not given, weak real effects (power curve), generators that miss real structure, and a swarm that runs out of credibility on pure noise. All measured on the Benchmark tab.
* **Is the betting data real?** It looks machine-generated (synthetic names, uniform odds); the UI says so. The finding is about *that file*, not real markets.
