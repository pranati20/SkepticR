# Skeptic: an AI researcher that audits its own luck

**Continuous integration for claims.** An agent swarm hunts for patterns; a Court then tries to prove each one is luck before anything is called a discovery. It audits the *whole search*, not just the winning result.

> "Finding patterns in data is easy; knowing which ones to trust is hard." (Brevan Howard challenge)

The core insight: the more ideas an agent tries, the more certain it is that something looks brilliant by chance. Skeptic records **every** idea tried (the ledger), re-runs the **entire search** in many "null worlds" where nothing real exists, and only believes a finding if it beats the best result luck could produce, and then survives a swarm of Prosecutors. Verdicts are always **DISCOVERY**, **LUCK** or **CAN'T TELL YET** (insufficient power).

## Honesty guardrails (also in the UI)
* Skeptic **reduces false discoveries and quantifies uncertainty. It does not prove truth.**
* Null-world validity depends on the null generator preserving the data's real structure. Assumptions per generator are in `backend/court/null_worlds.py`.
* Backtests are not forecasts. **No trading advice.** Simulated data is labelled as a stand-in everywhere.
* Known methods, cited: White's Reality Check, Hansen's SPA, Deflated Sharpe Ratio (Bailey & Lopez de Prado), Benjamini-Hochberg, Romano-Wolf step-down, Foster & Stine alpha-investing. Our contribution is applying them to an autonomous agent's own search as one integrated, measurable system.

## Team setup (start here)
The code imports itself as the package `skeptic`, so the repo folder **must be named `skeptic`** and commands are run from its **parent** folder.

```bash
# 1. Clone into a parent folder, naming the clone "skeptic"
mkdir whack && cd whack
git clone https://github.com/pranati20/SkepticR.git skeptic

# 2. Backend (Python 3.12 recommended)
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell   (macOS/Linux: source .venv/bin/activate)
pip install -r skeptic/requirements.txt

# 3. Run the app (UI is pre-built and committed in skeptic/frontend/dist)
python -m uvicorn skeptic.backend.app:app --port 8000
#    -> open http://localhost:8000

# 4. Optional: frontend dev mode with hot reload (second terminal, backend still running)
cd skeptic/frontend
npm install
npm run dev                         # -> http://localhost:5173 (proxies /api and /ws to :8000)
npm run build                       # rebuild frontend/dist, then commit it

# 5. Tests (from the parent "whack" folder)
python -m pytest skeptic/tests -q
```
Working together: create a branch per change (`git checkout -b my-change`), and run `npm run build` and commit `frontend/dist` if you touched the frontend. No API key is needed; see below for real Claude.

## Run it (offline, no API key)
```bash
pip install -r skeptic/requirements.txt
python -m skeptic.demo                       # narrated 3-minute story in the terminal (replays recorded runs)
python -m uvicorn skeptic.backend.app:app --port 8000   # UI + API at http://localhost:8000 (frontend is pre-built in frontend/dist)
python -m pytest skeptic/tests -q            # statistics incl. known-answer tests; UI smoke test: skeptic/tests/ui/ui_smoke.py
```
Rebuild the UI: `cd skeptic/frontend && npm install && npm run build` (dev server: `npm run dev`, proxies to :8000).
Re-record demo runs: `python -m skeptic.demo --record`. Re-run the benchmark: `python -m skeptic.backend.benchmark.run_benchmark --seeds 20`.
Librarian (paper search) is **live by default**: Crossref + arXiv, plus OpenAlex if `OPENALEX_API_KEY` is set; every paper shown is a real API result with a DOI/arXiv link, and the curated index is a labelled fallback. Set `SKEPTIC_OFFLINE=1` to disable network use.
Real Claude instead of the mock: `SKEPTIC_USE_LLM=1 ANTHROPIC_API_KEY=...` (cheap model for Hunters/Cleaner, stronger for Prosecutor/Judge/report; every call cached on disk by prompt hash; live token + cost meter). Live paper search additionally uses Crossref when online (cached).

## Architecture
```
Raw data -> Cleaner -> Hunter swarm (+ Librarian) -> LEDGER
                                                       |
            Null-world engine (zero tokens) <----------+
                        |
            Alpha-investing level -> Prosecutor swarm -> Judge -> Verdict + confession + pre-registration + strategy card
                        |
                  Live UI (swarm, meters, Graveyard)
```
| Piece | Where | LLM? |
|---|---|---|
| Ledger (SQLite): every test, every cleaning choice, seeds, tokens | `backend/ledger.py` | no |
| Null worlds: sign-flip, stationary bootstrap (pnl); label permutation incl. within confounder strata; circular shift, IAAFT surrogate, placebo dates (raw features) | `backend/court/null_worlds.py` | no |
| Max-statistic p-values (single-step + Romano-Wolf step-down), White RC, Hansen SPA, Deflated Sharpe, BH, Bonferroni | `court/null_worlds.py`, `court/stats.py` | no |
| Alpha-investing "Credibility Meter" (prior scales the stake, never the null test) | `court/alpha_investing.py` | no |
| Judge: verdict, calibrated confidence (1 - q-value from the empirical null), power, "how much more data", confession line | `court/judge.py`, `court/power.py` | no |
| Pre-registration: SHA-256 of spec + direction + timestamp, scored on a slice never touched by search | `court/prereg.py` | no |
| Prosecutors: leakage, survivorship (re-run on point-in-time universe), costs, regime/years, concentration, confounder/beta (alpha re-audited across the whole search), fragility; Simpson, peeking, placebo | `prosecutors/attacks.py`, `adapters/tabular.py`, `adapters/betting.py` | spec review only |
| Librarian: finds papers, checks they apply to *this* scenario (sample overlap, post-publication decay, universe), feeds the Prosecutors | `backend/literature.py` | optional |
| Hunters (6 families) emitting structured JSON, safe feature DSL (`ast` whitelist, **no eval/exec**), Cleaner | `hunters/base.py`, `features_dsl.py`, `cleaner.py` | yes (mock offline) |
| LLM layer: providers, role routing, disk cache, schema validation, token meter, `MockLLM` | `backend/llm.py` | n/a |

## Domains
| Scenario | Data | Showcases |
|---|---|---|
| Planted-signal benchmark | 500 simulated signals: 5 real, 10 confounded decoys, 5 lookahead/survivorship traps; fat tails, GARCH-like vol, cross-correlation | the proof (below) |
| Market swarm | simulated equities with planted alpha, factor premia, dirty data, delistings | full swarm; Prosecutor catches survivorship by re-running on the true universe |
| Real equities | 117 large caps 2010-2025 (yfinance, pre-downloaded) | honest "nothing survives"; survivor-universe warning + hindsight-winner stress test |
| Sports betting | the Kaggle "Sports Betting Predictive Analysis Dataset 2025" (appears synthetic) | ROI looks +37% but a random side earns the same: placebo kills it |
| A/B test | simulated, 8 metrics x segments, 10 peeks | peeking attack; one real segment found |
| Health claim | simulated confounded data | Simpson's paradox reversal under stratification |
| Replication mode | paste a paper claim | extract -> Librarian -> rebuild -> judge -> decay after publication |
| Test my claim | your CSV | generic tabular Court (stratified permutation null) |

<!--BENCH-->
## Results: synthetic benchmark (20 seeds, ground truth known)
500 signals over 200 assets, 1260 search days + 252 held out; 5 planted real signals (Sharpe 1.8, 2.2, 2.6, 3.0, 3.5), 10 confounded decoys, 5 traps. Mean per run, ± 95% CI over seeds; FDR pooled over runs.

| Method | claimed | true | false | precision | recall | FDR | false = decoy / trap / noise |
|---|---|---|---|---|---|---|---|
| Naive analyst (p<0.05) | 50.3 | 5.00 | 45.3 | 0.10 ± 0.01 | 1.00 ± 0.00 | 90% | 9.5 / 5.0 / 30.8 |
| Single train/test split | 13.5 | 4.25 | 9.2 | 0.32 ± 0.03 | 0.85 ± 0.07 | 69% | 3.9 / 4.7 / 0.7 |
| Bonferroni | 12.1 | 4.30 | 7.8 | 0.37 ± 0.03 | 0.86 ± 0.07 | 64% | 3.0 / 4.8 / 0.1 |
| Benjamini-Hochberg | 16.2 | 4.65 | 11.6 | 0.31 ± 0.04 | 0.93 ± 0.05 | 71% | 5.2 / 4.9 / 1.6 |
| Deflated Sharpe | 5.2 | 2.50 | 2.7 | 0.50 ± 0.06 | 0.50 ± 0.07 | 52% | 0.3 / 2.4 / 0.0 |
| Null-world audit only | 11.2 | 4.15 | 7.1 | 0.38 ± 0.04 | 0.83 ± 0.07 | 63% | 2.4 / 4.7 / 0.1 |
| **SKEPTIC (full court)** | 4.0 | 3.95 | 0.0 | 1.00 ± 0.00 | 0.79 ± 0.08 | 0% | 0.0 / 0.0 / 0.0 |

* **Calibration** (stated confidence vs actual hit rate, all hypotheses, all seeds): stated 0.02 → actual 0.00 (n=1942); stated 0.05 → actual 0.00 (n=7951); stated 0.74 → actual 0.60 (n=10); stated 0.88 → actual 0.79 (n=96). Confidence is modestly optimistic in the top bin.
* **Power** (Skeptic, detection rate by true Sharpe): SR 0.8: 0%, SR 1.2: 6%, SR 1.8: 34%, SR 2.2: 75%, SR 2.7: 88%, SR 3.4: 100%, SR 4.6: 100%, SR 6.2: 100%. Safety is paid for with lower power on weak signals.
* Per run, 0.85 of the 5 real signals were called LUCK and 0.20 CAN'T TELL YET.
* Measured failure modes: **Hidden confounders pass** (5 of 10 decoys declared DISCOVERY when VAL and MOM are hidden (vs 0 with full factors)); **Null-world validity depends on the null generator** (family-wise false alarm rate on pure noise with volatility regimes: 2% (nominal 5%)). See the Benchmark tab for all of them.
<!--/BENCH-->

* **Read the 0% FDR with care:** it is 0 false out of ~80 claimed discoveries (20 seeds x ~4). Zero observed does not mean zero error: the one-sided 95% upper bound on the false-discovery rate is about 3.7% (rule of three). More seeds would tighten it.
* "Confidence" in the UI is an **evidence score (1 - Storey q-value)**, not a calibrated probability; it runs optimistic at the top (see Calibration).


## Repo layout
```
skeptic/
  backend/    app.py (FastAPI + WebSocket)  runner.py (scenarios, event stream)  llm.py  ledger.py  cleaner.py  literature.py
              search.py  features_dsl.py  hunters/  prosecutors/  court/ (null_worlds, stats, alpha_investing, judge, power, prereg, cards, pipeline)
              adapters/ (synthetic, market_panel, equities, betting, tabular, claims_data)  benchmark/
  frontend/   React + Vite + TypeScript + Tailwind + framer-motion + Recharts
  data/       equities/ (yfinance prices), sports_betting/ (Kaggle CSV)      demo/  benchmark.json + recorded runs
  tests/      pytest (known-answer statistics, DSL safety, prosecutors, Court end to end) + ui/ui_smoke.py (Playwright)
  demo.py     python -m skeptic.demo
```
The 3-minute presentation script is in [DEMO_SCRIPT.md](DEMO_SCRIPT.md).

## Known limitations (in addition to the failure modes measured above)
* Synthetic-benchmark traps (leakage, survivorship) carry spec flags; on the market panel the Prosecutor detects them from data (AST scan, next-day IC plausibility, point-in-time re-run).
* Free price data contains only today's survivors, so survivorship on real equities is flagged and stress-tested, not corrected.
* The market-swarm "ground truth" counts a hypothesis as real if its PnL tracks a planted effect; many near-duplicate variants of one effect inflate the count of "real" ideas, so recall there is low by construction. Costs kill the planted short-term reversal.
* Cost-meter prices for the Claude models are estimates; edit `PRICES` in `llm.py`.
* The bundled paper index lists only facts the authors are confident of; verify citations before relying on them.
