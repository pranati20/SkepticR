import { useReducer, useState } from 'react'
import { initialState, reduce } from './state'
import { useRun } from './useRun'
import CourtView from './views/CourtView'
import SplitView from './views/SplitView'
import BenchmarkView from './views/BenchmarkView'
import ReplicationView from './views/ReplicationView'
import UploadView from './views/UploadView'

const SCENARIOS = [
  { id: 'market', label: 'Market swarm (simulated)' },
  { id: 'synthetic', label: 'Planted-signal benchmark' },
  { id: 'equities', label: 'Real equities 2010-25' },
  { id: 'betting', label: 'Sports betting (Kaggle)' },
  { id: 'health', label: 'Health claim (Simpson)' },
  { id: 'ab_test', label: 'A/B test with peeking' },
]
const TABS = [
  ['court', 'Court'], ['split', 'Naive AI vs Skeptic'], ['bench', 'Benchmark'], ['replicate', 'Replication'], ['upload', 'Test my claim'],
] as const

export default function App() {
  const [s, dispatch] = useReducer(reduce, undefined, initialState)
  const run = useRun(dispatch)
  const [tab, setTab] = useState<(typeof TABS)[number][0]>('court')
  const [scenario, setScenario] = useState('market')
  const [mode, setMode] = useState<'demo' | 'live'>('demo')
  const go = () => (mode === 'demo' ? run.demo(scenario) : run.live(scenario))
  const showControls = tab === 'court' || tab === 'split'
  return (
    <div className="mx-auto max-w-[1500px] px-4 pb-16 pt-4">
      <header className="mb-4 flex flex-wrap items-center gap-x-6 gap-y-2">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">⚖️ Skeptic</h1>
          <div className="-mt-0.5 text-xs text-mute">continuous integration for claims · an AI researcher that audits its own luck</div>
        </div>
        <nav className="flex flex-wrap gap-1 rounded-xl border border-line bg-panel p-1">
          {TABS.map(([id, label]) => (
            <button key={id} onClick={() => setTab(id)} className={`rounded-lg px-3 py-1.5 text-sm transition ${tab === id ? 'bg-accent/90 font-semibold text-ink' : 'text-slate-300 hover:text-white'}`}>{label}</button>
          ))}
        </nav>
        <div className="ml-auto text-xs text-mute">LLM: <span className="text-slate-300">MockLLM (offline)</span> · {run.source || 'idle'}</div>
      </header>

      {showControls && (
        <div className="card mb-4 flex flex-wrap items-center gap-3 p-3">
          <select value={scenario} onChange={(e) => setScenario(e.target.value)} className="btn">{SCENARIOS.map((x) => <option key={x.id} value={x.id}>{x.label}</option>)}</select>
          <div className="flex rounded-lg border border-line p-0.5 text-sm">
            {(['demo', 'live'] as const).map((m) => (<button key={m} onClick={() => setMode(m)} className={`rounded-md px-3 py-1 ${mode === m ? 'bg-panel2 text-white' : 'text-mute'}`}>{m === 'demo' ? 'Demo mode (replay)' : 'Live run'}</button>))}
          </div>
          <button className="btn-primary" disabled={run.running} onClick={go}>{mode === 'demo' ? '▶ Replay cached run' : '▶ Run live'}</button>
          {run.running && <button className="btn" onClick={run.stop}>■ stop</button>}
          <button className="btn" onClick={run.reset}>reset</button>
          <div className="ml-auto flex items-center gap-1 text-xs text-mute">speed
            {[0.5, 1, 2, 4, 100].map((x) => <button key={x} onClick={() => run.setSpeed(x)} className={`rounded px-2 py-1 ${run.speed === x ? 'bg-panel2 text-white' : ''}`}>{x === 100 ? 'instant' : `${x}x`}</button>)}</div>
          {s.meta && <div className="basis-full text-sm"><b>{s.meta.title}</b> <span className="text-mute">· {s.meta.blurb}</span></div>}
        </div>
      )}

      {tab === 'court' && <CourtView s={s} />}
      {tab === 'split' && <SplitView s={s} />}
      {tab === 'bench' && <BenchmarkView />}
      {tab === 'replicate' && <ReplicationView s={s} running={run.running} run={(p) => run.live('replication', p)} demo={() => run.demo('replication')} />}
      {tab === 'upload' && <UploadView running={run.running} run={(p) => run.live('upload', p)} goCourt={() => setTab('court')} />}

      <footer className="mt-10 rounded-2xl border border-line bg-panel/60 p-4 text-xs leading-relaxed text-mute">
        <div className="h-label mb-1">Honesty guardrails</div>
        <ul className="list-disc space-y-0.5 pl-4">
          <li>Skeptic reduces false discoveries and quantifies uncertainty. It does not prove truth.</li>
          <li>Null-world validity depends on the generator preserving the data's real structure (sign-flip, stationary bootstrap, label permutation each have stated assumptions).</li>
          <li>Backtests are not forecasts. Nothing here is trading advice. Simulated data is labelled as such.</li>
          <li>Built on White's Reality Check, Hansen's SPA, the Deflated Sharpe Ratio, Benjamini–Hochberg and Foster–Stine alpha-investing. Our contribution is applying them to an autonomous agent's own search as one integrated, measurable system.</li>
        </ul>
      </footer>
    </div>
  )
}
