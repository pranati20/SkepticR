import { useReducer, useState } from 'react'
import { initialState, reduce } from './state'
import { useRun } from './useRun'
import CourtView from './views/CourtView'
import SplitView from './views/SplitView'
import PapersView from './views/PapersView'
import MethodsView from './views/MethodsView'
import ConcludeView from './views/ConcludeView'
import ProofView from './views/ProofView'
import Backdrop from './Backdrop'

const SCENARIOS = [
  { id: 'market', label: 'Stock-picking AI (simulated stocks)' },
  { id: 'synthetic', label: 'Test with known answers (500 signals)' },
  { id: 'equities', label: 'Real stocks 2010-25' },
  { id: 'betting', label: 'Sports betting' },
  { id: 'health', label: 'Health claim (a trap in the data)' },
  { id: 'ab_test', label: 'A/B test (peeking trap)' },
]
const TABS = [
  ['conclude', 'Find a conclusion'], ['court', 'Watch it work'], ['split', 'AI vs SkepticR'], ['proof', 'Proof'],
  ['methods', 'Improve the algorithm'], ['papers', 'Papers it used'],
] as const

export default function App() {
  const [s, dispatch] = useReducer(reduce, undefined, initialState)
  const run = useRun(dispatch)
  type Tab = (typeof TABS)[number][0]
  const [tab, setTab] = useState<Tab>('conclude')
  const [seen, setSeen] = useState<Set<Tab>>(() => new Set<Tab>(['conclude']))
  const open = (t: Tab) => { setSeen((p) => (p.has(t) ? p : new Set(p).add(t))); setTab(t) }
  const [scenario, setScenario] = useState('market')
  const [mode, setMode] = useState<'demo' | 'live'>('demo')
  const go = () => (mode === 'demo' ? run.demo(scenario) : run.live(scenario, {}))
  const showControls = tab === 'court' || tab === 'split'
  return (
    <div className="mx-auto max-w-[1500px] px-4 pb-16 pt-4">
      <Backdrop />
      <header className="mb-6 border-b border-line">
       <div className="flex flex-wrap items-center gap-x-6 gap-y-3 pb-4 pt-2">
        <div className="flex items-center gap-3">
          {/* LOGO: the file is frontend/public/logo.svg (also the browser-tab icon); replace that file to change it */}
          <img src="/logo.svg" alt="SkepticR logo" className="h-11 w-11 shrink-0" />
          <div>
            <h1 className="font-display text-[26px] font-bold leading-none tracking-tight text-white">Skeptic<span className="text-gold">R</span></h1>
            <div className="mt-1 text-[11px] font-medium uppercase tracking-[0.16em] text-mute">signal or luck · statistical audit</div>
          </div>
        </div>
        <div className="ml-auto flex items-center gap-3 font-mono text-xs text-mute">{run.source || ''}
          {s.done && run.last && s.proposals.length > 0 && (
            <a className="btn-primary" href={`/api/report?${run.last.run ? 'run=' + run.last.run : 'demo=' + run.last.demo}`} download>Export report (PDF)</a>
          )}</div>
       </div>
        <nav className="flex flex-wrap items-center gap-1">
          {TABS.map(([id, label]) => (
            <button key={id} onClick={() => open(id)} className={`-mb-px border-b-2 px-3 py-2.5 text-sm font-medium transition-colors ${tab === id ? 'border-gold text-white' : 'border-transparent text-slate-400 hover:text-white'}`}>{label}</button>
          ))}
          {run.running && tab !== 'conclude' && tab !== 'court' && <span className="flex items-center gap-1.5 px-2 text-xs text-accent"><span className="h-1.5 w-1.5 animate-pulseGlow rounded-full bg-accent" />running…</span>}
        </nav>
      </header>

      {showControls && (
        <div className="card mb-4 flex flex-wrap items-center gap-3 p-3">
          <span className="text-sm text-slate-300">1. Pick an example</span>
          <select value={scenario} onChange={(e) => setScenario(e.target.value)} className="btn">{SCENARIOS.map((x) => <option key={x.id} value={x.id}>{x.label}</option>)}</select>
          <span className="text-sm text-slate-300">2.</span>
          <button className="btn-primary" disabled={run.running} onClick={go}>Start</button>
          {run.running && <button className="btn" onClick={run.stop}>Stop</button>}
          <button className="btn" onClick={run.reset}>Clear</button>
          <details className="ml-auto text-xs text-mute">
            <summary className="cursor-pointer">Options</summary>
            <div className="mt-2 flex flex-wrap items-center gap-3 rounded-lg border border-line bg-panel2 p-2">
              <div className="flex rounded-lg border border-line p-0.5 text-sm">
                {(['demo', 'live'] as const).map((m) => (<button key={m} onClick={() => setMode(m)} className={`rounded-md px-3 py-1 ${mode === m ? 'bg-panel text-white' : 'text-mute'}`}>{m === 'demo' ? 'Quick replay' : 'Run for real'}</button>))}
              </div>
              <div className="flex items-center gap-1">speed
                {[0.5, 1, 2, 4, 100].map((x) => <button key={x} onClick={() => run.setSpeed(x)} className={`rounded px-2 py-1 ${run.speed === x ? 'bg-panel text-white' : ''}`}>{x === 100 ? 'instant' : `${x}x`}</button>)}</div>
            </div>
          </details>
          {s.meta && <div className="basis-full text-sm"><b>{s.meta.title}</b> <span className="text-mute">· {s.meta.blurb}</span></div>}
        </div>
      )}

      {/* Tabs stay mounted once opened, so switching tabs never loses a form, an upload or an in-progress search. */}
      {seen.has('court') && <div hidden={tab !== 'court'}><CourtView s={s} /></div>}
      {seen.has('methods') && <div hidden={tab !== 'methods'}><MethodsView /></div>}
      {seen.has('split') && <div hidden={tab !== 'split'}><SplitView s={s} /></div>}
      {seen.has('proof') && <div hidden={tab !== 'proof'}><ProofView /></div>}
      {seen.has('papers') && <div hidden={tab !== 'papers'}><PapersView active={tab === 'papers'} /></div>}
      <div hidden={tab !== 'conclude'}><ConcludeView s={s} running={run.running} run={(sc, p) => { setScenario(sc); run.live(sc, p) }} goCourt={() => open('court')} goProof={() => open('proof')} /></div>

      <footer className="card mt-10 p-4 text-xs leading-relaxed text-mute">
        <details><summary className="h-label cursor-pointer">Important limits (click to read)</summary>
        <ul className="mt-1 list-disc space-y-0.5 pl-4">
          <li>Skeptic reduces false discoveries and quantifies uncertainty. It does not prove truth.</li>
          <li>Null-world validity depends on the generator preserving the data's real structure (sign-flip, stationary bootstrap, label permutation each have stated assumptions).</li>
          <li>Backtests are not forecasts. Nothing here is trading advice. Simulated data is labelled as such.</li>
          <li>Built on White's Reality Check, Hansen's SPA, the Deflated Sharpe Ratio, Benjamini–Hochberg and Foster–Stine alpha-investing. Our contribution is applying them to an autonomous agent's own search as one integrated, measurable system.</li>
        </ul></details>
      </footer>
    </div>
  )
}
