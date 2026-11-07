import { useReducer, useState } from 'react'
import { initialState, reduce } from './state'
import { useRun } from './useRun'
import CourtView from './views/CourtView'
import SplitView from './views/SplitView'
import PapersView from './views/PapersView'
import ChatView from './views/ChatView'
import MethodsView from './views/MethodsView'
import ConcludeView from './views/ConcludeView'

const SCENARIOS = [
  { id: 'market', label: 'Stock-picking AI (simulated stocks)' },
  { id: 'synthetic', label: 'Test with known answers (500 signals)' },
  { id: 'equities', label: 'Real stocks 2010-25' },
  { id: 'betting', label: 'Sports betting' },
  { id: 'health', label: 'Health claim (a trap in the data)' },
  { id: 'ab_test', label: 'A/B test (peeking trap)' },
]
const TABS = [
  ['conclude', 'Find a conclusion'], ['ask', 'Find papers & data'], ['court', 'Watch it work'],
  ['split', 'AI vs Skeptic'], ['methods', 'Improve the algorithm'], ['papers', 'Papers it used'],
] as const

export default function App() {
  const [s, dispatch] = useReducer(reduce, undefined, initialState)
  const run = useRun(dispatch)
  const [tab, setTab] = useState<(typeof TABS)[number][0]>('conclude')
  const [scenario, setScenario] = useState('market')
  const [mode, setMode] = useState<'demo' | 'live'>('demo')
  const [chatCtx, setChatCtx] = useState<{ question: string; papers: any[] } | null>(null)
  const go = () => (mode === 'demo' ? run.demo(scenario) : run.live(scenario, chatCtx ? { context: chatCtx } : {}))
  const pickScenario = (t: string) => {
    t = t.toLowerCase()
    return /\b(bet|betting|odds|sportsbook|sports)\b/.test(t) ? 'betting' : /a\/b|peeking|conversion|experiment/.test(t) ? 'ab_test'
      : /patient|treatment|supplement|clinical|health/.test(t) ? 'health' : 'market'
  }
  const auditFromChat = (question: string, sources: any[]) => {
    const ctx = { question, papers: sources.filter((s) => s.kind === 'paper') }
    const sc = pickScenario(question + ' ' + ctx.papers.map((p) => p.title).join(' '))
    setChatCtx(ctx); setScenario(sc); setMode('live'); setTab('court')
    run.live(sc, { context: ctx })
  }
  const showControls = tab === 'court' || tab === 'split'
  return (
    <div className="mx-auto max-w-[1500px] px-4 pb-16 pt-4">
      <header className="mb-4 flex flex-wrap items-center gap-x-6 gap-y-2">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">⚖️ Skeptic</h1>
          <div className="-mt-0.5 text-xs text-mute">checks whether a pattern in data is real, or just luck</div>
        </div>
        <nav className="flex flex-wrap gap-1 rounded-xl border border-line bg-panel p-1">
          {TABS.map(([id, label]) => (
            <button key={id} onClick={() => setTab(id)} className={`rounded-lg px-3 py-1.5 text-sm transition ${tab === id ? 'bg-accent/90 font-semibold text-ink' : 'text-slate-300 hover:text-white'}`}>{label}</button>
          ))}
        </nav>
        <div className="ml-auto flex items-center gap-3 text-xs text-mute">{run.source || ''}
          {s.done && run.last && s.proposals.length > 0 && (
            <a className="btn-primary" href={`/api/report?${run.last.run ? 'run=' + run.last.run : 'demo=' + run.last.demo}`} download>⬇ Export report (PDF)</a>
          )}</div>
      </header>

      {showControls && (
        <div className="card mb-4 flex flex-wrap items-center gap-3 p-3">
          {chatCtx && (
            <div className="flex w-full items-start gap-2 rounded-lg border border-accent/40 bg-accent/5 p-2 text-xs text-slate-200">
              <div className="flex-1">
                <b>Using your chat:</b> “{chatCtx.question.slice(0, 120)}” · {chatCtx.papers.length} paper(s) will be given to the Prosecutors as evidence, and the AI will test the idea types your question mentions.
                {scenario !== 'market' && scenario !== 'synthetic' && scenario !== 'equities' && <div className="text-amber-300">This example isn’t about stocks, so the papers are shown but not used in the checks.</div>}
                {mode === 'demo' && <div className="text-amber-300">“Quick replay” ignores your chat. Switch to “Run for real” under Options.</div>}
              </div>
              <button className="btn px-2 py-0.5" onClick={() => setChatCtx(null)}>✕ remove</button>
            </div>
          )}
          <span className="text-sm text-slate-300">1. Pick an example</span>
          <select value={scenario} onChange={(e) => setScenario(e.target.value)} className="btn">{SCENARIOS.map((x) => <option key={x.id} value={x.id}>{x.label}</option>)}</select>
          <span className="text-sm text-slate-300">2.</span>
          <button className="btn-primary" disabled={run.running} onClick={go}>▶ Start</button>
          {run.running && <button className="btn" onClick={run.stop}>■ Stop</button>}
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

      <div hidden={tab !== 'ask'}><ChatView onAudit={auditFromChat} /></div>
      {tab === 'court' && <CourtView s={s} />}
      {tab === 'methods' && <MethodsView />}
      {tab === 'split' && <SplitView s={s} />}
      {tab === 'papers' && <PapersView />}
      {tab === 'conclude' && <ConcludeView s={s} running={run.running} run={(sc, p) => { setScenario(sc); run.live(sc, p) }} goCourt={() => setTab('court')} />}

      <footer className="mt-10 rounded-2xl border border-line bg-panel/60 p-4 text-xs leading-relaxed text-mute">
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
