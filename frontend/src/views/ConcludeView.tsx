import { useState } from 'react'
import { RunState } from '../state'
import UploadView from './UploadView'
import CourtView from './CourtView'

const EXAMPLES = [
  'Does the new design increase conversion, and for whom?',
  'Do supplement users really have better health scores?',
  'Is there a stock pattern that is real and not just luck?',
]
const DATASETS = [
  { id: 'ab_test', label: 'A/B test (website experiment)' },
  { id: 'health', label: 'Health data (supplements)' },
  { id: 'market', label: 'Simulated stocks' },
  { id: 'equities', label: 'Real stocks 2010-25' },
  { id: 'betting', label: 'Sports betting' },
  { id: 'synthetic', label: 'Test data with known answers' },
]

function Conclusion({ c }: { c: any }) {
  const none = c.status === 'none'
  return (
    <div className={`card p-4 ${none ? 'border-amber-400/40' : 'border-disc/40'}`}>
      <div className="mb-1 flex flex-wrap items-center gap-2">
        <span className="text-lg font-bold">{none ? 'No valid conclusion' : 'Conclusion from the valid findings'}</span>
        <span className={`rounded-full border px-2 py-0.5 text-[11px] ${none ? 'border-amber-400 text-amber-300' : 'border-disc text-disc'}`}>
          {c.n_valid} of {c.n_tested} candidate findings survived
        </span>
      </div>
      <div className="text-xs text-mute">Your question: “{c.question}”</div>
      <div className="mt-3 text-xs font-semibold uppercase tracking-wide text-accent">Detailed write-up</div>
      <div className="mt-1 whitespace-pre-wrap text-sm leading-relaxed text-slate-100">{c.text}</div>
      {c.findings.length > 0 && (
        <div className="mt-4">
          <div className="h-label mb-1">Valid findings used (the only ones the conclusion is based on)</div>
          <div className="space-y-1">
            {c.findings.map((f: any) => (
              <div key={f.id} className="flex flex-wrap items-baseline gap-x-3 rounded-lg border border-line bg-panel2/50 px-3 py-1.5 text-sm">
                <span className="rounded bg-ink px-1.5 text-[10px] font-semibold text-accent">{f.ref}</span>
                <span className="min-w-0 flex-1">{f.claim}{f.direction ? ` · ${f.direction}` : ''}</span>
                <span className="text-[11px] text-mute">evidence {Math.round(f.evidence * 100)}% · passed {f.checks_passed} attacks{f.flags.length ? ` · caution: ${f.flags.join(', ')}` : ''}</span>
              </div>
            ))}
          </div>
        </div>
      )}
      <div className="mt-3 text-[11px] text-mute">
        Left out and not used: {c.n_luck} that looked like luck or bias, {c.n_unclear} with too little data to judge. Written by: {c.llm}.
        Valid means it survived our tests; it is not proof of cause and not advice.
      </div>
    </div>
  )
}

export default function ConcludeView({ s, run, running, goCourt }: { s: RunState; run: (scenario: string, params: any) => void; running: boolean; goCourt: () => void }) {
  const [q, setQ] = useState('')
  const [src, setSrc] = useState<'csv' | 'example'>('example')
  const [ds, setDs] = useState('ab_test')
  const question = q.trim()
  const started = s.meta !== null || running
  return (
    <div className="mx-auto max-w-4xl space-y-4">
      <div>
        <h2 className="text-lg font-semibold">What do you want to conclude from your data?</h2>
        <div className="text-sm text-slate-300">
          1. Say what you want to find out. 2. Give it data. 3. Skeptic tests every possible finding and throws out the ones that are just luck or bias.
          4. The conclusion is written <b>only from what survived</b>.
        </div>
      </div>
      <div className="card p-4">
        <div className="h-label mb-1">1 · Your question</div>
        <textarea value={q} onChange={(e) => setQ(e.target.value.slice(0, 500))} rows={2} placeholder="e.g. Does the new design increase conversion, and for whom?"
          className="w-full rounded-lg border border-line bg-ink p-3 text-sm outline-none focus:border-accent" />
        <div className="mt-2 flex flex-wrap gap-1.5">{EXAMPLES.map((x) => <button key={x} className="btn px-2 py-0.5 text-[11px]" onClick={() => setQ(x)}>{x}</button>)}</div>
      </div>
      <div className="card p-4">
        <div className="h-label mb-1">2 · Your data</div>
        <div className="mb-3 flex gap-2">
          <button className={`btn ${src === 'example' ? 'border-accent text-white' : ''}`} onClick={() => setSrc('example')}>Use an example dataset</button>
          <button className={`btn ${src === 'csv' ? 'border-accent text-white' : ''}`} onClick={() => setSrc('csv')}>Upload my CSV</button>
        </div>
        {src === 'example' ? (
          <div className="flex flex-wrap items-center gap-3">
            <select value={ds} onChange={(e) => setDs(e.target.value)} className="btn">{DATASETS.map((d) => <option key={d.id} value={d.id}>{d.label}</option>)}</select>
            <button className="btn-primary" disabled={running || !question} onClick={() => run(ds, { question })}>Find the valid conclusion</button>
            {!question && <span className="text-xs text-mute">Write your question first.</span>}
          </div>
        ) : question ? (
          <UploadView externalClaim={question} label="Find the valid conclusion" running={running} run={(p) => run('upload', { ...p, question })} goCourt={goCourt} />
        ) : <div className="text-xs text-mute">Write your question first.</div>}
      </div>

      {started && !s.conclusion && (
        <div className="card animate-pulseGlow p-4 text-sm text-slate-300">
          {s.error ? <span className="text-luck">{s.error}</span> : <>Working: <b>{s.phase}</b>. {s.proposals.length > 0 && `${s.proposals.length} candidate findings so far; `}{Object.keys(s.verdicts).length} judged…</>}
        </div>
      )}
      {s.conclusion && <Conclusion c={s.conclusion} />}
      {s.done && !s.conclusion && !s.error && <div className="text-sm text-mute">The run finished without a conclusion.</div>}
      {started && (
        <details className="card p-3">
          <summary className="cursor-pointer text-sm text-slate-300">Show the full audit (every idea, every attack)</summary>
          <div className="mt-3"><CourtView s={s} /></div>
        </details>
      )}
    </div>
  )
}
