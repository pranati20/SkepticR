import { useState } from 'react'
import { RunState } from '../state'
import UploadView from './UploadView'
import CourtView from './CourtView'
import { useBenchmark, isNaive, isOurs } from '../benchmark'

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
  const outl = c.kind === 'outliers'
  return (
    <div className={`card p-4 ${none ? 'border-amber-400/40' : 'border-disc/40'}`}>
      <div className="mb-1 flex flex-wrap items-center gap-2">
        <span className="text-lg font-bold">{outl ? 'Outliers in the data' : none ? 'No valid conclusion' : 'Conclusion from the valid findings'}</span>
        <span className={`rounded-full border px-2 py-0.5 text-[11px] ${none ? 'border-amber-400 text-amber-300' : 'border-disc text-disc'}`}>
          {outl ? 'described from the data' : `${c.n_valid} of ${c.n_tested} candidate findings survived`}
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

/** The public dataset the Data Scout found for a question-only run, and every candidate it looked at. */
function FoundDataset({ d }: { d: any }) {
  const ok = d.status === 'found'
  const tried: any[] = d.tried || []
  return (
    <div className={`card p-4 ${ok ? 'border-gold/40' : 'border-amber-400/40'}`}>
      <div className="mb-1 flex flex-wrap items-center gap-2">
        <span className="text-lg font-bold">{ok ? 'Data found online' : 'No usable public dataset found'}</span>
        {ok && <span className="rounded-full border border-gold px-2 py-0.5 text-[11px] text-gold">{d.source}</span>}
      </div>
      {ok && (
        <>
          <a href={d.url} target="_blank" rel="noreferrer" className="text-sm font-semibold text-blue-300 hover:text-white">{d.title} ↗</a>
          <div className="mt-2 grid gap-2 text-sm sm:grid-cols-2">
            <div className="rounded-lg border border-line bg-panel2/50 px-3 py-2"><div className="h-label">Cause tested</div>{d.treat_rule}<div className="text-[11px] text-mute">column “{d.treat}”</div></div>
            <div className="rounded-lg border border-line bg-panel2/50 px-3 py-2"><div className="h-label">Effect measured</div>{d.outcome_rule}<div className="text-[11px] text-mute">column “{d.outcome}”</div></div>
          </div>
          <div className="mt-2 text-xs text-slate-300">
            {d.rows?.toLocaleString()} rows · {d.n_columns} columns · columns chosen by {d.mapped_by}{d.why ? `: ${d.why}` : ''}
            {d.note ? <span className="text-mute"> · {d.note}</span> : null}
          </div>
        </>
      )}
      {tried.length > 0 && (
        <details className="mt-3" open={!ok}>
          <summary className="cursor-pointer text-xs text-mute">Searched {(d.sources || []).join(', ')}: looked at {tried.length} candidate{tried.length > 1 ? 's' : ''}</summary>
          <div className="mt-2 space-y-1">
            {tried.map((t, i) => (
              <div key={i} className="flex flex-wrap items-baseline gap-x-3 rounded border border-line bg-ink/60 px-2 py-1 text-xs">
                <span className={t.chosen ? 'text-gold' : t.status === 'usable' ? 'text-disc' : 'text-luck'}>{t.chosen ? 'used' : t.status}</span>
                {t.url ? <a href={t.url} target="_blank" rel="noreferrer" className="min-w-0 flex-1 text-slate-200 hover:text-white">{t.title}</a> : <span className="min-w-0 flex-1">{t.title}</span>}
                <span className="text-mute">{t.source}{t.reason ? ` · ${t.reason}` : t.mapping ? ` · ${t.mapping}` : ''}</span>
              </div>
            ))}
          </div>
        </details>
      )}
    </div>
  )
}

/** Headline + three numbers from the recorded benchmark (nothing typed in: all read from /api/benchmark). */
function Hero({ goProof }: { goProof: () => void }) {
  const { data: d } = useBenchmark()
  const rows = d ? Object.entries<any>(d.table) : []
  const naive = rows.find(([n]) => isNaive(n))?.[1]
  const ours = rows.find(([n]) => isOurs(n))?.[1]
  const stat = (v: string, label: string, gold = false) => (
    <div className="border-l border-line pl-4">
      <div className={`font-mono text-3xl font-semibold tabular-nums ${gold ? 'text-gold' : 'text-white'}`}>{v}</div>
      <div className="mt-0.5 text-xs leading-snug text-slate-300">{label}</div>
    </div>
  )
  return (
    <section className="pb-4 pt-6">
      <div className="h-label mb-3">Statistical audit for AI-driven research</div>
      <h2 className="text-4xl font-bold leading-[1.1] md:text-5xl">Is it a signal,<br />or is it <span className="text-gold">luck</span>?</h2>
      <p className="mt-4 max-w-2xl text-[15px] leading-relaxed text-slate-300">
        An AI that tests thousands of ideas will always find one that looks brilliant by chance. SkepticR re-runs the entire search in worlds where nothing is real,
        puts every survivor on trial, and writes its conclusion only from what is left.
      </p>
      {naive && ours && (
        <div className="mt-6 grid max-w-3xl grid-cols-1 gap-4 sm:grid-cols-3">
          {stat(`${Math.round(naive.fdr.mean * 100)}%`, 'of a naive analyst’s “discoveries” are false')}
          {stat(ours.fp.mean.toFixed(1), 'false discoveries per run with SkepticR', true)}
          {stat(`${Math.round(ours.recall.mean * 100)}%`, 'of the real signals it still finds')}
        </div>
      )}
      {d && <button onClick={goProof} className="mt-4 text-sm font-medium text-blue-300 hover:text-white">
        Benchmark: {d.setup?.signals} signals, {d.n_seeds} seeds, ground truth known. See the proof →</button>}
    </section>
  )
}

export default function ConcludeView({ s, run, running, goCourt, goProof }: { s: RunState; run: (scenario: string, params: any) => void; running: boolean; goCourt: () => void; goProof: () => void }) {
  const [q, setQ] = useState('')
  const [src, setSrc] = useState<'web' | 'csv' | 'example'>('web')
  const [ds, setDs] = useState('ab_test')
  const question = q.trim()
  const started = s.meta !== null || running
  return (
    <div className="mx-auto max-w-4xl space-y-4">
      <Hero goProof={goProof} />
      <div className="pt-2">
        <h2 className="text-lg font-semibold">Try it: what do you want to conclude from your data?</h2>
        <div className="text-sm text-slate-300">
          1. Say what you want to find out. 2. Give it data. 3. SkepticR tests every possible finding and throws out the ones that are just luck or bias.
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
        <div className="mb-3 flex flex-wrap gap-2">
          <button className={`btn ${src === 'web' ? 'border-accent text-white' : ''}`} onClick={() => setSrc('web')}>Find the data for me</button>
          <button className={`btn ${src === 'example' ? 'border-accent text-white' : ''}`} onClick={() => setSrc('example')}>Use an example dataset</button>
          <button className={`btn ${src === 'csv' ? 'border-accent text-white' : ''}`} onClick={() => setSrc('csv')}>Upload my CSV</button>
        </div>
        {src === 'web' ? (
          <div>
            <div className="mb-3 text-xs text-slate-300">
              No dataset needed. The Data Scout searches public data (OpenML, Hugging Face, Zenodo) for a table with your cause and effect, then the Court judges it as usual.
              Ask about one specific cause, e.g. “Does smoking increase heart disease?” or “Does education increase income?”.
            </div>
            <div className="flex flex-wrap items-center gap-3">
              <button className="btn-primary" disabled={running || !question} onClick={() => run('discover', { question })}>Search the internet and conclude</button>
              {!question && <span className="text-xs text-mute">Write your question first.</span>}
            </div>
          </div>
        ) : src === 'example' ? (
          <div className="flex flex-wrap items-center gap-3">
            <select value={ds} onChange={(e) => setDs(e.target.value)} className="btn">{DATASETS.map((d) => <option key={d.id} value={d.id}>{d.label}</option>)}</select>
            <button className="btn-primary" disabled={running || !question} onClick={() => run(ds, { question })}>Find the valid conclusion</button>
            {!question && <span className="text-xs text-mute">Write your question first.</span>}
          </div>
        ) : question ? (
          <UploadView externalClaim={question} label="Find the valid conclusion" running={running} run={(p) => run('upload', { ...p, question })} goCourt={goCourt} />
        ) : <div className="text-xs text-mute">Write your question first.</div>}
      </div>

      {s.dataset && <FoundDataset d={s.dataset} />}
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
