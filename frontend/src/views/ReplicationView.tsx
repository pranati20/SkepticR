import { useState } from 'react'
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { RunState } from '../state'
import { Banner, Card, Pill, fmt, pct } from '../ui'

const EXAMPLES = [
  'Short-term reversal: stocks that fell over the past week tend to rebound over the next week.',
  'Momentum: stocks with high returns over the past 12 months (skipping the latest month) continue to outperform.',
  'Low-volatility anomaly: stocks with low recent volatility earn higher risk-adjusted returns than volatile stocks.',
  'Value: stocks with high book-to-market ratios outperform expensive stocks.',
]

export default function ReplicationView({ s, run, demo, running }: { s: RunState; run: (p: any) => void; demo: () => void; running: boolean }) {
  const [text, setText] = useState(EXAMPLES[0])
  const [dataset, setDataset] = useState('market')
  const ex = s.replication.extracted, dec = s.replication.decay
  const variants = Object.values(s.verdicts)
  const primary: any = s.verdicts['rep_0']
  const lit = s.literature.filter((l) => l.family === 'replication').slice(-1)[0]
  return (
    <div className="space-y-4">
      <Card title="Paste a paper's claim">
        <textarea value={text} onChange={(e) => setText(e.target.value)} rows={3} className="w-full rounded-lg border border-line bg-ink p-3 text-sm outline-none focus:border-accent" />
        <div className="mt-2 flex flex-wrap items-center gap-2">
          {EXAMPLES.map((e, i) => <button key={i} className="btn text-xs" onClick={() => setText(e)}>{e.split(':')[0]}</button>)}
          <select value={dataset} onChange={(e) => setDataset(e.target.value)} className="btn ml-auto"><option value="market">Simulated world (effect decays after publication)</option><option value="equities">Real equities 2010-2025</option></select>
          <button className="btn-primary" disabled={running} onClick={() => run({ text, dataset })}>Extract · rebuild · judge</button>
          <button className="btn" disabled={running} onClick={demo}>Replay example</button>
        </div>
      </Card>
      {s.error && <Banner tone="rose">{s.error}</Banner>}
      {ex && (
        <div className="grid gap-4 lg:grid-cols-2">
          <Card title="1 · Extracted hypothesis (safe DSL, never eval'd)">
            <div className="text-lg font-semibold">{ex.extracted.name}</div>
            <code className="mt-1 block rounded bg-ink px-2 py-1 text-sm text-blue-300">{ex.formula}</code>
            <div className="mt-2 text-sm text-slate-300">{ex.extracted.rationale}</div>
            <div className="mt-1 text-xs text-mute">Source: {ex.extracted.source || 'unknown'} {ex.extracted.publication_year ? `(${ex.extracted.publication_year})` : ''}</div>
          </Card>
          <Card title="2 · Librarian: does the paper apply to this scenario?">
            {lit ? (<div className="space-y-2">
              <div className="text-[11px] text-mute">{lit.live ? 'Live search' : 'Offline'}{lit.search_status?.length ? ' · ' + lit.search_status.join(' · ') : ''} · these papers are passed to the Skeptics' literature attack</div>
              {lit.papers.map((p: any, i: number) => (<div key={i} className="text-sm"><b>{p.authors}</b> ({p.year}) {p.url ? <a href={p.url} target="_blank" rel="noreferrer" className="text-blue-300 underline">{p.title}</a> : <span className="text-slate-300">{p.title}</span>}<div className="text-[11px] text-mute">{p.source}{p.sample ? ` · sample ${p.sample.join('–')}` : ''}{p.universe ? ` · ${p.universe}` : ''}{p.applicability != null ? ` · applicability ${p.applicability}` : ''}</div>{p.used && <div className="mt-1 rounded border-l-2 border-accent/60 bg-ink/50 px-2 py-1 text-[11px] leading-snug text-slate-300"><b className="text-accent">What we used:</b> {p.used}</div>}</div>))}
              {lit.caveats?.map((c: string, i: number) => <div key={i} className="text-[12px] text-amber-300/90">• {c}</div>)}
            </div>) : <div className="text-sm text-mute">…</div>}
          </Card>
        </div>)}
      {dec && (
        <Card title="3 · Did it hold, and did it decay after publication?">
          <div className="grid gap-4 lg:grid-cols-3">
            <div className="space-y-2">
              <div className="flex items-center gap-2"><Pill verdict={dec.primary_verdict} /><span className="text-sm text-rose-300">{dec.cause_of_death}</span></div>
              <Banner tone="sky">“{dec.confession}”</Banner>
              {dec.sharpe_pre !== undefined ? (<div className="text-sm">Sharpe before publication <b>{fmt(dec.sharpe_pre)}</b> → after <b>{fmt(dec.sharpe_post)}</b> · difference t = {fmt(dec.decay_t, 1)}<div className="mt-1 text-lg font-semibold text-amber-300">{dec.verdict}</div></div>)
                : (<div className="text-sm text-slate-300">{dec.note}<div className="mt-1">Full-sample Sharpe {fmt(dec.sharpe_full)}</div></div>)}
            </div>
            <div className="lg:col-span-2 h-56">
              {dec.rolling_sharpe && (<ResponsiveContainer><LineChart data={dec.rolling_sharpe.map((y: number, i: number) => ({ i, y }))}><CartesianGrid stroke="#22304d" /><XAxis dataKey="i" hide /><YAxis tick={{ fontSize: 10, fill: '#64748b' }} width={30} />
                <Tooltip contentStyle={{ background: '#101a2e', border: '1px solid #22304d' }} formatter={(v: any) => fmt(v)} /><ReferenceLine y={0} stroke="#475569" />
                {dec.pub_index !== undefined && <ReferenceLine x={dec.pub_index} stroke="#e8b04b" strokeDasharray="4 4" label={{ value: 'published', fill: '#e8b04b', fontSize: 10 }} />}
                <Line dataKey="y" stroke="#3b82f6" strokeWidth={2} dot={false} /></LineChart></ResponsiveContainer>)}
              <div className="text-[11px] text-mute">Rolling 1-year Sharpe of the as-published construction.</div>
            </div>
          </div>
        </Card>)}
      {variants.length > 0 && (
        <Card title="4 · Everything the replicator tried (all counted against the claim)">
          <div className="space-y-1">{variants.map((v: any) => (
            <div key={v.hypothesis_id} className="flex items-center gap-2 text-sm"><Pill verdict={v.verdict} /><span className="flex-1 truncate">{v.description}</span><span className="font-mono text-xs text-mute">p_adj {fmt(v.p_adj, 2)} · luck {pct(v.p_luck_rate)}</span></div>))}</div>
        </Card>)}
    </div>
  )
}
