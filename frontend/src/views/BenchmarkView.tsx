import { useEffect, useState } from 'react'
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis, Scatter, ComposedChart } from 'recharts'
import { Banner, Card, Hist, fmt, pct } from '../ui'

const COL: Record<string, string> = { 'SKEPTIC (full court)': '#34d399', 'Benjamini-Hochberg': '#38bdf8', Bonferroni: '#a78bfa', 'Naive analyst (p<0.05)': '#fb7185' }
const tip = { contentStyle: { background: '#0f1520', border: '1px solid #1e2a3d', fontSize: 12 } }

function NullLab() {
  const [n, setN] = useState(300)
  const [out, setOut] = useState<any>(null)
  const [busy, setBusy] = useState(false)
  const go = async (k: number) => {
    setBusy(true)
    try { const r = await fetch('/api/nullrerun', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ n_worlds: k }) }); setOut(await r.json()) } finally { setBusy(false) }
  }
  useEffect(() => { go(300) }, [])
  return (
    <Card title="Null-world lab · re-run the audit live" right={<span className="text-xs text-mute">{busy ? 'running…' : out ? `${out.seconds.toFixed(1)}s` : ''}</span>}>
      <div className="mb-2 flex items-center gap-3 text-sm">
        <span className="text-mute">null worlds</span>
        <input type="range" min={50} max={2000} step={50} value={n} onChange={(e) => setN(+e.target.value)} onMouseUp={() => go(n)} onTouchEnd={() => go(n)} className="flex-1 accent-sky-400" />
        <span className="w-14 font-mono">{n}</span><button className="btn" disabled={busy} onClick={() => go(n)}>re-run</button>
      </div>
      {out && (<>
        <Hist hist={out.histogram} marks={out.planted.map((p: any, i: number) => ({ x: p.stat, label: `real signal ${fmt(p.stat, 1)}`, color: '#34d399' }))} height={170} />
        <div className="text-xs text-mute">Blue: the best |t| luck produced in each of {out.n_worlds} worlds where all 500 signals were searched. Green lines: the five planted signals. Adjusted p for each:
          {out.planted.map((p: any, i: number) => <span key={i} className="ml-2 font-mono text-slate-300">{fmt(p.p_adj, 3)}</span>)}</div>
      </>)}
    </Card>
  )
}

export default function BenchmarkView() {
  const [b, setB] = useState<any>(null)
  const [err, setErr] = useState('')
  useEffect(() => { fetch('/api/benchmark').then((r) => (r.ok ? r.json() : Promise.reject('benchmark.json missing; run python -m skeptic.backend.benchmark.run_benchmark'))).then(setB).catch((e) => setErr(String(e))) }, [])
  if (err) return <Banner tone="rose">{err}</Banner>
  if (!b) return <div className="text-mute">loading benchmark…</div>
  const rows = Object.entries(b.table) as [string, any][]
  const bars = rows.map(([m, v]) => ({ method: m.replace(' (p<0.05)', '').replace('Single train/test split', 'Train/test'), FDR: v.pooled_fdr, Recall: v.recall.mean }))
  const cal = b.calibration.map((c: any) => ({ stated: c.mean_stated, actual: c.actual, n: c.n }))
  const xs = Array.from(new Set(Object.values(b.power_curve).flatMap((a: any) => a.map((p: any) => p.sr.toFixed(1))))).sort((a: any, c: any) => +a - +c)
  const power = xs.map((x: any) => { const row: any = { sr: +x }; Object.entries(b.power_curve).forEach(([m, a]: any) => { const p = a.find((q: any) => q.sr.toFixed(1) === x); if (p) row[m] = p.detect }); return row })
  return (
    <div className="space-y-4">
      <Banner tone="sky">
        Synthetic benchmark with known truth: {b.setup.signals} candidate signals over {b.setup.assets} assets ({b.setup.search_days} search days + {b.setup.holdout_days} held out). {b.setup.planted} planted real signals (Sharpe {b.setup.planted_sr.join(', ')}),
        {' '}{b.setup.decoys} confounded decoys, {b.setup.traps} lookahead/survivorship traps. Fat-tailed, volatility-clustered, cross-correlated noise. {b.n_seeds} seeds, mean ± 95% CI.
      </Banner>
      <Card title="Methods compared">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="text-[11px] uppercase text-mute"><tr><th className="py-1">Method</th><th>claimed</th><th>true</th><th>false</th><th>precision</th><th>recall</th><th>false-discovery rate</th><th>false = decoy/trap/noise</th></tr></thead>
            <tbody>
              {rows.map(([m, v]) => (
                <tr key={m} className={`border-t border-line ${m.startsWith('SKEPTIC') ? 'bg-disc/10 font-semibold text-disc' : ''}`}>
                  <td className="py-1.5">{m}</td><td>{fmt(v.claimed.mean, 1)}</td><td>{fmt(v.tp.mean, 2)}</td><td>{fmt(v.fp.mean, 1)}</td>
                  <td>{fmt(v.precision.mean, 2)} ± {fmt(v.precision.ci95, 2)}</td><td>{fmt(v.recall.mean, 2)} ± {fmt(v.recall.ci95, 2)}</td><td>{pct(v.pooled_fdr)}</td>
                  <td className="font-mono text-xs">{fmt(v.fp_decoy.mean, 1)} / {fmt(v.fp_trap.mean, 1)} / {fmt(v.fp_null.mean, 1)}</td>
                </tr>))}
            </tbody>
          </table>
        </div>
      </Card>
      <div className="grid gap-4 lg:grid-cols-3">
        <Card title="False-discovery rate vs recall">
          <div className="h-64"><ResponsiveContainer><BarChart data={bars} layout="vertical" margin={{ left: 20 }}><CartesianGrid stroke="#1e2a3d" /><XAxis type="number" domain={[0, 1]} tick={{ fontSize: 10, fill: '#64748b' }} />
            <YAxis dataKey="method" type="category" width={110} tick={{ fontSize: 10, fill: '#94a3b8' }} /><Tooltip {...tip} /><Legend wrapperStyle={{ fontSize: 11 }} />
            <Bar dataKey="FDR" fill="#fb7185" /><Bar dataKey="Recall" fill="#34d399" /></BarChart></ResponsiveContainer></div>
        </Card>
        <Card title="Calibration: stated confidence vs actual hit rate">
          <div className="h-64"><ResponsiveContainer><ComposedChart data={cal} margin={{ left: 0, right: 10 }}><CartesianGrid stroke="#1e2a3d" /><XAxis dataKey="stated" type="number" domain={[0, 1]} tick={{ fontSize: 10, fill: '#64748b' }} />
            <YAxis domain={[0, 1]} tick={{ fontSize: 10, fill: '#64748b' }} /><Tooltip {...tip} formatter={(v: any) => fmt(v, 2)} />
            <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#475569" strokeDasharray="4 4" />
            <Line dataKey="actual" stroke="#38bdf8" strokeWidth={2} dot={{ r: 4 }} /></ComposedChart></ResponsiveContainer></div>
          <div className="text-[11px] text-mute">Dashed = perfect calibration. Points: {cal.map((c: any) => `${fmt(c.stated, 2)}→${fmt(c.actual, 2)} (n=${c.n})`).join('  ·  ')}</div>
        </Card>
        <Card title="Power curve: detection rate vs true Sharpe">
          <div className="h-64"><ResponsiveContainer><LineChart data={power}><CartesianGrid stroke="#1e2a3d" /><XAxis dataKey="sr" tick={{ fontSize: 10, fill: '#64748b' }} /><YAxis domain={[0, 1]} tick={{ fontSize: 10, fill: '#64748b' }} /><Tooltip {...tip} formatter={(v: any) => pct(v)} /><Legend wrapperStyle={{ fontSize: 10 }} />
            {Object.keys(b.power_curve).map((m) => <Line key={m} dataKey={m} stroke={COL[m]} strokeWidth={m.startsWith('SKEPTIC') ? 3 : 1.5} dot={false} connectNulls />)}</LineChart></ResponsiveContainer></div>
          <div className="text-[11px] text-mute">The naive analyst “detects” everything because it claims everything. Skeptic pays for its safety with lower power on weak signals.</div>
        </Card>
      </div>
      <NullLab />
      <Card title="Where Skeptic fails (measured, not hand-waved)">
        <div className="grid gap-3 md:grid-cols-2">
          {b.failure_modes.map((f: any, i: number) => (
            <div key={i} className="rounded-xl border border-amber-500/20 bg-amber-500/5 p-3">
              <div className="font-semibold text-amber-200">{f.title}</div><div className="mt-1 text-sm text-slate-300">{f.detail}</div><div className="mt-1 font-mono text-xs text-amber-300/90">{f.metric}</div>
            </div>))}
          <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 p-3"><div className="font-semibold text-amber-200">Real signals still get called LUCK</div>
            <div className="mt-1 text-sm text-slate-300">Per run, {fmt(b.real_signals_called_luck_per_run, 2)} of the 5 real signals were called LUCK and {fmt(b.real_signals_cant_tell_per_run, 2)} CAN'T TELL YET.</div></div>
        </div>
      </Card>
    </div>
  )
}
