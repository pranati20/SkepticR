import { useState } from 'react'
import { useBenchmark, isOurs, displayName, upperBoundFdr } from '../benchmark'

const pct = (x: number, d = 0) => `${(100 * x).toFixed(d)}%`
/** SkepticR's bar shows its error TARGET (a fixed design choice), not the measured rate; the measured value stays in the table and tooltip. */
const OUR_TARGET_FDR = 0.05

/** Horizontal bars: share of claimed discoveries that were false, per method. Ours in gold, the rest neutral. */
function FdrBars({ rows }: { rows: { name: string; fdr: number; claimed: number; tp: number; fp: number }[] }) {
  const [hover, setHover] = useState<number | null>(null)
  const W = 640, L = 190, R = 56, rowH = 34, T = 8, H = T + rows.length * rowH + 24
  const X = (v: number) => L + v * (W - L - R)
  const ticks = [0, 0.25, 0.5, 0.75, 1]
  return (
    <div className="relative">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="False discovery rate by method">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={X(t)} x2={X(t)} y1={T} y2={H - 22} stroke="#22304d" />
            <text x={X(t)} y={H - 6} textAnchor="middle" fontSize="11" fill="#a9b4c7">{pct(t)}</text>
          </g>
        ))}
        {rows.map((r, i) => {
          const ours = isOurs(r.name)
          const y = T + i * rowH + 7, w = Math.max(0, X(ours ? OUR_TARGET_FDR : r.fdr) - L)
          return (
            <g key={r.name} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}>
              <rect x={0} y={y - 6} width={W} height={rowH} fill={hover === i ? 'rgba(255,255,255,0.03)' : 'transparent'} />
              <text x={L - 12} y={y + 14} textAnchor="end" fontSize="12.5" fontWeight={ours ? 700 : 500} fill={ours ? '#ffffff' : '#cbd3e1'}>{displayName(r.name)}</text>
              {w > 0 ? <rect x={L} y={y + 2} width={w} height={18} rx={3} fill={ours ? '#c9a45c' : '#3a4a6b'} />
                : <line x1={L} x2={L} y1={y} y2={y + 22} stroke="#c9a45c" strokeWidth={3} />}
              <text x={L + w + 8} y={y + 15} fontSize="12" fontWeight={ours ? 700 : 500} fill={ours ? '#c9a45c' : '#a9b4c7'} className="font-mono">{ours ? `≤${pct(OUR_TARGET_FDR)} (target)` : pct(r.fdr)}</text>
            </g>
          )
        })}
      </svg>
      {hover !== null && (
        <div className="pointer-events-none absolute right-2 top-2 rounded-md border border-line bg-ink/95 px-3 py-2 text-xs text-slate-200">
          <div className="font-semibold text-white">{displayName(rows[hover].name)}</div>
          <div>{rows[hover].claimed.toFixed(1)} claimed per run · {rows[hover].tp.toFixed(2)} real · {rows[hover].fp.toFixed(1)} false</div>
          {isOurs(rows[hover].name) && <div className="text-mute">Bar shows the ≤{pct(OUR_TARGET_FDR)} target; measured: {pct(rows[hover].fdr)}</div>}
        </div>
      )}
    </div>
  )
}

export default function ProofView() {
  const { data: d, error } = useBenchmark()
  if (error) return <div className="text-sm text-luck">{error}</div>
  if (!d) return <div className="text-mute">loading…</div>
  const st = d.setup || {}
  const rows = Object.entries<any>(d.table).map(([name, r]) => ({
    name, claimed: r.claimed.mean, tp: r.tp.mean, fp: r.fp.mean, precision: r.precision.mean, recall: r.recall.mean, recallCi: r.recall.ci95, fdr: r.fdr.mean,
  }))
  const byFdr = [...rows].sort((a, b) => b.fdr - a.fdr)
  const ours = rows.find((r) => isOurs(r.name))
  const bound = ours ? upperBoundFdr(ours, d.n_seeds) : null
  return (
    <div className="mx-auto max-w-5xl space-y-5">
      <div>
        <div className="h-label mb-1">The proof · ground truth known</div>
        <h2 className="text-2xl font-semibold">How many “discoveries” are actually luck?</h2>
        <p className="mt-1 max-w-3xl text-sm text-slate-300">
          {st.signals} candidate signals over {st.assets} assets: {st.planted} are real (Sharpe {st.planted_sr?.join(', ')}), {st.decoys} are confounded decoys, {st.traps} are lookahead or survivorship traps, the rest are noise.
          Every method gets the same search; we know the answer, so we can count its mistakes. Averaged over {d.n_seeds} seeds with {d.n_worlds} null worlds each.
        </p>
      </div>

      <div className="grid gap-5 lg:grid-cols-5">
        <div className="card p-4 lg:col-span-3">
          <div className="h-label mb-2">Share of claimed discoveries that were false</div>
          <FdrBars rows={byFdr} />
          <div className="mt-1 text-[11px] text-mute">Lower is better. Hover a bar for counts per run.</div>
        </div>
        <div className="card flex flex-col justify-center gap-4 p-5 lg:col-span-2">
          {ours && (<>
            <div>
              <div className="font-mono text-5xl font-semibold text-gold">{ours.fp.toFixed(1)}</div>
              <div className="text-sm text-slate-300">false discoveries per run for SkepticR, out of {ours.claimed.toFixed(1)} claimed</div>
            </div>
            <div>
              <div className="font-mono text-3xl font-semibold text-white">{pct(ours.recall)}</div>
              <div className="text-sm text-slate-300">of the real signals still found (± {pct(ours.recallCi)})</div>
            </div>
            {bound !== null && <div className="border-t border-line pt-3 text-xs leading-relaxed text-mute">
              Zero observed is not zero error: over {Math.round(ours.claimed * d.n_seeds)} claims the one-sided 95% upper bound on the false-discovery rate is about {pct(bound, 1)}.
            </div>}
          </>)}
        </div>
      </div>

      <div className="card overflow-x-auto p-4">
        <div className="h-label mb-2">Full table · mean per run</div>
        <table className="w-full text-sm">
          <thead><tr className="border-b border-line text-left text-[11px] uppercase tracking-wide text-mute">
            <th className="py-2 pr-3 font-semibold">Method</th><th className="px-3 text-right font-semibold">Claimed</th><th className="px-3 text-right font-semibold">Real</th>
            <th className="px-3 text-right font-semibold">False</th><th className="px-3 text-right font-semibold">Precision</th><th className="px-3 text-right font-semibold">Recall</th><th className="pl-3 text-right font-semibold">False share</th>
          </tr></thead>
          <tbody className="font-mono tabular-nums">
            {rows.map((r) => (
              <tr key={r.name} className={`border-b border-line/60 ${isOurs(r.name) ? 'text-gold' : 'text-slate-200'}`}>
                <td className={`py-2 pr-3 font-sans ${isOurs(r.name) ? 'font-semibold' : ''}`}>{displayName(r.name)}</td>
                <td className="px-3 text-right">{r.claimed.toFixed(1)}</td><td className="px-3 text-right">{r.tp.toFixed(2)}</td><td className="px-3 text-right">{r.fp.toFixed(1)}</td>
                <td className="px-3 text-right">{r.precision.toFixed(2)}</td><td className="px-3 text-right">{r.recall.toFixed(2)}</td><td className="pl-3 text-right">{pct(r.fdr)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {d.failure_modes?.length > 0 && (
        <div className="card p-4">
          <div className="h-label mb-2">Where it can still fail (measured, not hidden)</div>
          <div className="grid gap-3 md:grid-cols-2">
            {d.failure_modes.map((f: any) => (
              <div key={f.title} className="rounded-md border border-line p-3">
                <div className="text-sm font-semibold text-white">{f.title}</div>
                {f.metric && <div className="mt-0.5 font-mono text-xs text-gold">{f.metric}</div>}
                <div className="mt-1 text-xs leading-relaxed text-slate-300">{f.detail}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
