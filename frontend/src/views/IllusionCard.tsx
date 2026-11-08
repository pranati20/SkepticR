import { useEffect, useMemo, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { RunState } from '../state'

const usd = (x: number, sign = false) => `${x < 0 ? '−' : sign && x > 0 ? '+' : ''}$${Math.abs(Math.round(x)).toLocaleString()}`

/** Counts from 0 to `target` after `delay` ms; restarts whenever `key` changes. */
function useCountUp(target: number, key: string, delay = 0, ms = 1400) {
  const [v, setV] = useState(0)
  useEffect(() => {
    setV(0)
    let raf = 0
    const t0 = performance.now() + delay
    const tick = (t: number) => {
      const p = Math.min(1, Math.max(0, (t - t0) / ms))
      setV(target * (1 - Math.pow(1 - p, 3)))
      if (p < 1) raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [key, target, delay, ms])
  return v
}

/** "Nice" axis ticks (1/2/5 × 10^k) spanning [lo, hi]. */
function ticks(lo: number, hi: number, want = 4) {
  const span = hi - lo || Math.abs(hi) || 1
  const raw = span / want, mag = Math.pow(10, Math.floor(Math.log10(raw)))
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? 10 * mag
  const out: number[] = []
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v)
  return out
}
const short = (x: number) => {
  const a = Math.abs(x), s = x < 0 ? '−' : ''
  return a >= 1e6 ? `${s}$${+(a / 1e6).toFixed(1)}M` : a >= 1e3 ? `${s}$${+(a / 1e3).toFixed(1)}k` : `${s}$${Math.round(a)}`
}

/**
 * Cumulative P&L in dollars: what the backtest promised (green) vs what the strategy really delivers (red),
 * both scaled by the user's stake. The curves end exactly at the two totals shown beside the chart, and the
 * bracket at the right edge is the gap between them: the cost of the illusion.
 */
function Chart({ a, b, stake, nDays, unit, k }: { a: number[]; b: number[]; stake: number; nDays: number; unit: string; k: string }) {
  const A = a.map((v) => v * stake), B = b.map((v) => v * stake)
  const W = 640, H = 220, L = 54, R = 92, T = 12, Bm = 26
  const all = [...A, ...B, 0]
  let lo = Math.min(...all), hi = Math.max(...all)
  const pad = (hi - lo || Math.abs(hi) || 1) * 0.06; lo -= pad; hi += pad
  const X = (i: number, n: number) => L + (i / Math.max(1, n - 1)) * (W - L - R)
  const Y = (v: number) => T + (1 - (v - lo) / (hi - lo || 1)) * (H - T - Bm)
  const path = (arr: number[]) => arr.map((v, i) => `${i ? 'L' : 'M'}${X(i, arr.length).toFixed(1)},${Y(v).toFixed(1)}`).join(' ')
  // shaded area between the two curves (both are sampled on the same days)
  const n = Math.min(A.length, B.length)
  const area = n > 1 ? `${path(A.slice(0, n))} ${B.slice(0, n).map((v, i) => `L${X(n - 1 - i, n).toFixed(1)},${Y(B[n - 1 - i]).toFixed(1)}`).join(' ')} Z` : ''
  const endA = A[A.length - 1] ?? 0, endB = B[B.length - 1] ?? 0, gap = endA - endB
  const xe = W - R, ya = Y(endA), yb = Y(endB)
  // keep the two end labels from overlapping
  const sep = 13, mid = (ya + yb) / 2
  const la = Math.abs(ya - yb) < sep ? mid + (ya <= yb ? -sep / 2 : sep / 2) : ya
  const lb = Math.abs(ya - yb) < sep ? mid + (ya <= yb ? sep / 2 : -sep / 2) : yb
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={`Backtest ${short(endA)} versus reality ${short(endB)}`}>
      {ticks(lo, hi).map((t) => (
        <g key={t}>
          <line x1={L} x2={xe} y1={Y(t)} y2={Y(t)} stroke={t === 0 ? '#475569' : '#1c2636'} strokeDasharray={t === 0 ? '4 4' : undefined} />
          <text x={L - 6} y={Y(t) + 3} textAnchor="end" fontSize="10" fill="#9ba7bb" className="font-mono">{short(t)}</text>
        </g>
      ))}
      <text x={L} y={H - 8} fontSize="10" fill="#9ba7bb">day 1</text>
      <text x={xe} y={H - 8} textAnchor="end" fontSize="10" fill="#9ba7bb">day {nDays.toLocaleString()} ({unit})</text>
      {area && <motion.path key={'g' + k} d={area} fill={gap > 0 ? '#f16d7f' : '#9ba7bb'} initial={{ opacity: 0 }} animate={{ opacity: 0.1 }} transition={{ delay: 3.0, duration: 0.6 }} />}
      <motion.path key={'a' + k} d={path(A)} fill="none" stroke="#3ecf8e" strokeWidth={2.2} strokeLinejoin="round" initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 1.6, ease: 'easeInOut' }} />
      <motion.path key={'b' + k} d={path(B)} fill="none" stroke="#f16d7f" strokeWidth={2.2} strokeLinejoin="round" initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 1.4, delay: 1.7, ease: 'easeInOut' }} />
      <motion.g key={'e' + k} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 3.1 }}>
        <circle cx={xe} cy={ya} r={3.5} fill="#3ecf8e" />
        <circle cx={xe} cy={yb} r={3.5} fill="#f16d7f" />
        <text x={xe + 8} y={la + 3.5} fontSize="11" fontWeight="600" fill="#3ecf8e" className="font-mono">{short(endA)}</text>
        <text x={xe + 8} y={lb + 3.5} fontSize="11" fontWeight="600" fill="#f16d7f" className="font-mono">{short(endB)}</text>
        {gap > 0 && Math.abs(ya - yb) > 2 * sep && (
          <g>
            <line x1={xe + 4} x2={xe + 4} y1={ya + 6} y2={yb - 6} stroke="#f16d7f" strokeWidth={1} strokeDasharray="2 3" />
            <text x={xe + 8} y={mid - 2} fontSize="9" fill="#f6a3ae">cost</text>
            <text x={xe + 8} y={mid + 10} fontSize="11" fontWeight="700" fill="#f6a3ae" className="font-mono">{short(gap)}</text>
          </g>
        )}
      </motion.g>
    </svg>
  )
}

export default function IllusionCard({ s }: { s: RunState }) {
  const items = s.illusions
  const [idx, setIdx] = useState(0)
  // No auto-advance by default: the card stays on the strategy you were looking at, including across tab switches.
  const [auto, setAuto] = useState(false)
  const [stakeTxt, setStakeTxt] = useState<string | null>(null)
  const hover = useRef(false)
  const box = useRef<HTMLDivElement>(null)
  const cur = items[Math.min(idx, items.length - 1)]
  const bets = cur?.kind === 'bets'
  const stake = Number(stakeTxt ?? (bets ? 100 : 10000)) || 0
  const key = cur ? cur.hypothesis_id + ':' + stake : ''
  const backtest = cur ? cur.in_total * stake : 0
  const real = cur ? cur.real_total * stake : 0
  const cBack = useCountUp(backtest, key, 0)
  const cReal = useCountUp(real, key, 1700)
  const gap = backtest - real

  useEffect(() => { setIdx(0) }, [s.meta?.title])
  useEffect(() => {
    if (!auto || items.length < 2) return
    // only advance while the card is actually on screen (hidden tab / collapsed section => offsetParent is null)
    const t = setInterval(() => { if (!hover.current && box.current?.offsetParent && !document.hidden) setIdx((i) => (i + 1) % items.length) }, 9000)
    return () => clearInterval(t)
  }, [auto, items.length])

  const agg = useMemo(() => {
    if (!items.length) return null
    const st = (it: any) => Number(stakeTxt ?? (it.kind === 'bets' ? 100 : 10000)) || 0
    const a = items.reduce((x: number, it: any) => x + it.in_total * st(it), 0) / items.length
    const r = items.reduce((x: number, it: any) => x + it.real_total * st(it), 0) / items.length
    return { a, r }
  }, [items, stakeTxt])

  return (
    <div ref={box} className="card overflow-hidden border-rose-500/30 bg-gradient-to-br from-rose-950/30 via-panel to-panel p-4">
      <div className="mb-2 flex flex-wrap items-center gap-3">
        <div>
          <div className="text-lg font-bold tracking-tight">💸 The Cost of Illusion</div>
          <div className="text-xs text-mute">The Judge throws out a strategy that <i>looked</i> profitable. This is what believing it would have cost.</div>
        </div>
        {cur && (
          <label className="ml-auto flex items-center gap-2 text-xs text-slate-300">
            {bets ? 'Money per bet' : 'Money you’d put in'} $
            <input value={stakeTxt ?? (bets ? '100' : '10000')} onChange={(e) => setStakeTxt(e.target.value.replace(/[^0-9]/g, '').slice(0, 9))}
              className="w-24 rounded-md border border-line bg-ink px-2 py-1 text-right font-mono text-sm outline-none focus:border-accent" inputMode="numeric" />
          </label>
        )}
      </div>

      {!cur ? (
        <div className="py-6 text-center text-sm text-mute">
          {s.done ? 'No lucky-looking strategy was thrown out in this example (or it has no money backtest).' : 'Waiting for the Judge to throw out a lucky-looking strategy…'}
        </div>
      ) : (
        <div onMouseEnter={() => (hover.current = true)} onMouseLeave={() => (hover.current = false)}>
          <AnimatePresence mode="wait">
            <motion.div key={cur.hypothesis_id} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} className="grid gap-4 lg:grid-cols-5">
              <div className="relative lg:col-span-3">
                <div className="mb-1 truncate text-sm text-slate-200">{cur.description}</div>
                <Chart a={cur.in_curve} b={cur.real_curve} stake={stake} nDays={cur.n_in} unit={bets ? 'days of bets' : 'trading days'} k={key} />
                <div className="flex flex-wrap gap-4 text-[11px] text-mute">
                  <span><span className="text-disc">━</span> what the backtest said (cumulative)</span>
                  <span><span className="text-luck">━</span> what it really delivers: {cur.basis}</span>
                  {gap > 0 && <span><span className="text-luck/60">▮</span> shaded gap = cost of the illusion</span>}
                </div>
                <motion.div initial={{ scale: 2.4, opacity: 0, rotate: -18 }} animate={{ scale: 1, opacity: 1, rotate: -9 }} transition={{ delay: 3.2, type: 'spring', stiffness: 220, damping: 14 }}
                  className="pointer-events-none absolute right-[17%] top-9 rounded-md border-4 border-rose-500/80 px-3 py-1 text-center font-black uppercase tracking-widest text-rose-400/90" key={'stamp' + key}>
                  <div className="text-xl leading-none">Thrown out</div><div className="mt-0.5 text-[10px] font-semibold tracking-normal normal-case">{cur.cause}</div>
                </motion.div>
              </div>
              <div className="space-y-3 lg:col-span-2">
                <div className="rounded-xl border border-disc/30 bg-disc/5 p-3">
                  <div className="text-[11px] uppercase tracking-wide text-mute">The backtest promised</div>
                  <div className="font-mono text-3xl font-semibold text-disc">{usd(cBack, true)}</div>
                </div>
                <div className="rounded-xl border border-rose-500/30 bg-rose-500/5 p-3">
                  <div className="text-[11px] uppercase tracking-wide text-mute">Reality check ({cur.basis})</div>
                  <div className={`font-mono text-3xl font-semibold ${real < 0 ? 'text-luck' : 'text-slate-200'}`}>{usd(cReal, true)}</div>
                </div>
                <motion.div key={'gap' + key} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 3.1 }} className="text-sm leading-snug text-slate-200">
                  {gap > 0
                    ? <>Believing it would have cost you <b className="text-luck">{usd(gap)}</b> of profit that was never there.</>
                    : <>The profit did not come from skill: the strategy earns about the same without the “insight”. Don’t credit the model with <b>{usd(backtest)}</b>.</>}
                  <div className="mt-1 text-[11px] text-mute">Over {cur.n_in.toLocaleString()} {bets ? 'days of bets' : 'trading days'}, before fees unless stated. Past results, not a forecast.</div>
                </motion.div>
              </div>
            </motion.div>
          </AnimatePresence>
          <div className="mt-3 flex flex-wrap items-center gap-2 border-t border-line pt-2 text-xs text-mute">
            <button className="btn px-2 py-0.5" onClick={() => { setAuto(false); setIdx((idx - 1 + items.length) % items.length) }}>◀</button>
            <span>strategy {Math.min(idx, items.length - 1) + 1} of {items.length}{cur.n_lucky_total > items.length ? ` (worst ${items.length} of ${cur.n_lucky_total} lucky-looking)` : ''}</span>
            <button className="btn px-2 py-0.5" onClick={() => { setAuto(false); setIdx((idx + 1) % items.length) }}>▶</button>
            <button className="btn px-2 py-0.5" onClick={() => setAuto(!auto)}>{auto ? 'pause' : 'auto-play'}</button>
            {agg && items.length > 1 && <span className="ml-auto text-slate-300">Across these {items.length}: backtests averaged <b className="text-disc">{usd(agg.a, true)}</b>, reality <b>{usd(agg.r, true)}</b></span>}
          </div>
        </div>
      )}
    </div>
  )
}
