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

function Chart({ a, b, k }: { a: number[]; b: number[]; k: string }) {
  const W = 640, H = 190, P = 8
  const all = [...a, ...b, 0]
  const lo = Math.min(...all), hi = Math.max(...all), span = hi - lo || 1
  const X = (i: number, n: number) => P + (i / Math.max(1, n - 1)) * (W - 2 * P)
  const Y = (v: number) => H - P - ((v - lo) / span) * (H - 2 * P)
  const path = (arr: number[]) => arr.map((v, i) => `${i ? 'L' : 'M'}${X(i, arr.length).toFixed(1)},${Y(v).toFixed(1)}`).join(' ')
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
      <line x1={P} x2={W - P} y1={Y(0)} y2={Y(0)} stroke="#334155" strokeDasharray="4 4" />
      <text x={W - P} y={Y(0) - 3} textAnchor="end" fontSize="9" fill="#64748b">break-even</text>
      <motion.path key={'a' + k} d={path(a)} fill="none" stroke="#34d399" strokeWidth={2.2} initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 1.6, ease: 'easeInOut' }} />
      <motion.path key={'b' + k} d={path(b)} fill="none" stroke="#fb7185" strokeWidth={2.2} strokeDasharray="1" initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 1.4, delay: 1.7, ease: 'easeInOut' }} />
    </svg>
  )
}

export default function IllusionCard({ s }: { s: RunState }) {
  const items = s.illusions
  const [idx, setIdx] = useState(0)
  const [auto, setAuto] = useState(true)
  const [stakeTxt, setStakeTxt] = useState<string | null>(null)
  const hover = useRef(false)
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
    const t = setInterval(() => { if (!hover.current) setIdx((i) => (i + 1) % items.length) }, 9000)
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
    <div className="card overflow-hidden border-rose-500/30 bg-gradient-to-br from-rose-950/30 via-panel to-panel p-4">
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
                <Chart a={cur.in_curve} b={cur.real_curve} k={key} />
                <div className="flex gap-4 text-[11px] text-mute"><span><span className="text-disc">━</span> what the backtest said</span><span><span className="text-luck">━</span> what it really delivers</span></div>
                <motion.div initial={{ scale: 2.4, opacity: 0, rotate: -18 }} animate={{ scale: 1, opacity: 1, rotate: -9 }} transition={{ delay: 3.2, type: 'spring', stiffness: 220, damping: 14 }}
                  className="pointer-events-none absolute right-4 top-10 rounded-md border-4 border-rose-500/80 px-3 py-1 text-center font-black uppercase tracking-widest text-rose-400/90" key={'stamp' + key}>
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
