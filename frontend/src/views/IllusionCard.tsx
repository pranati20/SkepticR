import { useEffect, useMemo, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { RunState } from '../state'

const usd = (x: number, sign = false) => `${x < 0 ? '−' : sign && x > 0 ? '+' : ''}$${Math.abs(Math.round(x)).toLocaleString()}`
const usdShort = (x: number) => {
  const a = Math.abs(x), s = x < 0 ? '−' : ''
  return a >= 1e6 ? `${s}$${+(a / 1e6).toFixed(1)}M` : a >= 1e3 ? `${s}$${+(a / 1e3).toFixed(1)}k` : `${s}$${Math.round(a)}`
}
/** Plain number with a sign; whole numbers for counts, 3 significant digits otherwise. */
const num = (x: number, whole: boolean, sign = false) => {
  const a = Math.abs(x)
  const body = whole ? Math.round(a).toLocaleString() : a === 0 ? '0' : a >= 100 ? a.toFixed(0) : a >= 1 ? a.toFixed(2) : a.toPrecision(2)
  return `${x < 0 && body !== '0' ? '−' : sign && x > 0 && body !== '0' ? '+' : ''}${body}`
}

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

/**
 * Two curves already in display units: what the tested data promised (green) vs what is really there (red).
 * The curves end exactly at the two totals shown beside the chart; the bracket at the right edge is the gap: the cost of the illusion.
 * Both are stretched across the full width (for effects: each is sampled at the same shares of its own sample).
 */
function Chart({ A, B, fmt, xLeft, xRight, zeroLabel, k, gap, colorB = '#e5677a' }: {
  A: number[]; B: number[]; fmt: (x: number) => string; xLeft: string; xRight: string; zeroLabel?: string; k: string; gap: number; colorB?: string
}) {
  const W = 640, H = 220, L = 54, R = 92, T = 12, Bm = 26
  const all = [...A, ...B, 0]
  let lo = Math.min(...all), hi = Math.max(...all)
  const pad = (hi - lo || Math.abs(hi) || 1) * 0.06; lo -= pad; hi += pad
  const X = (i: number, n: number) => L + (i / Math.max(1, n - 1)) * (W - L - R)
  const Y = (v: number) => T + (1 - (v - lo) / (hi - lo || 1)) * (H - T - Bm)
  const path = (arr: number[]) => arr.map((v, i) => `${i ? 'L' : 'M'}${X(i, arr.length).toFixed(1)},${Y(v).toFixed(1)}`).join(' ')
  const n = Math.min(A.length, B.length)
  const area = n > 1 && A.length === B.length
    ? `${path(A)} ${B.slice().reverse().map((v, i) => `L${X(n - 1 - i, n).toFixed(1)},${Y(v).toFixed(1)}`).join(' ')} Z` : ''
  const endA = A[A.length - 1] ?? 0, endB = B[B.length - 1] ?? 0
  const xe = W - R, ya = Y(endA), yb = Y(endB)
  const sep = 13, mid = (ya + yb) / 2
  const la = Math.abs(ya - yb) < sep ? mid + (ya <= yb ? -sep / 2 : sep / 2) : ya
  const lb = Math.abs(ya - yb) < sep ? mid + (ya <= yb ? sep / 2 : -sep / 2) : yb
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={`Promised ${fmt(endA)} versus reality ${fmt(endB)}`}>
      {ticks(lo, hi).map((t) => (
        <g key={t}>
          <line x1={L} x2={xe} y1={Y(t)} y2={Y(t)} stroke={t === 0 ? '#475569' : '#1a2540'} strokeDasharray={t === 0 ? '4 4' : undefined} />
          <text x={L - 6} y={Y(t) + 3} textAnchor="end" fontSize="10" fill="#a9b4c7" className="font-mono">{fmt(t)}</text>
        </g>
      ))}
      {zeroLabel && <text x={L + 4} y={Y(0) - 4} fontSize="9" fill="#a9b4c7">{zeroLabel}</text>}
      <text x={L} y={H - 8} fontSize="10" fill="#a9b4c7">{xLeft}</text>
      <text x={xe} y={H - 8} textAnchor="end" fontSize="10" fill="#a9b4c7">{xRight}</text>
      {area && <motion.path key={'g' + k} d={area} fill={gap > 0 ? '#e5677a' : '#a9b4c7'} initial={{ opacity: 0 }} animate={{ opacity: 0.1 }} transition={{ delay: 3.0, duration: 0.6 }} />}
      <motion.path key={'a' + k} d={path(A)} fill="none" stroke="#34c38f" strokeWidth={2.2} strokeLinejoin="round" initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 1.6, ease: 'easeInOut' }} />
      <motion.path key={'b' + k} d={path(B)} fill="none" stroke={colorB} strokeWidth={2.2} strokeLinejoin="round" initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 1.4, delay: 1.7, ease: 'easeInOut' }} />
      <motion.g key={'e' + k} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 3.1 }}>
        <circle cx={xe} cy={ya} r={3.5} fill="#34c38f" />
        <circle cx={xe} cy={yb} r={3.5} fill={colorB} />
        <text x={xe + 8} y={la + 3.5} fontSize="11" fontWeight="600" fill="#34c38f" className="font-mono">{fmt(endA)}</text>
        <text x={xe + 8} y={lb + 3.5} fontSize="11" fontWeight="600" fill={colorB} className="font-mono">{fmt(endB)}</text>
        {gap > 0 && Math.abs(ya - yb) > 2 * sep && (
          <g>
            <line x1={xe + 4} x2={xe + 4} y1={Math.min(ya, yb) + 6} y2={Math.max(ya, yb) - 6} stroke="#e5677a" strokeWidth={1} strokeDasharray="2 3" />
            <text x={xe + 8} y={mid - 2} fontSize="9" fill="#f6a3ae">overstated</text>
            <text x={xe + 8} y={mid + 10} fontSize="11" fontWeight="700" fill="#f6a3ae" className="font-mono">{fmt(gap).replace(/^[+−]/, '')}</text>
          </g>
        )}
      </motion.g>
    </svg>
  )
}

/** Everything that differs between a money illusion (backtest P&L) and an effect illusion (a group difference). */
function view(cur: any, scaleTxt: string | null) {
  if (cur.kind === 'effect') {
    const bin = !!cur.binary
    const scale = bin ? Number(scaleTxt ?? 1000) || 0 : 1
    const d = cur.direction ?? 1
    const promised = cur.in_total * scale, real = cur.real_total * scale
    const f = (x: number, sign = true) => num(x, bin, sign)
    const per = bin ? `per ${scale.toLocaleString()} people` : `in average ${cur.outcome}`
    const what = bin ? `extra “${cur.outcome}” cases ${per} with “${cur.treat}”` : `difference ${per}, “${cur.treat}” vs the rest`
    return {
      A: cur.in_curve.map((v: number) => v * scale), B: cur.real_curve.map((v: number) => v * scale),
      promised, real, gap: d * (promised - real), fmt: (x: number) => f(x), big: (x: number) => f(x),
      xLeft: 'first people read', xRight: `whole group (tested ${cur.n_in.toLocaleString()} · fresh ${cur.n_hold.toLocaleString()})`, zeroLabel: 'no effect',
      promisedLabel: 'The tested data promised', realLabel: 'On fresh data', what,
      legendA: 'effect in the data it was found in (running)', legendB: `the same comparison on ${cur.n_hold.toLocaleString()} people held back`,
      scaleLabel: bin ? 'Out of how many people' : null, scaleDefault: '1000',
      subtitle: 'The Judge refuses a finding that looked real. This is how much believing it would overstate the effect.',
      story: (gap: number) => Math.sign(real) !== Math.sign(promised) && Math.abs(real) > 1e-9
        ? <>On fresh data the effect points <b className="text-luck">the other way</b> ({f(real)} instead of {f(promised)}). Believing it would get the direction wrong.</>
        : gap > 0
          ? <>Believing it would overstate the effect by <b className="text-luck">{f(gap, false)}</b> {bin ? `cases ${per}` : `in average ${cur.outcome}`}.</>
          : <>Fresh data shows a similar effect, but the finding failed the Judge’s check ({cur.cause}), so it is not counted as proven.</>,
      foot: `In ${cur.segment}. Tested on ${cur.n_in.toLocaleString()} people, checked on ${cur.n_hold.toLocaleString()} it never saw. An association, not proof of cause.`,
      unitItem: 'finding', empty: 'No finding that looked real was refused in this run.',
    }
  }
  const bets = cur.kind === 'bets'
  const stake = Number(scaleTxt ?? (bets ? 100 : 10000)) || 0
  const promised = cur.in_total * stake, real = cur.real_total * stake
  const unit = bets ? 'days of bets' : 'trading days'
  return {
    A: cur.in_curve.map((v: number) => v * stake), B: cur.real_curve.map((v: number) => v * stake),
    promised, real, gap: promised - real, fmt: usdShort, big: (x: number) => usd(x, true),
    xLeft: 'day 1', xRight: `day ${cur.n_in.toLocaleString()} (${unit})`, zeroLabel: undefined,
    promisedLabel: 'The backtest promised', realLabel: `Reality check (${cur.basis})`, what: '',
    legendA: 'what the backtest said (cumulative)', legendB: `what it really delivers: ${cur.basis}`,
    scaleLabel: bets ? 'Money per bet $' : 'Money you’d put in $', scaleDefault: bets ? '100' : '10000',
    subtitle: 'The Judge throws out a strategy that looked profitable. This is what believing it would have cost.',
    story: (gap: number) => gap > 0
      ? <>Believing it would have cost you <b className="text-luck">{usd(gap)}</b> of profit that was never there.</>
      : <>The profit did not come from skill: the strategy earns about the same without the “insight”. Don’t credit the model with <b>{usd(promised)}</b>.</>,
    foot: `Over ${cur.n_in.toLocaleString()} ${unit}, before fees unless stated. Past results, not a forecast.`,
    unitItem: 'strategy', empty: 'No lucky-looking strategy was thrown out in this run.',
  }
}

/** Non-money run where nothing refused was overstated: the headline finding, tested vs fresh data. */
function HeldUp({ c, scaleTxt, setScaleTxt }: { c: any; scaleTxt: string | null; setScaleTxt: (x: string) => void }) {
  const v = view({ ...c, kind: 'effect', cause: c.verdict }, scaleTxt)
  const k = c.hypothesis_id + ':' + (scaleTxt ?? '')
  const ok = c.verdict === 'DISCOVERY'
  return (
    <div>
      <div className="mb-3 rounded-lg border border-disc/30 bg-disc/5 px-3 py-2 text-sm text-slate-200">
        <b className="text-disc">No illusion to price.</b>{' '}
        {c.n_refused > 0 ? `${c.n_refused} finding${c.n_refused > 1 ? 's were' : ' was'} refused, but none of them shrank on fresh data. ` : 'Nothing that looked real was refused. '}
        Here is the headline finding instead: what the tested data said vs the same comparison on people it never saw.
      </div>
      <div className="grid gap-4 lg:grid-cols-5">
        <div className="lg:col-span-3">
          <div className="mb-1 truncate text-sm text-slate-200">{c.description}</div>
          <Chart A={v.A} B={v.B} fmt={v.fmt} xLeft={v.xLeft} xRight={v.xRight} zeroLabel={v.zeroLabel} k={k} gap={0} colorB="#c9a45c" />
          <div className="flex flex-wrap gap-4 text-[11px] text-mute">
            <span><span className="text-disc">━</span> {v.legendA}</span>
            <span><span className="text-gold">━</span> {v.legendB}</span>
          </div>
        </div>
        <div className="space-y-3 lg:col-span-2">
          {v.scaleLabel && (
            <label className="flex items-center gap-2 text-xs text-slate-300">{v.scaleLabel}
              <input value={scaleTxt ?? v.scaleDefault} onChange={(e) => setScaleTxt(e.target.value.replace(/[^0-9]/g, '').slice(0, 9))}
                className="w-24 rounded-md border border-line bg-ink px-2 py-1 text-right font-mono text-sm outline-none focus:border-accent" inputMode="numeric" />
            </label>
          )}
          <div className="rounded-xl border border-disc/30 bg-disc/5 p-3">
            <div className="text-[11px] uppercase tracking-wide text-mute">The tested data said</div>
            <div className="font-mono text-3xl font-semibold text-disc">{v.big(v.promised)}</div>
            <div className="text-[11px] text-slate-300">{v.what}</div>
          </div>
          <div className="rounded-xl border border-gold/30 bg-gold/5 p-3">
            <div className="text-[11px] uppercase tracking-wide text-mute">On fresh data</div>
            <div className="font-mono text-3xl font-semibold text-gold">{v.big(v.real)}</div>
            <div className="text-[11px] text-slate-300">{c.basis}</div>
          </div>
          <div className="text-[11px] text-mute">{ok ? 'The Judge accepted this finding.' : `Judge’s verdict: ${c.verdict}.`} {v.foot}</div>
        </div>
      </div>
    </div>
  )
}

export default function IllusionCard({ s }: { s: RunState }) {
  const items = s.illusions
  const [idx, setIdx] = useState(0)
  // No auto-advance by default: the card stays on the item you were looking at, including across tab switches.
  const [auto, setAuto] = useState(false)
  const [scaleTxt, setScaleTxt] = useState<string | null>(null)
  const hover = useRef(false)
  const box = useRef<HTMLDivElement>(null)
  const cur = items[Math.min(idx, items.length - 1)]
  const v = cur ? view(cur, scaleTxt) : null
  const key = cur ? cur.hypothesis_id + ':' + (scaleTxt ?? '') : ''
  const cBack = useCountUp(v?.promised ?? 0, key, 0)
  const cReal = useCountUp(v?.real ?? 0, key, 1700)
  const effect = cur?.kind === 'effect'

  useEffect(() => { setIdx(0); setScaleTxt(null) }, [s.meta?.title])
  useEffect(() => {
    if (!auto || items.length < 2) return
    // only advance while the card is actually on screen (hidden tab / collapsed section => offsetParent is null)
    const t = setInterval(() => { if (!hover.current && box.current?.offsetParent && !document.hidden) setIdx((i) => (i + 1) % items.length) }, 9000)
    return () => clearInterval(t)
  }, [auto, items.length])

  const agg = useMemo(() => {
    if (!items.length) return null
    // only average items measured in the same unit (same kind and, for effects, the same outcome)
    const unitOf = (it: any) => it.kind + (it.kind === 'effect' ? '|' + it.outcome : '')
    if (new Set(items.map(unitOf)).size > 1) return null
    const vs = items.map((it: any) => view(it, scaleTxt))
    return { a: vs.reduce((x, y) => x + y.promised, 0) / vs.length, r: vs.reduce((x, y) => x + y.real, 0) / vs.length, big: vs[0].big }
  }, [items, scaleTxt])

  return (
    <div ref={box} className="card overflow-hidden p-4">
      <div className="mb-2 flex flex-wrap items-center gap-3">
        <div>
          <div className="text-lg font-bold tracking-tight">The Cost of Illusion</div>
          <div className="text-xs text-mute">{v ? v.subtitle : 'What believing a finding the Judge refused would have cost.'}</div>
        </div>
        {v?.scaleLabel && (
          <label className="ml-auto flex items-center gap-2 text-xs text-slate-300">
            {v.scaleLabel}
            <input value={scaleTxt ?? v.scaleDefault} onChange={(e) => setScaleTxt(e.target.value.replace(/[^0-9]/g, '').slice(0, 9))}
              className="w-24 rounded-md border border-line bg-ink px-2 py-1 text-right font-mono text-sm outline-none focus:border-accent" inputMode="numeric" />
          </label>
        )}
      </div>

      {(!cur || !v) && s.effectCheck ? (
        <HeldUp c={s.effectCheck} scaleTxt={scaleTxt} setScaleTxt={setScaleTxt} />
      ) : !cur || !v ? (
        <div className="py-6 text-center text-sm text-mute">
          {s.done ? 'Nothing that looked real was refused in this run, so there is no illusion to price.' : 'Waiting for the Judge to refuse a finding that looked real…'}
        </div>
      ) : (
        <div onMouseEnter={() => (hover.current = true)} onMouseLeave={() => (hover.current = false)}>
          <AnimatePresence mode="wait">
            <motion.div key={cur.hypothesis_id} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} className="grid gap-4 lg:grid-cols-5">
              <div className="relative lg:col-span-3">
                <div className="mb-1 truncate text-sm text-slate-200">{cur.description}</div>
                <Chart A={v.A} B={v.B} fmt={v.fmt} xLeft={v.xLeft} xRight={v.xRight} zeroLabel={v.zeroLabel} k={key} gap={v.gap} />
                <div className="flex flex-wrap gap-4 text-[11px] text-mute">
                  <span><span className="text-disc">━</span> {v.legendA}</span>
                  <span><span className="text-luck">━</span> {v.legendB}</span>
                  {v.gap > 0 && <span><span className="text-luck/60">▮</span> shaded gap = cost of the illusion</span>}
                </div>
                <motion.div initial={{ scale: 2.4, opacity: 0, rotate: -18 }} animate={{ scale: 1, opacity: 1, rotate: -9 }} transition={{ delay: 3.2, type: 'spring', stiffness: 220, damping: 14 }}
                  className="pointer-events-none absolute right-[17%] top-9 rounded-md border-4 border-rose-500/80 px-3 py-1 text-center font-black uppercase tracking-widest text-rose-400/90" key={'stamp' + key}>
                  <div className="text-xl leading-none">{effect ? 'Refused' : 'Thrown out'}</div><div className="mt-0.5 text-[10px] font-semibold tracking-normal normal-case">{cur.cause}</div>
                </motion.div>
              </div>
              <div className="space-y-3 lg:col-span-2">
                <div className="rounded-xl border border-disc/30 bg-disc/5 p-3">
                  <div className="text-[11px] uppercase tracking-wide text-mute">{v.promisedLabel}</div>
                  <div className="font-mono text-3xl font-semibold text-disc">{v.big(cBack)}</div>
                  {v.what && <div className="text-[11px] text-slate-300">{v.what}</div>}
                </div>
                <div className="rounded-xl border border-rose-500/30 bg-rose-500/5 p-3">
                  <div className="text-[11px] uppercase tracking-wide text-mute">{v.realLabel}</div>
                  <div className={`font-mono text-3xl font-semibold ${v.gap > 0 ? 'text-luck' : 'text-slate-200'}`}>{v.big(cReal)}</div>
                  {effect && <div className="text-[11px] text-slate-300">{cur.basis}</div>}
                </div>
                <motion.div key={'gap' + key} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 3.1 }} className="text-sm leading-snug text-slate-200">
                  {v.story(v.gap)}
                  <div className="mt-1 text-[11px] text-mute">{v.foot}</div>
                </motion.div>
              </div>
            </motion.div>
          </AnimatePresence>
          <div className="mt-3 flex flex-wrap items-center gap-2 border-t border-line pt-2 text-xs text-mute">
            <button className="btn px-2 py-0.5" onClick={() => { setAuto(false); setIdx((idx - 1 + items.length) % items.length) }}>◀</button>
            <span>{v.unitItem} {Math.min(idx, items.length - 1) + 1} of {items.length}{cur.n_lucky_total > items.length ? ` (top ${items.length} of ${cur.n_lucky_total} refused)` : ''}</span>
            <button className="btn px-2 py-0.5" onClick={() => { setAuto(false); setIdx((idx + 1) % items.length) }}>▶</button>
            <button className="btn px-2 py-0.5" onClick={() => setAuto(!auto)}>{auto ? 'pause' : 'auto-play'}</button>
            {agg && items.length > 1 && <span className="ml-auto text-slate-300">Across these {items.length}: promised on average <b className="text-disc">{agg.big(agg.a)}</b>, reality <b>{agg.big(agg.r)}</b></span>}
          </div>
        </div>
      )}
    </div>
  )
}
