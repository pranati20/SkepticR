import { motion, AnimatePresence } from 'framer-motion'
import { ReactNode, useEffect, useState } from 'react'
import { VERDICT_COLOR } from './state'

export const fmt = (x: any, d = 2) => (x === null || x === undefined || Number.isNaN(x) ? '–' : Number(x).toFixed(d))
export const pct = (x: any, d = 0) => (x === null || x === undefined ? '–' : `${(100 * Number(x)).toFixed(d)}%`)

export function Card({ title, right, children, className = '' }: { title?: string; right?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={`card p-4 ${className}`}>
      {(title || right) && (
        <header className="mb-3 flex items-center justify-between">
          <h3 className="h-label">{title}</h3>
          {right}
        </header>
      )}
      {children}
    </section>
  )
}

export function Stat({ label, value, sub, color }: { label: string; value: ReactNode; sub?: ReactNode; color?: string }) {
  return (
    <div>
      <div className="h-label">{label}</div>
      <motion.div key={String(value)} initial={{ opacity: 0.4, y: -4 }} animate={{ opacity: 1, y: 0 }} className="font-mono text-2xl font-semibold tabular-nums" style={{ color }}>
        {value}
      </motion.div>
      {sub && <div className="text-xs text-mute">{sub}</div>}
    </div>
  )
}

export const VERDICT_PLAIN: Record<string, string> = { DISCOVERY: 'Looks real', LUCK: 'Probably luck', "CAN'T TELL YET": 'Not enough data' }

export function Pill({ verdict }: { verdict: string }) {
  const c = VERDICT_COLOR[verdict] || '#94a3b8'
  return (
    <span className="rounded-full border px-2 py-0.5 text-[11px] font-semibold tracking-wide" style={{ color: c, borderColor: c + '66', background: c + '14' }}>
      {VERDICT_PLAIN[verdict] || verdict}
    </span>
  )
}

export function Banner({ children, tone = 'amber' }: { children: ReactNode; tone?: 'amber' | 'rose' | 'sky' }) {
  const t = { amber: 'border-amber-500/40 bg-amber-500/10 text-amber-200', rose: 'border-rose-500/40 bg-rose-500/10 text-rose-200', sky: 'border-blue-500/40 bg-blue-500/10 text-blue-200' }[tone]
  return <div className={`rounded-xl border px-3 py-2 text-xs leading-relaxed ${t}`}>{children}</div>
}

/** Credibility Meter: a semicircular gauge of alpha-investing wealth that drains and refills. */
export function Gauge({ value, max, start }: { value: number; max: number; start: number }) {
  const frac = Math.max(0, Math.min(1, max ? value / max : 0))
  const R = 78, cx = 100, cy = 98
  const pt = (f: number) => { const a = Math.PI * (1 - f); return [cx + R * Math.cos(a), cy - R * Math.sin(a)] }
  const [x1, y1] = pt(0), [x3, y3] = pt(1), [sx, sy] = pt(Math.min(1, start / max))
  const ratio = start ? value / start : 1
  const col = ratio > 0.7 ? '#34c38f' : ratio > 0.3 ? '#e8b04b' : '#e5677a'
  return (
    <svg viewBox="0 0 200 118" className="w-full">
      <path d={`M ${x1} ${y1} A ${R} ${R} 0 0 1 ${x3} ${y3}`} stroke="#22304d" strokeWidth="14" fill="none" strokeLinecap="round" />
      <motion.path d={`M ${x1} ${y1} A ${R} ${R} 0 0 1 ${x3} ${y3}`} stroke={col} strokeWidth="14" fill="none" strokeLinecap="round"
        initial={false} animate={{ pathLength: Math.max(0.001, frac) }} transition={{ duration: 0.6 }} />
      <line x1={sx} y1={sy} x2={cx + (sx - cx) * 0.8} y2={cy + (sy - cy) * 0.8} stroke="#94a3b8" strokeWidth="1.5" />
      <text x={cx} y={cy - 18} textAnchor="middle" className="fill-white font-mono" fontSize="26" fontWeight="600">{value.toFixed(3)}</text>
      <text x={cx} y={cy - 2} textAnchor="middle" className="fill-slate-400" fontSize="9">alpha-wealth (started {start.toFixed(2)})</text>
    </svg>
  )
}

/** Null-world histogram: the best result luck produced in each world, with observed statistics marked. */
export function Hist({ hist, marks = [], height = 150 }: { hist: { counts: number[]; edges: number[]; p95?: number }; marks?: { x: number; label: string; color?: string }[]; height?: number }) {
  if (!hist) return null
  const W = 420, H = height, pad = 22
  const maxC = Math.max(...hist.counts, 1)
  const x0 = hist.edges[0], x1 = hist.edges[hist.edges.length - 1]
  const X = (v: number) => pad + ((v - x0) / (x1 - x0 || 1)) * (W - 2 * pad)
  return (
    <svg viewBox={`0 0 ${W} ${H + 24}`} className="w-full">
      {hist.counts.map((c, i) => {
        const bx = X(hist.edges[i]), bw = Math.max(1, X(hist.edges[i + 1]) - X(hist.edges[i]) - 1), bh = (c / maxC) * (H - 30)
        return <motion.rect key={i} x={bx} width={bw} fill="#3b82f6" opacity="0.55" initial={{ height: 0, y: H }} animate={{ height: bh, y: H - bh }} transition={{ duration: 0.5, delay: i * 0.01 }} />
      })}
      {hist.p95 !== undefined && (<g><line x1={X(hist.p95)} x2={X(hist.p95)} y1={6} y2={H} stroke="#e8b04b" strokeDasharray="3 3" /><text x={X(hist.p95)} y={H + 14} textAnchor="middle" fontSize="9" fill="#e8b04b">luck beats this 5%</text></g>)}
      {marks.map((m, i) => (
        <g key={i}>
          <line x1={X(m.x)} x2={X(m.x)} y1={0} y2={H} stroke={m.color || '#34c38f'} strokeWidth="2" />
          <text x={Math.min(W - 40, X(m.x))} y={10 + (i % 3) * 10} fontSize="9" fill={m.color || '#34c38f'} textAnchor={X(m.x) > W - 90 ? 'end' : 'start'}> {m.label}</text>
        </g>
      ))}
      <line x1={pad} x2={W - pad} y1={H} y2={H} stroke="#334155" />
      <text x={pad} y={H + 14} fontSize="9" fill="#64748b">{x0.toFixed(1)}</text>
      <text x={W - pad} y={H + 14} fontSize="9" fill="#64748b" textAnchor="end">{x1.toFixed(1)}</text>
    </svg>
  )
}

const CAUSE_ICON: Record<string, string> = {
  beta: '📈', costs: '💸', 'multiple': '🎲', leak: '🔮', survivor: '🧟', regime: '🌪️', concentr: '🎯', fragil: '🧊', simpson: '🔀', peeking: '👀', placebo: '🎭', sign: '↩️',
}
export const causeIcon = (c: string) => { const k = Object.keys(CAUSE_ICON).find((k) => (c || '').toLowerCase().includes(k)); return k ? CAUSE_ICON[k] : '🪦' }

export function Tombstone({ g, onClick }: { g: any; onClick: () => void }) {
  return (
    <motion.button layout initial={{ opacity: 0, y: 20, scale: 0.9 }} animate={{ opacity: 1, y: 0, scale: 1 }} whileHover={{ y: -3 }} onClick={onClick}
      className="group relative flex h-[124px] flex-col items-center rounded-t-[56px] rounded-b-lg border border-line bg-panel2 px-3 pb-2 pt-4 text-center hover:border-rose-400/60">
      <div className="h-px w-6 bg-luck/70" />
      <div className="mt-1 text-[10px] font-semibold uppercase tracking-wide text-rose-300">{g.cause}</div>
      <div className="mt-1 line-clamp-2 text-[10px] leading-snug text-slate-300">{g.description || g.hypothesis_id}</div>
      <div className="mt-auto font-mono text-[9px] text-mute">naive p={g.p_naive !== undefined ? Number(g.p_naive).toExponential(0) : '–'}</div>
    </motion.button>
  )
}

export function Drawer({ open, onClose, children }: { open: boolean; onClose: () => void; children: ReactNode }) {
  useEffect(() => { const f = (e: KeyboardEvent) => e.key === 'Escape' && onClose(); window.addEventListener('keydown', f); return () => window.removeEventListener('keydown', f) }, [onClose])
  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div className="fixed inset-0 z-40 bg-black/50" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={onClose} />
          <motion.aside className="fixed right-0 top-0 z-50 h-full w-full max-w-xl overflow-y-auto border-l border-line bg-panel p-5" initial={{ x: 540 }} animate={{ x: 0 }} exit={{ x: 540 }} transition={{ type: 'spring', damping: 28, stiffness: 260 }}>
            <button className="btn float-right" onClick={onClose}>close ✕</button>
            {children}
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  )
}

export function useNow(ms = 250) { const [t, setT] = useState(Date.now()); useEffect(() => { const i = setInterval(() => setT(Date.now()), ms); return () => clearInterval(i) }, [ms]); return t }

export const ATTACK_NAMES: Record<string, string> = {
  leakage: 'Leakage', survivorship: 'Survivorship', costs: 'Costs', regime: 'Regime', concentration: 'Concentration', confounder: 'Confounder / beta',
  fragility: 'Fragility', spec_review: 'Spec review (LLM)', literature: 'Literature', simpson: "Simpson's paradox", peeking: 'Peeking', placebo: 'Placebo',
}
