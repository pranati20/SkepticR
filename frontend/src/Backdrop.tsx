import { useMemo } from 'react'

/**
 * Page background that shows the product's idea at a glance:
 * many faint "null worlds" (random walks with no edge, i.e. pure luck) and one gold path that
 * beats the best of them, with a dashed line at the best result luck produced.
 * Generated from a seeded PRNG, so it is identical on every load; nothing is hand-placed.
 */
const W = 1600, H = 900, STEPS = 140, WORLDS = 42

function rng(seed: number) {            // mulberry32
  return () => { seed |= 0; seed = (seed + 0x6d2b79f5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296 }
}
function gauss(r: () => number) { const u = Math.max(r(), 1e-12), v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v) }

export default function Backdrop() {
  const { nulls, gold, luckMax, Y, X } = useMemo(() => {
    const r = rng(20251108)
    const walk = (drift: number) => { let v = 0; const out = [0]; for (let i = 1; i <= STEPS; i++) { v += drift + gauss(r); out.push(v) } return out }
    const nulls = Array.from({ length: WORLDS }, () => walk(0))
    const luckMax = Math.max(...nulls.map((w) => w[w.length - 1]))
    const spread = Math.max(...nulls.flat().map(Math.abs))
    // the real signal: same noise, plus a drift that carries it clearly past the best lucky world
    const gold = walk((luckMax + 0.6 * spread) / STEPS)
    const lo = Math.min(...nulls.flat(), ...gold), hi = Math.max(...nulls.flat(), ...gold)
    const Y = (v: number) => H * 0.92 - ((v - lo) / (hi - lo)) * H * 0.8
    const X = (i: number) => (i / STEPS) * W
    return { nulls, gold, luckMax, Y, X }
  }, [])
  const path = (w: number[]) => w.map((v, i) => `${i ? 'L' : 'M'}${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join('')
  return (
    <div aria-hidden className="pointer-events-none fixed inset-0 -z-10 overflow-hidden">
      <div className="absolute inset-0 bg-[radial-gradient(1200px_700px_at_70%_-10%,rgba(59,130,246,0.14),transparent_60%),radial-gradient(900px_600px_at_0%_100%,rgba(201,164,92,0.07),transparent_60%)]" />
      {/* dimmed behind the centre column so text stays readable; full strength at the page edges */}
      <div className="absolute inset-0 [mask-image:linear-gradient(to_right,black_0%,rgba(0,0,0,0.3)_22%,rgba(0,0,0,0.22)_50%,rgba(0,0,0,0.3)_78%,black_100%)]">
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="absolute inset-0 h-full w-full [mask-image:linear-gradient(to_bottom,black_0%,black_55%,transparent_100%)]">
        <g fill="none" stroke="#7f93b8" strokeWidth={1} strokeOpacity={0.14} vectorEffect="non-scaling-stroke">
          {nulls.map((w, i) => <path key={i} d={path(w)} vectorEffect="non-scaling-stroke" />)}
        </g>
        <line x1={0} x2={W} y1={Y(luckMax)} y2={Y(luckMax)} stroke="#a9b4c7" strokeOpacity={0.18} strokeDasharray="6 8" vectorEffect="non-scaling-stroke" />
        <path d={path(gold)} fill="none" stroke="#c9a45c" strokeWidth={1.6} strokeOpacity={0.55} vectorEffect="non-scaling-stroke" pathLength={1} className="backdrop-draw" />
      </svg>
      </div>
    </div>
  )
}
