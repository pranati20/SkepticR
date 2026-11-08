import { motion } from 'framer-motion'
import { RunState, VERDICT_COLOR } from '../state'
import { Banner, Card, fmt } from '../ui'

/** Naive AI vs Skeptic: the same claims, left as announced by a p<0.05 analyst, right as judged by the Court. */
export default function SplitView({ s }: { s: RunState }) {
  const claims: any[] = s.naive?.claims || []
  const judged = claims.filter((c) => s.verdicts[c.id])
  const survive = claims.filter((c) => s.verdicts[c.id]?.verdict === 'DISCOVERY').length
  const killed = claims.filter((c) => s.verdicts[c.id]?.verdict === 'LUCK').length
  const maybe = claims.filter((c) => s.verdicts[c.id]?.verdict === "CAN'T TELL YET").length
  if (!s.naive) return <Card><div className="text-sm text-mute">Start a run (Replay demo works offline) and this screen fills as the naive analyst announces its “discoveries” and the Court answers them.</div></Card>
  return (
    <div className="space-y-4">
      {s.meta?.standin && <Banner>Simulated stand-in data with known truth. {s.truth ? `Of the SkepticR survivors, ${s.truth.score.tp} of ${s.truth.score.claimed} match a planted effect.` : ''}</Banner>}
      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Naive AI analyst (anything with p < 0.05)" className="border-rose-500/20">
          <div className="mb-3 flex items-end gap-3">
            <motion.div key={claims.length} initial={{ scale: 1.4 }} animate={{ scale: 1 }} className="font-mono text-6xl font-bold text-amber-300">{claims.length}</motion.div>
            <div className="pb-2 text-sm text-slate-300">“discoveries!” out of {s.naive.total} ideas tried.<br /><span className="text-xs text-mute">No correction for how many ideas it tried.</span></div>
          </div>
          <div className="max-h-[520px] space-y-1 overflow-y-auto pr-1">
            {claims.map((c) => (
              <div key={c.id} className="flex items-center gap-2 rounded-md bg-amber-300/5 px-2 py-1 text-[13px]">
                <span className="text-gold">•</span><span className="flex-1 truncate">{c.text}</span><span className="font-mono text-[11px] text-mute">t={fmt(c.t, 1)}</span>
              </div>))}
          </div>
        </Card>
        <Card title="Skeptic (audits the whole search)" className="border-emerald-500/20">
          <div className="mb-3 flex items-end gap-3">
            <motion.div key={survive} initial={{ scale: 1.4 }} animate={{ scale: 1 }} className="font-mono text-6xl font-bold text-disc">{survive}</motion.div>
            <div className="pb-2 text-sm text-slate-300">survive<br /><span className="text-xs text-mute">{killed} killed · {maybe} can't tell · {claims.length - judged.length} still being judged</span></div>
          </div>
          <div className="max-h-[520px] space-y-1 overflow-y-auto pr-1">
            {claims.map((c) => {
              const v = s.verdicts[c.id]
              const col = v ? VERDICT_COLOR[v.verdict] : '#475569'
              return (
                <motion.div key={c.id} layout className="flex items-center gap-2 rounded-md px-2 py-1 text-[13px]" style={{ background: col + '12' }}>
                  <span style={{ color: col }}>{!v ? '…' : v.verdict === 'DISCOVERY' ? '✔' : v.verdict === 'LUCK' ? '✖' : '?'}</span>
                  <span className={`flex-1 truncate ${v?.verdict === 'LUCK' ? 'text-slate-500 line-through' : ''}`}>{c.text}</span>
                  <span className="text-[11px]" style={{ color: col }}>{v ? (v.cause_of_death || v.verdict) : 'judging'}</span>
                </motion.div>)
            })}
          </div>
        </Card>
      </div>
    </div>
  )
}
