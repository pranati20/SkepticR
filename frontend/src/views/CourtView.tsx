import { useMemo, useState } from 'react'
import { motion } from 'framer-motion'
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { RunState, VERDICT_COLOR, counts } from '../state'
import { ATTACK_NAMES, Banner, Card, Drawer, Gauge, Hist, Pill, Stat, Tombstone, fmt, pct, useNow } from '../ui'

// ---------------------------------------------------------------- swarm graph
const PROS = ['leakage', 'survivorship', 'costs', 'regime', 'concentration', 'confounder', 'fragility', 'literature', 'simpson', 'peeking', 'placebo']

function SwarmGraph({ s }: { s: RunState }) {
  const now = useNow(200)
  const fams = useMemo(() => Array.from(new Set(s.proposals.map((p) => p.family))), [s.proposals.length])
  const usedAttacks = useMemo(() => {
    const set = new Set<string>(); Object.values(s.prosecutions).forEach((a: any) => a.forEach((x: any) => set.add(x.attack)))
    return PROS.filter((p) => set.has(p) || (s.phase === 'idle' && ['leakage', 'costs', 'confounder'].includes(p)))
  }, [s.prosecutions, s.phase])
  const cols = [
    { x: 40, nodes: [{ id: 'Cleaner', label: 'Cleaner' }, { id: 'Librarian', label: 'Librarian' }] },
    { x: 150, nodes: (fams.length ? fams : ['hunters']).map((f) => ({ id: `${f}-hunter`, label: f.replace('_', ' ') })) },
    { x: 260, nodes: [{ id: 'Court', label: 'Null worlds' }, { id: 'Credibility', label: 'α-wealth' }] },
    { x: 370, nodes: (usedAttacks.length ? usedAttacks : ['leakage']).map((a) => ({ id: a, label: ATTACK_NAMES[a] || a })) },
    { x: 470, nodes: [{ id: 'Judge', label: 'Judge' }] },
  ]
  const H = 250
  const pos: Record<string, [number, number]> = {}
  cols.forEach((c) => c.nodes.forEach((n, i) => { pos[n.id] = [c.x, ((i + 1) * H) / (c.nodes.length + 1)] }))
  const edges: [string, string][] = []
  const link = (a: typeof cols[0], b: typeof cols[0]) => a.nodes.forEach((x) => b.nodes.forEach((y) => edges.push([x.id, y.id])))
  for (let i = 0; i < cols.length - 1; i++) link(cols[i], cols[i + 1])
  const hot = (id: string) => now - (s.active[id] || 0) < 1400
  return (
    <svg viewBox={`0 0 520 ${H}`} className="w-full">
      {edges.map(([a, b], i) => {
        const [x1, y1] = pos[a], [x2, y2] = pos[b]; const on = hot(a) || hot(b)
        return <line key={i} x1={x1 + 24} y1={y1} x2={x2 - 24} y2={y2} stroke={on ? '#38bdf8' : '#1e2a3d'} strokeWidth={on ? 1.6 : 0.8} opacity={on ? 0.95 : 0.55} />
      })}
      {cols.map((c) => c.nodes.map((n) => {
        const [x, y] = pos[n.id]; const on = hot(n.id)
        return (
          <g key={n.id}>
            <motion.rect x={x - 26} y={y - 9} width={52} height={18} rx={9} fill={on ? '#0c4a6e' : '#0f1520'} stroke={on ? '#38bdf8' : '#2a3a55'} animate={{ opacity: on ? 1 : 0.8 }} />
            <text x={x} y={y + 3} textAnchor="middle" fontSize="7" fill={on ? '#e0f2fe' : '#94a3b8'}>{n.label.length > 12 ? n.label.slice(0, 11) + '…' : n.label}</text>
          </g>
        )
      }))}
      {['LEDGER', 'NULL AUDIT', 'PROSECUTE', 'VERDICT'].map((t, i) => (<text key={t} x={[95, 205, 315, 420][i]} y={H - 4} textAnchor="middle" fontSize="6.5" fill="#475569" letterSpacing="1">{t}</text>))}
    </svg>
  )
}

function Feed({ s }: { s: RunState }) {
  const items = s.feed.slice(-9).reverse()
  return (
    <div className="space-y-1.5">
      {items.length === 0 && <div className="text-sm text-mute">Waiting for a run… press “Replay demo” or “Run live”.</div>}
      {items.map((f) => (
        <motion.div key={f.n} initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} className="text-[13px] leading-snug">
          <span className="font-semibold text-accent">{f.agent}</span> <span className="text-slate-300">{f.text}</span>
        </motion.div>
      ))}
    </div>
  )
}

// ---------------------------------------------------------------- meters
function TokenMeter({ s }: { s: RunState }) {
  const t = s.tokens
  const roles = t ? Object.entries(t.by_role) : []
  const totalTok = t?.total_tokens || 0
  const stat = t?.zero_token_ops || 0
  const calls = t?.llm_calls || 0
  const share = stat + calls ? stat / (stat + calls) : 0
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-3 gap-3">
        <Stat label="LLM tokens" value={totalTok.toLocaleString()} sub={`${calls} calls`} />
        <Stat label="Cost" value={`$${fmt(t?.total_cost_usd ?? 0, 3)}`} sub="MockLLM = $0" />
        <Stat label="Zero-token ops" value={stat.toLocaleString()} color="#34d399" sub="null worlds, backtests, attacks" />
      </div>
      <div>
        <div className="mb-1 flex justify-between text-[11px] text-mute"><span>statistics (no tokens)</span><span>LLM calls</span></div>
        <div className="flex h-2.5 overflow-hidden rounded-full bg-line"><motion.div className="bg-disc" animate={{ width: `${share * 100}%` }} /><div className="flex-1 bg-accent/70" /></div>
        <div className="mt-1 text-[11px] text-mute">{pct(share, 4)} of operations needed no tokens</div>
      </div>
      <div className="space-y-1">
        {roles.map(([role, r]: any) => (
          <div key={role} className="flex items-center gap-2 text-xs">
            <span className="w-20 text-mute">{role}</span>
            <div className="h-1.5 flex-1 rounded bg-line"><div className="h-1.5 rounded bg-accent" style={{ width: `${Math.min(100, ((r.tokens_in + r.tokens_out) / Math.max(1, totalTok)) * 100)}%` }} /></div>
            <span className="w-16 text-right font-mono text-slate-400">{(r.tokens_in + r.tokens_out).toLocaleString()}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function LedgerTicker({ s }: { s: RunState }) {
  const L = s.ledger
  return (
    <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
      <Stat label="Ideas tried" value={(L.tests || 0).toLocaleString()} sub="every test, every agent" />
      <Stat label="Cleaning choices" value={L.cleaning_choices || 0} sub={`${L.cleaning_paths || 1} forking paths`} />
      <Stat label="Effective search" value={(L.effective || L.tests || 0).toLocaleString()} color="#38bdf8" sub="what luck must beat" />
      <Stat label="Agents searching" value={L.agents || 0} sub="10 agents = 10x the luck" />
    </div>
  )
}

// ---------------------------------------------------------------- detail drawer
export function Detail({ s, id }: { s: RunState; id: string }) {
  const v = s.verdicts[id]
  if (!v) return <div className="mt-8 text-mute">No verdict yet for {id}.</div>
  const attacks = s.prosecutions[id] || v.attacks || []
  const audit = s.audits[Math.min(v.round ?? 0, s.audits.length - 1)] || s.audits[s.audits.length - 1]
  const card = s.cards[id], pr = s.prereg[id]
  const lit = attacks.find((a: any) => a.attack === 'literature')
  const obs = Math.abs(v.statistic ?? 0)
  return (
    <div className="mt-2 space-y-4 pb-10">
      <div>
        <div className="h-label">{v.family}</div>
        <h2 className="mt-1 text-lg font-semibold leading-snug">{v.description || id}</h2>
        {v.formula && <code className="mt-1 block break-all rounded bg-ink px-2 py-1 text-xs text-sky-300">{v.formula}</code>}
        <div className="mt-2 flex items-center gap-2"><Pill verdict={v.verdict} />{v.cause_of_death && <span className="text-sm text-rose-300">{v.cause_of_death}</span>}</div>
      </div>
      <Banner tone={v.verdict === 'DISCOVERY' ? 'sky' : 'amber'}>“{v.confession}”</Banner>
      <div className="grid grid-cols-4 gap-3">
        <Stat label="Adj. p" value={fmt(v.p_adj, 3)} sub={`level ${fmt(v.level, 3)}`} />
        <Stat label="Evidence score (1-q, not a probability)" value={pct(v.confidence)} />
        <Stat label="Null worlds" value={v.n_null_worlds} />
        <Stat label="Search size" value={v.search_size} />
      </div>
      {v.more_data_days && <Banner>CAN'T TELL YET: roughly {Number(v.more_data_days).toLocaleString()} {s.meta?.panel?.unit || 'more observations (days/participants)'} would be needed for an effect of this size to clear the bar. Power now ≈ {pct(v.power_at_observed)}.</Banner>}
      {audit && (
        <div>
          <div className="h-label mb-1">Best result luck produced, across {v.n_null_worlds} null worlds (whole search re-run each time)</div>
          <Hist hist={audit.histogram} marks={[{ x: obs, label: `this idea ${fmt(obs, 1)}` }]} />
          <div className="text-[11px] text-mute">If the blue bars are to the left of the green line, luck rarely matches it anywhere in the search.</div>
        </div>
      )}
      <div>
        <div className="h-label mb-2">Prosecutor results</div>
        <div className="space-y-1.5">
          {attacks.map((a: any, i: number) => (
            <details key={i} className="rounded-lg border border-line bg-panel2/60 px-3 py-2">
              <summary className="flex cursor-pointer items-center gap-2 text-sm">
                <span>{a.survived ? '✅' : a.severity === 'high' ? '☠️' : '⚠️'}</span>
                <span className="font-medium">{ATTACK_NAMES[a.attack] || a.attack}</span>
                <span className="text-[10px] uppercase text-mute">{a.severity}</span>
                <span className={`ml-auto text-xs ${a.survived ? 'text-disc' : 'text-rose-300'}`}>{a.survived ? 'survived' : 'FAILED'}</span>
              </summary>
              <pre className="mt-2 max-h-52 overflow-auto whitespace-pre-wrap break-words text-[11px] text-slate-400">{JSON.stringify(a.evidence, (k, x) => (typeof x === 'number' ? Math.round(x * 1000) / 1000 : x), 1)}</pre>
            </details>
          ))}
          {attacks.length === 0 && <div className="text-xs text-mute">Not sent to the Prosecutors: it never came close to beating the null.</div>}
        </div>
      </div>
      {lit && lit.evidence.papers?.length > 0 && (
        <div>
          <div className="h-label mb-1">Librarian: papers checked against this scenario</div>
          {lit.evidence.papers.map((p: any, i: number) => (<div key={i} className="text-xs text-slate-300">📄 {p.authors} ({p.year}) — {p.title} <span className="text-mute">· applicability {p.applicability ?? '–'}</span></div>))}
          {lit.evidence.caveats?.map((c: string, i: number) => <div key={i} className="mt-1 text-[11px] text-amber-300/90">• {c}</div>)}
        </div>
      )}
      {pr && (
        <Banner tone="sky">
          Pre-registered <code>{pr.hash.slice(0, 16)}…</code> before touching the held-out slice. Out-of-sample: {pr.oos.n_days} obs, score {fmt(pr.oos.oos_sharpe, 2)}, p={fmt(pr.oos.p_one_sided, 3)} → <b>{pr.oos.confirmed ? 'CONFIRMED' : 'NOT confirmed'}</b>
        </Banner>
      )}
      {card && (
        <div>
          <div className="h-label mb-1">Strategy card · Sharpe {fmt(card.sharpe_gross)} gross / {fmt(card.sharpe_net)} net · deflated {pct(card.deflated_sharpe_prob)} · maxDD {pct(card.max_drawdown_net, 1)}</div>
          <div className="h-36"><ResponsiveContainer><AreaChart data={card.equity_net.map((y: number, i: number) => ({ i, net: y, gross: card.equity_gross[i] }))}>
            <CartesianGrid stroke="#1e2a3d" /><XAxis dataKey="i" hide /><YAxis width={42} tick={{ fontSize: 10, fill: '#64748b' }} /><Tooltip contentStyle={{ background: '#0f1520', border: '1px solid #1e2a3d' }} />
            <Area dataKey="gross" stroke="#64748b" fill="#64748b22" /><Area dataKey="net" stroke="#34d399" fill="#34d39922" /></AreaChart></ResponsiveContainer></div>
          <div className="text-[11px] text-mute">{card.capacity_note} {card.disclaimer}</div>
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------- main view
export default function CourtView({ s, compact = false }: { s: RunState; compact?: boolean }) {
  const [sel, setSel] = useState<string | null>(null)
  const c = counts(s)
  const audit = s.audits[s.audits.length - 1]
  const discs = Object.values(s.verdicts).filter((v: any) => v.verdict === 'DISCOVERY')
  const maybe = Object.values(s.verdicts).filter((v: any) => v.verdict === "CAN'T TELL YET")
  const topMarks = (audit?.top_observed || []).slice(0, 3).map((t: any) => ({ x: t.stat, label: `best found ${fmt(t.stat, 1)}`, color: '#34d399' }))
  return (
    <div className="space-y-4">
      {s.error && <Banner tone="rose">{s.error}</Banner>}
      {s.meta?.standin && <Banner>{s.meta?.panel?.note || 'Simulated stand-in data.'} All results are demonstrations; ground truth {s.truth ? 'is' : 'may be'} known to the simulator.</Banner>}
      {s.meta?.panel?.survivorship_bias && <Banner tone="rose">Survivorship warning: this universe contains only companies that are still listed today.</Banner>}
      {s.dataQuality && (
        <Banner tone="sky">
          <b>Cleaner:</b> {s.dataQuality.steps?.map((x: any) => `${x.step.replace('_', ' ')} → ${x.mode ?? x.pct ?? x.days ?? ''} (${(x.rows_affected || 0).toLocaleString()} rows)`).join('  ·  ')}. Every choice counts as a forking path.
          {s.dataQuality.warning && <span className="text-amber-300"> {s.dataQuality.warning}</span>}
        </Banner>
      )}
      <div className="grid gap-4 xl:grid-cols-3">
        <Card title="Live swarm" right={<span className="text-xs text-mute">{s.phase}</span>} className="xl:col-span-2">
          <SwarmGraph s={s} />
          <div className="mt-2 border-t border-line pt-3"><Feed s={s} /></div>
          {s.backtest && !s.done && s.backtest.done < s.backtest.total && <div className="mt-2 text-xs text-mute">backtesting {s.backtest.done}/{s.backtest.total} (zero tokens)…</div>}
        </Card>
        <Card title="Credibility Meter" right={<span className="text-[11px] text-mute">Foster–Stine alpha-investing</span>}>
          <Gauge value={s.credibility.wealth} max={Math.max(s.credibility.wealth0 * 3, s.credibility.wealth, 0.1)} start={s.credibility.wealth0 || 0.09} />
          <div className="text-xs text-mute">Each round of tests spends wealth; validated discoveries earn some back. Low wealth forces the swarm to be selective.</div>
          {audit && <div className="mt-2 text-xs">Current level to beat: <span className="font-mono text-accent">{fmt(audit.level, 3)}</span></div>}
        </Card>
      </div>
      <Card title="Ledger"><LedgerTicker s={s} /></Card>
      <div className="grid gap-4 xl:grid-cols-3">
        <Card title="Verdicts" className="xl:col-span-1">
          <div className="grid grid-cols-3 gap-2 text-center">
            {[['DISCOVERY', c.DISCOVERY], ['LUCK', c.LUCK], ["CAN'T TELL YET", c.CANT]].map(([k, n]: any) => (
              <div key={k} className="rounded-xl border border-line p-2" style={{ background: VERDICT_COLOR[k] + '10' }}>
                <div className="font-mono text-3xl font-semibold" style={{ color: VERDICT_COLOR[k] }}>{n}</div><div className="text-[10px] tracking-wide text-mute">{k}</div>
              </div>))}
          </div>
          {s.truth && (
            <div className="mt-3 rounded-lg border border-disc/30 bg-disc/5 p-2 text-xs">
              <b className="text-disc">Ground truth check:</b> {s.truth.score.tp} of {s.truth.score.claimed} discoveries are real, {s.truth.score.fp} false.
              <div className="text-mute">The naive analyst claimed {s.naive?.count ?? '–'}.</div>
            </div>)}
          <div className="mt-3 space-y-2">
            {discs.slice(0, compact ? 4 : 8).map((v: any) => (
              <motion.button key={v.hypothesis_id} initial={{ opacity: 0, scale: 0.97 }} animate={{ opacity: 1, scale: 1 }} onClick={() => setSel(v.hypothesis_id)} className="block w-full rounded-lg border border-disc/30 bg-disc/5 p-2 text-left hover:border-disc">
                <div className="flex items-center gap-2"><Pill verdict="DISCOVERY" /><span className="truncate text-sm">{v.description}</span></div>
                <div className="mt-1 text-[11px] italic text-slate-400">“{v.confession}”</div>
                {s.prereg[v.hypothesis_id] && <div className="mt-1 text-[11px] text-sky-300">pre-registered {s.prereg[v.hypothesis_id].hash.slice(0, 10)}… → out-of-sample {s.prereg[v.hypothesis_id].oos.confirmed ? 'confirmed ✓' : 'not confirmed ✗'}</div>}
              </motion.button>))}
            {maybe.slice(0, 3).map((v: any) => (
              <button key={v.hypothesis_id} onClick={() => setSel(v.hypothesis_id)} className="block w-full rounded-lg border border-maybe/30 bg-maybe/5 p-2 text-left hover:border-maybe">
                <div className="flex items-center gap-2"><Pill verdict="CAN'T TELL YET" /><span className="truncate text-sm">{v.description}</span></div>
              </button>))}
          </div>
        </Card>
        <Card title="Null-world evidence" className="xl:col-span-1">
          {audit ? (<>
            <div className="text-xs text-mute mb-1">Round {audit.round + 1}: the best result luck produced when the whole {audit.search_size}-idea search was re-run in null worlds where nothing is real.</div>
            <Hist hist={audit.histogram} marks={topMarks} height={170} />
          </>) : <div className="text-sm text-mute">No audit yet.</div>}
        </Card>
        <Card title="Cost & tokens" className="xl:col-span-1"><TokenMeter s={s} /></Card>
      </div>
      {s.portfolio && s.portfolio.ids.length > 1 && (
        <Card title="Survivors as a portfolio">
          <div className="text-xs text-slate-300">Combined Sharpe <b>{fmt(s.portfolio.combined_sharpe)}</b> vs average single {fmt(s.portfolio.mean_individual_sharpe)}. Correlation matrix of {s.portfolio.ids.length} survivors:</div>
          <div className="mt-2 overflow-x-auto"><div className="inline-grid gap-px" style={{ gridTemplateColumns: `repeat(${s.portfolio.ids.length}, 22px)` }}>
            {s.portfolio.correlation.flat().map((x: number, i: number) => (<div key={i} title={fmt(x)} className="h-[22px] w-[22px] rounded-sm" style={{ background: `rgba(56,189,248,${Math.abs(x)})` }} />))}
          </div></div>
        </Card>)}
      <Card title={`The Graveyard · ${s.graveyard.length} killed ideas`} right={<span className="text-xs text-mute">click a tombstone for the evidence</span>}>
        {s.graveyard.length === 0 && <div className="text-sm text-mute">Nothing has died yet.</div>}
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5 xl:grid-cols-6">
          {s.graveyard.map((g) => <Tombstone key={g.hypothesis_id} g={g} onClick={() => setSel(g.hypothesis_id)} />)}
        </div>
      </Card>
      <Drawer open={!!sel} onClose={() => setSel(null)}>{sel && <Detail s={s} id={sel} />}</Drawer>
    </div>
  )
}
