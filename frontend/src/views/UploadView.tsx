import { useState } from 'react'
import { Banner, Card } from '../ui'

export default function UploadView({ run, running, goCourt }: { run: (p: any) => void; running: boolean; goCourt: () => void }) {
  const [csv, setCsv] = useState('')
  const [name, setName] = useState('')
  const [prof, setProf] = useState<any>(null)
  const [claim, setClaim] = useState('')
  const [treat, setTreat] = useState('')
  const [outcome, setOutcome] = useState('')
  const [subs, setSubs] = useState<string[]>([])
  const [strata, setStrata] = useState<string[]>([])
  const [err, setErr] = useState('')

  const profile = async (text: string, c = claim) => {
    setErr('')
    try {
      const r = await fetch('/api/upload', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ csv: text, claim: c }) })
      if (!r.ok) throw new Error(await r.text())
      const p = await r.json(); setProf(p); setTreat(p.suggest.treat || ''); setOutcome(p.suggest.outcome || ''); setSubs(p.suggest.subgroups || [])
    } catch (e: any) { setErr(String(e.message || e)) }
  }
  const onFile = async (f?: File | null) => { if (!f) return; const t = await f.text(); setCsv(t); setName(f.name); profile(t) }
  const sample = async () => { const r = await fetch('/api/sample/health'); const t = await r.text(); setCsv(t); setName('sample_supplements.csv'); setClaim('Supplement users have lower health_score than non-users'); profile(t, 'supplement health_score') }
  const toggle = (arr: string[], set: (x: string[]) => void, c: string) => set(arr.includes(c) ? arr.filter((x) => x !== c) : [...arr, c])
  return (
    <div className="space-y-4">
      <Banner tone="sky">Upload a CSV and describe your claim. It goes through the same Court: your analysis is counted as one search over every subgroup and metric, tested in null worlds (treatment labels permuted, within confounder strata if you name them), then prosecuted. Data stays on this machine; nothing is stored.</Banner>
      <Card title="1 · Your data and claim">
        <div className="flex flex-wrap items-center gap-2">
          <input type="file" accept=".csv,text/csv" onChange={(e) => onFile(e.target.files?.[0])} className="text-sm" />
          <button className="btn" onClick={sample}>or load a sample (health, with a hidden confounder)</button>
          {name && <span className="text-xs text-mute">{name} · {prof?.rows?.toLocaleString()} rows</span>}
        </div>
        <textarea value={claim} onChange={(e) => setClaim(e.target.value)} placeholder="Describe the claim, e.g. “users on the new plan spend more”" rows={2} className="mt-3 w-full rounded-lg border border-line bg-ink p-3 text-sm outline-none focus:border-accent" />
        {err && <div className="mt-2"><Banner tone="rose">{err}</Banner></div>}
      </Card>
      {prof && (
        <Card title="2 · Map the columns">
          <div className="grid gap-4 md:grid-cols-2">
            <label className="text-sm">Treatment / exposure (two groups)<select value={treat} onChange={(e) => setTreat(e.target.value)} className="btn mt-1 block w-full"><option value=""></option>{prof.columns.map((c: any) => <option key={c.name}>{c.name}</option>)}</select></label>
            <label className="text-sm">Outcome (numeric)<select value={outcome} onChange={(e) => setOutcome(e.target.value)} className="btn mt-1 block w-full"><option value=""></option>{prof.columns.filter((c: any) => c.numeric).map((c: any) => <option key={c.name}>{c.name}</option>)}</select></label>
            <div><div className="h-label mb-1">Subgroups to search (each one is a forking path)</div><div className="flex flex-wrap gap-1.5">{prof.columns.filter((c: any) => c.name !== treat && c.name !== outcome).map((c: any) => (<button key={c.name} onClick={() => toggle(subs, setSubs, c.name)} className={`rounded-full border px-2 py-0.5 text-xs ${subs.includes(c.name) ? 'border-accent bg-accent/20 text-white' : 'border-line text-mute'}`}>{c.name}</button>))}</div></div>
            <div><div className="h-label mb-1">Known confounders (permute treatment within these; enables Simpson check)</div><div className="flex flex-wrap gap-1.5">{prof.columns.filter((c: any) => c.name !== treat && c.name !== outcome && c.n_unique <= 12).map((c: any) => (<button key={c.name} onClick={() => toggle(strata, setStrata, c.name)} className={`rounded-full border px-2 py-0.5 text-xs ${strata.includes(c.name) ? 'border-amber-400 bg-amber-400/20 text-white' : 'border-line text-mute'}`}>{c.name}</button>))}</div></div>
          </div>
          <div className="mt-4 flex items-center gap-3">
            <button className="btn-primary" disabled={running || !treat || !outcome} onClick={() => { run({ csv, treat, outcome, subgroups: subs, strata, claim }); goCourt() }}>Put my claim on trial</button>
            <span className="text-xs text-mute">Results appear in the Court tab.</span>
          </div>
          <div className="mt-3 overflow-x-auto"><table className="text-xs"><thead><tr>{prof.columns.map((c: any) => <th key={c.name} className="px-2 text-left text-mute">{c.name}</th>)}</tr></thead><tbody>{prof.preview.map((r: any, i: number) => <tr key={i}>{prof.columns.map((c: any) => <td key={c.name} className="px-2 font-mono text-slate-400">{String(r[c.name]).slice(0, 14)}</td>)}</tr>)}</tbody></table></div>
        </Card>)}
    </div>
  )
}
