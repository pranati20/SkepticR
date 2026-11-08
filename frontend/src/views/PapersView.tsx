import { useEffect, useMemo, useState } from 'react'

type Entry = { title: string; authors: string; year: number | string | null; venue: string; url: string | null; source: string; kind: string; snippet: string; uses: number; contexts: string[]; used?: string[]; last_seen: number }

const ago = (t: number) => { const s = Math.max(1, Math.round(Date.now() / 1000 - t)); return s < 90 ? `${s}s ago` : s < 5400 ? `${Math.round(s / 60)} min ago` : s < 129600 ? `${Math.round(s / 3600)} h ago` : `${Math.round(s / 86400)} d ago` }

export default function PapersView({ active = true }: { active?: boolean }) {
  const [d, setD] = useState<{ looked_at: Entry[] } | null>(null)
  const [err, setErr] = useState('')
  const [q, setQ] = useState('')
  const [kind, setKind] = useState<'all' | 'paper' | 'dataset'>('all')
  useEffect(() => { if (!active) return; fetch('/api/library').then((r) => (r.ok ? r.json() : Promise.reject('could not load'))).then(setD).catch((e) => setErr(String(e))) }, [active])
  const rows = useMemo(() => (d?.looked_at || []).filter((e) => (kind === 'all' || e.kind === kind) && (e.title + e.authors + e.contexts.join(' ')).toLowerCase().includes(q.toLowerCase())), [d, q, kind])
  if (err) return <div className="text-sm text-luck">{err}</div>
  if (!d) return <div className="text-mute">loading…</div>
  return (
    <div className="mx-auto max-w-4xl space-y-6">
      <div>
        <h2 className="text-lg font-semibold">Papers & datasets it has looked at</h2>
        <div className="text-sm text-slate-300">Everything the AI found while checking ideas or answering your questions. Each one is a real search result with a link; nothing is made up. Open and read a paper before relying on it.</div>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Filter by title, author or question…" className="flex-1 rounded-lg border border-line bg-panel px-3 py-2 text-sm outline-none focus:border-accent" />
        {(['all', 'paper', 'dataset'] as const).map((k) => <button key={k} onClick={() => setKind(k)} className={`btn ${kind === k ? 'border-accent text-white' : ''}`}>{k === 'all' ? 'All' : k === 'paper' ? 'Papers' : 'Datasets'}</button>)}
      </div>
      {rows.length === 0 && (
        <div className="card p-6 text-center text-sm text-mute">
          {d.looked_at.length === 0 ? 'Nothing yet. Run an example in “Watch it work”, check a paper, or ask the assistant a question, and what it finds will show up here.' : 'No matches.'}
        </div>
      )}
      <div className="space-y-2">
        {rows.map((e) => (
          <div key={(e.url || e.title)} className="card p-3">
            <div className="flex items-start gap-2">
              <span className="mt-0.5 rounded border border-gold/40 px-1.5 text-[10px] font-semibold uppercase tracking-wide text-gold">{e.kind === 'dataset' ? 'Data' : 'Paper'}</span>
              <div className="min-w-0 flex-1">
                {e.url ? <a href={e.url} target="_blank" rel="noreferrer" className="text-sm font-medium text-blue-300 hover:underline">{e.title}</a> : <span className="text-sm font-medium">{e.title}</span>}
                <div className="text-[11px] text-mute">{[e.authors, e.year, e.venue].filter(Boolean).join(' · ')} · {e.source}</div>
                {e.used?.map((u, i) => <div key={i} className="mt-1 rounded border-l-2 border-accent/60 bg-ink/50 px-2 py-1 text-[11px] leading-snug text-slate-300"><b className="text-accent">What we used:</b> {u}</div>)}
                <div className="mt-1.5 flex flex-wrap gap-1.5">
                  {e.contexts.map((c, i) => <span key={i} className="rounded-full border border-line bg-panel2 px-2 py-0.5 text-[10px] text-slate-300">{c}</span>)}
                </div>
              </div>
              <div className="shrink-0 text-right text-[11px] text-mute"><div>used {e.uses}×</div><div>{ago(e.last_seen)}</div></div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
