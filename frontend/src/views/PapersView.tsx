import { useEffect, useMemo, useState } from 'react'

type Entry = { title: string; authors: string; year: number | string | null; venue: string; url: string | null; source: string; kind: string; snippet: string; uses: number; contexts: string[]; last_seen: number }
type Ref = { title: string; authors: string; year: number; venue: string; claim: string; later: string }

const ago = (t: number) => { const s = Math.max(1, Math.round(Date.now() / 1000 - t)); return s < 90 ? `${s}s ago` : s < 5400 ? `${Math.round(s / 60)} min ago` : s < 129600 ? `${Math.round(s / 3600)} h ago` : `${Math.round(s / 86400)} d ago` }

export default function PapersView() {
  const [d, setD] = useState<{ looked_at: Entry[]; reference: Ref[] } | null>(null)
  const [err, setErr] = useState('')
  const [q, setQ] = useState('')
  const [kind, setKind] = useState<'all' | 'paper' | 'dataset'>('all')
  useEffect(() => { fetch('/api/library').then((r) => (r.ok ? r.json() : Promise.reject('could not load'))).then(setD).catch((e) => setErr(String(e))) }, [])
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
              <span className="mt-0.5">{e.kind === 'dataset' ? '🗂️' : '📄'}</span>
              <div className="min-w-0 flex-1">
                {e.url ? <a href={e.url} target="_blank" rel="noreferrer" className="text-sm font-medium text-sky-300 hover:underline">{e.title}</a> : <span className="text-sm font-medium">{e.title}</span>}
                <div className="text-[11px] text-mute">{[e.authors, e.year, e.venue].filter(Boolean).join(' · ')} · {e.source}</div>
                {e.snippet && <div className="mt-1 line-clamp-2 text-xs text-slate-400">{e.snippet}</div>}
                <div className="mt-1.5 flex flex-wrap gap-1.5">
                  {e.contexts.map((c, i) => <span key={i} className="rounded-full border border-line bg-panel2 px-2 py-0.5 text-[10px] text-slate-300">{c}</span>)}
                </div>
              </div>
              <div className="shrink-0 text-right text-[11px] text-mute"><div>used {e.uses}×</div><div>{ago(e.last_seen)}</div></div>
            </div>
          </div>
        ))}
      </div>
      <details className="card p-3">
        <summary className="cursor-pointer text-sm font-medium">Built-in reference list ({d.reference.length}) · what the Prosecutors always check against</summary>
        <div className="mt-2 text-[11px] text-amber-300/90">Hand-written summaries, not live search results. Verify before citing.</div>
        <div className="mt-2 space-y-2">
          {d.reference.map((r) => (
            <div key={r.title} className="text-sm"><b>{r.authors}</b> ({r.year}) <span className="text-slate-300">{r.title}</span><div className="text-[11px] text-mute">{r.venue} · {r.claim}{r.later ? ` Later work: ${r.later}` : ''}</div></div>
          ))}
        </div>
      </details>
    </div>
  )
}
