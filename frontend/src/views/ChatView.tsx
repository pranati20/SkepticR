import { useEffect, useRef, useState } from 'react'

type Source = { kind: 'paper' | 'dataset'; title: string; by: string; year: string | number | null; venue: string; url: string; snippet: string; source: string }
type Msg = { role: 'user' | 'assistant'; content: string; sources?: Source[]; steps?: any[]; llm?: string; error?: string | null }

const EXAMPLES = [
  'Does short-term reversal still work in stocks? Find papers.',
  'Find a dataset I could use to test whether sports betting odds are beatable.',
  'Is there evidence that A/B test peeking inflates false positives?',
  'Find datasets about patient outcomes and treatment, and papers on confounding.',
]

function SourceCard({ s, n }: { s: Source; n: number }) {
  return (
    <a href={s.url} target="_blank" rel="noreferrer" className="block rounded-lg border border-line bg-panel2/60 p-2 hover:border-accent/60">
      <div className="flex items-start gap-2">
        <span className="mt-0.5 rounded bg-ink px-1.5 text-[10px] font-semibold text-accent">S{n}</span>
        <div className="min-w-0">
          <div className="text-sm text-sky-300 underline-offset-2 hover:underline">{s.kind === 'paper' ? '📄' : '🗂️'} {s.title}</div>
          <div className="text-[11px] text-mute">{[s.by, s.year, s.venue].filter(Boolean).join(' · ')} · {s.source}</div>
          {s.snippet && <div className="mt-1 line-clamp-2 text-xs text-slate-400">{s.snippet}</div>}
        </div>
      </div>
    </a>
  )
}

export default function ChatView({ onAudit }: { onAudit?: (question: string, sources: Source[]) => void }) {
  const [msgs, setMsgs] = useState<Msg[]>([])
  const [text, setText] = useState('')
  const [busy, setBusy] = useState(false)
  const end = useRef<HTMLDivElement>(null)
  useEffect(() => { end.current?.scrollIntoView({ behavior: 'smooth' }) }, [msgs, busy])

  const send = async (q: string) => {
    q = q.trim()
    if (!q || busy) return
    const next: Msg[] = [...msgs, { role: 'user', content: q }]
    setMsgs(next); setText(''); setBusy(true)
    try {
      const r = await fetch('/api/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: next.map((m) => ({ role: m.role, content: m.content })) }) })
      if (!r.ok) throw new Error(await r.text())
      const d = await r.json()
      setMsgs([...next, { role: 'assistant', content: d.answer, sources: d.sources, steps: d.steps, llm: d.llm, error: d.error }])
    } catch (e: any) {
      setMsgs([...next, { role: 'assistant', content: `Something went wrong: ${String(e.message || e).slice(0, 200)}` }])
    } finally { setBusy(false) }
  }

  return (
    <div className="mx-auto max-w-3xl">
      <div className="mb-3 text-sm text-slate-300">
        Ask a question in plain English. The assistant searches the internet for <b>research papers</b> and <b>datasets</b> on its own and shows
        you the real links. It never makes up a source.
      </div>
      <div className="card min-h-[320px] space-y-4 p-4">
        {msgs.length === 0 && (
          <div>
            <div className="mb-2 text-sm text-mute">Try one:</div>
            <div className="flex flex-wrap gap-2">
              {EXAMPLES.map((e) => <button key={e} className="btn text-left" onClick={() => send(e)}>{e}</button>)}
            </div>
          </div>
        )}
        {msgs.map((m, i) => (
          <div key={i} className={m.role === 'user' ? 'flex justify-end' : ''}>
            <div className={`max-w-[92%] rounded-2xl px-3 py-2 text-sm leading-relaxed ${m.role === 'user' ? 'bg-accent/90 text-ink' : 'border border-line bg-panel2/70 text-slate-200'}`}>
              <div className="whitespace-pre-wrap">{m.content}</div>
              {m.sources && m.sources.length > 0 && (
                <div className="mt-3 space-y-1.5">
                  <div className="h-label">Sources found ({m.sources.length}) · click to open</div>
                  {m.sources.map((s, j) => <SourceCard key={s.url} s={s} n={j + 1} />)}
                  {onAudit && m.sources.some((s) => s.kind === 'paper') && (
                    <div className="mt-2 rounded-lg border border-accent/40 bg-accent/5 p-2">
                      <button className="btn-primary" onClick={() => onAudit(msgs.slice(0, i).reverse().find((x) => x.role === 'user')?.content || '', m.sources!)}>Send to “Watch it work” →</button>
                      <div className="mt-1 text-[11px] text-mute">Runs a real audit on the idea in your question and hands these papers to the Skeptics as evidence.</div>
                    </div>
                  )}
                </div>
              )}
              {m.role === 'assistant' && m.llm && (
                <details className="mt-2 text-[11px] text-mute">
                  <summary className="cursor-pointer">How I searched</summary>
                  <div>Model: {m.llm}</div>
                  {m.steps?.map((st, k) => <div key={k}>{st.tool === 'search_papers' ? 'Papers' : 'Datasets'} for “{st.query}”: {st.found} found ({st.status?.join(', ')})</div>)}
                  <div>Sources come straight from the search sites, not from the AI's memory. Always open and read them before relying on them.</div>
                </details>
              )}
            </div>
          </div>
        ))}
        {busy && <div className="animate-pulseGlow text-sm text-mute">Searching papers and datasets…</div>}
        <div ref={end} />
      </div>
      <form className="mt-3 flex gap-2" onSubmit={(e) => { e.preventDefault(); send(text) }}>
        <input value={text} onChange={(e) => setText(e.target.value)} maxLength={2000} placeholder="Ask about a claim, a paper, or a dataset…"
          className="flex-1 rounded-lg border border-line bg-panel px-3 py-2 text-sm outline-none focus:border-accent" />
        <button className="btn-primary" disabled={busy || !text.trim()}>Ask</button>
        {msgs.length > 0 && <button type="button" className="btn" onClick={() => setMsgs([])}>New chat</button>}
      </form>
    </div>
  )
}
