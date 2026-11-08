import { useEffect, useRef, useState } from 'react'
import katex from 'katex'
import 'katex/dist/katex.min.css'

type Eq = { latex: string; section: string }
type Paper = { used?: string; title: string; authors: string; year: number | null; venue: string; url: string | null; source: string; abstract: string; cited_by?: number | null; equations: Eq[]; equation_note: string }
type Topic = { id: string; title: string; component: string; papers: Paper[]; status: string[] }
type Data = { topics: Topic[]; conclusion: { text: string; llm: string; style: string } }

function Tex({ src }: { src: string }) {
  const ref = useRef<HTMLDivElement>(null)
  const [fallback, setFallback] = useState(false)
  useEffect(() => {
    try { katex.render(src, ref.current!, { displayMode: true, throwOnError: true, strict: 'ignore', trust: false, output: 'html' }) } catch { setFallback(true) }
  }, [src])
  return fallback ? <pre className="overflow-x-auto rounded bg-ink p-2 text-[11px] text-slate-300">{src}</pre> : <div ref={ref} className="overflow-x-auto py-1 text-sm" />
}

const STYLES = ['short bullet points, one per Skeptic component', 'a table: component, best paper, what it adds', 'three sentences for a portfolio manager', 'a prioritised to-do list for the developers']

export default function MethodsView() {
  const [style, setStyle] = useState(STYLES[0])
  const [d, setD] = useState<Data | null>(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')

  const scan = async () => {
    setBusy(true); setErr('')
    try {
      const r = await fetch('/api/methods', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ style }) })
      if (!r.ok) throw new Error(await r.text())
      setD(await r.json())
    } catch (e: any) { setErr(String(e.message || e).slice(0, 200)) } finally { setBusy(false) }
  }
  useEffect(() => { scan() }, [])   // eslint-disable-line react-hooks/exhaustive-deps

  let n = 0
  return (
    <div className="mx-auto max-w-4xl space-y-5">
      <div>
        <h2 className="text-lg font-semibold">Papers that could make SkepticR’s own math stronger</h2>
        <div className="text-sm text-slate-300">Skeptic searches the research literature by itself for the weak points of its own statistics (listed below). You can’t choose the search:
          you only choose how the conclusion is written. Equations are copied from the papers’ own LaTeX, never written by the AI. Where a paper has no readable full text we say so and link it.</div>
      </div>
      <div className="card flex flex-wrap items-center gap-2 p-3">
        <label className="text-sm text-slate-300">How should the conclusion be shown?</label>
        <input value={style} onChange={(e) => setStyle(e.target.value)} maxLength={300} className="min-w-[260px] flex-1 rounded-lg border border-line bg-panel px-3 py-1.5 text-sm outline-none focus:border-accent" />
        <button className="btn-primary" disabled={busy} onClick={scan}>{busy ? 'Searching…' : 'Update conclusion'}</button>
        <div className="flex w-full flex-wrap gap-1.5">{STYLES.map((x) => <button key={x} className="btn px-2 py-0.5 text-[11px]" onClick={() => setStyle(x)}>{x}</button>)}</div>
      </div>
      {err && <div className="text-sm text-luck">{err}</div>}
      {busy && !d && <div className="animate-pulseGlow text-sm text-mute">Searching OpenAlex, Crossref and arXiv, and reading equations…</div>}
      {d && (
        <>
          <div className="card border-accent/40 p-4">
            <div className="h-label mb-1">Conclusion</div>
            <div className="whitespace-pre-wrap text-sm leading-relaxed text-slate-200">{d.conclusion.text}</div>
            <div className="mt-2 text-[11px] text-mute">Written by: {d.conclusion.llm}. Citations [S#] follow the order of the papers below. Reading a paper’s abstract is not the same as validating its method on our data.</div>
          </div>
          {d.topics.map((t) => (
            <div key={t.id} className="card p-4">
              <div className="flex flex-wrap items-baseline gap-2">
                <div className="font-semibold">{t.title}</div>
                <span className="rounded-full border border-accent/40 px-2 py-0.5 text-[10px] text-accent">strengthens: {t.component}</span>
              </div>
              <div className="text-[11px] text-mute">{t.status.join(' · ')}</div>
              {t.papers.length === 0 && <div className="mt-2 text-sm text-mute">No relevant paper came back from the live search.</div>}
              {t.papers.map((p) => { n += 1; return (
                <div key={p.title + n} className="mt-3 rounded-lg border border-line bg-panel2/50 p-3">
                  <div className="flex items-start gap-2">
                    <span className="mt-0.5 rounded bg-ink px-1.5 text-[10px] font-semibold text-accent">S{n}</span>
                    <div className="min-w-0 flex-1">
                      {p.url ? <a href={p.url} target="_blank" rel="noreferrer" className="text-sm font-medium text-blue-300 hover:underline">{p.title}</a> : <span className="text-sm font-medium">{p.title}</span>}
                      <div className="text-[11px] text-mute">{[p.authors, p.year, p.venue].filter(Boolean).join(' · ')} · {p.source}{p.cited_by ? ` · cited ${p.cited_by}×` : ''}</div>
                      {p.abstract && <div className="mt-1 line-clamp-3 text-xs text-slate-400">{p.abstract}</div>}
                      {p.used && <div className="mt-1 rounded border-l-2 border-accent/60 bg-ink/50 px-2 py-1 text-[11px] leading-snug text-slate-300"><b className="text-accent">What we used:</b> {p.used}</div>}
                    </div>
                  </div>
                  {p.equations.length > 0 ? (
                    <div className="mt-2 border-t border-line pt-2">
                      <div className="text-[11px] uppercase tracking-wide text-mute">Equations from the paper</div>
                      {p.equations.map((e, i) => (
                        <div key={i} className="mt-1">
                          {e.section && <div className="text-[10px] text-mute">§ {e.section}</div>}
                          <Tex src={e.latex} />
                          <details className="text-[10px] text-mute"><summary className="cursor-pointer">LaTeX source</summary><pre className="overflow-x-auto whitespace-pre-wrap">{e.latex}</pre></details>
                        </div>
                      ))}
                    </div>
                  ) : <div className="mt-2 text-[11px] text-mute">{p.equation_note}</div>}
                </div>
              ) })}
            </div>
          ))}
        </>
      )}
    </div>
  )
}
