import { useCallback, useEffect, useRef, useState } from 'react'

// Pacing so the swarm is watchable. Per-event delay in ms (scaled by speed; 0 = instant).
const DELAY: Record<string, number> = {
  proposal: 34, agent: 380, null_audit: 700, prosecution: 110, verdict: 14, graveyard: 80, credibility: 450, ledger: 12, tokens: 4,
  prereg: 200, strategy_card: 250, naive: 500, meta: 300, truth_check: 600, literature: 300, illusion: 450, conclusion: 300, replication: 500, data_quality: 350, backtest_progress: 30,
}

export type Mode = 'demo' | 'live'

export function useRun(dispatch: (a: any) => void) {
  const [running, setRunning] = useState(false)
  const [speed, setSpeed] = useState(1)
  const [source, setSource] = useState<string>('')
  const [last, setLast] = useState<{ run?: string; demo?: string } | null>(null)
  const queue = useRef<any[]>([])
  const timer = useRef<any>(null)
  const speedRef = useRef(1)
  const wsRef = useRef<WebSocket | null>(null)
  const gen = useRef(0)
  speedRef.current = speed

  const pump = useCallback(() => {
    if (timer.current) return
    const step = () => {
      const ev = queue.current.shift()
      if (!ev) { timer.current = null; return }
      dispatch(ev)
      if (ev.type === 'done') setRunning(false)
      const base = DELAY[ev.type] ?? 10
      const d = speedRef.current >= 50 ? 0 : base / speedRef.current
      timer.current = setTimeout(() => { timer.current = null; step() }, d)
    }
    step()
  }, [dispatch])

  const stop = useCallback(() => {
    gen.current++
    queue.current = []
    if (timer.current) { clearTimeout(timer.current); timer.current = null }
    wsRef.current?.close(); wsRef.current = null
    setRunning(false)
  }, [])

  const reset = useCallback(() => { stop(); dispatch({ type: '__reset' }) }, [stop, dispatch])

  const demo = useCallback(async (scenario: string) => {
    reset(); setRunning(true); setSource(`replay: ${scenario}`); setLast({ demo: scenario })
    const my = ++gen.current
    try {
      const r = await fetch(`/api/demo/${scenario}`)
      if (!r.ok) throw new Error('recorded run missing; run: python -m skeptic.demo --record')
      const evs = await r.json()
      if (my !== gen.current) return
      queue.current.push(...evs); pump()
    } catch (e: any) { dispatch({ type: 'error', message: String(e.message || e) }); setRunning(false) }
  }, [reset, pump, dispatch])

  const live = useCallback(async (scenario: string, params: any = {}, seed = 0) => {
    reset(); setRunning(true); setSource(`live: ${scenario}`); setLast(null)
    const my = ++gen.current
    try {
      const r = await fetch('/api/run', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ scenario, params, seed }) })
      if (!r.ok) throw new Error(await r.text())
      const { run_id } = await r.json()
      setLast({ run: run_id })
      if (my !== gen.current) return
      const proto = location.protocol === 'https:' ? 'wss' : 'ws'
      const ws = new WebSocket(`${proto}://${location.host}/ws/${run_id}`)
      wsRef.current = ws
      let sawDone = false
      ws.onmessage = (m) => { if (my === gen.current) { const ev = JSON.parse(m.data); if (ev.type === 'done') sawDone = true; queue.current.push(ev); pump() } }
      ws.onclose = () => {
        // a socket that closes without a `done` event must not leave the UI spinning forever
        if (my === gen.current && !sawDone) { queue.current.push({ type: 'error', message: 'Connection to the backend closed before the run finished.' }, { type: 'done' }); pump() }
      }
      ws.onerror = () => dispatch({ type: 'error', message: 'WebSocket error: is the backend running? (uvicorn skeptic.backend.app:app)' })
    } catch (e: any) { dispatch({ type: 'error', message: String(e.message || e) }); setRunning(false) }
  }, [reset, pump, dispatch])

  useEffect(() => () => stop(), [stop])
  return { running, speed, setSpeed, source, last, demo, live, stop, reset }
}
