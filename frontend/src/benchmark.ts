import { useEffect, useState } from 'react'

/** The recorded synthetic benchmark (served by GET /api/benchmark). Fetched once and shared. */
let cache: Promise<any> | null = null
const load = () => (cache ??= fetch('/api/benchmark').then((r) => (r.ok ? r.json() : Promise.reject(new Error('benchmark not recorded')))))

export function useBenchmark() {
  const [data, setData] = useState<any>(null)
  const [error, setError] = useState('')
  useEffect(() => { load().then(setData).catch((e) => { cache = null; setError(String(e.message || e)) }) }, [])
  return { data, error }
}

/** Our row in the benchmark table is the full court (the generator names it "SKEPTIC (full court)"). */
export const isOurs = (name: string) => /^skeptic\b/i.test(name)
export const isNaive = (name: string) => /naive/i.test(name)
export const displayName = (name: string) => (isOurs(name) ? name.replace(/^skeptic\b/i, 'SkepticR') : name)

/** One-sided 95% upper bound on the false-discovery rate. With zero false out of n claims: rule of three (3/n). */
export function upperBoundFdr(r: { claimed: number; fp: number }, seeds: number) {
  const n = Math.round(r.claimed * seeds)
  if (!n) return null
  return r.fp === 0 ? Math.min(1, 3 / n) : null
}
