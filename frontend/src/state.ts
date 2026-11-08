// Event-stream reducer: every backend event folds into one RunState the views render.
export type Ev = { type: string; [k: string]: any }

export interface RunState {
  meta: any
  dataQuality: any
  naive: any
  proposals: any[]
  feed: { n: number; agent: string; text: string }[]
  active: Record<string, number>
  ledger: any
  credibility: { wealth: number; wealth0: number; history: number[] }
  tokens: any
  audits: any[]
  prosecutions: Record<string, any[]>
  verdicts: Record<string, any>
  graveyard: any[]
  prereg: Record<string, any>
  cards: Record<string, any>
  portfolio: any
  truth: any
  literature: any[]
  methods: any | null
  illusions: any[]
  effectCheck: any | null         // non-money runs with no illusion: the headline finding, tested vs fresh data
  conclusion: any | null
  dataset: any | null            // prompt-only flow: the public dataset the Data Scout found (and what it tried)
  replication: { extracted?: any; decay?: any }
  backtest: { done: number; total: number } | null
  phase: string
  done: boolean
  error: string | null
  n: number
}

export const initialState = (): RunState => ({
  meta: null, dataQuality: null, naive: null, proposals: [], feed: [], active: {}, ledger: { tests: 0, cleaning_choices: 0, agents: 0, potential_paths: 0 },
  credibility: { wealth: 0, wealth0: 0, history: [] }, tokens: null, audits: [], prosecutions: {}, verdicts: {}, graveyard: [], prereg: {}, cards: {},
  portfolio: null, truth: null, literature: [], methods: null, illusions: [], effectCheck: null, conclusion: null, dataset: null, replication: {}, backtest: null, phase: 'idle', done: false, error: null, n: 0,
})

const researcherNode = (fam: string) => `${fam}-researcher`

export function reduce(s: RunState, a: any): RunState {
  if (a.type === '__reset') return initialState()
  const e: Ev = a
  const now = Date.now()
  const n = s.n + 1
  const act = (names: string[]) => { const m = { ...s.active }; names.forEach((x) => (m[x] = now)); return m }
  switch (e.type) {
    case 'meta': return { ...s, n, meta: e, phase: 'starting' }
    case 'data_quality': return { ...s, n, dataQuality: e, active: act(['Cleaner']) }
    case 'agent':
      return { ...s, n, active: act([e.agent]), feed: [...s.feed.slice(-80), { n, agent: e.agent, text: e.text }], phase: e.agent }
    case 'proposal':
      return { ...s, n, proposals: [...s.proposals, e], active: act([researcherNode(e.family), 'Researchers']),
        feed: e.source_paper ? [...s.feed.slice(-80), { n, agent: e.agent, text: `proposed "${e.description}" (${e.source_paper})` }] : s.feed }
    case 'backtest_progress': return { ...s, n, backtest: { done: e.done, total: e.total }, active: act(['Backtester']) }
    case 'ledger': return { ...s, n, ledger: e }
    case 'naive': return { ...s, n, naive: e }
    case 'conclusion': return { ...s, n, conclusion: e, active: act(['Judge']) }
    case 'dataset': return { ...s, n, dataset: e, active: act(['Data Scout']) }
    case 'illusion': return { ...s, n, illusions: [...s.illusions, e], active: act(['Judge']) }
    case 'effect_check': return { ...s, n, effectCheck: e, active: act(['Judge']) }
    case 'methods': return { ...s, n, methods: e, active: act(['Librarian']) }
    case 'literature': return { ...s, n, literature: [...s.literature, e], active: act(['Librarian']) }
    case 'null_audit': return { ...s, n, audits: [...s.audits, e], active: act(['Court']), phase: 'Court' }
    case 'prosecution': {
      const names = e.attacks.map((x: any) => x.attack)
      return { ...s, n, prosecutions: { ...s.prosecutions, [e.hypothesis_id]: e.attacks }, active: act(['Skeptics', ...names]) }
    }
    case 'verdict': return { ...s, n, verdicts: { ...s.verdicts, [e.hypothesis_id]: e }, active: act(['Judge']) }
    case 'graveyard': return { ...s, n, graveyard: [...s.graveyard, e] }
    case 'credibility': return { ...s, n, credibility: { wealth: e.wealth, wealth0: e.wealth0, history: [...s.credibility.history, e.wealth] }, active: act(['Credibility']) }
    case 'tokens': return { ...s, n, tokens: e }
    case 'prereg': return { ...s, n, prereg: { ...s.prereg, [e.hypothesis_id]: e } }
    case 'strategy_card': return { ...s, n, cards: { ...s.cards, [e.hypothesis_id]: e } }
    case 'portfolio': return { ...s, n, portfolio: e }
    case 'truth_check': return { ...s, n, truth: e }
    case 'replication':
      return { ...s, n, replication: e.stage === 'extracted' ? { ...s.replication, extracted: e } : e.stage === 'decay' ? { ...s.replication, decay: e } : s.replication,
        active: act(['Extractor', 'Replicator']) }
    case 'error': return { ...s, n, error: e.message, done: true }
    case 'done': return { ...s, n, done: true, phase: 'done' }
    default: return { ...s, n }
  }
}

export const VERDICT_COLOR: Record<string, string> = { DISCOVERY: '#34c38f', LUCK: '#e5677a', "CAN'T TELL YET": '#e8b04b' }

export function counts(s: RunState) {
  const c = { DISCOVERY: 0, LUCK: 0, CANT: 0 }
  Object.values(s.verdicts).forEach((v: any) => { if (v.verdict === 'DISCOVERY') c.DISCOVERY++; else if (v.verdict === 'LUCK') c.LUCK++; else c.CANT++ })
  return c
}
