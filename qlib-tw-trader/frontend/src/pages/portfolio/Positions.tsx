import { useEffect, useState, useCallback, useMemo } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import {
  Loader2,
  AlertCircle,
  Search,
  Briefcase,
  History,
  ArrowUpRight,
  ArrowDownRight,
  TrendingUp,
  ChevronLeft,
  ChevronRight,
  Zap,
  GitBranch,
} from 'lucide-react'
import {
  portfolioApi,
  PredictionHistoryItem,
  PredictionSignal,
  TodayPredictionStatus,
} from '@/api/client'

const PAGE_SIZE = 10

/** Compare two days' top-K picks to derive buy/sell actions */
function deriveTradeActions(
  current: PredictionSignal[],
  previous: PredictionSignal[],
  topK: number,
): { symbol: string; name: string | null; action: 'buy' | 'sell'; score: number }[] {
  const currentTop = new Set(current.slice(0, topK).map((s) => s.symbol))
  const previousTop = new Set(previous.slice(0, topK).map((s) => s.symbol))

  const actions: { symbol: string; name: string | null; action: 'buy' | 'sell'; score: number }[] = []

  // New entries = buy
  for (const sig of current.slice(0, topK)) {
    if (!previousTop.has(sig.symbol)) {
      actions.push({ symbol: sig.symbol, name: sig.name, action: 'buy', score: sig.score })
    }
  }

  // Exited = sell
  for (const sig of previous.slice(0, topK)) {
    if (!currentTop.has(sig.symbol)) {
      actions.push({ symbol: sig.symbol, name: sig.name, action: 'sell', score: sig.score })
    }
  }

  return actions
}

/** Calculate how many consecutive days (predictions) a stock has been in top-K */
function calcDaysHeld(
  symbol: string,
  history: PredictionHistoryItem[],
  topK: number,
): number {
  let count = 0
  for (const item of history) {
    const topSymbols = item.top_picks.slice(0, topK).map((s) => s.symbol)
    if (topSymbols.includes(symbol)) {
      count++
    } else {
      break
    }
  }
  return count
}

export function Positions() {
  const [todayStatus, setTodayStatus] = useState<TodayPredictionStatus | null>(null)
  const [history, setHistory] = useState<PredictionHistoryItem[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [historyPage, setHistoryPage] = useState(0)

  const topK = 10

  const fetchData = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const [statusRes, historyRes] = await Promise.all([
        portfolioApi.todayStatus(),
        portfolioApi.history(30),
      ])
      setTodayStatus(statusRes)
      setHistory(historyRes.items)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load data')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchData()
  }, [fetchData])

  // Current holdings from latest prediction
  const currentHoldings = useMemo(() => {
    const prediction = todayStatus?.prediction
    if (!prediction) return []
    return prediction.signals.slice(0, topK).map((sig) => ({
      ...sig,
      daysHeld: calcDaysHeld(sig.symbol, history, topK),
    }))
  }, [todayStatus, history, topK])

  // Filter holdings by search
  const filteredHoldings = useMemo(() => {
    if (!searchQuery.trim()) return currentHoldings
    const q = searchQuery.trim().toLowerCase()
    return currentHoldings.filter(
      (h) =>
        h.symbol.toLowerCase().includes(q) ||
        (h.name && h.name.toLowerCase().includes(q)),
    )
  }, [currentHoldings, searchQuery])

  // Trade history: derive buy/sell from consecutive prediction days
  const tradeHistory = useMemo(() => {
    const trades: {
      date: string
      symbol: string
      name: string | null
      action: 'buy' | 'sell'
      score: number
    }[] = []

    for (let i = 0; i < history.length - 1; i++) {
      const current = history[i]
      const previous = history[i + 1]
      const actions = deriveTradeActions(current.top_picks, previous.top_picks, topK)
      for (const a of actions) {
        trades.push({ date: current.trade_date, ...a })
      }
    }

    // First day in history: all top-K are "buy"
    if (history.length > 0) {
      const oldest = history[history.length - 1]
      for (const sig of oldest.top_picks.slice(0, topK)) {
        trades.push({
          date: oldest.trade_date,
          symbol: sig.symbol,
          name: sig.name,
          action: 'buy',
          score: sig.score,
        })
      }
    }

    return trades
  }, [history, topK])

  // Paginated trade history
  const pagedTrades = useMemo(() => {
    const start = historyPage * PAGE_SIZE
    return tradeHistory.slice(start, start + PAGE_SIZE)
  }, [tradeHistory, historyPage])

  const totalTradePages = Math.max(1, Math.ceil(tradeHistory.length / PAGE_SIZE))

  // Holdings timeline: for each stock in history, mark which days it was held
  const timelineDates = useMemo(() => {
    return history.map((h) => h.trade_date).reverse()
  }, [history])

  const timelineStocks = useMemo(() => {
    // Collect all stocks that ever appeared in top-K
    const stockMap = new Map<string, { name: string | null; held: Set<string> }>()
    for (const item of history) {
      for (const sig of item.top_picks.slice(0, topK)) {
        if (!stockMap.has(sig.symbol)) {
          stockMap.set(sig.symbol, { name: sig.name, held: new Set() })
        }
        stockMap.get(sig.symbol)!.held.add(item.trade_date)
      }
    }

    // Sort by most recent appearance, then frequency
    return Array.from(stockMap.entries())
      .sort((a, b) => {
        const aLatest = Math.max(...Array.from(a[1].held).map((d) => new Date(d).getTime()))
        const bLatest = Math.max(...Array.from(b[1].held).map((d) => new Date(d).getTime()))
        if (bLatest !== aLatest) return bLatest - aLatest
        return b[1].held.size - a[1].held.size
      })
      .slice(0, 30) // Limit to 30 stocks for readability
  }, [history, topK])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="heading text-2xl">Positions</h1>
        <p className="subheading mt-1">
          Current holdings and trade history based on model predictions
        </p>
      </div>

      {/* Error */}
      {error && (
        <div className="p-3 rounded-lg bg-red-50 border border-red-100">
          <div className="flex items-center gap-2">
            <AlertCircle className="h-4 w-4 text-red" />
            <p className="text-sm text-red">{error}</p>
          </div>
        </div>
      )}

      {/* No prediction available */}
      {!todayStatus?.has_prediction && (
        <Card>
          <CardContent className="pt-6">
            <div className="flex flex-col items-center py-8">
              <div className="icon-box icon-box-blue w-12 h-12 mb-4">
                <Briefcase className="h-6 w-6" />
              </div>
              <p className="font-semibold text-lg">No Predictions Yet</p>
              <p className="text-sm text-muted-foreground mt-1">
                Generate predictions on the Predictions page to see positions here.
              </p>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Current Holdings Panel */}
      {todayStatus?.has_prediction && todayStatus.prediction && (
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle className="flex items-center gap-2 text-base">
              <Briefcase className="h-4 w-4 text-blue" />
              Current Holdings (Top {topK})
            </CardTitle>
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <span>Trade Date: <span className="font-semibold text-foreground">{todayStatus.prediction.trade_date}</span></span>
              <span className="text-border">|</span>
              <span>Model: <span className="font-semibold text-foreground">{todayStatus.prediction.model_name}</span></span>
              {todayStatus.prediction.is_fallback && (
                <span className="badge badge-yellow text-xs flex items-center gap-1">
                  <GitBranch className="h-3 w-3" />
                  Fallback
                </span>
              )}
              {todayStatus.prediction.is_incremental && (
                <span className="badge badge-blue text-xs flex items-center gap-1">
                  <Zap className="h-3 w-3" />
                  +{todayStatus.prediction.incremental_days}d
                </span>
              )}
            </div>
          </CardHeader>
          <CardContent className="p-0">
            {/* Search within holdings */}
            <div className="px-5 py-3 border-b border-border">
              <div className="relative w-64">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <input
                  type="text"
                  className="input w-full text-sm pl-9"
                  placeholder="Search holdings..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
              </div>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-border bg-secondary/50">
                    <th className="table-header px-5 py-3 text-left w-16">Rank</th>
                    <th className="table-header px-5 py-3 text-left">Symbol</th>
                    <th className="table-header px-5 py-3 text-left">Name</th>
                    <th className="table-header px-5 py-3 text-right">Score</th>
                    <th className="table-header px-5 py-3 text-right">Days Held</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredHoldings.map((h) => (
                    <tr key={h.symbol} className="table-row bg-green-50/30">
                      <td className="table-cell px-5">
                        <span className="font-semibold text-green">#{h.rank}</span>
                      </td>
                      <td className="table-cell px-5">
                        <span className="font-semibold mono">{h.symbol}</span>
                      </td>
                      <td className="table-cell px-5">
                        <span className="text-muted-foreground">{h.name || '---'}</span>
                      </td>
                      <td className="table-cell px-5 text-right">
                        <span
                          className={`mono font-semibold ${
                            h.score > 0 ? 'text-green' : h.score < 0 ? 'text-red' : ''
                          }`}
                        >
                          {h.score.toFixed(6)}
                        </span>
                      </td>
                      <td className="table-cell px-5 text-right">
                        <span className="mono">{h.daysHeld}d</span>
                      </td>
                    </tr>
                  ))}
                  {filteredHoldings.length === 0 && (
                    <tr>
                      <td colSpan={5} className="text-center py-8 text-muted-foreground">
                        {searchQuery ? 'No matching holdings' : 'No holdings'}
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Trade History */}
      {tradeHistory.length > 0 && (
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle className="flex items-center gap-2 text-base">
              <History className="h-4 w-4 text-purple" />
              Trade History ({tradeHistory.length} actions)
            </CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-border bg-secondary/50">
                    <th className="table-header px-5 py-3 text-left">Date</th>
                    <th className="table-header px-5 py-3 text-left">Symbol</th>
                    <th className="table-header px-5 py-3 text-left">Name</th>
                    <th className="table-header px-5 py-3 text-center">Action</th>
                    <th className="table-header px-5 py-3 text-right">Score</th>
                  </tr>
                </thead>
                <tbody>
                  {pagedTrades.map((t, idx) => (
                    <tr key={`${t.date}-${t.symbol}-${t.action}-${idx}`} className="table-row">
                      <td className="table-cell px-5">
                        <span className="mono text-sm">{t.date}</span>
                      </td>
                      <td className="table-cell px-5">
                        <span className="font-semibold mono">{t.symbol}</span>
                      </td>
                      <td className="table-cell px-5">
                        <span className="text-muted-foreground">{t.name || '---'}</span>
                      </td>
                      <td className="table-cell px-5 text-center">
                        {t.action === 'buy' ? (
                          <span className="inline-flex items-center gap-1 text-xs font-semibold text-green bg-green-50 px-2 py-0.5 rounded-full">
                            <ArrowUpRight className="h-3 w-3" />
                            BUY
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1 text-xs font-semibold text-red bg-red-50 px-2 py-0.5 rounded-full">
                            <ArrowDownRight className="h-3 w-3" />
                            SELL
                          </span>
                        )}
                      </td>
                      <td className="table-cell px-5 text-right">
                        <span
                          className={`mono font-semibold ${
                            t.score > 0 ? 'text-green' : t.score < 0 ? 'text-red' : ''
                          }`}
                        >
                          {t.score.toFixed(6)}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {/* Pagination */}
            {totalTradePages > 1 && (
              <div className="flex items-center justify-between px-5 py-3 border-t border-border">
                <span className="text-xs text-muted-foreground">
                  Page {historyPage + 1} of {totalTradePages}
                </span>
                <div className="flex gap-1">
                  <button
                    className="btn btn-ghost text-xs px-2 py-1"
                    disabled={historyPage === 0}
                    onClick={() => setHistoryPage((p) => p - 1)}
                  >
                    <ChevronLeft className="h-4 w-4" />
                  </button>
                  <button
                    className="btn btn-ghost text-xs px-2 py-1"
                    disabled={historyPage >= totalTradePages - 1}
                    onClick={() => setHistoryPage((p) => p + 1)}
                  >
                    <ChevronRight className="h-4 w-4" />
                  </button>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Holdings Timeline */}
      {timelineStocks.length > 0 && timelineDates.length > 1 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <TrendingUp className="h-4 w-4 text-green" />
              Holdings Timeline
            </CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-border bg-secondary/50">
                    <th className="table-header px-3 py-2 text-left sticky left-0 bg-secondary/50 z-10 min-w-[120px]">
                      Stock
                    </th>
                    {timelineDates.map((d) => (
                      <th key={d} className="table-header px-1 py-2 text-center min-w-[40px]">
                        <span className="mono">{d.slice(5)}</span>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {timelineStocks.map(([symbol, info]) => (
                    <tr key={symbol} className="table-row">
                      <td className="table-cell px-3 sticky left-0 bg-white z-10">
                        <div>
                          <span className="font-semibold mono">{symbol}</span>
                          {info.name && (
                            <span className="text-muted-foreground ml-1">{info.name}</span>
                          )}
                        </div>
                      </td>
                      {timelineDates.map((d) => {
                        const isHeld = info.held.has(d)
                        return (
                          <td key={d} className="px-1 py-1.5 text-center">
                            {isHeld ? (
                              <div
                                className="w-full h-5 rounded-sm bg-green/20 border border-green/30"
                                title={`${symbol} held on ${d}`}
                              />
                            ) : (
                              <div className="w-full h-5" />
                            )}
                          </td>
                        )
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
