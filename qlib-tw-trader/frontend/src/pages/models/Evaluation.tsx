import { useEffect, useState, useCallback, useMemo } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import {
  Loader2,
  RefreshCw,
  AlertCircle,
  TrendingUp,
  Activity,
  BarChart3,
  Download,
  Brain,
} from 'lucide-react'
import {
  walkForwardApi,
  factorApi,
  WalkForwardSummary,
  WalkForwardItem,
  Factor,
} from '@/api/client'
import { RollingIcChart } from '@/components/charts/RollingIcChart'
import { CumulativeReturnChart } from '@/components/charts/CumulativeReturnChart'
import { FactorImportanceChart } from '@/components/charts/FactorImportanceChart'

export function Evaluation() {
  const [summary, setSummary] = useState<WalkForwardSummary | null>(null)
  const [backtests, setBacktests] = useState<WalkForwardItem[]>([])
  const [factors, setFactors] = useState<Factor[]>([])
  const [selectedId, setSelectedId] = useState<number | undefined>(undefined)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchSummary = useCallback(async (backtestId?: number) => {
    try {
      const data = await walkForwardApi.summary(backtestId)
      setSummary(data)
      setError(null)
    } catch (err) {
      setSummary(null)
      setError(err instanceof Error ? err.message : 'Failed to load summary')
    }
  }, [])

  const fetchAll = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const [btRes, factorRes] = await Promise.all([
        walkForwardApi.list(50),
        factorApi.list(undefined, true),
      ])
      setBacktests(btRes.items.filter((bt) => bt.status === 'completed'))
      setFactors(factorRes.items)
      await fetchSummary(selectedId)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load data')
    } finally {
      setLoading(false)
    }
  }, [fetchSummary, selectedId])

  useEffect(() => {
    fetchAll()
  }, [fetchAll])

  const handleSelectBacktest = async (id: number) => {
    setSelectedId(id)
    await fetchSummary(id)
  }

  const topFactors = useMemo(() => {
    return [...factors]
      .filter((f) => f.times_evaluated > 0)
      .sort((a, b) => b.selection_rate - a.selection_rate)
      .slice(0, 20)
  }, [factors])

  const handleExportCsv = () => {
    if (!summary) return
    const rows = [
      ['Metric', 'Value'],
      ['Backtest ID', String(summary.backtest_id)],
      ['Period', `${summary.start_week_id} ~ ${summary.end_week_id}`],
      ['Total Weeks', String(summary.total_weeks)],
      ['Mean IC', String(summary.mean_ic)],
      ['ICIR', String(summary.icir)],
      ['IC>0%', `${summary.ic_positive_rate}%`],
      ['Cumulative Return', `${summary.cumulative_return}%`],
      ['Market Return', `${summary.market_return}%`],
      ['Excess Return', `${summary.excess_return}%`],
      ['Annualized Return', summary.annualized_return !== null ? `${summary.annualized_return}%` : 'N/A'],
      ['Annualized Excess', summary.annualized_excess !== null ? `${summary.annualized_excess}%` : 'N/A'],
      ['Sharpe Ratio', summary.sharpe_ratio !== null ? String(summary.sharpe_ratio) : 'N/A'],
      ['Max Drawdown', summary.max_drawdown !== null ? `${summary.max_drawdown}%` : 'N/A'],
      ['Win Rate', summary.win_rate !== null ? `${summary.win_rate}%` : 'N/A'],
      ['Total Trades', summary.total_trades !== null ? String(summary.total_trades) : 'N/A'],
    ]
    const csv = rows.map((r) => r.join(',')).join('\n')
    downloadFile(csv, 'evaluation-metrics.csv', 'text/csv')
  }

  const handleExportJson = () => {
    if (!summary) return
    const data = {
      weekly_points: summary.weekly_points,
      equity_curve: summary.equity_curve,
    }
    downloadFile(JSON.stringify(data, null, 2), 'evaluation-charts.json', 'application/json')
  }

  const formatPercent = (value: number | null | undefined) => {
    if (value === null || value === undefined) return '---'
    const prefix = value >= 0 ? '+' : ''
    return `${prefix}${value.toFixed(2)}%`
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h1 className="text-lg font-semibold">Model Evaluation</h1>
          {summary && (
            <span className="text-sm text-muted-foreground">
              #{summary.backtest_id} ({summary.start_week_id} ~ {summary.end_week_id})
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {/* Backtest selector */}
          <select
            className="input text-sm w-48"
            value={selectedId ?? ''}
            onChange={(e) => {
              const val = e.target.value
              if (val) handleSelectBacktest(Number(val))
            }}
          >
            <option value="">Latest completed</option>
            {backtests.map((bt) => (
              <option key={bt.id} value={bt.id}>
                #{bt.id} {bt.start_week_id}~{bt.end_week_id}
              </option>
            ))}
          </select>
          <button onClick={fetchAll} className="btn btn-ghost btn-sm" title="Refresh">
            <RefreshCw className="h-4 w-4" />
          </button>
          <button
            onClick={handleExportCsv}
            disabled={!summary}
            className="btn btn-ghost btn-sm disabled:opacity-50"
            title="Export CSV"
          >
            <Download className="h-4 w-4" />
            CSV
          </button>
          <button
            onClick={handleExportJson}
            disabled={!summary}
            className="btn btn-ghost btn-sm disabled:opacity-50"
            title="Export JSON"
          >
            <Download className="h-4 w-4" />
            JSON
          </button>
        </div>
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

      {!summary && !error && (
        <div className="flex flex-col items-center justify-center h-64 text-muted-foreground">
          <Activity className="h-12 w-12 mb-3 opacity-30" />
          <p>No completed backtests found. Run a walk-forward backtest first.</p>
        </div>
      )}

      {summary && (
        <>
          {/* Row 1: IC Summary + Strategy Metrics */}
          <div className="grid grid-cols-2 gap-4">
            {/* IC Summary */}
            <Card>
              <CardHeader className="py-3">
                <CardTitle className="flex items-center gap-2 text-base">
                  <Activity className="h-4 w-4 text-blue" />
                  IC Summary
                </CardTitle>
              </CardHeader>
              <CardContent className="pt-4">
                <div className="grid grid-cols-3 gap-3">
                  <MetricBox
                    label="Mean IC"
                    value={summary.mean_ic.toFixed(4)}
                    highlight
                  />
                  <MetricBox
                    label="ICIR"
                    value={summary.icir.toFixed(4)}
                  />
                  <MetricBox
                    label="IC > 0"
                    value={`${summary.ic_positive_rate.toFixed(1)}%`}
                    color={summary.ic_positive_rate >= 50}
                  />
                </div>
              </CardContent>
            </Card>

            {/* Strategy Metrics */}
            <Card>
              <CardHeader className="py-3">
                <CardTitle className="flex items-center gap-2 text-base">
                  <TrendingUp className="h-4 w-4 text-blue" />
                  Strategy Performance
                </CardTitle>
              </CardHeader>
              <CardContent className="pt-4">
                <div className="grid grid-cols-4 gap-3">
                  <MetricBox
                    label="Ann. Return"
                    value={formatPercent(summary.annualized_return)}
                    color={summary.annualized_return !== null ? summary.annualized_return >= 0 : undefined}
                  />
                  <MetricBox
                    label="Ann. Excess"
                    value={formatPercent(summary.annualized_excess)}
                    color={summary.annualized_excess !== null ? summary.annualized_excess >= 0 : undefined}
                    highlight
                  />
                  <MetricBox
                    label="Sharpe"
                    value={summary.sharpe_ratio?.toFixed(2) ?? '---'}
                  />
                  <MetricBox
                    label="Max DD"
                    value={summary.max_drawdown !== null ? `-${summary.max_drawdown.toFixed(1)}%` : '---'}
                    color={false}
                  />
                </div>
                <div className="grid grid-cols-4 gap-3 mt-3">
                  <MetricBox
                    label="Cumulative"
                    value={formatPercent(summary.cumulative_return)}
                    color={summary.cumulative_return >= 0}
                  />
                  <MetricBox
                    label="Market"
                    value={formatPercent(summary.market_return)}
                  />
                  <MetricBox
                    label="Win Rate"
                    value={summary.win_rate !== null ? `${summary.win_rate.toFixed(0)}%` : '---'}
                  />
                  <MetricBox
                    label="Weeks"
                    value={String(summary.total_weeks)}
                  />
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Row 2: Rolling IC Chart */}
          <Card>
            <CardHeader className="py-3">
              <CardTitle className="flex items-center gap-2 text-base">
                <Activity className="h-4 w-4 text-blue" />
                Rolling IC
                <span className="text-xs text-muted-foreground font-normal">
                  ({summary.weekly_points.length} weeks)
                </span>
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-4">
              <RollingIcChart data={summary.weekly_points} height={200} />
            </CardContent>
          </Card>

          {/* Row 3: Cumulative Return Chart */}
          <Card>
            <CardHeader className="py-3">
              <CardTitle className="flex items-center gap-2 text-base">
                <TrendingUp className="h-4 w-4 text-blue" />
                Cumulative Return (Strategy vs Market)
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-4">
              <CumulativeReturnChart data={summary.weekly_points} height={250} />
            </CardContent>
          </Card>

          {/* Row 4: Factor Importance */}
          <Card>
            <CardHeader className="py-3">
              <CardTitle className="flex items-center gap-2 text-base">
                <Brain className="h-4 w-4 text-blue" />
                Top-20 Factors by Selection Rate
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-4">
              {topFactors.length > 0 ? (
                <FactorImportanceChart factors={topFactors} height={400} />
              ) : (
                <div className="flex items-center justify-center h-24 text-muted-foreground text-sm">
                  <BarChart3 className="h-8 w-8 mr-3 opacity-30" />
                  No factor data available
                </div>
              )}
            </CardContent>
          </Card>
        </>
      )}
    </div>
  )
}

function MetricBox({
  label,
  value,
  color,
  highlight,
}: {
  label: string
  value: string
  color?: boolean
  highlight?: boolean
}) {
  return (
    <div className={`p-2 rounded text-center ${highlight ? 'bg-blue/10 border border-blue/20' : 'bg-secondary/50'}`}>
      <p className="text-[10px] text-muted-foreground">{label}</p>
      <p className={`font-semibold text-sm font-mono ${
        color === true ? 'text-green' : color === false ? 'text-red' : ''
      }`}>
        {value}
      </p>
    </div>
  )
}

function downloadFile(content: string, filename: string, mimeType: string) {
  const blob = new Blob([content], { type: mimeType })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}
