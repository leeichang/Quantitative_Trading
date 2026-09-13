import { useEffect, useState, useCallback, useMemo } from 'react'
import { Link } from 'react-router-dom'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import {
  Target,
  Loader2,
  Play,
  TrendingUp,
  AlertCircle,
  CheckCircle2,
  Search,
  ArrowRight,
  Zap,
  GitBranch,
} from 'lucide-react'
import { portfolioApi, TodayPredictionStatus, PredictionSignal } from '@/api/client'
import { useJobs } from '@/hooks/useJobs'

export function Predictions() {
  const [status, setStatus] = useState<TodayPredictionStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [generating, setGenerating] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const { activeJob, clearJob } = useJobs()

  const fetchStatus = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await portfolioApi.todayStatus()
      setStatus(res)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load prediction status')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchStatus()
  }, [fetchStatus])

  // 監聽 predict 任務完成
  useEffect(() => {
    if (!activeJob) return
    if (activeJob.job_type !== 'predict') return

    if (activeJob.status === 'completed') {
      clearJob(activeJob.id)
      setGenerating(false)
      fetchStatus()
    } else if (activeJob.status === 'failed') {
      setError(activeJob.error || 'Prediction failed')
      clearJob(activeJob.id)
      setGenerating(false)
    }
  }, [activeJob, clearJob, fetchStatus])

  const handleGenerate = async () => {
    setGenerating(true)
    setError(null)
    try {
      await portfolioApi.generateToday()
      // 進度由 useJobs 追蹤
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start prediction')
      setGenerating(false)
    }
  }

  // 過濾 signals
  const filteredSignals = useMemo(() => {
    const signals = status?.prediction?.signals
    if (!signals) return []
    if (!searchQuery.trim()) return signals
    const q = searchQuery.trim().toLowerCase()
    return signals.filter(
      (s: PredictionSignal) =>
        s.symbol.toLowerCase().includes(q) ||
        (s.name && s.name.toLowerCase().includes(q)),
    )
  }, [status?.prediction?.signals, searchQuery])

  // 判斷是否在預測任務進行中
  const isPredicting = generating || (activeJob?.job_type === 'predict' && activeJob.status === 'running')
  const predictProgress = isPredicting && activeJob?.job_type === 'predict' ? activeJob.progress : 0
  const predictMessage = isPredicting && activeJob?.job_type === 'predict' ? activeJob.message : null

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
        <h1 className="heading text-2xl">Today's Predictions</h1>
        {status && (
          <p className="subheading mt-1">
            {status.today} ({status.week_id})
          </p>
        )}
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

      {/* Status Card */}
      {status && !status.has_prediction && (
        <Card>
          <CardContent className="pt-6">
            {/* 無模型 */}
            {!status.model_available && (
              <div className="flex flex-col items-center py-8">
                <div className="icon-box icon-box-red w-12 h-12 mb-4">
                  <AlertCircle className="h-6 w-6" />
                </div>
                <p className="font-semibold text-lg">No Model Available</p>
                <p className="text-sm text-muted-foreground mt-1">
                  {status.message}
                </p>
                <Link
                  to="/models/training"
                  className="btn btn-primary mt-4 text-sm"
                >
                  Go to Training
                  <ArrowRight className="h-4 w-4 ml-1" />
                </Link>
              </div>
            )}

            {/* 有模型，可產生 */}
            {status.model_available && !isPredicting && (
              <div className="flex flex-col items-center py-8">
                <div className="icon-box icon-box-blue w-12 h-12 mb-4">
                  <Target className="h-6 w-6" />
                </div>
                <p className="font-semibold text-lg">Ready to Predict</p>
                <div className="flex items-center gap-2 mt-2 text-sm text-muted-foreground">
                  <span>Model: <span className="font-semibold text-foreground">{status.model_name}</span></span>
                  <span>({status.model_week})</span>
                  {status.is_fallback && (
                    <span className="badge badge-yellow text-xs">Fallback</span>
                  )}
                </div>
                <button
                  onClick={handleGenerate}
                  className="btn btn-primary mt-4"
                >
                  <Play className="h-4 w-4" />
                  Generate Predictions
                </button>
              </div>
            )}

            {/* 產生中 */}
            {isPredicting && (
              <div className="flex flex-col items-center py-8">
                <Loader2 className="h-8 w-8 animate-spin text-primary mb-4" />
                <p className="font-semibold text-lg">Generating Predictions...</p>
                <p className="text-sm text-muted-foreground mt-1">
                  {predictMessage || 'Starting...'}
                </p>
                <div className="w-full max-w-md mt-4">
                  <div className="h-2 rounded-full bg-secondary overflow-hidden">
                    <div
                      className="h-full rounded-full bg-primary transition-all duration-300"
                      style={{ width: `${predictProgress}%` }}
                    />
                  </div>
                  <p className="text-xs text-muted-foreground text-center mt-1">
                    {Math.round(predictProgress)}%
                  </p>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* 已有預測：metadata + search + table */}
      {status?.has_prediction && status.prediction && (
        <>
          {/* Metadata */}
          <Card>
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <CheckCircle2 className="h-5 w-5 text-green" />
                  <div>
                    <p className="font-semibold">Predictions Generated</p>
                    <p className="text-xs text-muted-foreground">
                      {status.prediction.created_at &&
                        new Date(status.prediction.created_at).toLocaleString('zh-TW', {
                          timeZone: 'Asia/Taipei',
                        })}
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-3 text-sm">
                  <div className="text-right">
                    <p className="text-xs text-muted-foreground">Model</p>
                    <p className="font-semibold">{status.prediction.model_name}</p>
                  </div>
                  <div className="text-right">
                    <p className="text-xs text-muted-foreground">Feature Date</p>
                    <p className="font-semibold">{status.prediction.feature_date}</p>
                  </div>
                  <div className="flex gap-1">
                    {status.prediction.is_fallback && (
                      <span className="badge badge-yellow text-xs flex items-center gap-1">
                        <GitBranch className="h-3 w-3" />
                        Fallback
                      </span>
                    )}
                    {status.prediction.is_incremental && (
                      <span className="badge badge-blue text-xs flex items-center gap-1">
                        <Zap className="h-3 w-3" />
                        +{status.prediction.incremental_days}d
                      </span>
                    )}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Search + Table */}
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle className="flex items-center gap-2 text-base">
                <TrendingUp className="h-4 w-4 text-green" />
                Signals ({filteredSignals.length}
                {searchQuery && ` / ${status.prediction.signals.length}`})
              </CardTitle>
              <div className="relative w-64">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <input
                  type="text"
                  className="input w-full text-sm pl-9"
                  placeholder="Search symbol or name..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
              </div>
            </CardHeader>
            <CardContent className="p-0">
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-border bg-secondary/50">
                      <th className="table-header px-5 py-3 text-left w-16">Rank</th>
                      <th className="table-header px-5 py-3 text-left">Symbol</th>
                      <th className="table-header px-5 py-3 text-left">Name</th>
                      <th className="table-header px-5 py-3 text-right">Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredSignals.map((sig: PredictionSignal) => (
                      <tr
                        key={sig.symbol}
                        className={`table-row ${sig.rank <= 10 ? 'bg-green-50/50' : ''}`}
                      >
                        <td className="table-cell px-5">
                          <span className={`font-semibold ${sig.rank <= 10 ? 'text-green' : ''}`}>
                            #{sig.rank}
                          </span>
                        </td>
                        <td className="table-cell px-5">
                          <span className="font-semibold mono">{sig.symbol}</span>
                        </td>
                        <td className="table-cell px-5">
                          <span className="text-muted-foreground">{sig.name || '---'}</span>
                        </td>
                        <td className="table-cell px-5 text-right">
                          <span
                            className={`mono font-semibold ${
                              sig.score > 0 ? 'text-green' : sig.score < 0 ? 'text-red' : ''
                            }`}
                          >
                            {sig.score.toFixed(6)}
                          </span>
                        </td>
                      </tr>
                    ))}
                    {filteredSignals.length === 0 && (
                      <tr>
                        <td colSpan={4} className="text-center py-8 text-muted-foreground">
                          {searchQuery ? 'No matching stocks' : 'No signals'}
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  )
}
