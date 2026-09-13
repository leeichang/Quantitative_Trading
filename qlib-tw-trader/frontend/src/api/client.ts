const API_BASE = '/api/v1'

async function request<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    headers: {
      'Content-Type': 'application/json',
    },
    ...options,
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({}))
    // FastAPI 使用 detail，其他可能用 error.message
    const message = error.detail || error.error?.message || error.message || `HTTP ${response.status}`
    throw new Error(message)
  }

  return response.json()
}

export const api = {
  get: <T>(endpoint: string) => request<T>(endpoint),
  post: <T>(endpoint: string, data: unknown) =>
    request<T>(endpoint, {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  put: <T>(endpoint: string, data: unknown) =>
    request<T>(endpoint, {
      method: 'PUT',
      body: JSON.stringify(data),
    }),
  patch: <T>(endpoint: string, data?: unknown) =>
    request<T>(endpoint, {
      method: 'PATCH',
      body: data ? JSON.stringify(data) : undefined,
    }),
  delete: (endpoint: string) =>
    fetch(`${API_BASE}${endpoint}`, { method: 'DELETE' }),
}

// Types
export interface DatasetStatus {
  name: string
  earliest_date: string | null
  latest_date: string | null
  is_fresh: boolean
}

export interface StockItem {
  stock_id: string
  is_fresh: boolean
}

export interface DataStatusResponse {
  datasets: DatasetStatus[]
  stocks: StockItem[]
  checked_at: string
}

export interface HealthResponse {
  status: string
  timestamp: string
  version: string
}

export interface SyncRequest {
  stock_id: string
  start_date: string
  end_date: string
  datasets?: string[]
}

export interface SyncResult {
  dataset: string
  records_fetched: number
  success: boolean
  error: string | null
}

export interface SyncResponse {
  stock_id: string
  results: SyncResult[]
  total_records: number
  synced_at: string
}

// Factor Types
export interface Factor {
  id: string
  name: string
  display_name: string | null
  category: string
  description: string | null
  formula: string
  selection_rate: number
  times_selected: number
  times_evaluated: number
  enabled: boolean
  created_at: string
}

export interface FactorDetail extends Factor {
  selection_history: {
    model_id: string
    trained_at: string
    selected: boolean
  }[]
}

export interface FactorListResponse {
  items: Factor[]
  total: number
}

export interface FactorCreate {
  name: string
  display_name?: string
  category?: string
  description?: string
  formula: string
}

export interface FactorUpdate {
  name?: string
  display_name?: string
  category?: string
  description?: string
  formula?: string
}

// API functions
export const systemApi = {
  health: () => api.get<HealthResponse>('/system/health'),
  dataStatus: () => api.get<DataStatusResponse>('/system/data-status'),
  sync: (data: SyncRequest) => api.post<SyncResponse>('/system/sync', data),
}

export interface ValidateResponse {
  valid: boolean
  error?: string
  fields_used: string[]
  operators_used: string[]
  warnings: string[]
}

export interface SeedResponse {
  success: boolean
  inserted: number
  message: string
}

export interface AvailableFieldsResponse {
  fields: string[]
  operators: string[]
}

export interface DeduplicateResponse {
  success: boolean
  total_factors: number
  kept_factors: number
  disabled_factors: number
  disabled_names: string[]
  message: string
}

export const factorApi = {
  list: (category?: string, enabled?: boolean) => {
    const params = new URLSearchParams()
    if (category) params.set('category', category)
    if (enabled !== undefined) params.set('enabled', String(enabled))
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.get<FactorListResponse>(`/factors${query}`)
  },
  get: (id: number) => api.get<FactorDetail>(`/factors/${id}`),
  create: (data: FactorCreate) => api.post<Factor>('/factors', data),
  update: (id: number, data: FactorUpdate) => api.put<Factor>(`/factors/${id}`, data),
  delete: (id: number) => api.delete(`/factors/${id}`),
  toggle: (id: number) => api.patch<Factor>(`/factors/${id}/toggle`),
  validate: (expression: string) => api.post<ValidateResponse>('/factors/validate', { expression }),
  seed: (force?: boolean) => {
    const query = force ? '?force=true' : ''
    return api.post<SeedResponse>(`/factors/seed${query}`, {})
  },
  available: () => api.get<AvailableFieldsResponse>('/factors/available'),
  dedup: (threshold?: number) => {
    const query = threshold ? `?threshold=${threshold}` : ''
    return api.post<DeduplicateResponse>(`/factors/dedup${query}`, {})
  },
}

// Model Types
export interface Period {
  start: string
  end: string
}

export interface ModelMetrics {
  ic: number | null
  icir: number | null
}

export interface FactorSummary {
  id: string
  name: string
  display_name: string | null
  category: string
  ic_value: number | null
}

export interface SelectionInfo {
  method: string | null
  config: Record<string, unknown> | null
  stats: Record<string, unknown> | null
}

export interface Model {
  id: string
  name: string | null
  status: string
  trained_at: string
  factor_count: number | null
  factors: string[]
  train_period: Period | null
  valid_period: Period | null
  metrics: ModelMetrics
  training_duration_seconds: number | null
  candidate_factors: FactorSummary[]
  selected_factors: FactorSummary[]
  selection: SelectionInfo | null
}

export interface ModelSummary {
  id: string
  name: string | null
  status: string
  trained_at: string
  train_period: Period | null
  valid_period: Period | null
  metrics: ModelMetrics
  factor_count: number | null
  candidate_count: number | null
  selection_method: string | null
}

// 向後兼容
export type ModelHistoryItem = ModelSummary

export interface ModelHistoryResponse {
  items: ModelHistoryItem[]
  total: number
}

export interface TrainRequest {
  week_id: string  // "2026W05"
}

export interface TrainBatchRequest {
  year: string  // "2025"
}

// Week Types
export interface WeekModel {
  id: string
  name: string
  model_ic: number
  factor_count: number
  factor_pool_hash: string | null
  is_outdated: boolean
}

export interface WeekSlot {
  week_id: string
  valid_end: string
  valid_start: string
  train_end: string
  train_start: string
  status: 'trained' | 'trainable' | 'insufficient_data'
  model: WeekModel | null
}

export interface DataRange {
  start: string
  end: string
}

export interface WeeksResponse {
  slots: WeekSlot[]
  current_factor_pool_hash: string
  data_range: DataRange
}

export interface TrainResponse {
  job_id: string
  status: string
  message: string
}

export interface DeleteResponse {
  status: string
  id: string
}

// Quality Types
export interface QualityMetricsItem {
  training_run_id: number
  week_id: string | null
  factor_jaccard_sim: number | null
  factor_overlap_count: number | null
  ic_moving_avg_5w: number | null
  ic_moving_std_5w: number | null
  icir_5w: number | null
  has_warning: boolean
  warning_type: string | null
  warning_message: string | null
  created_at: string | null
}

export interface QualityResponse {
  items: QualityMetricsItem[]
  thresholds: {
    jaccard_min: number
    ic_std_max: number
    icir_min: number
  }
}

export const modelApi = {
  get: (id: string) => api.get<Model>(`/models/${id}`),
  delete: (id: string) => api.delete(`/models/${id}`).then(res => res.json() as Promise<DeleteResponse>),
  deleteAll: () => api.delete('/models/all').then(res => res.json() as Promise<{ deleted_count: number }>),

  // 週訓練
  weeks: () => api.get<WeeksResponse>('/models/weeks'),
  train: (data: TrainRequest) => api.post<TrainResponse>('/models/train', data),
  trainBatch: (data: TrainBatchRequest) => api.post<TrainResponse>('/models/train-batch', data),

  history: (limit?: number) => {
    const query = limit ? `?limit=${limit}` : ''
    return api.get<ModelHistoryResponse>(`/models/history${query}`)
  },

  // 品質監控
  quality: (limit?: number) => {
    const query = limit ? `?limit=${limit}` : ''
    return api.get<QualityResponse>(`/models/quality${query}`)
  },
}

// Portfolio Types
export interface PredictionRequest {
  model_id: number
  top_k: number
  trade_date?: string
}

export interface PredictionSignal {
  rank: number
  symbol: string
  name: string | null
  score: number
}

export interface PredictionsResponse {
  trade_date: string
  feature_date: string
  model_name: string
  signals: PredictionSignal[]
}

export interface TodayPredictionDetail {
  trade_date: string
  feature_date: string
  model_name: string
  model_week: string
  is_fallback: boolean
  is_incremental: boolean
  incremental_days: number | null
  signals: PredictionSignal[]
  created_at: string | null
}

export interface TodayPredictionStatus {
  today: string
  week_id: string
  has_prediction: boolean
  prediction: TodayPredictionDetail | null
  model_available: boolean
  model_name: string | null
  model_week: string | null
  is_fallback: boolean
  message: string | null
}

export interface PredictionHistoryItem {
  trade_date: string
  feature_date: string
  model_name: string
  model_week: string
  is_fallback: boolean
  is_incremental: boolean
  incremental_days: number | null
  signal_count: number
  top_picks: PredictionSignal[]
  created_at: string | null
}

export interface PredictionHistoryResponse {
  items: PredictionHistoryItem[]
  total: number
}

export const portfolioApi = {
  generatePredictions: (data: PredictionRequest) =>
    api.post<PredictionsResponse>('/predictions/generate', data),
  todayStatus: () =>
    api.get<TodayPredictionStatus>('/predictions/today'),
  generateToday: () =>
    api.post<{ job_id: string; status: string }>('/predictions/today/generate', {}),
  history: (limit?: number) => {
    const query = limit ? `?limit=${limit}` : ''
    return api.get<PredictionHistoryResponse>(`/predictions/history${query}`)
  },
}

// Dashboard Types
export interface DashboardSummary {
  factors: {
    total: number
    enabled: number
    low_selection_count: number
  }
  model: {
    last_trained_at: string | null
    days_since_training: number | null
    needs_retrain: boolean
    factor_count: number | null
    ic: number | null
    icir: number | null
  }
  prediction: {
    date: string | null
    buy_signals: number
    sell_signals: number
    top_pick: { symbol: string; score: number } | null
  }
  data_status: {
    is_complete: boolean
    last_updated: string | null
    missing_count: number
  }
  performance: {
    today_return: number | null
    mtd_return: number | null
    ytd_return: number | null
    total_return: number | null
  }
}

export const dashboardApi = {
  summary: () => api.get<DashboardSummary>('/dashboard/summary'),
}

// Job Types
export interface JobItem {
  id: string
  job_type: string
  status: string
  progress: number
  message: string | null
  started_at: string | null
  completed_at: string | null
}

export interface JobDetail extends JobItem {
  result: string | null
}

export interface JobListResponse {
  items: JobItem[]
  total: number
}

export const jobApi = {
  list: (limit?: number) => {
    const query = limit ? `?limit=${limit}` : ''
    return api.get<JobListResponse>(`/jobs${query}`)
  },
  get: (jobId: string) => api.get<JobDetail>(`/jobs/${jobId}`),
  cancel: (jobId: string) => api.delete(`/jobs/${jobId}`).then(res => res.json() as Promise<{ status: string; id: string }>),
}

// Dataset Types
export interface DatasetInfo {
  name: string
  display_name: string
  category: string
  source: string
  status: 'available' | 'needs_accumulation' | 'not_implemented' | 'pending'
  description: string | null
  requires_stock_id: boolean
}

export interface DatasetListResponse {
  datasets: DatasetInfo[]
  total: number
}

export interface TestResult {
  dataset: string
  success: boolean
  record_count: number
  sample_data: Record<string, unknown>[] | null
  error: string | null
}

export interface CategoryInfo {
  id: string
  name: string
  total: number
  available: number
}

// Universe Types
export interface StockInfo {
  stock_id: string
  name: string
  market_cap: number
  rank: number
}

export interface UniverseResponse {
  name: string
  description: string
  total: number
  stocks: StockInfo[]
  updated_at: string | null
}

export interface UniverseStats {
  total: number
  min_market_cap: number
  max_market_cap: number
  updated_at: string | null
}

export const universeApi = {
  get: () => api.get<UniverseResponse>('/universe'),
  stats: () => api.get<UniverseStats>('/universe/stats'),
  ids: () => api.get<{ stock_ids: string[]; total: number }>('/universe/ids'),
  sync: () => api.post<{ success: boolean; total: number; updated_at: string }>('/universe/sync', {}),
}

export const datasetsApi = {
  list: (category?: string, status?: string) => {
    const params = new URLSearchParams()
    if (category) params.set('category', category)
    if (status) params.set('status', status)
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.get<DatasetListResponse>(`/datasets${query}`)
  },
  test: (datasetName: string, stockId?: string, days?: number) => {
    const params = new URLSearchParams()
    if (stockId) params.set('stock_id', stockId)
    if (days) params.set('days', String(days))
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.get<TestResult>(`/datasets/test/${datasetName}${query}`)
  },
  categories: () => api.get<{ categories: CategoryInfo[] }>('/datasets/categories'),
}

// Sync Types
export interface SyncStatusItem {
  stock_id: string
  name: string
  rank: number
  earliest_date: string | null
  latest_date: string | null
  total_records: number
  missing_count: number
  coverage_pct: number
}

export interface SyncStatusResponse {
  trading_days: number
  start_date: string
  end_date: string
  stocks: SyncStatusItem[]
}

export interface SyncCalendarResponse {
  start_date: string
  end_date: string
  new_dates: number
  total_dates: number
}

export interface SyncStockResponse {
  stock_id: string
  fetched: number
  inserted: number
  missing_dates: string[]
}

export interface SyncBulkResponse {
  date: string
  total: number
  inserted: number
  days_synced?: number
  error: string | null
}

export interface SyncAllResponse {
  stocks: number
  total_inserted: number
  errors: { stock_id: string; error: string }[]
}

// 月營收專用
export interface MonthlyStatusItem {
  stock_id: string
  name: string
  rank: number
  earliest_month: string | null
  latest_month: string | null
  total_records: number
  missing_count: number
  coverage_pct: number
}

export interface MonthlyStatusResponse {
  expected_months: number
  start_year: number
  end_year: number
  stocks: MonthlyStatusItem[]
}

export interface MonthlyStockResponse {
  stock_id: string
  fetched: number
  inserted: number
  missing_months: string[]
}

export interface AllSyncStatusResponse {
  stock_daily: SyncStatusResponse
  per: SyncStatusResponse
  institutional: SyncStatusResponse
  margin: SyncStatusResponse
  adj: SyncStatusResponse
  shareholding: SyncStatusResponse
  securities_lending: SyncStatusResponse
  monthly_revenue: MonthlyStatusResponse
}

export const syncApi = {
  allStatus: () => api.get<AllSyncStatusResponse>('/sync/all-status'),
  status: (startDate?: string, endDate?: string) => {
    const params = new URLSearchParams()
    if (startDate) params.set('start_date', startDate)
    if (endDate) params.set('end_date', endDate)
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.get<SyncStatusResponse>(`/sync/status${query}`)
  },
  calendar: (startDate?: string, endDate?: string) => {
    const params = new URLSearchParams()
    if (startDate) params.set('start_date', startDate)
    if (endDate) params.set('end_date', endDate)
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.post<SyncCalendarResponse>(`/sync/calendar${query}`, {})
  },
  stock: (stockId: string, startDate?: string, endDate?: string) => {
    const params = new URLSearchParams()
    if (startDate) params.set('start_date', startDate)
    if (endDate) params.set('end_date', endDate)
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.post<SyncStockResponse>(`/sync/stock/${stockId}${query}`, {})
  },
  bulk: (targetDate?: string) => {
    const params = new URLSearchParams()
    if (targetDate) params.set('target_date', targetDate)
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.post<SyncBulkResponse>(`/sync/bulk${query}`, {})
  },
  all: (startDate?: string, endDate?: string) => {
    const params = new URLSearchParams()
    if (startDate) params.set('start_date', startDate)
    if (endDate) params.set('end_date', endDate)
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.post<SyncAllResponse>(`/sync/all${query}`, {})
  },
  // PER/PBR/殖利率
  perStatus: (startDate?: string, endDate?: string) => {
    const params = new URLSearchParams()
    if (startDate) params.set('start_date', startDate)
    if (endDate) params.set('end_date', endDate)
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.get<SyncStatusResponse>(`/sync/per/status${query}`)
  },
  perBulk: (targetDate?: string) => {
    const params = new URLSearchParams()
    if (targetDate) params.set('target_date', targetDate)
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.post<SyncBulkResponse>(`/sync/per/bulk${query}`, {})
  },
  perStock: (stockId: string, startDate?: string, endDate?: string) => {
    const params = new URLSearchParams()
    if (startDate) params.set('start_date', startDate)
    if (endDate) params.set('end_date', endDate)
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.post<SyncStockResponse>(`/sync/per/stock/${stockId}${query}`, {})
  },
  perAll: (startDate?: string, endDate?: string) => {
    const params = new URLSearchParams()
    if (startDate) params.set('start_date', startDate)
    if (endDate) params.set('end_date', endDate)
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.post<SyncAllResponse>(`/sync/per/all${query}`, {})
  },
  // 三大法人
  institutionalStatus: (startDate?: string, endDate?: string) => {
    const params = new URLSearchParams()
    if (startDate) params.set('start_date', startDate)
    if (endDate) params.set('end_date', endDate)
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.get<SyncStatusResponse>(`/sync/institutional/status${query}`)
  },
  institutionalBulk: (targetDate?: string) => {
    const params = new URLSearchParams()
    if (targetDate) params.set('target_date', targetDate)
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.post<SyncBulkResponse>(`/sync/institutional/bulk${query}`, {})
  },
  institutionalStock: (stockId: string, startDate?: string, endDate?: string) => {
    const params = new URLSearchParams()
    if (startDate) params.set('start_date', startDate)
    if (endDate) params.set('end_date', endDate)
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.post<SyncStockResponse>(`/sync/institutional/stock/${stockId}${query}`, {})
  },
  institutionalAll: (startDate?: string, endDate?: string) => {
    const params = new URLSearchParams()
    if (startDate) params.set('start_date', startDate)
    if (endDate) params.set('end_date', endDate)
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.post<SyncAllResponse>(`/sync/institutional/all${query}`, {})
  },
  // 融資融券
  marginStatus: (startDate?: string, endDate?: string) => {
    const params = new URLSearchParams()
    if (startDate) params.set('start_date', startDate)
    if (endDate) params.set('end_date', endDate)
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.get<SyncStatusResponse>(`/sync/margin/status${query}`)
  },
  marginBulk: (targetDate?: string) => {
    const params = new URLSearchParams()
    if (targetDate) params.set('target_date', targetDate)
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.post<SyncBulkResponse>(`/sync/margin/bulk${query}`, {})
  },
  marginStock: (stockId: string, startDate?: string, endDate?: string) => {
    const params = new URLSearchParams()
    if (startDate) params.set('start_date', startDate)
    if (endDate) params.set('end_date', endDate)
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.post<SyncStockResponse>(`/sync/margin/stock/${stockId}${query}`, {})
  },
  marginAll: (startDate?: string, endDate?: string) => {
    const params = new URLSearchParams()
    if (startDate) params.set('start_date', startDate)
    if (endDate) params.set('end_date', endDate)
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.post<SyncAllResponse>(`/sync/margin/all${query}`, {})
  },
  // 還原股價 (yfinance)
  adjStatus: (startDate?: string, endDate?: string) => {
    const params = new URLSearchParams()
    if (startDate) params.set('start_date', startDate)
    if (endDate) params.set('end_date', endDate)
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.get<SyncStatusResponse>(`/sync/adj/status${query}`)
  },
  adjBulk: (targetDate?: string) => {
    const params = new URLSearchParams()
    if (targetDate) params.set('target_date', targetDate)
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.post<SyncBulkResponse>(`/sync/adj/bulk${query}`, {})
  },
  adjStock: (stockId: string, startDate?: string, endDate?: string) => {
    const params = new URLSearchParams()
    if (startDate) params.set('start_date', startDate)
    if (endDate) params.set('end_date', endDate)
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.post<SyncStockResponse>(`/sync/adj/stock/${stockId}${query}`, {})
  },
  adjAll: (startDate?: string, endDate?: string) => {
    const params = new URLSearchParams()
    if (startDate) params.set('start_date', startDate)
    if (endDate) params.set('end_date', endDate)
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.post<SyncAllResponse>(`/sync/adj/all${query}`, {})
  },
  // 外資持股
  shareholdingStatus: (startDate?: string, endDate?: string) => {
    const params = new URLSearchParams()
    if (startDate) params.set('start_date', startDate)
    if (endDate) params.set('end_date', endDate)
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.get<SyncStatusResponse>(`/sync/shareholding/status${query}`)
  },
  shareholdingBulk: (targetDate?: string) => {
    const params = new URLSearchParams()
    if (targetDate) params.set('target_date', targetDate)
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.post<SyncBulkResponse>(`/sync/shareholding/bulk${query}`, {})
  },
  shareholdingStock: (stockId: string, startDate?: string, endDate?: string) => {
    const params = new URLSearchParams()
    if (startDate) params.set('start_date', startDate)
    if (endDate) params.set('end_date', endDate)
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.post<SyncStockResponse>(`/sync/shareholding/stock/${stockId}${query}`, {})
  },
  shareholdingAll: (startDate?: string, endDate?: string) => {
    const params = new URLSearchParams()
    if (startDate) params.set('start_date', startDate)
    if (endDate) params.set('end_date', endDate)
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.post<SyncAllResponse>(`/sync/shareholding/all${query}`, {})
  },
  // 借券明細
  securitiesLendingStatus: (startDate?: string, endDate?: string) => {
    const params = new URLSearchParams()
    if (startDate) params.set('start_date', startDate)
    if (endDate) params.set('end_date', endDate)
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.get<SyncStatusResponse>(`/sync/securities-lending/status${query}`)
  },
  securitiesLendingStock: (stockId: string, startDate?: string, endDate?: string) => {
    const params = new URLSearchParams()
    if (startDate) params.set('start_date', startDate)
    if (endDate) params.set('end_date', endDate)
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.post<SyncStockResponse>(`/sync/securities-lending/stock/${stockId}${query}`, {})
  },
  securitiesLendingAll: (startDate?: string, endDate?: string) => {
    const params = new URLSearchParams()
    if (startDate) params.set('start_date', startDate)
    if (endDate) params.set('end_date', endDate)
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.post<SyncAllResponse>(`/sync/securities-lending/all${query}`, {})
  },
  // 月營收
  monthlyRevenueStatus: (startYear?: number, endYear?: number) => {
    const params = new URLSearchParams()
    if (startYear) params.set('start_year', String(startYear))
    if (endYear) params.set('end_year', String(endYear))
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.get<MonthlyStatusResponse>(`/sync/monthly-revenue/status${query}`)
  },
  monthlyRevenueStock: (stockId: string, startYear?: number, endYear?: number) => {
    const params = new URLSearchParams()
    if (startYear) params.set('start_year', String(startYear))
    if (endYear) params.set('end_year', String(endYear))
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.post<MonthlyStockResponse>(`/sync/monthly-revenue/stock/${stockId}${query}`, {})
  },
  monthlyRevenueAll: (startYear?: number, endYear?: number) => {
    const params = new URLSearchParams()
    if (startYear) params.set('start_year', String(startYear))
    if (endYear) params.set('end_year', String(endYear))
    const query = params.toString() ? `?${params.toString()}` : ''
    return api.post<SyncAllResponse>(`/sync/monthly-revenue/all${query}`, {})
  },
}

// =============================================================================
// Walk-Forward Backtest Types
// =============================================================================

export interface EquityCurvePoint {
  date: string
  equity: number
  benchmark: number | null
  drawdown: number | null
}

export interface WeekStatus {
  week_id: string
  status: 'available' | 'missing' | 'not_allowed'
  model_name?: string
  valid_ic?: number
  fallback_week?: string
  fallback_model?: string
  reason?: string
}

export interface AvailableWeeksResponse {
  weeks: WeekStatus[]
  current_week_id: string
}

export interface WalkForwardRequest {
  start_week_id: string
  end_week_id: string
  initial_capital?: number
  max_positions?: number
  trade_price?: string
  enable_incremental?: boolean
  strategy?: string
}

export interface WalkForwardRunResponse {
  backtest_id: number
  job_id: string
  status: string
  message: string
}

export interface WalkForwardConfig {
  initial_capital: number
  max_positions: number
  trade_price: string
  enable_incremental: boolean
  strategy: string
}

export interface IcAnalysis {
  avg_valid_ic: number
  avg_live_ic: number
  ic_decay: number
  ic_correlation: number | null
}

export interface WalkForwardReturnMetrics {
  cumulative_return: number
  market_return: number
  excess_return: number
  sharpe_ratio: number | null
  max_drawdown: number | null
  win_rate: number | null
  total_trades: number | null
}

export interface WeeklyDetail {
  predict_week: string
  model_week: string
  model_name: string
  valid_ic: number | null
  live_ic: number | null
  ic_decay: number | null
  week_return: number | null
  market_return: number | null
  is_fallback: boolean
  incremental_days: number | null
}

export interface WalkForwardItem {
  id: number
  start_week_id: string
  end_week_id: string
  status: string
  config: WalkForwardConfig
  created_at: string
  completed_at: string | null
}

export interface WalkForwardDetail extends WalkForwardItem {
  ic_analysis: IcAnalysis | null
  return_metrics: WalkForwardReturnMetrics | null
  weekly_details: WeeklyDetail[] | null
  equity_curve: EquityCurvePoint[] | null
}

export interface WalkForwardListResponse {
  items: WalkForwardItem[]
  total: number
}

export interface WeeklySummaryPoint {
  predict_week: string
  live_ic: number | null
  week_return: number | null
  market_return: number | null
  cumulative_return: number | null
  cumulative_market: number | null
}

export interface WalkForwardSummary {
  backtest_id: number
  start_week_id: string
  end_week_id: string
  config: WalkForwardConfig
  total_weeks: number
  mean_ic: number
  icir: number
  ic_positive_rate: number
  annualized_return: number | null
  annualized_excess: number | null
  cumulative_return: number
  market_return: number
  excess_return: number
  sharpe_ratio: number | null
  max_drawdown: number | null
  win_rate: number | null
  total_trades: number | null
  weekly_points: WeeklySummaryPoint[]
  equity_curve: EquityCurvePoint[]
  created_at: string
  completed_at: string | null
}

export const walkForwardApi = {
  availableWeeks: () => api.get<AvailableWeeksResponse>('/backtest/walk-forward/available-weeks'),
  list: (limit?: number) => {
    const query = limit ? `?limit=${limit}` : ''
    return api.get<WalkForwardListResponse>(`/backtest/walk-forward${query}`)
  },
  get: (backtestId: number) => api.get<WalkForwardDetail>(`/backtest/walk-forward/${backtestId}`),
  summary: (backtestId?: number) => {
    const query = backtestId ? `?backtest_id=${backtestId}` : ''
    return api.get<WalkForwardSummary>(`/backtest/walk-forward/summary${query}`)
  },
  run: (data: WalkForwardRequest) => api.post<WalkForwardRunResponse>('/backtest/walk-forward', data),
  delete: (backtestId: number) => api.delete(`/backtest/walk-forward/${backtestId}`),
}
