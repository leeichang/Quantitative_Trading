import { useMemo } from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Legend,
} from 'recharts'
import type { WeeklySummaryPoint } from '@/api/client'

interface CumulativeReturnChartProps {
  data: WeeklySummaryPoint[]
  height?: number
}

export function CumulativeReturnChart({ data, height = 250 }: CumulativeReturnChartProps) {
  const chartData = useMemo(() => {
    return data.map((p) => ({
      week: p.predict_week,
      strategy: p.cumulative_return !== null ? Number(p.cumulative_return.toFixed(2)) : null,
      market: p.cumulative_market !== null ? Number(p.cumulative_market.toFixed(2)) : null,
    }))
  }, [data])

  if (chartData.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-muted-foreground text-sm">
        No return data available
      </div>
    )
  }

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={chartData} margin={{ top: 5, right: 10, left: 10, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
        <XAxis
          dataKey="week"
          tick={{ fontSize: 10 }}
          tickLine={false}
          axisLine={{ stroke: '#e5e7eb' }}
          interval={Math.max(0, Math.floor(chartData.length / 12) - 1)}
        />
        <YAxis
          tick={{ fontSize: 10 }}
          tickLine={false}
          axisLine={{ stroke: '#e5e7eb' }}
          tickFormatter={(value) => `${value.toFixed(1)}%`}
        />
        <Tooltip
          formatter={(value: number, name: string) => {
            const label = name === 'strategy' ? 'Strategy' : 'Market'
            return [`${value.toFixed(2)}%`, label]
          }}
          labelFormatter={(label) => `Week: ${label}`}
        />
        <Legend
          formatter={(value) => (value === 'strategy' ? 'Strategy' : 'Market')}
        />
        <ReferenceLine y={0} stroke="#9ca3af" strokeDasharray="5 5" />
        <Line
          type="monotone"
          dataKey="strategy"
          stroke="#3b82f6"
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 4, fill: '#3b82f6' }}
        />
        <Line
          type="monotone"
          dataKey="market"
          stroke="#9ca3af"
          strokeWidth={1.5}
          dot={false}
          strokeDasharray="5 5"
          activeDot={{ r: 3, fill: '#9ca3af' }}
        />
      </LineChart>
    </ResponsiveContainer>
  )
}
