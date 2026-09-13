import { useMemo } from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Cell,
} from 'recharts'
import type { WeeklySummaryPoint } from '@/api/client'

interface RollingIcChartProps {
  data: WeeklySummaryPoint[]
  height?: number
}

export function RollingIcChart({ data, height = 200 }: RollingIcChartProps) {
  const chartData = useMemo(() => {
    return data
      .filter((p) => p.live_ic !== null)
      .map((p) => ({
        week: p.predict_week,
        ic: p.live_ic,
      }))
  }, [data])

  if (chartData.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-muted-foreground text-sm">
        No IC data available
      </div>
    )
  }

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={chartData} margin={{ top: 5, right: 10, left: 10, bottom: 5 }}>
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
          tickFormatter={(value) => value.toFixed(2)}
        />
        <Tooltip
          formatter={(value: number) => [value.toFixed(4), 'Live IC']}
          labelFormatter={(label) => `Week: ${label}`}
        />
        <ReferenceLine y={0} stroke="#9ca3af" strokeDasharray="5 5" />
        <Bar dataKey="ic" maxBarSize={20}>
          {chartData.map((entry, index) => (
            <Cell
              key={index}
              fill={(entry.ic ?? 0) >= 0 ? '#22c55e' : '#ef4444'}
              fillOpacity={0.8}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
