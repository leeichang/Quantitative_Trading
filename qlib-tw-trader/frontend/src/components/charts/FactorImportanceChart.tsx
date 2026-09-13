import { useMemo } from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'
import type { Factor } from '@/api/client'

interface FactorImportanceChartProps {
  factors: Factor[]
  height?: number
}

export function FactorImportanceChart({ factors, height = 400 }: FactorImportanceChartProps) {
  const chartData = useMemo(() => {
    return factors.map((f) => ({
      name: f.display_name || f.name,
      rate: Number((f.selection_rate * 100).toFixed(1)),
      category: f.category,
    })).reverse() // reverse for horizontal bar: top item at top
  }, [factors])

  if (chartData.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-muted-foreground text-sm">
        No factor data
      </div>
    )
  }

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart
        data={chartData}
        layout="vertical"
        margin={{ top: 5, right: 30, left: 120, bottom: 5 }}
      >
        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" horizontal={false} />
        <XAxis
          type="number"
          tick={{ fontSize: 10 }}
          tickLine={false}
          axisLine={{ stroke: '#e5e7eb' }}
          tickFormatter={(value) => `${value}%`}
          domain={[0, 100]}
        />
        <YAxis
          type="category"
          dataKey="name"
          tick={{ fontSize: 10 }}
          tickLine={false}
          axisLine={{ stroke: '#e5e7eb' }}
          width={110}
        />
        <Tooltip
          formatter={(value: number) => [`${value.toFixed(1)}%`, 'Selection Rate']}
          labelFormatter={(label) => `Factor: ${label}`}
        />
        <Bar
          dataKey="rate"
          fill="#3b82f6"
          fillOpacity={0.8}
          radius={[0, 4, 4, 0]}
          maxBarSize={16}
        />
      </BarChart>
    </ResponsiveContainer>
  )
}
