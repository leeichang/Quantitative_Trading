import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/**
 * 檢查資料是否為最新
 * 邏輯：latest_date >= 昨天 就是新鮮的
 */
export function isDataFresh(latestDate: string | null): boolean {
  if (!latestDate) return false

  const yesterday = new Date()
  yesterday.setDate(yesterday.getDate() - 1)
  yesterday.setHours(0, 0, 0, 0)

  const latest = new Date(latestDate)
  latest.setHours(0, 0, 0, 0)

  return latest >= yesterday
}
