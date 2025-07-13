// File: /mnt/data/ChartContext.tsx
"use client"

import React, { createContext, useState, ReactNode, useContext } from "react"

// Define the shape of our shared context
interface ChartContextType {
  selectedCharts: string[]                        // IDs of charts user has chosen
  setSelectedCharts: React.Dispatch<React.SetStateAction<string[]>>
  chartComments: Record<string, string>           // User comments per chart
  setChartComments: React.Dispatch<React.SetStateAction<Record<string, string>>>
  chartData: Record<string, any | null>           // Cached API data for each chart (null if failed)
  loadChartData: (chartId: string) => Promise<void>// Function to fetch & cache data
}

// Create the context
const ChartContext = createContext<ChartContextType | undefined>(undefined)

// Provider component that holds all shared state + generic fetch logic
export function ChartProvider({ children }: { children: ReactNode }) {
  const [selectedCharts, setSelectedCharts] = useState<string[]>([])
  const [chartComments, setChartComments] = useState<Record<string, string>>({})
  const [chartData, setChartData] = useState<Record<string, any | null>>({})

  // Generic loader: fetch from /api/{chartId}
  const loadChartData = async (chartId: string) => {
    // skip if already loaded or attempted
    if (chartData.hasOwnProperty(chartId)) return
    const url = `/api/${chartId}`
    try {
      const res = await fetch(url)
      if (!res.ok) {
        console.error(`Error loading ${chartId}: HTTP ${res.status}`)
        setChartData(prev => ({ ...prev, [chartId]: null }))
        return
      }
      const data = await res.json()
      setChartData(prev => ({ ...prev, [chartId]: data }))
    } catch (err) {
      console.error(`Error loading ${chartId}:`, err)
      setChartData(prev => ({ ...prev, [chartId]: null }))
    }
  }

  return (
    <ChartContext.Provider
      value={{ selectedCharts, setSelectedCharts, chartComments, setChartComments, chartData, loadChartData }}
    >
      {children}
    </ChartContext.Provider>
  )
}

// Hook for consuming the chart context
export function useChartContext() {
  const context = useContext(ChartContext)
  if (!context) throw new Error("useChartContext must be used within a ChartProvider")
  return context
}
