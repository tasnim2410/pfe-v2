"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from "recharts"

const forecastData = [
  { year: "2020", patents: 150, research: 80, predicted: false },
  { year: "2021", patents: 180, research: 95, predicted: false },
  { year: "2022", patents: 220, research: 110, predicted: false },
  { year: "2023", patents: 280, research: 130, predicted: false },
  { year: "2024", patents: 340, research: 155, predicted: true },
  { year: "2025", patents: 410, research: 180, predicted: true },
  { year: "2026", patents: 490, research: 210, predicted: true },
]

import PatentPublicationChart from "@/components/filing_vs_publication";
import TVPTwoCharts from "@/components/TVPcharts";
import ProphetForecast from "@/components/prophet_forecast";
import ArimaxQuadraticClient from "@/components/arimax_quadratic_client";
export default function Forecasting() {
  return (
    <div className="container mx-auto px-6 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Technology Forecasting</h1>
        <p className="text-gray-600">Predictive analysis of future patent and research trends</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>ARIMAX Quadratic Model Forecast</CardTitle>
          </CardHeader>
          <CardContent>
            <ArimaxQuadraticClient />
          </CardContent>
        </Card>
        
      </div>
      <div className="mt-8">
        <Card className="w-full">
          <CardHeader>
            <CardTitle>Patent Filings vs. Publications</CardTitle>
          </CardHeader>
          <CardContent>
            <PatentPublicationChart />
          </CardContent>
        </Card>
        <div className="mt-8">
          <Card className="w-full">
            <CardHeader>
              <CardTitle>Forecasted Patent & Publication Trends</CardTitle>
            </CardHeader>
            <CardContent>
              <TVPTwoCharts />
              <div className="mt-8">
                <Card className="w-full">
                  <CardHeader>
                    <CardTitle>Prophet Model Forecast</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ProphetForecast/>
                  </CardContent>
                </Card>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
