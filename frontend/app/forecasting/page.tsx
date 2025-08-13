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

export default function Forecasting() {
  return (
    <div className="container mx-auto px-6 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Technology Forecasting</h1>
        <p className="text-gray-600">Predictive analysis of future patent and research trends</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* <Card>
          <CardHeader>
            <CardTitle>Patent Trend Forecast</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={forecastData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="year" />
                <YAxis />
                <Tooltip />
                <Line
                  type="monotone"
                  dataKey="patents"
                  stroke="#8884d8"
                  strokeWidth={2}
                    
                />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Research Publication Forecast</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={400}>
              <AreaChart data={forecastData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="year" />
                <YAxis />
                <Tooltip />
                <Area type="monotone" dataKey="research" stroke="#82ca9d" fill="#82ca9d" fillOpacity={0.6} />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card> */}

        {/* <Card className="lg:col-span-2">
          <CardHeader>  
            <CardTitle>Combined Forecast Analysis</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={forecastData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="year" />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="patents" stroke="#8884d8" strokeWidth={2} name="Patents" />
                <Line type="monotone" dataKey="research" stroke="#82ca9d" strokeWidth={2} name="Research" />
              </LineChart>
            </ResponsiveContainer>
            <div className="mt-4 p-4 bg-blue-50 rounded-lg">
              <h3 className="font-semibold text-blue-900 mb-2">Forecast Insights</h3>
              <ul className="text-sm text-blue-800 space-y-1">
                <li>• Patent applications expected to grow by 75% over the next 3 years</li>
                <li>• Research publications showing steady 15% annual growth</li>
                <li>• Strong correlation between research output and patent filings</li>
                <li>• Peak innovation period predicted for 2025-2026</li>
              </ul>
            </div>
          </CardContent>
        </Card> */}
      </div>
      <div className="mt-8">
        <Card>
          <CardHeader>
            <CardTitle>Patent Filings vs. Publications</CardTitle>
          </CardHeader>
          <CardContent>
            <PatentPublicationChart />
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
