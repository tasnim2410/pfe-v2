"use client";

import React, { useState, useEffect } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Area,
  ComposedChart,
  Scatter
} from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";

// Interface definitions
interface ForecastPoint {
  year: number;
  predicted_patents: number;
}

interface EvaluationPoint {
  actual: number;
  year: number;
  yhat: number;
}

interface GrowthData {
  label: string;
  percent: number;
  window: {
    end_year: number;
    start_year: number;
  };
}

interface HistoryDataPoint {
  ds: string;
  patents: number;
  publications: number;
  year: number;
}

interface PatentDataPoint {
  count: number;
  ds: string;
  year: number;
}

interface PublicationDataPoint {
  count: number;
  ds: string;
  year: number;
}

interface ApiResponse {
  applied_truncation: {
    pat_trunc_effective: number;
    pat_trunc_requested: number;
    pub_trunc_effective: number;
    pub_trunc_requested: number;
  };
  evaluation: {
    mspe_percent: number;
    past_growth_comparison: {
      delta_percent: number;
      labels: {
        actual: string;
        model: string;
      };
      past_actual_percent: number;
      past_model_percent: number;
      window: {
        end_year: number;
        start_year: number;
      };
    };
    points: EvaluationPoint[];
    test_years: number[];
  };
  forecast: ForecastPoint[];
  forecast_horizon: number;
  growth: {
    current_actual: GrowthData;
    current_with_forecast: GrowthData;
    past_actual: GrowthData;
  };
  history: {
    last_history_year: number;
    merged: HistoryDataPoint[];
    patents: PatentDataPoint[];
    publications: PublicationDataPoint[];
    years_used_for_model: number[];
  };
  ok: boolean;
  params: {
    horizon: number;
    split_year: number;
    test_years: number | null;
  };
  start_year: number;
}

interface ChartDataPoint {
  year: number;
  historical?: number;
  actual?: number;
  predicted?: number;
  forecast?: number;
  type: 'history' | 'test' | 'forecast' | 'connecting';
}

const LSTMForecastSeries: React.FC = () => {
  const [data, setData] = useState<ChartDataPoint[]>([]);
  const [apiResponse, setApiResponse] = useState<ApiResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [patTrunc, setPatTrunc] = useState(1);
  const [pubTrunc, setPubTrunc] = useState(1);
  const [testYears, setTestYears] = useState(2);
  const [horizon, setHorizon] = useState(3);

  function valueFmt(v?: number) {
    return typeof v === "number" ? v.toFixed(2) : v;
  }

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const portRes = await fetch('/backend_port.txt');
      const port = (await portRes.text()).trim();
      const response = await fetch(`http://localhost:${port}/api/lstm_forecast_sereis`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          pat_trunc: patTrunc,
          pub_trunc: pubTrunc,
          test_years: testYears,
          horizon: horizon
        }),
      });



      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result: ApiResponse = await response.json();
      
      if (result.ok) {
        setApiResponse(result);
        
        // Transform data for the chart
        const chartData: ChartDataPoint[] = [];
        
        // Create a map to store all data points by year
        const dataMap = new Map<number, ChartDataPoint>();
        
        // Add historical data from merged history
        result.history.merged.forEach(point => {
          dataMap.set(point.year, {
            year: point.year,
            historical: point.patents,
            type: 'history'
          });
        });
        
        // Get last historical values for connection
        const lastHistYear = Math.max(...result.history.merged.map(d => d.year));
        const lastHistValue = result.history.merged.find(d => d.year === lastHistYear)?.patents ?? 0;

        // Add forecast points with connection point
        const forecastWithConnection = [
          // Add connection point
          { year: lastHistYear, forecast: lastHistValue },
          // Add forecast points that come after history
          ...result.forecast
            .filter(point => point.year > lastHistYear)
            .map(point => ({
              year: point.year,
              forecast: point.predicted_patents,
            }))
        ];

        forecastWithConnection.forEach(point => {
          const existingPoint = dataMap.get(point.year);
          if (existingPoint) {
            existingPoint.forecast = point.forecast;
          } else {
            dataMap.set(point.year, {
              year: point.year,
              forecast: point.forecast,
              type: 'forecast'
            });
          }
        });
        
        // Convert map to array and sort by year
        const sortedData = Array.from(dataMap.values()).sort((a, b) => a.year - b.year);
        
        setData(sortedData);
      } else {
        throw new Error('API returned error status');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unknown error occurred');
    } finally {
      setLoading(false);
    }
  };

  // Don't fetch on mount - let user configure parameters first


  return (
    <Card className="w-full">
      <CardHeader className="flex flex-col gap-4">
        <CardTitle>LSTM Forecast</CardTitle>
        
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="flex items-center gap-2">
              <Label htmlFor="patTrunc" className="text-sm whitespace-nowrap min-w-[120px]">
                Patent Truncation:
              </Label>
              <Input
                id="patTrunc"
                type="number"
                min="1"
                max="10"
                value={patTrunc}
                onChange={(e) => setPatTrunc(parseInt(e.target.value) || 1)}
                className="w-16"
              />
            </div>
            <div className="flex items-center gap-2">
              <Label htmlFor="pubTrunc" className="text-sm whitespace-nowrap min-w-[120px]">
                Publication Truncation:
              </Label>
              <Input
                id="pubTrunc"
                type="number"
                min="1"
                max="10"
                value={pubTrunc}
                onChange={(e) => setPubTrunc(parseInt(e.target.value) || 1)}
                className="w-16"
              />
            </div>
            <div className="flex items-center gap-2">
              <Label htmlFor="testYears" className="text-sm whitespace-nowrap min-w-[120px]">
                Test Years:
              </Label>
              <Input
                id="testYears"
                type="number"
                min="1"
                max="5"
                value={testYears}
                onChange={(e) => setTestYears(parseInt(e.target.value) || 2)}
                className="w-16"
              />
            </div>
            <div className="flex items-center gap-2">
              <Label htmlFor="horizon" className="text-sm whitespace-nowrap min-w-[120px]">
                Forecast Horizon:
              </Label>
              <Input
                id="horizon"
                type="number"
                min="1"
                max="10"
                value={horizon}
                onChange={(e) => setHorizon(parseInt(e.target.value) || 3)}
                className="w-16"
              />
            </div>
          </div>
        </div>
        
        <div className="flex flex-wrap items-center gap-4">
          <Button onClick={fetchData} disabled={loading}>
            {loading ? "Loading..." : "Run Forecast"}
          </Button>
        </div>
      </CardHeader>
      
      <CardContent className="space-y-4">
        {loading && <div className="text-sm text-gray-500">Loading forecast…</div>}
        {error && <div className="text-sm text-red-600">Error: {error}</div>}
        
        {!loading && !error && apiResponse && (
          <>
            {/* Growth Rate Comparison */}
            {apiResponse.growth && (
              <div className="p-4 bg-blue-50 rounded-lg">
                <h3 className="font-semibold text-lg mb-2">Growth Rate Analysis</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <h4 className="font-medium">Past Growth (Actual)</h4>
                    <div className="flex items-center gap-2">
                      <span className="text-sm">Period:</span>
                      <Badge variant="outline">
                        {apiResponse.growth.past_actual.window.start_year} - {apiResponse.growth.past_actual.window.end_year}
                      </Badge>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-sm">Trend:</span>
                      <Badge variant={apiResponse.growth.past_actual.label === "Rapid" ? "default" : "secondary"}>
                        {apiResponse.growth.past_actual.label.replace('_', ' ')}
                      </Badge>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-sm">Rate:</span>
                      <span className="font-semibold text-blue-700">
                        {apiResponse.growth.past_actual.percent.toFixed(2)}%
                      </span>
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    <h4 className="font-medium">Forecasted Growth</h4>
                    <div className="flex items-center gap-2">
                      <span className="text-sm">Period:</span>
                      <Badge variant="outline">
                        {apiResponse.growth.current_with_forecast.window.start_year} - {apiResponse.growth.current_with_forecast.window.end_year}
                      </Badge>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-sm">Trend:</span>
                      <Badge variant={apiResponse.growth.current_with_forecast.label === "Rapid" ? "default" : "secondary"}>
                        {apiResponse.growth.current_with_forecast.label.replace('_', ' ')}
                      </Badge>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-sm">Rate:</span>
                      <span className="font-semibold text-purple-700">
                        {apiResponse.growth.current_with_forecast.percent.toFixed(2)}%
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            )}
            
            <div className="w-full h-[380px]">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={data} margin={{ left: 10, right: 20, top: 10, bottom: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="year"
                    type="number"
                    domain={['dataMin', 'dataMax']}
                    tickFormatter={(year) => year.toString()}
                  />
                  <YAxis />
                  <Tooltip 
                    formatter={(value, name) => [
                      valueFmt(value as number), 
                      name === 'actual' ? 'Actual Patents' : 
                      name === 'predicted' ? 'Predicted (Test)' : 
                      'Forecasted Patents'
                    ]}
                    labelFormatter={(year) => `Year: ${year}`}
                  />
                  <Legend />
                  
                  {/* Historical trend line */}
                  <Line
                    name="Historical Patents"
                    type="monotone"
                    dataKey="historical"
                    stroke="#1f2937"
                    strokeWidth={2}
                    dot={{ fill: '#1f2937', strokeWidth: 2, r: 3 }}
                    isAnimationActive={false}
                    connectNulls={true}
                  />
                  
                  {/* Forecast line */}
                  <Line 
                    name="Forecasted Patents"
                    type="monotone" 
                    dataKey="forecast" 
                    stroke="#10b981" 
                    strokeWidth={3}
                    strokeDasharray="8 4"
                    dot={{ fill: '#10b981', strokeWidth: 2, r: 4 }}
                    isAnimationActive={false}
                    connectNulls={true}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
            
            <div className="mt-2 text-sm text-gray-800 grid sm:grid-cols-2 gap-4">
              <div className="space-y-1">
                <div className="font-semibold">Model Performance</div>
                <div>
                  MSPE:&nbsp;
                  <span className="tabular-nums">
                    {apiResponse.evaluation.mspe_percent.toFixed(2)}%
                  </span>
                </div>
              </div>
              
              <div className="space-y-1">
                <div className="font-semibold">Model Details</div>
                <div>
                  Forecast Horizon:&nbsp;
                  <span className="tabular-nums">
                    {apiResponse.forecast_horizon} years
                  </span>
                </div>
                <div>
                  Test Years:&nbsp;
                  <span className="tabular-nums">
                    {apiResponse.evaluation.test_years.join(', ')}
                  </span>
                </div>
              </div>
            </div>
            
            <div className="text-xs text-gray-500">
              Note: Gray line shows historical patent data and green dashed line shows forecasted values. 
              Lines are connected to show trend continuity.
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
};

export default LSTMForecastSeries;