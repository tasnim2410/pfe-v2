"use client";

import React, { useEffect, useMemo, useState } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
  Area,
  Scatter,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";

// Types
type HistoryPoint = { year: number; count: number };
type ForecastPoint = { year: number; yhat: number; yhat_lower?: number; yhat_upper?: number };
type TestPoint = { year: number; actual: number; yhat?: number; yhat_with_pub_reg?: number };
type GrowthWindow = { start_year: number; end_year: number };
type GrowthData = { 
  label: string; 
  percent: number; 
  window: GrowthWindow;
};

type ProphetResponse = {
  best_lag: number;
  tech?: string;
  xcorr?: number;
  growth?: {
    current: GrowthData;
    past: GrowthData;
  };
  metrics: {
    mae_patents_baseline: number;
    rmse_patents_baseline: number;
    mae_patents_with_reg: number;
    rmse_patents_with_reg: number;
    mae_pubs: number;
    rmse_pubs: number;
    ampe_patents_baseline?: number;
    ampe_patents_with_reg?: number;
    ampe_pubs?: number;
  };
  patents: {
    history: HistoryPoint[];
    forecast: {
      baseline: ForecastPoint[];
      with_pub_ar: ForecastPoint[];
    };
    test?: TestPoint[];
  };
  publications: {
    history: HistoryPoint[];
    forecast: ForecastPoint[];
    test?: TestPoint[];
  };
};

type ChartDataPoint = {
  year: number;
  // Patents data
  patents_history?: number;
  patents_forecast?: number;
  patents_test_actual?: number;
  patents_test_pred?: number;
  // Publications data
  pubs_history?: number;
  pubs_forecast?: number;
  pubs_lower?: number;
  pubs_upper?: number;
  pubs_test_actual?: number;
  pubs_test_pred?: number;
};

const CURRENT_YEAR = 2025; // Hardcoded to current year

function valueFmt(v?: number) {
  return typeof v === "number" ? v.toFixed(2) : v;
}

export default function ProphetForecast() {
  const [data, setData] = useState<ProphetResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [tab, setTab] = useState<"patents" | "publications">("patents");
  const [error, setError] = useState<string | null>(null);
  const [horizon, setHorizon] = useState(5);
  const [pubTail, setPubTail] = useState(1);
  const [patTail, setPatTail] = useState(3);
  const [splitYear, setSplitYear] = useState(2021);
  const [testYears, setTestYears] = useState(1);
  const [evalStartYear, setEvalStartYear] = useState(2022);
  const [evalEndYear, setEvalEndYear] = useState(2022);
  const [requestType, setRequestType] = useState<"split" | "eval">("split");

  // Fetch data from API with parameters
  const fetchData = async () => {
    try {
      setLoading(true);
      setError(null);
      const portRes = await fetch("/backend_port.txt");
      const port = (await portRes.text()).trim();
      const backendUrl = `http://localhost:${port}/api/prophet_forecast`;
      
      let requestBody = {};
      if (requestType === "split") {
        requestBody = {
          horizon,
          pub_tail: pubTail,
          pat_tail: patTail,
          split_year: splitYear,
          test_years: testYears
        };
      } else {
        requestBody = {
          horizon,
          pub_tail: pubTail,
          pat_tail: patTail,
          eval_start_year: evalStartYear,
          eval_end_year: evalEndYear
        };
      }
      
      const res = await fetch(backendUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      });
      if (!res.ok) {
        const t = await res.text();
        throw new Error(`HTTP ${res.status} - ${t || "Request failed"}`);
      }
      const json: ProphetResponse = await res.json();
      setData(json);
    } catch (e: any) {
      setError(e?.message ?? "Failed to load forecast");
    } finally {
      setLoading(false);
    }
  };

  // Fetch data on mount and when parameters change
  useEffect(() => {
    fetchData();
  }, [requestType]);

  // Process data for charts
  const { patentsData, publicationsData } = useMemo(() => {
    if (!data) return { patentsData: [], publicationsData: [] };

    // Process patents data
    const patentsHistory = data.patents.history.map(item => ({
      year: item.year,
      patents_history: item.count,
    }));

    // Get last historical values for connection
    const lastPatentHistYear = Math.max(...data.patents.history.map(d => d.year));
    const lastPatentHistValue = data.patents.history.find(d => d.year === lastPatentHistYear)?.count ?? 0;

    const patentsForecast = [
      // Add connection point
      { year: lastPatentHistYear, patents_forecast: lastPatentHistValue },
      // Add forecast points that come after history
      ...data.patents.forecast.with_pub_ar
        .filter(item => item.year > lastPatentHistYear)
        .map(item => ({
          year: item.year,
          patents_forecast: item.yhat,
        }))
    ];

    const patentsTest = data.patents.test?.map(item => ({
      year: item.year,
      patents_test_actual: item.actual,
      patents_test_pred: item.yhat_with_pub_reg,
    })) || [];

    // Process publications data
    const pubsHistory = data.publications.history.map(item => ({
      year: item.year,
      pubs_history: item.count,
    }));

    // Get last historical values for connection
    const lastPubHistYear = Math.max(...data.publications.history.map(d => d.year));
    const lastPubHistValue = data.publications.history.find(d => d.year === lastPubHistYear)?.count ?? 0;

    const pubsForecast = [
      // Add connection point
      { 
        year: lastPubHistYear, 
        pubs_forecast: lastPubHistValue,
        pubs_lower: lastPubHistValue,
        pubs_upper: lastPubHistValue
      },
      // Add forecast points that come after history
      ...data.publications.forecast
        .filter(item => item.year > lastPubHistYear)
        .map(item => ({
          year: item.year,
          pubs_forecast: item.yhat,
          pubs_lower: item.yhat_lower,
          pubs_upper: item.yhat_upper,
        }))
    ];

    const pubsTest = data.publications.test?.map(item => ({
      year: item.year,
      pubs_test_actual: item.actual,
      pubs_test_pred: item.yhat,
    })) || [];

    // Combine data
    const allYears = new Set([
      ...patentsHistory.map(d => d.year),
      ...patentsForecast.map(d => d.year),
      ...patentsTest.map(d => d.year),
      ...pubsHistory.map(d => d.year),
      ...pubsForecast.map(d => d.year),
      ...pubsTest.map(d => d.year),
    ]);

    const patentsData: ChartDataPoint[] = [];
    const publicationsData: ChartDataPoint[] = [];

    Array.from(allYears).sort().forEach(year => {
      // Patents data point
      const patentsPoint: ChartDataPoint = { year };
      const historyPt = patentsHistory.find(d => d.year === year);
      const forecastPt = patentsForecast.find(d => d.year === year);
      const testPt = patentsTest.find(d => d.year === year);

      if (historyPt) patentsPoint.patents_history = historyPt.patents_history;
      if (forecastPt) patentsPoint.patents_forecast = forecastPt.patents_forecast;
      if (testPt) {
        patentsPoint.patents_test_actual = testPt.patents_test_actual;
        patentsPoint.patents_test_pred = testPt.patents_test_pred;
      }

      patentsData.push(patentsPoint);

      // Publications data point
      const pubsPoint: ChartDataPoint = { year };
      const pubsHistoryPt = pubsHistory.find(d => d.year === year);
      const pubsForecastPt = pubsForecast.find(d => d.year === year);
      const pubsTestPt = pubsTest.find(d => d.year === year);

      if (pubsHistoryPt) pubsPoint.pubs_history = pubsHistoryPt.pubs_history;
      if (pubsForecastPt) {
        pubsPoint.pubs_forecast = pubsForecastPt.pubs_forecast;
        pubsPoint.pubs_lower = pubsForecastPt.pubs_lower;
        pubsPoint.pubs_upper = pubsForecastPt.pubs_upper;
      }
      if (pubsTestPt) {
        pubsPoint.pubs_test_actual = pubsTestPt.pubs_test_actual;
        pubsPoint.pubs_test_pred = pubsTestPt.pubs_test_pred;
      }

      publicationsData.push(pubsPoint);
    });

    return { 
      patentsData: patentsData.sort((a, b) => a.year - b.year), 
      publicationsData: publicationsData.sort((a, b) => a.year - b.year) 
    };
  }, [data]);

  return (
    <Card className="w-full">
      <CardHeader className="flex flex-col gap-4">
        <CardTitle>Prophet Forecast</CardTitle>
        
        <Tabs value={requestType} onValueChange={(v) => setRequestType(v as "split" | "eval")}>
          <TabsList>
            <TabsTrigger value="split">Split Year Configuration</TabsTrigger>
            <TabsTrigger value="eval">Evaluation Range Configuration</TabsTrigger>
          </TabsList>
          
          <TabsContent value="split" className="space-y-4 mt-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="flex items-center gap-2">
                <Label htmlFor="horizon" className="text-sm whitespace-nowrap min-w-[100px]">
                  Forecast Horizon:
                </Label>
                <Input
                  id="horizon"
                  type="number"
                  min="1"
                  max="20"
                  value={horizon}
                  onChange={(e) => setHorizon(parseInt(e.target.value))}
                  className="w-16"
                />
              </div>
              <div className="flex items-center gap-2">
                <Label htmlFor="pubTail" className="text-sm whitespace-nowrap min-w-[100px]">
                  Pub Tail:
                </Label>
                <Input
                  id="pubTail"
                  type="number"
                  min="1"
                  max="10"
                  value={pubTail}
                  onChange={(e) => setPubTail(parseInt(e.target.value))}
                  className="w-16"
                />
              </div>
              <div className="flex items-center gap-2">
                <Label htmlFor="patTail" className="text-sm whitespace-nowrap min-w-[100px]">
                  Pat Tail:
                </Label>
                <Input
                  id="patTail"
                  type="number"
                  min="1"
                  max="10"
                  value={patTail}
                  onChange={(e) => setPatTail(parseInt(e.target.value))}
                  className="w-16"
                />
              </div>
              <div className="flex items-center gap-2">
                <Label htmlFor="splitYear" className="text-sm whitespace-nowrap min-w-[100px]">
                  Split Year:
                </Label>
                <Input
                  id="splitYear"
                  type="number"
                  min="2000"
                  max="2030"
                  value={splitYear}
                  onChange={(e) => setSplitYear(parseInt(e.target.value))}
                  className="w-20"
                />
              </div>
              <div className="flex items-center gap-2">
                <Label htmlFor="testYears" className="text-sm whitespace-nowrap min-w-[100px]">
                  Test Years:
                </Label>
                <Input
                  id="testYears"
                  type="number"
                  min="1"
                  max="10"
                  value={testYears}
                  onChange={(e) => setTestYears(parseInt(e.target.value))}
                  className="w-16"
                />
              </div>
            </div>
          </TabsContent>
          
          <TabsContent value="eval" className="space-y-4 mt-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="flex items-center gap-2">
                <Label htmlFor="horizon2" className="text-sm whitespace-nowrap min-w-[100px]">
                  Forecast Horizon:
                </Label>
                <Input
                  id="horizon2"
                  type="number"
                  min="1"
                  max="20"
                  value={horizon}
                  onChange={(e) => setHorizon(parseInt(e.target.value))}
                  className="w-16"
                />
              </div>
              <div className="flex items-center gap-2">
                <Label htmlFor="pubTail2" className="text-sm whitespace-nowrap min-w-[100px]">
                  Pub Tail:
                </Label>
                <Input
                  id="pubTail2"
                  type="number"
                  min="1"
                  max="10"
                  value={pubTail}
                  onChange={(e) => setPubTail(parseInt(e.target.value))}
                  className="w-16"
                />
              </div>
              <div className="flex items-center gap-2">
                <Label htmlFor="patTail2" className="text-sm whitespace-nowrap min-w-[100px]">
                  Pat Tail:
                </Label>
                <Input
                  id="patTail2"
                  type="number"
                  min="1"
                  max="10"
                  value={patTail}
                  onChange={(e) => setPatTail(parseInt(e.target.value))}
                  className="w-16"
                />
              </div>
              <div className="flex items-center gap-2">
                <Label htmlFor="evalStartYear" className="text-sm whitespace-nowrap min-w-[100px]">
                  Eval Start Year:
                </Label>
                <Input
                  id="evalStartYear"
                  type="number"
                  min="2000"
                  max="2030"
                  value={evalStartYear}
                  onChange={(e) => setEvalStartYear(parseInt(e.target.value))}
                  className="w-20"
                />
              </div>
              <div className="flex items-center gap-2">
                <Label htmlFor="evalEndYear" className="text-sm whitespace-nowrap min-w-[100px]">
                  Eval End Year:
                </Label>
                <Input
                  id="evalEndYear"
                  type="number"
                  min="2000"
                  max="2030"
                  value={evalEndYear}
                  onChange={(e) => setEvalEndYear(parseInt(e.target.value))}
                  className="w-20"
                />
              </div>
            </div>
          </TabsContent>
        </Tabs>
        
        <div className="flex flex-wrap items-center gap-4">
          <Button onClick={fetchData} disabled={loading}>
            {loading ? "Loading..." : "Run Forecast"}
          </Button>
          
          <div className="flex flex-wrap items-center gap-2">
            <Button
              variant={tab === "patents" ? "default" : "outline"}
              onClick={() => setTab("patents")}
            >
              Patents
            </Button>
            <Button
              variant={tab === "publications" ? "default" : "outline"}
              onClick={() => setTab("publications")}
            >
              Publications
            </Button>
          </div>
        </div>
      </CardHeader>
      
      <CardContent className="space-y-4">
        {loading && <div className="text-sm text-gray-500">Loading forecast…</div>}
        {error && <div className="text-sm text-red-600">Error: {error}</div>}
        
        {!loading && !error && data && (
          <>
            {/* Growth Rate Comparison */}
            {data.growth && (
              <div className="p-4 bg-blue-50 rounded-lg">
                <h3 className="font-semibold text-lg mb-2">Growth Rate Comparison</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <h4 className="font-medium">Past Growth</h4>
                    <div className="flex items-center gap-2">
                      <span className="text-sm">Period:</span>
                      <Badge variant="outline">
                        {data.growth.past.window.start_year} - {data.growth.past.window.end_year}
                      </Badge>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-sm">Trend:</span>
                      <Badge variant={data.growth.past.label === "Rapid" ? "default" : "secondary"}>
                        {data.growth.past.label}
                      </Badge>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-sm">Rate:</span>
                      <span className="font-semibold text-blue-700">
                        {data.growth.past.percent.toFixed(2)}%
                      </span>
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    <h4 className="font-medium">Forecasted Growth</h4>
                    <div className="flex items-center gap-2">
                      <span className="text-sm">Period:</span>
                      <Badge variant="outline">
                        {data.growth.current.window.start_year} - {data.growth.current.window.end_year}
                      </Badge>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-sm">Trend:</span>
                      <Badge variant={data.growth.current.label === "Rapid" ? "default" : "secondary"}>
                        {data.growth.current.label}
                      </Badge>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-sm">Rate:</span>
                      <span className="font-semibold text-green-700">
                        {data.growth.current.percent.toFixed(2)}%
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            )}
            
            {tab === "patents" ? (
              <div className="w-full h-[380px]">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={patentsData} margin={{ left: 10, right: 20, top: 10, bottom: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="year" />
                    <YAxis />
                    <Tooltip
                      formatter={(v, name) => [valueFmt(v as number), String(name)]}
                      labelFormatter={(label) => `Year: ${label}`}
                    />
                    <Legend />
                    
                    {/* History */}
                    <Line
                      name="Patents (history)"
                      type="monotone"
                      dataKey="patents_history"
                      dot={false}
                      stroke="#111827"
                      strokeWidth={2}
                      isAnimationActive={false}
                    />
                    
                    {/* Forecast */}
                    <Line
                      name="Patents (forecast)"
                      type="monotone"
                      dataKey="patents_forecast"
                      dot={false}
                      stroke="#10b981"
                      strokeWidth={2}
                      isAnimationActive={false}
                    />
                    
                    {/* Test actual values */}
                    <Scatter
                      name="Test (actual)"
                      dataKey="patents_test_actual"
                      fill="#ef4444"
                      stroke="#ef4444"
                      strokeWidth={2}
                    />
                    
                    {/* Test predictions */}
                    <Scatter
                      name="Test (predicted)"
                      dataKey="patents_test_pred"
                      fill="#3b82f6"
                      stroke="#3b82f6"
                      strokeWidth={2}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <div className="w-full h-[380px]">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={publicationsData} margin={{ left: 10, right: 20, top: 10, bottom: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="year" />
                    <YAxis />
                    <Tooltip
                      formatter={(v, name) => [valueFmt(v as number), String(name)]}
                      labelFormatter={(label) => `Year: ${label}`}
                    />
                    <Legend />
                    
                    {/* History */}
                    <Line
                      name="Publications (history)"
                      type="monotone"
                      dataKey="pubs_history"
                      dot={false}
                      stroke="#111827"
                      strokeWidth={2}
                      isAnimationActive={false}
                    />
                    
                    {/* Confidence interval */}
                    <Area
                      name="Forecast interval"
                      type="monotone"
                      dataKey="pubs_upper"
                      stroke="none"
                      fillOpacity={0.12}
                      fill="#10b981"
                      activeDot={false}
                      isAnimationActive={false}
                    />
                    <Area
                      type="monotone"
                      dataKey="pubs_lower"
                      stroke="none"
                      fillOpacity={0.12}
                      fill="#10b981"
                      activeDot={false}
                      isAnimationActive={false}
                    />
                    
                    {/* Forecast */}
                    <Line
                      name="Publications (forecast)"
                      type="monotone"
                      dataKey="pubs_forecast"
                      dot={false}
                      stroke="#10b981"
                      strokeWidth={2}
                      isAnimationActive={false}
                    />
                    
                    {/* Test actual values */}
                    <Scatter
                      name="Test (actual)"
                      dataKey="pubs_test_actual"
                      fill="#ef4444"
                      stroke="#ef4444"
                      strokeWidth={2}
                    />
                    
                    {/* Test predictions */}
                    <Scatter
                      name="Test (predicted)"
                      dataKey="pubs_test_pred"
                      fill="#3b82f6"
                      stroke="#3b82f6"
                      strokeWidth={2}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}
            
            <div className="mt-2 text-sm text-gray-800 grid sm:grid-cols-2 gap-4">
              <div className="space-y-1">
                <div className="font-semibold">Patents Metrics</div>
                <div>
                  MAE With pubs AR:&nbsp;
                  <span className="tabular-nums">
                    {data.metrics.mae_patents_with_reg.toFixed(2)}
                  </span>
                </div>
                <div>
                  RMSE With pubs AR:&nbsp;
                  <span className="tabular-nums">
                    {data.metrics.rmse_patents_with_reg.toFixed(2)}
                  </span>
                </div>
                {data.metrics.ampe_patents_with_reg && (
                  <div>
                    AMPE With pubs AR:&nbsp;
                    <span className="tabular-nums">
                      {data.metrics.ampe_patents_with_reg.toFixed(2)}
                    </span>
                  </div>
                )}
              </div>
              
              <div className="space-y-1">
                <div className="font-semibold">Publications Metrics</div>
                <div>
                  MAE:&nbsp;
                  <span className="tabular-nums">
                    {data.metrics.mae_pubs.toFixed(2)}
                  </span>
                </div>
                <div>
                  RMSE:&nbsp;
                  <span className="tabular-nums">
                    {data.metrics.rmse_pubs.toFixed(2)}
                  </span>
                </div>
                {data.metrics.ampe_pubs && (
                  <div>
                    AMPE:&nbsp;
                    <span className="tabular-nums">
                      {data.metrics.ampe_pubs.toFixed(2)}
                    </span>
                  </div>
                )}
                {typeof data.xcorr === "number" && (
                  <div>
                    Corr(pubs→patents): <span className="tabular-nums">{data.xcorr.toFixed(3)}</span>
                    {typeof data.best_lag === "number" && (
                      <> &nbsp;• Best lag: <span className="tabular-nums">{data.best_lag}</span> years</>
                    )}
                  </div>
                )}
              </div>
            </div>
            
            <div className="text-xs text-gray-500">
              Note: Black line shows historical values, green line shows forecasts, 
              red dots show actual test values, and blue dots show predicted test values.
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}