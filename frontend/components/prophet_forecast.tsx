



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
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

// Types
type HistoryPoint = { year: number; count: number };
type ForecastPoint = { year: number; yhat: number; yhat_lower?: number; yhat_upper?: number };

type ProphetResponse = {
  best_lag: number;
  tech?: string;
  xcorr?: number;
  metrics: {
    mae_patents_baseline: number;
    rmse_patents_baseline: number;
    mae_patents_with_reg: number;
    rmse_patents_with_reg: number;
    mae_pubs: number;
    rmse_pubs: number;
  };
  patents: {
    history: HistoryPoint[];
    forecast: {
      baseline: ForecastPoint[];
      with_pub_ar: ForecastPoint[];
    };
  };
  publications: {
    history: HistoryPoint[];
    forecast: ForecastPoint[];
  };
};

type ChartDataPoint = {
  year: number;
  // Patents data
  patents_history?: number;
  patents_baseline?: number;
  patents_with_pub_ar?: number;
  // Publications data
  pubs_history?: number;
  pubs_forecast?: number;
  pubs_lower?: number;
  pubs_upper?: number;
};

const CURRENT_YEAR = 2025; // Hardcoded to current year

function valueFmt(v?: number) {
  return typeof v === "number" ? v.toFixed(2) : v;
}

export default function ProphetForecast() {
  const [data, setData] = useState<ProphetResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [tab, setTab] = useState<"patents" | "publications">("patents");
  const [showBaseline, setShowBaseline] = useState(true);
  const [showWithReg, setShowWithReg] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [horizon, setHorizon] = useState(7); // Default horizon value
  const [pubTail, setPubTail] = useState(1); // Default pub_tail value
  const [patTail, setPatTail] = useState(1); // Default pat_tail value

  // Fetch data from API with all parameters
  const fetchData = async (horizonValue: number, pubTailValue: number, patTailValue: number) => {
    try {
      setLoading(true);
      setError(null);
      const portRes = await fetch("/backend_port.txt");
      const port = (await portRes.text()).trim();
      const backendUrl = `http://localhost:${port}/api/prophet_forecast`;
      const res = await fetch(backendUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          horizon: horizonValue, 
          pub_tail: pubTailValue, 
          pat_tail: patTailValue 
        }),
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

  // Handle input changes
  const handleHorizonChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseInt(e.target.value);
    if (!isNaN(value) && value > 0 && value <= 20) {
      setHorizon(value);
    }
  };

  const handlePubTailChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseInt(e.target.value);
    if (!isNaN(value) && value >= 0 && value <= 10) {
      setPubTail(value);
    }
  };

  const handlePatTailChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseInt(e.target.value);
    if (!isNaN(value) && value >= 0 && value <= 10) {
      setPatTail(value);
    }
  };

  // Handle forecast button click
  const handleForecast = () => {
    fetchData(horizon, pubTail, patTail);
  };

  // Process data for charts - ensure forecast connects to history
  const { patentsData, publicationsData } = useMemo(() => {
    if (!data) return { patentsData: [], publicationsData: [] };

    // Find the last historical year for patents and publications
    const lastPatentsHistoryYear = Math.max(...data.patents.history.map(d => d.year));
    const lastPublicationsHistoryYear = Math.max(...data.publications.history.map(d => d.year));

    // Process patents data - include the last historical point in forecast data
    const patentsHistory = data.patents.history.map(item => ({
      year: item.year,
      patents_history: item.count,
    }));

    // Get the last historical value to connect forecasts
    const lastPatentsValue = data.patents.history.find(d => d.year === lastPatentsHistoryYear)?.count || 0;

    // Create connection points for seamless transition
    const patentsForecastBaseline = [
      // Add connection point at last historical year
      {
        year: lastPatentsHistoryYear,
        patents_baseline: lastPatentsValue,
      },
      // Add all forecast points starting from the next year
      ...data.patents.forecast.baseline
        .filter(item => item.year > lastPatentsHistoryYear)
        .map(item => ({
          year: item.year,
          patents_baseline: item.yhat,
        }))
    ];

    const patentsForecastWithAR = [
      // Add connection point at last historical year
      {
        year: lastPatentsHistoryYear,
        patents_with_pub_ar: lastPatentsValue,
      },
      // Add all forecast points starting from the next year
      ...data.patents.forecast.with_pub_ar
        .filter(item => item.year > lastPatentsHistoryYear)
        .map(item => ({
          year: item.year,
          patents_with_pub_ar: item.yhat,
        }))
    ];

    // Process publications data - include the last historical point in forecast data
    const pubsHistory = data.publications.history.map(item => ({
      year: item.year,
      pubs_history: item.count,
    }));

    // Get the last historical value to connect forecasts
    const lastPubsValue = data.publications.history.find(d => d.year === lastPublicationsHistoryYear)?.count || 0;

    const pubsForecast = [
      // Add connection point at last historical year
      {
        year: lastPublicationsHistoryYear,
        pubs_forecast: lastPubsValue,
        pubs_lower: lastPubsValue,
        pubs_upper: lastPubsValue,
      },
      // Add all forecast points starting from the next year
      ...data.publications.forecast
        .filter(item => item.year > lastPublicationsHistoryYear)
        .map(item => ({
          year: item.year,
          pubs_forecast: item.yhat,
          pubs_lower: item.yhat_lower,
          pubs_upper: item.yhat_upper,
        }))
    ];

    // Combine data
    const allYears = new Set([
      ...patentsHistory.map(d => d.year),
      ...patentsForecastBaseline.map(d => d.year),
      ...patentsForecastWithAR.map(d => d.year),
      ...pubsHistory.map(d => d.year),
      ...pubsForecast.map(d => d.year),
    ]);

    const patentsData: ChartDataPoint[] = [];
    const publicationsData: ChartDataPoint[] = [];

    Array.from(allYears).sort().forEach(year => {
      // Patents data point
      const patentsPoint: ChartDataPoint = { year };
      const historyPt = patentsHistory.find(d => d.year === year);
      const baselinePt = patentsForecastBaseline.find(d => d.year === year);
      const withARPt = patentsForecastWithAR.find(d => d.year === year);

      if (historyPt) patentsPoint.patents_history = historyPt.patents_history;
      if (baselinePt) patentsPoint.patents_baseline = baselinePt.patents_baseline;
      if (withARPt) patentsPoint.patents_with_pub_ar = withARPt.patents_with_pub_ar;

      if (historyPt || baselinePt || withARPt) {
        patentsData.push(patentsPoint);
      }

      // Publications data point
      const pubsPoint: ChartDataPoint = { year };
      const pubsHistoryPt = pubsHistory.find(d => d.year === year);
      const pubsForecastPt = pubsForecast.find(d => d.year === year);

      if (pubsHistoryPt) pubsPoint.pubs_history = pubsHistoryPt.pubs_history;
      if (pubsForecastPt) {
        pubsPoint.pubs_forecast = pubsForecastPt.pubs_forecast;
        pubsPoint.pubs_lower = pubsForecastPt.pubs_lower;
        pubsPoint.pubs_upper = pubsForecastPt.pubs_upper;
      }

      if (pubsHistoryPt || pubsForecastPt) {
        publicationsData.push(pubsPoint);
      }
    });

    return { patentsData, publicationsData };
  }, [data]);

  return (
    <Card className="w-full">
      <CardHeader className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <CardTitle>Prophet Forecast</CardTitle>
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-2">
            <Label htmlFor="horizon" className="text-sm whitespace-nowrap">
              Forecast Horizon:
            </Label>
            <Input
              id="horizon"
              type="number"
              min="1"
              max="20"
              value={horizon}
              onChange={handleHorizonChange}
              className="w-16"
            />
            <span className="text-sm whitespace-nowrap">years</span>
          </div>
          <div className="flex items-center gap-2">
            <Label htmlFor="pub-tail" className="text-sm whitespace-nowrap">
              Pub Tail:
            </Label>
            <Input
              id="pub-tail"
              type="number"
              min="0"
              max="10"
              value={pubTail}
              onChange={handlePubTailChange}
              className="w-16"
            />
          </div>
          <div className="flex items-center gap-2">
            <Label htmlFor="pat-tail" className="text-sm whitespace-nowrap">
              Pat Tail:
            </Label>
            <Input
              id="pat-tail"
              type="number"
              min="0"
              max="10"
              value={patTail}
              onChange={handlePatTailChange}
              className="w-16"
            />
          </div>
          <Button 
            onClick={handleForecast} 
            disabled={loading}
            className="bg-blue-600 hover:bg-blue-700"
          >
            {loading ? "Loading..." : "Generate Forecast"}
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
            {tab === "patents" && (
              <div className="flex items-center gap-3 pl-2">
                <label className="flex items-center gap-1 text-sm">
                  <input
                    type="checkbox"
                    checked={showBaseline}
                    onChange={(e) => setShowBaseline(e.target.checked)}
                  />
                  Baseline
                </label>
                <label className="flex items-center gap-1 text-sm">
                  <input
                    type="checkbox"
                    checked={showWithReg}
                    onChange={(e) => setShowWithReg(e.target.checked)}
                  />
                  With pubs AR
                </label>
              </div>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {loading && <div className="text-sm text-gray-500">Loading forecast…</div>}
        {error && <div className="text-sm text-red-600">Error: {error}</div>}
        {!loading && !error && data && (
          <>
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
                    <Line
                      name="Patents (history)"
                      type="monotone"
                      dataKey="patents_history"
                      dot={false}
                      stroke="#111827"
                      strokeWidth={2}
                      isAnimationActive={false}
                    />
                    {showBaseline && (
                      <Line
                        name="Patents (baseline)"
                        type="monotone"
                        dataKey="patents_baseline"
                        dot={false}
                        stroke="#2563eb"
                        strokeWidth={2}
                        isAnimationActive={false}
                      />
                    )}
                    {showWithReg && (
                      <Line
                        name="Patents (with pubs AR)"
                        type="monotone"
                        dataKey="patents_with_pub_ar"
                        dot={false}
                        stroke="#f59e0b"
                        strokeWidth={2}
                        isAnimationActive={false}
                      />
                    )}
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
                    <Line
                      name="Publications (history)"
                      type="monotone"
                      dataKey="pubs_history"
                      dot={false}
                      stroke="#111827"
                      strokeWidth={2}
                      isAnimationActive={false}
                    />
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
                    <Line
                      name="Publications (forecast)"
                      type="monotone"
                      dataKey="pubs_forecast"
                      dot={false}
                      stroke="#10b981"
                      strokeWidth={2}
                      isAnimationActive={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}
            <div className="mt-2 text-sm text-gray-800 grid sm:grid-cols-2 gap-2">
              <div className="space-y-1">
                <div className="font-semibold">Patents (MAE / RMSE)</div>
                <div>
                  Baseline:&nbsp;
                  <span className="tabular-nums">
                    {data.metrics.mae_patents_baseline.toFixed(2)} /{" "}
                    {data.metrics.rmse_patents_baseline.toFixed(2)}
                  </span>
                </div>
                <div>
                  With pubs AR:&nbsp;
                  <span className="tabular-nums">
                    {data.metrics.mae_patents_with_reg.toFixed(2)} /{" "}
                    {data.metrics.rmse_patents_with_reg.toFixed(2)}
                  </span>
                </div>
              </div>
              <div className="space-y-1">
                <div className="font-semibold">Publications (MAE / RMSE)</div>
                <div className="tabular-nums">
                  {data.metrics.mae_pubs.toFixed(2)} / {data.metrics.rmse_pubs.toFixed(2)}
                </div>
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
              Note: Forecast lines are connected to historical data for a continuous visualization.
              Forecasting {horizon} years ahead.
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}