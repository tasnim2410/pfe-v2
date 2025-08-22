"use client";

import React, { useEffect, useMemo, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  ReferenceLine,
  Legend,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

/** ---------- Types that match your API shape ---------- */
type ForecastResponse = {
  ok: boolean;
  data?: {
    ok: boolean;
    model: string;
    order: number[]; // e.g., [p,d,q]
    metrics?: { MAE?: number; RMSE?: number };
    original_data: {
      years: number[];
      patent_count: number[];
      pub_count?: number[];
    };
    test?: {
      years: number[];
      yhat: number[];
      yhat_lower?: number[];
      yhat_upper?: number[];
    };
    train?: {
      years: number[];
      fitted: number[];
    };
  };
  stderr?: string;
};

type ChartRow = {
  year: number;
  actual?: number;   // original_data.patent_count (black line)
  forecast?: number; // test.yhat (colored dashed line) — only for test years
  lower?: number;    // yhat_lower — only for test years
  upper?: number;    // yhat_upper — only for test years
};

/** Small formatter for axis/tooltip */
const fmt = (n?: number) => (typeof n === "number" ? Math.round(n) : n);

/**
 * ARIMAX Forecast (Patents)
 *
 * What it does:
 * - Calls /api/arimax/forecast on your backend (port read from /backend_port.txt).
 * - Plots ORIGINAL series (original_data.patent_count) in BLACK.
 * - Plots TEST predictions (test.years + test.yhat) in ACCENT COLOR, only on those years.
 * - Starts the chart at year 2000 for readability.
 * - If yhat_lower/yhat_upper exist, shows a light confidence band for test years.
 */
export default function ArimaxForecast({
  className,
  title = "ARIMAX Forecast (Patents)",
}: {
  className?: string;
  title?: string;
}) {
  const [data, setData] = useState<ForecastResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    const run = async () => {
      setLoading(true);
      setErr(null);
      try {
        // Read backend port written by your server tooling
        const portRes = await fetch("/backend_port.txt");
        const port = (await portRes.text()).trim();

        // Call the ARIMAX endpoint (no body needed per your example)
        const res = await fetch(`http://localhost:${port}/api/arimax/forecast`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          cache: "no-store",
        });

        const json: ForecastResponse = await res.json();
        if (!json?.ok && !json?.data?.ok) setErr("Backend returned not ok.");
        setData(json);
      } catch (e: any) {
        setErr(e?.message || "Failed to load forecast.");
      } finally {
        setLoading(false);
      }
    };
    run();
  }, []);

  const {
    rows,
    currentYear,
    mae,
    rmse,
    modelName,
    order,
  } = useMemo(() => {
    const now = new Date().getFullYear();

    const d = data?.data;
    const originalYears = d?.original_data?.years || [];
    const originalCounts = d?.original_data?.patent_count || [];

    const testYears = d?.test?.years || [];
    const yhat = d?.test?.yhat || [];
    const yL = d?.test?.yhat_lower || [];
    const yU = d?.test?.yhat_upper || [];

    // --- Maps for quick lookup ---
    const actualMap = new Map<number, number>();          // original_data Actuals
    originalYears.forEach((y, i) => actualMap.set(y, originalCounts[i]));

    const forecastMap = new Map<number, number>();        // test.yhat (Forecast)
    const lowerMap = new Map<number, number>();           // test.yhat_lower
    const upperMap = new Map<number, number>();           // test.yhat_upper
    testYears.forEach((y, i) => {
      forecastMap.set(y, yhat[i]);
      if (yL?.length) lowerMap.set(y, yL[i]);
      if (yU?.length) upperMap.set(y, yU[i]);
    });
    const testYearSet = new Set<number>(testYears);

    // --- Build rows, starting at 2000 ---
    const startYear = 2000;
    const allYears = Array.from(
      new Set<number>([...originalYears, ...testYears].filter((y) => y >= startYear))
    ).sort((a, b) => a - b);

    const rows: ChartRow[] = allYears.map((year) => ({
      year,
      // Always show the original series in black wherever it exists
      actual: actualMap.get(year),
      // Only show forecast line on test years
      forecast: testYearSet.has(year) ? forecastMap.get(year) : undefined,
      lower: testYearSet.has(year) ? lowerMap.get(year) : undefined,
      upper: testYearSet.has(year) ? upperMap.get(year) : undefined,
    }));

    return {
      rows,
      currentYear: now,
      mae: d?.metrics?.MAE,
      rmse: d?.metrics?.RMSE,
      modelName: d?.model,
      order: d?.order,
    };
  }, [data]);

  if (loading) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle>{title}</CardTitle>
        </CardHeader>
        <CardContent className="text-sm text-gray-500">Loading forecast…</CardContent>
      </Card>
    );
  }

  if (err) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle>{title}</CardTitle>
        </CardHeader>
        <CardContent className="text-sm text-red-600">{err}</CardContent>
      </Card>
    );
  }

  const info = (
    <div className="text-xs text-gray-500 space-x-4">
      {modelName && <span>Model: <b>{modelName}</b></span>}
      {order && <span>Order: <b>({order.join(", ")})</b></span>}
      {typeof mae === "number" && <span>MAE: <b>{mae.toFixed(2)}</b></span>}
      {typeof rmse === "number" && <span>RMSE: <b>{rmse.toFixed(2)}</b></span>}
    </div>
  );

  return (
    <Card className={className}>
      <CardHeader className="flex flex-col gap-1">
        <CardTitle>{title}</CardTitle>
        {info}
      </CardHeader>
      <CardContent className="h-[360px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={rows} margin={{ top: 10, right: 20, bottom: 10, left: 0 }}>
            {/* Grid + axes */}
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="year" tickFormatter={(v) => String(v)} interval="preserveStartEnd" />
            <YAxis tickFormatter={(v) => String(fmt(v))} />

            {/* Tooltip + legend */}
            <Tooltip
              formatter={(value: any, name) => [
                fmt(value),
                name === "actual" ? "Original (Actual)" :
                name === "forecast" ? "Forecast (Test yhat)" : name
              ]}
              labelFormatter={(label) => `Year: ${label}`}
            />
            <Legend />

            {/* Optional vertical reference at current year (visual cue only) */}
            <ReferenceLine x={currentYear} stroke="#999" strokeDasharray="4 4" label={`Now (${currentYear})`} />

            {/* Confidence band (only on test years, because others are undefined) */}
            <Area
              type="monotone"
              dataKey="upper"
              strokeOpacity={0}
              fillOpacity={0.12}
              activeDot={false}
              isAnimationActive={false}
              name="Upper 95%"
            />
            <Area
              type="monotone"
              dataKey="lower"
              strokeOpacity={0}
              fillOpacity={0.12}
              activeDot={false}
              isAnimationActive={false}
              name="Lower 95%"
            />

            {/* Original series in solid black */}
            <Line
              type="monotone"
              dataKey="actual"
              name="Original (Actual)"
              stroke="#111"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />

            {/* Forecast (test.yhat) in dashed accent color, only on test years */}
            <Line
              type="monotone"
              dataKey="forecast"
              name="Forecast (Test yhat)"
              stroke="#2563eb"
              strokeWidth={2}
              strokeDasharray="4 2"
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>

        {/* Footnote */}
        <div className="mt-2 text-[11px] text-gray-500">
          Note: Shaded area shows the forecast interval if provided by the API (only on test years).
        </div>
      </CardContent>
    </Card>
  );
}
