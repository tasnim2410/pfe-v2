"use client";

import { useEffect, useMemo, useState } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";

/**
 * Small helper: from arrays like [2022..2031, 2022..2031, 2022..2031]
 * detect the first repetition and keep only the first block.
 */
function uniqueBlock<T>(years: number[], values: T[]): { years: number[]; values: T[] } {
  if (!years || years.length === 0) return { years: [], values: [] };
  const first = years[0];
  let blockLen = years.length;
  for (let i = 1; i < years.length; i++) {
    if (years[i] === first) {
      blockLen = i;
      break;
    }
  }
  return {
    years: years.slice(0, blockLen),
    values: values.slice(0, blockLen),
  };
}

/**
 * Split a single series into two: Actual (<= actualUntilYear) and Forecast (> actualUntilYear).
 * We use null padding so Recharts draws two differently colored segments.
 */
function splitActualForecast(
  years: number[],
  values: number[],
  actualUntilYear: number
) {
  const rows = [];
  for (let i = 0; i < years.length; i++) {
    const y = years[i];
    if (y < actualUntilYear) {
      rows.push({ year: y, actual: values[i], forecast: null });
    } else if (y === actualUntilYear) {
      // At the boundary, show both actual and forecast for this year
      rows.push({ year: y, actual: values[i], forecast: values[i] });
    } else {
      rows.push({ year: y, actual: null, forecast: values[i] });
    }
  }
  return rows;
}

type SeriesPayload = { years: number[]; values: number[] };

type TVPModel = {
  p: number;
  historical: {
    years: number[];
    patent_count: number[];
    pub_count: number[];
  };
  forecast: {
    years: number[];
    patent_count: number[];
    pub_count: number[];
  };
};

type ApiPayload = {
  ok: boolean;
  data: {
    models: Record<string, TVPModel>;
    original_data: {
      years: number[];
      patent_count: number[];
      pub_count: number[];
    };
    // Backward compatibility
    patents: SeriesPayload;
    publications: SeriesPayload;
  };
};

export default function TVPTwoCharts({
  endpoint = "/api/tvp/forecast",
  actualUntilYear, // optional override; default = current year
  showModels = true, // whether to show separate models
}: {
  endpoint?: string;
  actualUntilYear?: number;
  showModels?: boolean;
}) {
  const [patents, setPatents] = useState<SeriesPayload | null>(null);
  const [pubs, setPubs] = useState<SeriesPayload | null>(null);
  const [models, setModels] = useState<Record<string, TVPModel>>({});
  const [selectedModel, setSelectedModel] = useState<string>("");
  const actualYear = useMemo(
    () => actualUntilYear ?? new Date().getFullYear(),
    [actualUntilYear]
  );

  useEffect(() => {
    (async () => {
      try {
        // Get backend port
        const portRes = await fetch("/backend_port.txt");
        const port = (await portRes.text()).trim();
        // Construct backend URL
        const backendUrl = `http://localhost:${port}/api/tvp/forecast`;
        const res = await fetch(backendUrl, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ "truncate_last_n": 4, "forecast_h": 10, "lags": [1, 2, 3] }),
        });
        const json: ApiPayload = await res.json();
        if (!json?.ok) throw new Error("Forecast API failed");

        // Handle new enhanced data structure
        if (json.data.models && Object.keys(json.data.models).length > 0) {
          setModels(json.data.models);
          
          // Set default selected model to first available
          const firstModelKey = Object.keys(json.data.models)[0];
          setSelectedModel(firstModelKey);
        }

        // Backward compatibility: use legacy data if available
        if (json.data.patents && json.data.publications) {
          const p = uniqueBlock(json.data.patents.years, json.data.patents.values);
          const r = uniqueBlock(json.data.publications.years, json.data.publications.values);
          setPatents(p);
          setPubs(r);
        }
      } catch (e) {
        console.error("TVPTwoCharts fetch error:", e);
        setPatents(null);
        setPubs(null);
      }
    })();
  }, []);

  // Data for selected model or fallback to legacy data
  function filterFrom2000<T>(years: number[], values: T[]): { years: number[]; values: T[] } {
    const filtered = years.reduce<{ years: number[]; values: T[] }>((acc, year, i) => {
      if (year >= 2010) {
        acc.years.push(year);
        acc.values.push(values[i]);
      }
      return acc;
    }, { years: [], values: [] });
    return filtered;
  }

  const currentModelData = useMemo(() => {
    if (showModels && selectedModel && models[selectedModel]) {
      const model = models[selectedModel];
      // Merge and filter from 2000
      const pat = filterFrom2000(
        [...model.historical.years, ...model.forecast.years],
        [...model.historical.patent_count, ...model.forecast.patent_count]
      );
      const pub = filterFrom2000(
        [...model.historical.years, ...model.forecast.years],
        [...model.historical.pub_count, ...model.forecast.pub_count]
      );
      return {
        patents: pat,
        publications: pub,
      };
    }
    // Legacy fallback: filter from 2000
    return {
      patents: patents ? filterFrom2000(patents.years, patents.values) : null,
      publications: pubs ? filterFrom2000(pubs.years, pubs.values) : null,
    };
  }, [showModels, selectedModel, models, patents, pubs]);


  const patentsRows = useMemo(() => {
    if (!currentModelData.patents) return [];
    return splitActualForecast(currentModelData.patents.years, currentModelData.patents.values, actualYear);
  }, [currentModelData.patents, actualYear]);

  const pubsRows = useMemo(() => {
    if (!currentModelData.publications) return [];
    return splitActualForecast(currentModelData.publications.years, currentModelData.publications.values, actualYear);
  }, [currentModelData.publications, actualYear]);

  return (
    <div className="space-y-6">
      {/* Model Selection */}
      {showModels && Object.keys(models).length > 0 && (
        <div className="flex flex-wrap gap-4 items-center">
          <span className="font-medium">TVP Model:</span>
          {Object.keys(models).map((modelKey) => {
            const model = models[modelKey];
            return (
              <button
                key={modelKey}
                onClick={() => setSelectedModel(modelKey)}
                className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                  selectedModel === modelKey
                    ? "bg-blue-500 text-white"
                    : "bg-gray-200 text-gray-700 hover:bg-gray-300"
                }`}
              >
                p={model.p}
              </button>
            );
          })}
        </div>
      )}


      {/* TVP Model Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Patents chart */}
        <div className="w-full h-[360px]">
          <h3 className="text-lg font-semibold mb-3">
            Patents {selectedModel && `(TVP p=${models[selectedModel]?.p})`}
          </h3>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={patentsRows}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="year" />
              <YAxis />
              <Tooltip />
              <Legend />
              {/* Actual segment */}
              <Line
                type="monotone"
                dataKey="actual"
                name={`Patents (≤ ${actualYear})`}
                stroke="#2C3E50"
                dot={false}
                strokeWidth={2}
                connectNulls={false}
              />
              {/* Forecast segment */}
              <Line
                type="monotone"
                dataKey="forecast"
                name={`Patents Forecast (> ${actualYear})`}
                stroke="#E67E22"
                dot={false}
                strokeWidth={2}
                connectNulls={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Publications chart */}
        <div className="w-full h-[360px]">
          <h3 className="text-lg font-semibold mb-3">
            Publications {selectedModel && `(TVP p=${models[selectedModel]?.p})`}
          </h3>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={pubsRows}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="year" />
              <YAxis />
              <Tooltip />
              <Legend />
              {/* Actual segment */}
              <Line
                type="monotone"
                dataKey="actual"
                name={`Publications (≤ ${actualYear})`}
                stroke="#2C3E50"
                dot={false}
                strokeWidth={2}
                connectNulls={false}
              />
              {/* Forecast segment */}
              <Line
                type="monotone"
                dataKey="forecast"
                name={`Publications Forecast (> ${actualYear})`}
                stroke="#2980B9"
                dot={false}
                strokeWidth={2}
                connectNulls={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
