import React, { useEffect, useMemo, useState } from "react";
import {
  ResponsiveContainer,
  LineChart, Line,
  XAxis, YAxis,
  CartesianGrid, Tooltip,
  ReferenceLine, ReferenceArea,
  Legend
} from "recharts";
import LoadingSpinner from "./LoadingSpinner";

/** Types that match the API payload */
type DivergenceItem = { from_year: number; to_year: number; divergence: number };
type WindowItem     = { start: number; end: number };
type ApiResponse = {
  divergence_data: DivergenceItem[];
  threshold: number;
  windows: WindowItem[];
};

const fmt = (v: number) => (typeof v === "number" ? v.toFixed(3) : "");

/**
 * Convert API divergence data to chart points.
 * We place each point at the "to_year" and keep a label "from–to" for the tooltip.
 */
function toSeries(divs: DivergenceItem[]) {
  const dedupKey = new Set<string>();
  const arr = divs
    .map(d => ({ year: d.to_year, label: `${d.from_year}–${d.to_year}`, divergence: d.divergence }))
    .filter(p => {
      const k = `${p.year}-${p.divergence.toFixed(6)}`;
      if (dedupKey.has(k)) return false;
      dedupKey.add(k);
      return true;
    })
    .sort((a, b) => a.year - b.year);
  return arr;
}

const ResearchTopicShiftTrend: React.FC = () => {
  const [data, setData] = useState<ApiResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [shade, setShade] = useState(true); // toggle to show/hide window shading

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        setLoading(true);
        setErr(null);

        // We use the same pattern as your other components: read backend port from a small file.
        const portRes = await fetch("/backend_port.txt");
        const port = (await portRes.text()).trim();

        const res = await fetch(`http://localhost:${port}/api/research/automatic_topic_shift`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const json: ApiResponse = await res.json();

        if (!cancelled) setData(json);
      } catch (e: any) {
        if (!cancelled) setErr(e?.message || "Failed to load topic shift data.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, []);

  // Prepare the chart series once, even if data is undefined (safe fallback).
  const series = useMemo(() => toSeries(data?.divergence_data || []), [data?.divergence_data]);
  const minYear = series.length ? Math.min(...series.map(s => s.year)) : 0;
  const maxYear = series.length ? Math.max(...series.map(s => s.year)) : 0;
  const yMax = series.length ? Math.max(1, Math.ceil((Math.max(...series.map(s => s.divergence)) + 0.05) * 100) / 100) : 1;

  if (loading) return <LoadingSpinner text="Loading Topic Shift..." />;
  if (err) return <div className="text-sm text-red-500">{err}</div>;
  if (!data || !data.divergence_data?.length)
    return <div className="text-sm text-gray-500">No topic-shift data available.</div>;

  return (
    <div className="w-full">
      {/* Small header: threshold + toggle */}
      <div className="flex items-center justify-between mb-2">
        <div className="text-xs text-gray-600">
          Threshold: <span className="font-semibold">{fmt(data.threshold)}</span>
        </div>
        <label className="flex items-center gap-2 text-xs text-gray-600 select-none cursor-pointer">
          <input type="checkbox" checked={shade} onChange={(e) => setShade(e.target.checked)} />
          Shade windows
        </label>
      </div>

      <div className="w-full h-[320px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={series} margin={{ top: 10, right: 20, bottom: 10, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="year"
              type="number"
              domain={[minYear, maxYear]}
              tickCount={8}
              tick={{ fontSize: 12 }}
            />
            <YAxis
              domain={[0, yMax]}
              tickFormatter={v => v.toFixed(2)}
              tick={{ fontSize: 12 }}
            />
            <Tooltip
              content={({ active, payload, label }) => {
                if (!active || !payload?.length) return null;
                const p = payload[0].payload;
                return (
                  <div className="rounded-md border bg-white px-3 py-2 text-xs shadow">
                    <div className="font-semibold mb-1">Year: {p.year}</div>
                    <div>Span: {p.label}</div>
                    <div>Divergence: <span className="font-semibold">{fmt(p.divergence)}</span></div>
                  </div>
                );
              }}
            />
            <Legend verticalAlign="top" height={20} />
            {/* Horizontal threshold line */}
            <ReferenceLine y={data.threshold} stroke="#94a3b8" strokeDasharray="4 4" label={{ value: "Threshold", position: "right", fontSize: 10 }} />
            {/* Optional: shade windows (light background bands) */}
            {shade && data.windows?.map((w, idx) => (
              <ReferenceArea
                key={`${w.start}-${w.end}-${idx}`}
                x1={w.start}
                x2={w.end}
                y1={0}
                y2={yMax}
                ifOverflow="extendDomain"
                fill="#d9e6b9"
                fillOpacity={0.22}
                strokeOpacity={0}
              />
            ))}
            <Line
              type="monotone"
              dataKey="divergence"
              name="Jensen–Shannon divergence"
              stroke="#6b8b1e"
              strokeWidth={2}
              dot={{ r: 2 }}
              activeDot={{ r: 4 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <p className="text-[11px] text-gray-500 mt-2">
        The line shows the year-to-year Jensen–Shannon divergence of vocabulary distributions.
        Bands indicate detected windows returned by the API.
      </p>
    </div>
  );
};

export default ResearchTopicShiftTrend;
