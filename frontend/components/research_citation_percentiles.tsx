import React, { useEffect, useMemo, useState } from "react";
import {
  ResponsiveContainer, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend
} from "recharts";

type Row = {
  year: number;
  n_papers: number;
  p50: number;
  p90: number;
  p99: number;
  top10_threshold: number;
  top10_share_of_citations: number; // 0..1
};

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

const ResearchCitationPercentiles: React.FC = () => {
  const [data, setData] = useState<Row[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        setLoading(true);
        setErr(null);
        const portRes = await fetch("/backend_port.txt");
        const port = (await portRes.text()).trim();
        const res = await fetch(`http://localhost:${port}/api/research/citation_percentiles`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const json: Row[] = await res.json();
        if (!cancelled) setData(json.sort((a, b) => a.year - b.year));
      } catch (e: any) {
        if (!cancelled) setErr(e?.message || "Failed to load data.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, []);

  const yMax = useMemo(() => {
    if (!data?.length) return 1;
    const m = Math.max(...data.map(d => Math.max(d.p99, d.p90, d.p50)));
    return Math.ceil((m + 1));
  }, [data]);

  if (loading) return <div className="text-xs text-gray-500">Loading percentiles…</div>;
  if (err) return <div className="text-xs text-red-500">{err}</div>;
  if (!data?.length) return <div className="text-xs text-gray-500">No data.</div>;

  return (
    <div className="w-full">
      <div className="text-sm font-semibold mb-2">Citation Percentiles & Top-10% Share</div>
      <div className="w-full h-[320px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 10, right: 40, bottom: 10, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="year" tick={{ fontSize: 12 }} />
            <YAxis yAxisId="left" domain={[0, yMax]} tick={{ fontSize: 12 }} />
            <YAxis yAxisId="right" orientation="right" domain={[0, 1]} tickFormatter={(v) => `${Math.round(v*100)}%`} tick={{ fontSize: 12 }} />
            <Tooltip
              content={({ active, payload }) => {
                if (!active || !payload?.length) return null;
                const p = payload[0].payload as Row;
                return (
                  <div className="rounded-md border bg-white px-3 py-2 text-xs shadow">
                    <div className="font-semibold mb-1">Year: {p.year}</div>
                    <div>P50: <b>{p.p50.toFixed(1)}</b></div>
                    <div>P90: <b>{p.p90.toFixed(1)}</b> (Top-10% threshold)</div>
                    <div>P99: <b>{p.p99.toFixed(1)}</b></div>
                    <div className="mt-1 text-[11px] text-gray-500">Top-10% share: <b>{pct(p.top10_share_of_citations)}</b> (n={p.n_papers})</div>
                  </div>
                );
              }}
            />
            <Legend />
            <Line yAxisId="left" type="monotone" dataKey="p50" name="P50" stroke="#4b5563" strokeWidth={2} dot={{ r: 2 }} />
            <Line yAxisId="left" type="monotone" dataKey="p90" name="P90 (Top-10% threshold)" stroke="#6b8b1e" strokeWidth={2} dot={{ r: 2 }} />
            <Line yAxisId="left" type="monotone" dataKey="p99" name="P99" stroke="#b45309" strokeWidth={2} dot={{ r: 2 }} />
            <Line yAxisId="right" type="monotone" dataKey="top10_share_of_citations" name="Top-10% Share" stroke="#2f6fb1" strokeWidth={2} dot={{ r: 2 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <p className="text-[11px] text-gray-500 mt-2">
        P90 is the **top-10%** threshold; “Top-10% Share” is what fraction of total citations those papers capture.
      </p>
    </div>
  );
};

export default ResearchCitationPercentiles;
