import React, { useEffect, useMemo, useState } from "react";
import {
  ResponsiveContainer, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend
} from "recharts";

type Row = {
  year: number;
  n_papers: number;
  total_citations: number;
  mean_citations: number;
  median_citations: number;
  mean_citations_per_year: number;
  median_citations_per_year: number;
};

const ResearchCitationsPerYearStats: React.FC = () => {
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
        const res = await fetch(`http://localhost:${port}/api/research/citations_per_year_stats`);
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
    const m = Math.max(
      ...data.map(d => Math.max(d.mean_citations_per_year, d.median_citations_per_year))
    );
    return Math.ceil((m + 0.1) * 10) / 10;
  }, [data]);

  if (loading) return <div className="text-xs text-gray-500">Loading CPY…</div>;
  if (err) return <div className="text-xs text-red-500">{err}</div>;
  if (!data?.length) return <div className="text-xs text-gray-500">No data.</div>;

  return (
    <div className="w-full">
      <div className="text-sm font-semibold mb-2">Citations per Year (Age-normalized)</div>
      <div className="w-full h-[320px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 10, right: 20, bottom: 10, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="year" tick={{ fontSize: 12 }} />
            <YAxis tick={{ fontSize: 12 }} domain={[0, yMax]} />
            <Tooltip
              content={({ active, payload }) => {
                if (!active || !payload?.length) return null;
                const p = payload[0].payload as Row;
                return (
                  <div className="rounded-md border bg-white px-3 py-2 text-xs shadow">
                    <div className="font-semibold mb-1">Year: {p.year}</div>
                    <div>Mean CPY: <b>{p.mean_citations_per_year.toFixed(2)}</b></div>
                    <div>Median CPY: <b>{p.median_citations_per_year.toFixed(2)}</b></div>
                    <div className="mt-1 text-[11px] text-gray-500">
                      n={p.n_papers}, total citations={p.total_citations}
                    </div>
                  </div>
                );
              }}
            />
            <Legend />
            <Line type="monotone" dataKey="mean_citations_per_year" name="Mean CPY" stroke="#6b8b1e" strokeWidth={2} dot={{ r: 2 }} />
            <Line type="monotone" dataKey="median_citations_per_year" name="Median CPY" stroke="#2f6fb1" strokeWidth={2} dot={{ r: 2 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <p className="text-[11px] text-gray-500 mt-2">
        CPY = total citations / (current_year − publication_year + 1). Levels the playing field across ages.
      </p>
    </div>
  );
};

export default ResearchCitationsPerYearStats;
