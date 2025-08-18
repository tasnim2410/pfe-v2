import React, { useEffect, useMemo, useState } from "react";
import {
  ResponsiveContainer, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend
} from "recharts";

type Row = {
  year: number;
  n_papers: number;
  gini: number; // 0..1
  top10_share_of_citations: number; // 0..1
  top1_share_of_citations: number;  // 0..1
};

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

const ResearchCitationInequality: React.FC = () => {
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
        const res = await fetch(`http://localhost:${port}/api/research/citation_inequality`);
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

  const years = useMemo(() => data?.map(d => d.year) ?? [], [data]);

  if (loading) return <div className="text-xs text-gray-500">Loading inequality…</div>;
  if (err) return <div className="text-xs text-red-500">{err}</div>;
  if (!data?.length) return <div className="text-xs text-gray-500">No data.</div>;

  return (
    <div className="w-full">
      <div className="text-sm font-semibold mb-2">Citation Inequality (Gini & Pareto Shares)</div>
      <div className="w-full h-[320px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 10, right: 20, bottom: 10, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="year" tick={{ fontSize: 12 }} />
            <YAxis domain={[0, 1]} tick={{ fontSize: 12 }} tickFormatter={(v) => `${Math.round(v*100)}%`} />
            <Tooltip
              content={({ active, payload }) => {
                if (!active || !payload?.length) return null;
                const p = payload[0].payload as Row;
                return (
                  <div className="rounded-md border bg-white px-3 py-2 text-xs shadow">
                    <div className="font-semibold mb-1">Year: {p.year}</div>
                    <div>Gini: <b>{pct(p.gini)}</b></div>
                    <div>Top-10% share: <b>{pct(p.top10_share_of_citations)}</b></div>
                    <div>Top-1% share: <b>{pct(p.top1_share_of_citations)}</b></div>
                    <div className="mt-1 text-[11px] text-gray-500">n={p.n_papers}</div>
                  </div>
                );
              }}
            />
            <Legend />
            <Line type="monotone" dataKey="gini" name="Gini" stroke="#6b8b1e" strokeWidth={2} dot={{ r: 2 }} />
            <Line type="monotone" dataKey="top10_share_of_citations" name="Top-10% Share" stroke="#2f6fb1" strokeWidth={2} dot={{ r: 2 }} />
            <Line type="monotone" dataKey="top1_share_of_citations" name="Top-1% Share" stroke="#b45309" strokeWidth={2} dot={{ r: 2 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <p className="text-[11px] text-gray-500 mt-2">
        Gini and Pareto shares (Top-10% / Top-1%) summarize how concentrated citations are each year.
      </p>
    </div>
  );
};

export default ResearchCitationInequality;
