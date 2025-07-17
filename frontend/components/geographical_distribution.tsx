// components/geographical_distribution.tsx
"use client";

import React, { useEffect, useState } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import LoadingSpinner from "./LoadingSpinner";

type ApiResp = { labels: string[]; datasets: { data: number[] }[] };

export default function GeographicalDistribution() {
  const [rows, setRows] = useState<{ iso: string; count: number }[] | null>(null);
  const [err,  setErr]  = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const port = (await (await fetch("/backend_port.txt")).text()).trim();
        const json: ApiResp = await fetch(
          `http://localhost:${port}/api/geographical_distribution`
        ).then(r => r.json());

        const data = json.labels.map((iso, i) => ({
          iso,
          count: json.datasets[0].data[i] ?? 0,
        }));
        setRows(data);
      } catch {
        setErr("Error loading geographical data.");
      }
    })();
  }, []);

  if (err)     return <div className="text-red-500">{err}</div>;
  if (!rows)   return <LoadingSpinner text="Loading chart…" size={28} />;

  return (
    <div className="w-full h-[340px]">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={rows} margin={{ top: 10, right: 30, left: 0, bottom: 20 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="iso" />
          <YAxis allowDecimals={false} />
          <Tooltip />
          <Bar dataKey="count" fill="#3182bd" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
