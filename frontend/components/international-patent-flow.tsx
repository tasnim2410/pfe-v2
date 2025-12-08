import React, { useEffect, useState } from "react";
import {
  Chart,
  BarElement,
  CategoryScale,
  LinearScale,
  Tooltip,
  Legend,
} from "chart.js";
import { Bar } from "react-chartjs-2";
import { schemeTableau10, schemeSet3 } from "d3-scale-chromatic";

/* register once */
Chart.register(BarElement, CategoryScale, LinearScale, Tooltip, Legend);

/* ----- API types ----- */
interface ApiResponse {
  origins: string[];    // columns (US, CN, …)
  receivers: string[];  // rows    (JP, CN, …)
  matrix: number[][];   // receiver x origin
}

/* ----- colour helper (enough unique colours) ----- */
const PALETTE = [...schemeTableau10, ...schemeSet3].flat();
const color = (i: number) => PALETTE[i % PALETTE.length];

/* ---------- component ---------- */
export const InternationalPatentFlowChart: React.FC<{ port?: number }> = ({
  port: overridePort,
}) => {
  const [api, setApi]     = useState<ApiResponse | null>(null);
  const [loading, setLoad] = useState(true);
  const [error, setErr]   = useState<string | null>(null);

  /* ─ fetch once ─ */
  useEffect(() => {
    let dead = false;
    (async () => {
      try {
        /* 1️⃣ port autodetect */
        let port = overridePort;
        if (!port) {
          const txt = await fetch("/backend_port.txt").then(r => r.text()).catch(() => "");
          const n   = parseInt(txt.trim(), 10);
          port      = Number.isFinite(n) ? n : 49473;
        }

        /* 2️⃣ data */
        const res = await fetch(`http://localhost:${port}/api/international_patent_flow`);
        if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`);
        const json: ApiResponse = await res.json();
        if (!dead) setApi(json);
      } catch (e: any) {
        if (!dead) setErr(e.message ?? String(e));
      } finally {
        if (!dead) setLoad(false);
      }
    })();
    return () => { dead = true; };
  }, [overridePort]);

  if (loading) return <div>Loading international patent flow…</div>;
  if (error)   return <div style={{ color: "#EA3C53" }}>{error}</div>;
  if (!api)    return null;

  /* ----- Filter to top 10 receivers by total patents ----- */
  const rowTotals = api.receivers.map((_, rowIdx) => 
    api.matrix[rowIdx]?.reduce((sum, val) => sum + val, 0) ?? 0
  );
  
  // Get indices sorted by total (descending), take top 10
  const sortedIndices = api.receivers
    .map((_, idx) => idx)
    .filter(idx => rowTotals[idx] > 0)
    .sort((a, b) => rowTotals[a] - rowTotals[b])  // ascending for bottom-to-top display
    .slice(-10);  // take top 10 (last 10 after ascending sort)
  
  const filteredReceivers = sortedIndices.map(idx => api.receivers[idx]);
  const filteredMatrix = sortedIndices.map(idx => api.matrix[idx]);

  /* ----- chart.js datasets ----- */
  const datasets = api.origins.map((origin, col) => ({
    label: origin,
    data : filteredReceivers.map((_, row) => filteredMatrix[row]?.[col] ?? 0),
    backgroundColor: color(col),
    borderWidth: 0,
  }));

  const data = { labels: filteredReceivers, datasets };

  const options: any = {
    indexAxis: "y" as const,
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: { 
        stacked: true, 
        reverse: true,  // Bars grow from right to left (mirror of protection matrix)
        title: { display: true, text: "Total Patents Received" }, 
        ticks: { color: "#3B3C3D" } 
      },
      y: { 
        stacked: true, 
        position: "right",  // Y-axis labels on the right side
        title: { display: true, text: "Receiving Country" }, 
        ticks: { color: "#3B3C3D", font: { weight: 600 } } 
      },
    },
    plugins: {
      tooltip: {
        mode: "nearest",
        intersect: false,
        callbacks: {
          label: (ctx: any) => ` ${ctx.dataset.label} → ${ctx.parsed.x}`,
        },
      },
      legend: { position: "right", labels: { boxWidth: 14 } },
    },
  };

  /* ----- card wrapper ----- */
  return (
    <div
      style={{
        background: "#fff",
        borderRadius: 18,
        boxShadow: "0 2px 18px #B2DBA422",
        padding: "0 16px 16px 16px",
        width: "100%",
        display: "flex",
        flexDirection: "column",
      }}
    >
      {/* <div
        style={{
          marginTop: 12,
          padding: "8px 28px",
          fontWeight: 700,
          fontSize: 20,
          background: "#232526",
          color: "#fff",
          borderRadius: 10,
          alignSelf: "center",
          boxShadow: "0 1px 8px #bdd24816",
        }}
      >
        International Patent Flow Analysis
      </div> */}
      <div style={{ textAlign: "center", marginTop: 4, fontSize: 13, color: "#666" }}>
        Where countries receive their patents from
      </div>

      <div style={{ height: 520, marginTop: 10 }}>
        <Bar data={data} options={options} />
      </div>
    </div>
  );
};

export default InternationalPatentFlowChart;
