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

// register once (needed by chart.js v3+)
Chart.register(BarElement, CategoryScale, LinearScale, Tooltip, Legend);

/* ---------- API payload shape ---------- */
interface ApiResponse {
  origins: string[];      // rows   (e.g. ["KR", "RU", …])
  filings: string[];      // cols   (e.g. ["JP", "CN", …])
  matrix: number[][];     // rows x cols counts
}

/* ---------- tiny colour helper ---------- */
const palette = [...schemeTableau10, ...schemeSet3].flat(); // plenty distinct
const colourFor = (idx: number) => palette[idx % palette.length];

/* ---------- component ---------- */
export const InternationalProtectionMatrixChart: React.FC<{ port?: number }> = ({
  port: overridePort,
}) => {
  const [api, setApi]     = useState<ApiResponse | null>(null);
  const [loading, setLoad] = useState(true);
  const [error, setErr]   = useState<string | null>(null);

  /* ─── fetch once ─── */
  useEffect(() => {
    let dead = false;
    (async () => {
      try {
        /* 1️⃣  resolve backend port */
        let port = overridePort;
        if (!port) {
          const txt = await fetch("/backend_port.txt")
            .then(r => r.text())
            .catch(() => "");
          const n = parseInt(txt.trim(), 10);
          port = Number.isFinite(n) ? n : 49473;
        }

        /* 2️⃣  get matrix data */
        const res = await fetch(
          `http://localhost:${port}/api/international_protection_matrix`
        );
        if (!res.ok)
          throw new Error(`HTTP ${res.status} ${res.statusText}`);
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

  if (loading) return <div>Loading patent-protection matrix…</div>;
  if (error)   return <div style={{ color: "#EA3C53" }}>{error}</div>;
  if (!api)    return null;

  /* ---------- build chart.js structures ---------- */
  const datasets = api.filings.map((filing, colIdx) => ({
    label: filing,
    data : api.origins.map((_, rowIdx) => api.matrix[rowIdx]?.[colIdx] ?? 0),
    backgroundColor: colourFor(colIdx),
    borderWidth: 0,
  }));

  const data = {
    labels: api.origins,   // y-axis
    datasets,
  };

  const options: any = {
    indexAxis: "y" as const,
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      tooltip: {
        mode: "nearest",
        intersect: false,
        callbacks: {
          label: (ctx: any) =>
            ` ${ctx.dataset.label} → ${ctx.parsed.x}`,
        },
      },
      legend: {
        position: "right",
        labels: { boxWidth: 14 },
      },
    },
    scales: {
      x: { stacked: true, ticks: { color: "#3B3C3D" } },
      y: { stacked: true, ticks: { color: "#3B3C3D", font: { weight: 600 } } },
    },
  };

  /* ---------- card wrapper ---------- */
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
      <div
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
        International Patent Protection Strategy
      </div>
      <div
        style={{
          textAlign: "center",
          marginTop: 4,
          fontSize: 13,
          color: "#666",
        }}
      >
        Origin countries and their international filing patterns
      </div>

      <div style={{ height: 500, marginTop: 10 }}>
        <Bar data={data} options={options} />
      </div>
    </div>
  );
};

export default InternationalProtectionMatrixChart;
