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
import { colorForCountry } from "../lib/country-colors";

// register once (needed by chart.js v3+)
Chart.register(BarElement, CategoryScale, LinearScale, Tooltip, Legend);

/* ---------- API payload shape ---------- */
interface ApiResponse {
  origins: string[];      // rows   (e.g. ["KR", "RU", …])
  filings: string[];      // cols   (e.g. ["JP", "CN", …])
  matrix: number[][];     // rows x cols counts
}

/* ---------- colour helper provided via shared lib (country-colors) ---------- */

/* ---------- component ---------- */
export const InternationalProtectionMatrixChart: React.FC<{ port?: number }> = ({
  port: overridePort,
}) => {
  const [api, setApi]     = useState<ApiResponse | null>(null);
  const [firstFilingsTotal, setFirstFilingsTotal] = useState<number | null>(null);
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

        /* 3️⃣  get total first filings count */
        const countRes = await fetch(
          `http://localhost:${port}/api/first_filings/count`
        );
        if (!countRes.ok)
          throw new Error(`HTTP ${countRes.status} ${countRes.statusText}`);
        const countJson: { count: number } = await countRes.json();
        if (!dead) setFirstFilingsTotal(countJson.count ?? 0);
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
  if (!api || firstFilingsTotal === null)    return null;

  /* ----- Filter to top 10 origins by total filings ----- */
  const rowTotals = api.origins.map((_, rowIdx) => 
    api.matrix[rowIdx]?.reduce((sum, val) => sum + val, 0) ?? 0
  );
  
  // Get indices sorted by total (ascending for bottom-to-top display), take top 10
  const sortedIndices = api.origins
    .map((_, idx) => idx)
    .filter(idx => rowTotals[idx] > 0)
    .sort((a, b) => rowTotals[a] - rowTotals[b])  // ascending for bottom-to-top display
    .slice(-10);  // take top 10 (last 10 after ascending sort)
  
  const filteredOrigins = sortedIndices.map(idx => api.origins[idx]);
  const filteredMatrix = sortedIndices.map(idx => api.matrix[idx]);
  const filingIndexByCode: Record<string, number> = Object.fromEntries(
    api.filings.map((c, i) => [c, i])
  );

  const rdLegend = filteredOrigins.map((code, rowIdx) => {
    const colIdx = filingIndexByCode[code];
    const row = filteredMatrix[rowIdx] ?? [];
    const diag = colIdx !== undefined ? (row[colIdx] ?? 0) : 0;
    const pct = firstFilingsTotal > 0 ? (diag / firstFilingsTotal) * 100 : 0;
    return { code, pct };
  }).sort((a, b) => b.pct - a.pct);

  /* ---------- build chart.js structures ---------- */
  const datasets = api.filings.map((filing, colIdx) => ({
    label: filing,
    data : filteredOrigins.map((_, rowIdx) => filteredMatrix[rowIdx]?.[colIdx] ?? 0),
    backgroundColor: colorForCountry(filing),
    borderWidth: 0,
  }));

  const data = {
    labels: filteredOrigins,   // y-axis
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
        position: "left",
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
        International Patent Protection Strategy
      </div> */}
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
      <div
        style={{
          marginTop: 16,
          padding: 16,
          border: "2px solid #BDD248",
          borderRadius: 12,
          background: "#fafafa",
        }}
      >
        <div
          style={{
            textAlign: "center",
            fontWeight: 700,
            fontSize: 16,
            color: "#232526",
            marginBottom: 12,
          }}
        >
          R&D Investment 
        </div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 10, justifyContent: "center" }}>
          {rdLegend.map((item) => (
            <div
              key={item.code}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 6,
                padding: "4px 8px",
                borderRadius: 8,
                background: "#f6f6f6ff",
                boxShadow: "0 1px 2px rgba(0,0,0,0.05)",
              }}
            >
              <div style={{ width: 12, height: 12, borderRadius: 3, background: colorForCountry(item.code) }} />
              <span style={{ fontWeight: 600, color: "#232526" }}>{item.code}</span>
              <span style={{ color: "#888" }}>→</span>
              <span style={{ color: "#555" }}>{item.pct.toFixed(1)}%</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default InternationalProtectionMatrixChart;
