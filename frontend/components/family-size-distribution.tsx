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

// Register Chart.js parts once
Chart.register(BarElement, CategoryScale, LinearScale, Tooltip, Legend);

interface ApiResponse {
  datasets: { data: number[]; label: string }[];
  labels: (number | string)[];
}

export const FamilySizeDistributionChart: React.FC<{ port?: number }> = ({
  port: overridePort,
}) => {
  const [chartData, setChartData] = useState<ApiResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  /* ─── Fetch once ─── */
  useEffect(() => {
    let dead = false;

    (async () => {
      try {
        /* 1️⃣ Resolve backend port */
        let port = overridePort;
        if (!port) {
          const txt = await fetch("/backend_port.txt")
            .then((r) => r.text())
            .catch(() => "");
          const n = parseInt(txt.trim(), 10);
          port = Number.isFinite(n) ? n : 49473;
        }

        /* 2️⃣ Fetch data */
        const res = await fetch(
          `http://localhost:${port}/api/family_size_distribution`
        );
        if (!res.ok)
          throw new Error(`HTTP ${res.status} ${res.statusText} while fetching`);
        const json: ApiResponse = await res.json();

        if (!dead) setChartData(json);
      } catch (e: any) {
        if (!dead) setError(e.message ?? String(e));
      } finally {
        if (!dead) setLoading(false);
      }
    })();

    return () => {
      dead = true;
    };
  }, [overridePort]);

  if (loading) return <div>Loading family-size distribution…</div>;
  if (error) return <div style={{ color: "#EA3C53" }}>{error}</div>;
  if (!chartData) return null;

  /* ─── Build chart.js props ─── */
  const data = {
    labels: chartData.labels.map(String), // ensure strings for category axis
    datasets: [
      {
        label: chartData.datasets[0].label,
        data: chartData.datasets[0].data,
        backgroundColor: "#F2D15F",
        borderColor: "#CBAA40",
        borderWidth: 1,
      },
    ],
  };

  const options: any = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: (ctx: any) => ` ${ctx.parsed.y} patents`,
        },
      },
    },
    scales: {
      x: {
        title: { text: "Family Size", display: true, color: "#3B3C3D" },
        grid: { display: false },
        ticks: { color: "#3B3C3D", font: { weight: 600 } },
      },
      y: {
        beginAtZero: true,
        title: { text: "Count", display: true, color: "#3B3C3D" },
        ticks: { color: "#3B3C3D" },
      },
    },
  };

  /* ─── Card wrapper consistent with other widgets ─── */
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
        Family Size Distribution
      </div> */}

      <div style={{ height: 280, marginTop: 10 }}>
        <Bar data={data} options={options} />
      </div>
    </div>
  );
};

export default FamilySizeDistributionChart;
