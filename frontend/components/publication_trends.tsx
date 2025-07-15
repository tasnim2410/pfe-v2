import React, { useEffect, useState } from "react";
import LoadingSpinner from "./LoadingSpinner";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  LineElement,
  PointElement,
  CategoryScale,
  LinearScale,
  Tooltip,
  Legend,
  Title,
} from "chart.js";

// Register Chart.js modules
ChartJS.register(LineElement, PointElement, CategoryScale, LinearScale, Tooltip, Legend, Title);

const CHART_BG = "#fff";
const ACCENT = "#232526";
const HIGHLIGHT = "#BDD248";

type ChartProps = { width?: number; height?: number };
const PublicationTrends: React.FC<ChartProps> = ({ width, height }) => {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setErr(null);

    fetch("/backend_port.txt")
      .then((res) => res.text())
      .then((port) =>
        fetch(`http://localhost:${port.trim()}/api/patents/first_filing_years`)
      )
      .then((res) => res.json())
      .then((json) => {
        if (!cancelled) setData(json);
        setLoading(false);
      })
      .catch(() => {
        if (!cancelled) setErr("Failed to fetch publication trend data.");
        setLoading(false);
      });

    return () => { cancelled = true; };
  }, []);

  if (loading) return <LoadingSpinner text="Loading Publication Trends..." />;
  if (err) return <div style={{ color: "#EA3C53", textAlign: "center" }}>{err}</div>;
  if (!data) return null;

  // Chart.js data
  const chartData = {
    labels: data.labels,
    datasets: data.datasets.map((ds: any) => ({
      ...ds,
      fill: false,
      borderColor: HIGHLIGHT,
      backgroundColor: HIGHLIGHT,
      pointBackgroundColor: ACCENT,
      pointRadius: 4,
      pointHoverRadius: 7,
      tension: 0.2,
    })),
  };

  // Chart.js options
  const options = {
    responsive: true,
    plugins: {
      legend: { display: false },
      title: {
        display: true,
        text: "Publication Trend by First Filing Year",
        color: ACCENT,
        font: { size: 22, weight: 700 },
        padding: { top: 15, bottom: 30 },
      },
      tooltip: {
        callbacks: {
          label: (ctx: any) => ` ${ctx.parsed.y} patents`,
        },
        backgroundColor: "#232526",
        titleColor: "#fff",
        bodyColor: "#BDD248",
      },
    },
    scales: {
      x: {
        title: {
          display: true,
          text: "Year",
          color: ACCENT,
          font: { size: 16, weight: 700 }
        },
        ticks: { color: ACCENT, font: { size: 13, weight: 600 } },
        grid: { color: "#eee" }
      },
      y: {
        title: {
          display: true,
          text: "Number of Patents",
          color: ACCENT,
          font: { size: 16, weight: 700 }
        },
        beginAtZero: true,
        ticks: { color: ACCENT, font: { size: 13, weight: 600 } },
        grid: { color: "#eee" }
      }
    }
  };

  return (
    <div style={{
      background: CHART_BG,
      borderRadius: 18,
      boxShadow: "0 2px 18px #B2DBA422",
      padding: 30,
      width: 800,
      maxWidth: "100%",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      minHeight: 440,
      margin: "0 auto"
    }}>
      <div style={{ width: "100%", maxWidth: 740, height: 360 }}>
        <Line data={chartData} options={options} />
      </div>
    </div>
  );
};

export default PublicationTrends;
