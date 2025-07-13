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

ChartJS.register(LineElement, PointElement, CategoryScale, LinearScale, Tooltip, Legend, Title);

const CHART_BG = "#fff";
const ACCENT = "#232526";
const COLORS = [
  "#3366CC", // Blue
  "#DC3912", // Red
  "#FF9900", // Orange
  "#109618", // Green
  "#990099", // Purple
  "#3B3EAC", // Indigo
  "#0099C6", // Cyan
  "#DD4477", // Pink
  "#66AA00", // Lime
  "#B82E2E", // Dark Red
  "#316395", // Steel Blue
  "#994499", // Violet
  "#22AA99", // Teal
  "#AAAA11", // Olive
  "#6633CC", // Deep Purple
  "#E67300", // Amber
  "#8B0707", // Maroon
  "#329262", // Emerald
  "#5574A6", // Slate Blue
  "#3B3EAC"  // Indigo
];

const FieldLegend = ({ fields }: { fields: string[] }) => (
  <div style={{
    display: "flex", gap: 22, margin: "18px 0 0 0", flexWrap: "wrap",
    justifyContent: "center"
  }}>
    {fields.map((f, i) => (
      <span key={f} style={{
        display: "flex", alignItems: "center", fontWeight: 600, fontSize: 16,
        color: ACCENT, gap: 8
      }}>
        <span style={{
          width: 16, height: 16, borderRadius: "50%",
          background: COLORS[i % COLORS.length], display: "inline-block", marginRight: 5,
          border: "1.5px solid #d8d8d8"
        }} />{f}
      </span>
    ))}
  </div>
);

const PatentFieldTrends: React.FC = () => {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [classifying, setClassifying] = useState(false);

  const loadData = async () => {
    setLoading(true);
    setErr(null);
    let cancelled = false;
    try {
      const portRes = await fetch("/backend_port.txt");
      const port = (await portRes.text()).trim();
      const apiRes = await fetch(`http://localhost:${port}/api/patent_field_trends`);
      if (!apiRes.ok) throw new Error();
      const json = await apiRes.json();
      if (!cancelled) setData(json);
      setLoading(false);
    } catch (e) {
      setErr("Failed to fetch patent field trends. You may need to classify the patents database.");
      setLoading(false);
    }
    return () => { cancelled = true; };
  };

  useEffect(() => { loadData(); }, []);

  const classify = async () => {
    setClassifying(true);
    setErr(null);
    try {
      const portRes = await fetch("/backend_port.txt");
      const port = (await portRes.text()).trim();
      const resp = await fetch(`http://localhost:${port}/classify_patents`, { method: "POST" });
      if (!resp.ok) throw new Error();
      await new Promise((res) => setTimeout(res, 1500));
      loadData();
    } catch {
      setErr("Failed to classify patents. Is the backend running?");
    }
    setClassifying(false);
  };

  if (loading) return <LoadingSpinner text="Loading Patent Field Trends..." />;
  if (err) return (
    <div style={{ color: "#EA3C53", textAlign: "center", margin: 22 }}>
      {err}
      <br />
      <button
        onClick={classify}
        disabled={classifying}
        style={{
          marginTop: 12, padding: "7px 26px", borderRadius: 9,
          background: "#BDD248", color: "#232526", fontWeight: 800,
          border: "none", fontSize: 17, cursor: "pointer",
          opacity: classifying ? 0.6 : 1,
          transition: "opacity .2s"
        }}
      >
        {classifying ? "Classifying..." : "Classify Now"}
      </button>
    </div>
  );
  if (!data) return null;

  const chartData = {
    labels: data.labels,
    datasets: data.datasets.map((ds: any, i: number) => ({
      ...ds,
      borderColor: COLORS[i % COLORS.length],
      backgroundColor: COLORS[i % COLORS.length],
      pointBackgroundColor: "#fff",
      pointBorderColor: COLORS[i % COLORS.length],
      pointRadius: 0,          // Points hidden by default
      pointHoverRadius: 7,     // Appear on hover
      borderWidth: 3,
      fill: false,
      tension: 0.20,
    })),
  };
  
  const options = {
    responsive: true,
    plugins: {
      legend: { display: false },
      title: {
        display: true,
        text: "Patent Trends by Field",
        color: ACCENT,
        font: { size: 22, weight:  700 },
        padding: { top: 12, bottom: 28 },
      },
      tooltip: {
        mode: 'index' as const,
        intersect: false,
        callbacks: {
          label: (ctx: any) =>
            `${ctx.dataset.label}: ${ctx.parsed.y} patents in ${ctx.label}`,
        },
        backgroundColor: "#232526",
        titleColor: "#fff",
        bodyColor: "#BDD248",
        borderColor: "#EA3C53",
        borderWidth: 1.2
      },
    },
    hover: {
      mode: 'index' as const,
      intersect: false,
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
      width: 860,
      maxWidth: "100%",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      minHeight: 480,
      margin: "0 auto"
    }}>
      <div style={{ width: "100%", maxWidth: 790, height: 400 }}>
        <Line data={chartData} options={options} />
      </div>
      <FieldLegend fields={data.datasets.map((d: any) => d.label)} />
    </div>
  );
};

export default PatentFieldTrends;
