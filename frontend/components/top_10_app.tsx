import React, { useEffect, useState } from "react";
import LoadingSpinner from "./LoadingSpinner";
import { Bar } from "react-chartjs-2";
import { Chart, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from "chart.js";

// Register necessary Chart.js components
Chart.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

export const Top10Applicants: React.FC = () => {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setErr(null);

    fetch("/backend_port.txt")
      .then(res => res.text())
      .then(port => {
        return fetch(`http://localhost:${port.trim()}/api/top_10_patent_applicants`);
      })
      .then(res => res.json())
      .then(json => {
        if (!cancelled) setData(json);
        setLoading(false);
      })
      .catch(() => {
        if (!cancelled) setErr("Failed to fetch data");
        setLoading(false);
      });

    return () => { cancelled = true; };
  }, []);

  if (loading) {
    return <LoadingSpinner text="Loading Top 10 Applicants..." style={{ marginTop: 24 }} />;
  }
  if (err) {
    return <div style={{color:"#EA3C53", textAlign:"center"}}>{err}</div>;
  }
  if (!data) return null;

  // Prepare Chart Data
  const chartData = {
    labels: data.labels,
    datasets: data.datasets.map((ds: any, i: number) => ({
      ...ds,
      backgroundColor: "#BDD248",
      borderRadius: 12,
      borderWidth: 0
    }))
  };

const options = {
  indexAxis: 'y' as const,
  responsive: true,
  maintainAspectRatio: false,    
  plugins: {
    legend: { display: false },
    title: { display: false },
    tooltip: {
      callbacks: {
        label: (ctx: any) => `${ctx.parsed.x} patents`
      }
    }
  },
  scales: {
    x: {
      beginAtZero: true,
      ticks: { color: "#232526", font: { size: 14, weight: 700 } },  // <--- weight as a number
      grid: { color: "#eee" }
    },
    y: {
      ticks: { color: "#232526", font: { size: 14, weight: 700 } },  // <--- weight as a number
      grid: { color: "#fff" }
    }
  }
};


return (
  <div style={{
    background: "#fff",
    borderRadius: 18,
    boxShadow: "0 2px 18px #B2DBA422",
    padding: 32,
    minWidth: 400,
    width: "fit-content",
    maxWidth: "100%",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    minHeight: 340,
    margin: "0 auto"
  }}>
    <div style={{ width: 400, height: 340, overflow: "visible" }}>
      <Bar data={chartData} options={options} />
    </div>
  </div>
);
};

export default Top10Applicants;
