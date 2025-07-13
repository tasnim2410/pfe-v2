import React, { useEffect, useState } from "react";
import LoadingSpinner from "./LoadingSpinner";
import { Bar } from "react-chartjs-2";
import { Chart, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from "chart.js";

// Register Chart.js modules
Chart.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

const PIE_COLORS = [
  "#BDD248", "#8ecae6", "#ffb703", "#219ebc", "#ea3c53",
  "#a0c4ff", "#b2dbb8", "#ffafcc", "#cdb4db", "#ffd6a5"
];

const TopKeywords: React.FC = () => {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/backend_port.txt")
      .then(res => res.text())
      .then(port =>
        fetch(`http://localhost:${port.trim()}/top_keyword`)
      )
      .then(res => res.json())
      .then(json => {
        setData(json.keywords);
        setLoading(false);
      });
  }, []);

  if (loading) return <div>Loading...</div>;
  if (!data) return <div>No data</div>;

  // Prepare the data for Chart.js
  const chartData = {
    labels: data.map((k: any) => k.keyword),
    datasets: [{
      data: data.map((k: any) => k.score),
      backgroundColor: PIE_COLORS,
      borderRadius: 10,
      barThickness: 28,
      label: "Keyword Score"
    }]
  };

  const options = {
    indexAxis: "y" as const,
    responsive: true,
    plugins: {
      legend: { display: false },
      title: { display: false },
      tooltip: {
        callbacks: {
          label: (ctx: any) =>
            `Score: ${ctx.parsed.x.toFixed(4)}`
        }
      }
    },
    scales: {
      x: {
        beginAtZero: true,
        ticks: { color: "#232526", font: { size: 16, weight: 700 } },
        grid: { color: "#eee" }
      },
      y: {
        ticks: { color: "#232526", font: { size: 16, weight: 700 } },
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
      minWidth: 520,
      width: "fit-content",
      maxWidth: "100%",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      minHeight: 420,
      margin: "0 auto"
    }}>
      <div style={{ width: '100%', overflowX: 'auto' }}>
        <div style={{ minWidth: 700, height: 420 }}>
          <Bar data={chartData} options={options} />
        </div>
      </div>
    </div>
  );
};

export default TopKeywords;
