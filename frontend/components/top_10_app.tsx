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
  layout: {
    padding: {
      left: 20
    }
  },
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
      ticks: { color: "#232526", font: { size: 12, weight: 600 } },
      grid: { color: "#eee" }
    },
    y: {
      ticks: { 
        color: "#232526", 
        font: { size: 10, weight: 500 },
        autoSkip: false,
        maxRotation: 0,
        minRotation: 0,
        callback: function(value: any, index: number) {
          const label = (this as any).getLabelForValue(value);
          if (label && label.length > 37) {
            // Split into multiple lines at ~37 chars
            const words = label.split(' ');
            const lines: string[] = [];
            let currentLine = '';
            for (const word of words) {
              if ((currentLine + ' ' + word).trim().length <= 37) {
                currentLine = (currentLine + ' ' + word).trim();
              } else {
                if (currentLine) lines.push(currentLine);
                currentLine = word;
              }
            }
            if (currentLine) lines.push(currentLine);
            return lines;
          }
          return label;
        }
      },
      grid: { color: "#fff" },
      afterFit: (scaleInstance: any) => {
        scaleInstance.width = 320;
      }
    }
  }
};


return (
  <div style={{
    background: "#fff",
    borderRadius: 18,
    boxShadow: "0 2px 18px #B2DBA422",
    padding: 5,
    minWidth: 600,
    width: "fit-content",
    maxWidth: "100%",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    minHeight: 340,
    margin: "0 auto"
  }}>
    <div style={{ width: 650, height: 420, overflow: "visible" }}>
      <Bar data={chartData} options={options} />
    </div>
    {data.percentage !== undefined && (
      <div style={{
        marginTop: 12,
        padding: "8px 16px",
        background: "#f5f5f5",
        borderRadius: 8,
        fontSize: 13,
        color: "#555",
        textAlign: "center"
      }}>
        Top 10 applicants represent <strong>{data.percentage}%</strong> of the top 100 applicants 
        ({data.top10_total} out of {data.top100_total} patents)
      </div>
    )}
  </div>
);
};

export default Top10Applicants;
