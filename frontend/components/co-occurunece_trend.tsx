import React, { useEffect, useState } from "react";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  LineElement,
  PointElement,
  CategoryScale,
  LinearScale,
  Tooltip,
  Legend
} from "chart.js";
import LoadingSpinner from "./LoadingSpinner";

// Register chart.js components (if not already)
ChartJS.register(LineElement, PointElement, CategoryScale, LinearScale, Tooltip, Legend);

const cardStyle: React.CSSProperties = {
  background: "#fff",
  borderRadius: 18,
  boxShadow: "0 2px 18px #B2DBA422",
  padding: 30,
  width: 860,
  maxWidth: "100%",
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  minHeight: 320,
  margin: "0 auto"
};

const tabStyle = (active: boolean): React.CSSProperties => ({
  padding: "10px 22px",
  borderRadius: 8,
  background: active ? "#BDD248" : "#f2f5ec",
  color: active ? "#232526" : "#63735A",
  fontWeight: 700,
  margin: "0 7px",
  border: "none",
  cursor: "pointer",
  fontSize: 16,
  boxShadow: active ? "0 2px 12px #BDD24866" : undefined,
  transition: "background 0.15s"
});

const COLORS = {
  line: "#EA3C53",
  emerging: "#3FB950",
  declining: "#EA3C53"
};

function groupByPair(data: any[]) {
  // Returns { [term_pair]: [{year, frequency}, ...] }
  const out: { [pair: string]: { year: number, frequency: number }[] } = {};
  data.forEach(({ term_pair, year, frequency }) => {
    if (!out[term_pair]) out[term_pair] = [];
    out[term_pair].push({ year, frequency });
  });
  // Sort years for each pair
  Object.values(out).forEach(list => list.sort((a, b) => a.year - b.year));
  return out;
}

const colorSlope = (slope: number, mode: "emerging" | "declining") =>
  mode === "emerging"
    ? COLORS.emerging
    : COLORS.declining;


const CooccurrenceTrends: React.FC = () => {
  const [data, setData] = useState<any | null>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [tab, setTab] = useState<"emerging" | "declining">("emerging");

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      setLoading(true);
      setErr(null);
      try {
        const portRes = await fetch("/backend_port.txt");
        const port = (await portRes.text()).trim();
        const apiRes = await fetch(`http://localhost:${port}/api/cooccurrence_trends`);
        if (!apiRes.ok) throw new Error();
        const json = await apiRes.json();
        if (!cancelled) setData(json);
        setLoading(false);
      } catch {
        setErr("Failed to fetch cooccurrence trends. Is the backend running?");
        setLoading(false);
      }
    };
    load();
    return () => { cancelled = true; };
  }, []);

  if (loading) return <LoadingSpinner text="Loading Co-occurrence Trends..." />;
  if (err) return (
    <div style={{ color: "#EA3C53", textAlign: "center", margin: 22 }}>
      {err}
    </div>
  );
  if (!data) return null;

  // Pick data for this tab
  const pairKey = tab === "emerging" ? "emerging_pairs" : "declining_pairs";
  const yearlyKey = tab === "emerging" ? "emerging" : "declining";
  const allTrends: { [pair: string]: { year: number, frequency: number }[] } = groupByPair(data[yearlyKey] || []);
  const pairs: any[] = data[pairKey] || [];
  pairs.sort((a, b) =>
    Math.abs(b.slope) - Math.abs(a.slope) ||
    (b.total_count || 0) - (a.total_count || 0)
  );

  return (
    <div style={cardStyle}>
      {/* Tabs for emerging/declining */}
      <div style={{ display: "flex", justifyContent: "center", marginBottom: 22 }}>
        <button style={tabStyle(tab === "emerging")} onClick={() => setTab("emerging")}>Emerging</button>
        <button style={tabStyle(tab === "declining")} onClick={() => setTab("declining")}>Declining</button>
      </div>
      <div style={{ width: "100%", maxWidth: 790, margin: "0 auto" }}>
        {pairs.length === 0 && (
          <div style={{ color: "#EA3C53", fontWeight: 700, textAlign: "center", fontSize: 19, marginTop: 25 }}>
            No {tab} keyword pairs detected.
          </div>
        )}
        {pairs.map((pair, idx) => {
          const label = `${pair.term1} & ${pair.term2}`;
          const yearly = allTrends[label] || [];
          if (!yearly.length) return null;

          // Prepare Chart.js data
          const chartData = {
            labels: yearly.map(y => y.year),
            datasets: [
              {
                label: label,
                data: yearly.map(y => y.frequency),
                borderColor: colorSlope(pair.slope, tab),
                backgroundColor: "rgba(191, 210, 72,0.14)",
                pointBackgroundColor: "#fff",
                pointBorderColor: colorSlope(pair.slope, tab),
                borderWidth: 3,
                fill: false,
                tension: 0.20,
                pointRadius: 0,
                pointHoverRadius: 5
              }
            ]
          };
          const options = {
            responsive: true,
            plugins: {
              legend: { display: false },
              tooltip: {
                callbacks: {
                  label: (ctx: any) =>
                    `Frequency: ${ctx.parsed.y} in ${ctx.label}`
                },
                backgroundColor: "#232526",
                titleColor: "#fff",
                bodyColor: "#BDD248",
                borderColor: "#EA3C53",
                borderWidth: 1.2
              }
            },
            scales: {
              x: {
                title: { display: false },
                ticks: { color: "#8fa68f", font: { size: 12, weight: 500 } },
                grid: { color: "#f2f5ec" }
              },
              y: {
                title: { display: false },
                beginAtZero: true,
                ticks: { color: "#8fa68f", font: { size: 12, weight: 500 } },
                grid: { color: "#f2f5ec" }
              }
            }
          };

          return (
            <div
              key={label}
              style={{
                background: "#f7f9f3",
                borderRadius: 10,
                padding: "17px 24px 9px 24px",
                margin: "0 0 22px 0",
                boxShadow: "0 1px 8px #BDD24822"
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
                <span style={{
                  fontWeight: 800,
                  fontSize: 17,
                  color: colorSlope(pair.slope, tab),
                  letterSpacing: 0.5
                }}>
                  {label}
                </span>
                <span style={{
                  fontWeight: 600,
                  fontSize: 14,
                  color: "#9BA29B"
                }}>
                  Slope: <span style={{ color: colorSlope(pair.slope, tab), fontWeight: 700 }}>{pair.slope >= 0 ? "+" : ""}{pair.slope.toFixed(2)}</span>
                  {"  "} | Total: <span style={{ color: "#232526" }}>{pair.total_count}</span>
                </span>
              </div>
              <div style={{ width: "100%", maxWidth: 350, minWidth: 180 }}>
                <Line data={chartData} options={options} height={110} />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default CooccurrenceTrends;
