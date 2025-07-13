
import LoadingSpinner from "./LoadingSpinner";

import React, { useEffect, useRef, useState } from "react";
import { Bar, getElementAtEvent } from "react-chartjs-2";
import {
  Chart,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
} from "chart.js";

/* ------------------------------------------------------------------
   1) Register Chart.js scales & plugins once at module load.
   ------------------------------------------------------------------ */
Chart.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

/* ------------------------------------------------------------------
   2) Brand-consistent palette for up to 10 IPC bars.
   ------------------------------------------------------------------ */
const IPC_COLORS = [
  "#9950CC", "#4A90E2", "#50CC7F", "#FFD166", "#EF476F",
  "#118AB2", "#06D6A0", "#FF61A6", "#B58840", "#7D6FFF"
];

/* Remove every non-alphanumeric character so the keys are chart-safe. */
const stripLabel = (str: string) => str.replace(/[^A-Za-z0-9]/g, "");

/* ------------------------------------------------------------------
   TopIPCCodes component
   ------------------------------------------------------------------ */
export const TopIPCCodes: React.FC = () => {
  /* ------------- Runtime state ------------- */
  const [data, setData] = useState<any>(null);          // API payload
  const [loading, setLoading] = useState(true);         // loading flag
  const [err, setErr] = useState<string | null>(null);  // fetch error (if any)

  /* NEW ➜ store the bar the user clicked to show its details later */
  const [selectedInfo, setSelectedInfo] = useState<any>(null);

  /* OPTIONAL ➜ keep a ref to the Chart instance in case you want to
     programmatically interact with it in the future (e.g., zoom/resize). */
  const chartRef = useRef<any>(null);

  /* ------------- Side-effect: fetch once on mount ------------- */
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setErr(null);

    /* First read the backend port from a static text file                     */
    /* then hit /api/top_ipc_codes on that port.                              */
    fetch("/backend_port.txt")
      .then(res => res.text())
      .then(port => fetch(`http://localhost:${port.trim()}/api/top_ipc_codes`))
      .then(res => res.json())
      .then(json => {
        if (!cancelled) setData(json);
        setLoading(false);
      })
      .catch(() => {
        if (!cancelled) setErr("Failed to fetch IPC codes");
        setLoading(false);
      });

    return () => { cancelled = true; };
  }, []);

  /* ------------- Early returns (UX guards) ------------- */
  if (loading) return <LoadingSpinner text="Loading Top IPC Codes..." />;
  if (err)      return <div style={{ color: "#EA3C53", textAlign: "center" }}>{err}</div>;
  if (!data)    return null;  // should not happen, but type-safe guard

  /* ------------- Prepare data for Chart.js ------------- */
  const cleanLabels = data.labels.map(stripLabel);  // sanitised X-axis keys

  /* Create lookup: { "H01M": { ipc_code, title, explanation, count } } */
  const ipcInfoMap: Record<string, any> = Object.fromEntries(
    data.ipc_info.map((info: any) => [stripLabel(info.ipc_code), { ...info }])
  );

  const chartData = {
    labels: cleanLabels,
    datasets: data.datasets.map((ds: any) => ({
      ...ds,
      backgroundColor: IPC_COLORS,
      borderRadius: 12,
      borderWidth: 0
    }))
  };

  /* ------------- Chart options ------------- */
  const options = {
    indexAxis: "y" as const,
    responsive: true,
    plugins: {
      legend: { display: false },
      title:  { display: false },
      tooltip: {
        callbacks: {
          /* Custom tooltip shows count + title + explanation + hint */
          label: (ctx: any) => {
            const code = ctx.label;
            return `${code}: ${ctx.parsed.x} patents\n(click for more details)`;
          }
        }
      }
    },
    scales: {
      x: {
        beginAtZero: true,
        ticks: { color: "#232526", font: { size: 14, weight: 700 } },
        grid:  { color: "#eee" }
      },
      y: {
        ticks: { color: "#232526", font: { size: 14, weight: 700 }, padding: 12 },
        grid:  { color: "#fff" }
      }
    }
  };

  /* -------------------- Render -------------------- */
  return (
    <div
      style={{
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
      }}
    >
      {/* ---------------------- Bar chart ---------------------- */}
      <div style={{ width: 500, height: 420, overflow: "visible" }}>
        <Bar
          ref={chartRef}
          data={chartData}
          options={options}
          onClick={(event) => {
            // Import getElementAtEvent from react-chartjs-2 at the top of the file
            const elements = getElementAtEvent(chartRef.current, event);
            if (!elements.length) return;
            const idx = elements[0].index;
            const codeKey = cleanLabels[idx];
            const info = ipcInfoMap[codeKey];
            setSelectedInfo({ ipc_code: codeKey, ...info });
          }}
        />
      </div>

      {/* --------------- Details panel (appears after click) --------------- */}
      {selectedInfo && (
        <div
          style={{
            marginTop: 24,
            padding: 20,
            background: "#F7FAFC",
            border: "1px solid #E2E8F0",
            borderRadius: 12,
            maxWidth: 500,
            position: 'relative'
          }}
        >
          <button
            onClick={() => setSelectedInfo(null)}
            style={{
              position: 'absolute',
              top: 12,
              right: 12,
              background: 'none',
              border: 'none',
              fontSize: 20,
              color: '#888',
              cursor: 'pointer',
              padding: 0
            }}
            aria-label="Close details"
            title="Close details"
          >
            ×
          </button>
          <h3 style={{ margin: "0 0 8px", color: "#118AB2" }}>
            {stripLabel(selectedInfo.ipc_code)}
          </h3>

          {selectedInfo.title && (
            <p style={{ margin: "4px 0" }}>
              <strong>Title:</strong> {selectedInfo.title}
            </p>
          )}

          {selectedInfo.explanation && (
            <p style={{ margin: "4px 0" }}>
              <strong>Explanation:</strong> {selectedInfo.explanation}
            </p>
          )}

          {/* count is returned by the backend as part of each ipc_info item */}
          <p style={{ margin: "4px 0" }}>
            <strong>Patent count:</strong> {selectedInfo.count}
          </p>
        </div>
      )}
    </div>
  );
};

export default TopIPCCodes;

