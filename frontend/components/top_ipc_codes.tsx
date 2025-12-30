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

  /* NEW ➜ comment section state */
  const [showGeneralComment, setShowGeneralComment] = useState(false);
  const [commentPosition, setCommentPosition] = useState({ x: 0, y: 0 });

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
  /* Aggregate duplicates: strip codes and sum patent counts for same code */
  const aggregatedData: Record<string, number> = {};
  const rawLabels: string[] = data.labels;
  const rawCounts: number[] = data.datasets[0]?.data || [];

  rawLabels.forEach((label: string, idx: number) => {
    const cleanCode = stripLabel(label);
    const count = rawCounts[idx] || 0;
    aggregatedData[cleanCode] = (aggregatedData[cleanCode] || 0) + count;
  });

  /* Sort by count descending and take top 10 */
  const sortedEntries = Object.entries(aggregatedData)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10);

  const cleanLabels = sortedEntries.map(([code]) => code);
  const aggregatedCounts = sortedEntries.map(([, count]) => count);

  /* Use total patents from database (returned by API) */
  const totalPatents = data.total_patents || aggregatedCounts.reduce((sum, count) => sum + count, 0);

  /* Create lookup: { "H01M": { ipc_code, title, explanation, count } } */
  const ipcInfoMap: Record<string, any> = {};
  if (data.ipc_info) {
    data.ipc_info.forEach((info: any) => {
      const cleanCode = stripLabel(info.ipc_code);
      if (!ipcInfoMap[cleanCode]) {
        ipcInfoMap[cleanCode] = { ...info };
      }
      // Update count with aggregated value if available
      if (aggregatedData[cleanCode]) {
        ipcInfoMap[cleanCode].count = aggregatedData[cleanCode];
      }
    });
  }

  const chartData = {
    labels: cleanLabels,
    datasets: [{
      data: aggregatedCounts,
      backgroundColor: IPC_COLORS,
      borderRadius: 12,
      borderWidth: 0
    }]
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
            const info = ipcInfoMap[code];
            const title = info?.title ? `\n${info.title}` : '';
            const percentage = ((ctx.parsed.x / totalPatents) * 100).toFixed(1);
            return `${code}: ${ctx.parsed.x} patents (${percentage}%)${title}`;
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
        margin: "0 auto",
        position: "relative" // Added for positioning the info icon
      }}
    >
      {/* Info icon for general comment */}
      <div
        style={{
          position: "absolute",
          top: 16,
          right: 16,
          width: 24,
          height: 24,
          borderRadius: "50%",
          backgroundColor: "#f0f0f0",
          border: "1px solid #ccc",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          cursor: "pointer",
          fontSize: "14px",
          fontWeight: "bold",
          color: "#666",
          zIndex: 10
        }}
        onMouseEnter={(e) => {
          const rect = e.currentTarget.getBoundingClientRect();
          setCommentPosition({ x: rect.left, y: rect.bottom });
          setShowGeneralComment(true);
        }}
        onMouseLeave={() => setShowGeneralComment(false)}
      >
        i
      </div>

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

          <p style={{ margin: "4px 0" }}>
            <strong>Percentage:</strong> {((selectedInfo.count / totalPatents) * 100).toFixed(1)}%
          </p>
        </div>
      )}

      {/* General Comment Block (on hover) */}
      {showGeneralComment && (
        <div
          style={{
            position: "fixed",
            left: commentPosition.x - 200,
            top: commentPosition.y + 5,
            width: 250,
            background: "#fff",
            border: "1px solid #ddd",
            borderRadius: "8px",
            padding: "15px",
            boxShadow: "0 4px 12px rgba(0,0,0,0.15)",
            zIndex: 10000,
            fontSize: "14px",
            lineHeight: "1.4"
          }}
          onMouseEnter={() => setShowGeneralComment(true)}
          onMouseLeave={() => setShowGeneralComment(false)}
        >
          <div style={{ fontWeight: "bold", marginBottom: "8px", color: "#333", fontSize: "15px" }}>
            📊 IPC Codes Analysis
          </div>
          <div style={{ color: "#555" }}>
            {(() => {
              if (sortedEntries.length === 0) {
                return "No IPC code data available for analysis.";
              }
              
              // Calculate top 3 IPC codes
              const top3 = sortedEntries.slice(0, 3);
              const totalTop3 = top3.reduce((sum, [, count]) => sum + count, 0);
              const top3Percentage = (totalTop3 / totalPatents * 100).toFixed(1);
              
              // Calculate dominance of top code
              const topCodePercentage = (top3[0][1] / totalPatents * 100).toFixed(1);
              
              return (
                <>
                  The most frequent IPC code is <strong>{top3[0][0]}</strong> with{" "}
                  <strong>{top3Percentage}%</strong> of all patents.
                  <br /><br />
                  The top 3 codes (
                  {top3.map(([code], idx) => (
                    <span key={code}>
                      {idx > 0 && idx === top3.length - 1 ? " and " : idx > 0 ? ", " : ""}
                      <strong>{code}</strong>
                    </span>
                  ))
                  }) account for <strong>{top3Percentage}%</strong> of total patents.
                  <br /><br />
                  {parseFloat(topCodePercentage) > 25 ? (
                    <>This indicates a strong focus on <strong>{top3[0][0]}</strong> technology.</>
                  ) : parseFloat(topCodePercentage) > 15 ? (
                    <>This shows moderate concentration in <strong>{top3[0][0]}</strong>.</>
                  ) : (
                    <>Patent distribution is relatively balanced across technology areas.</>
                  )}
                </>
              );
            })()}
          </div>
          <div style={{ 
            marginTop: "10px", 
            fontSize: "12px", 
            color: "#888",
            fontStyle: "italic",
            borderTop: "1px solid #eee",
            paddingTop: "8px"
          }}>
            💡 <strong>Tip:</strong> Click on bars to see detailed IPC code information
          </div>
        </div>
      )}
    </div>
  );
};

export default TopIPCCodes;