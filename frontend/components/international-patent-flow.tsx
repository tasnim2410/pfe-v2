import React, { useEffect, useState, useMemo } from "react";
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

/* register once */
Chart.register(BarElement, CategoryScale, LinearScale, Tooltip, Legend);

/* ----- API types ----- */
interface ApiResponse {
  origins: string[];    // columns (US, CN, …)
  receivers: string[];  // rows    (JP, CN, …)
  matrix: number[][];   // receiver x origin
}

/* ----- colour helper provided via shared lib (country-colors) ----- */

/* ---------- component ---------- */
export const InternationalPatentFlowChart: React.FC<{ port?: number }> = ({
  port: overridePort,
}) => {
  const [api, setApi] = useState<ApiResponse | null>(null);
  const [loading, setLoad] = useState(true);
  const [error, setErr] = useState<string | null>(null);
  const [hiddenReceivers, setHiddenReceivers] = useState<Set<string>>(new Set());

  /* ─ fetch once ─ */
  useEffect(() => {
    let dead = false;
    (async () => {
      try {
        /* 1️⃣ port autodetect */
        let port = overridePort;
        if (!port) {
          const txt = await fetch("/backend_port.txt").then(r => r.text()).catch(() => "");
          const n = parseInt(txt.trim(), 10);
          port = Number.isFinite(n) ? n : 49473;
        }

        /* 2️⃣ data */
        const res = await fetch(`http://localhost:${port}/api/international_patent_flow`);
        if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`);
        const json: ApiResponse = await res.json();
        if (!dead) setApi(json);
      } catch (e: any) {
        if (!dead) setErr(e.message ?? String(e));
      } finally {
        if (!dead) setLoad(false);
      }
    })();
    return () => { dead = true; };
  }, [overridePort]);

  /* ----- Filter to top 10 receivers by total patents ----- */
  const { filteredReceivers, filteredMatrix, allReceivers } = useMemo(() => {
    if (!api) return { filteredReceivers: [], filteredMatrix: [], allReceivers: [] };
    
    const rowTotals = api.receivers.map((_, rowIdx) => 
      api.matrix[rowIdx]?.reduce((sum, val) => sum + val, 0) ?? 0
    );
    
    // Get indices sorted by total (descending), take top 10
    const sortedIndices = api.receivers
      .map((_, idx) => idx)
      .filter(idx => rowTotals[idx] > 0)
      .sort((a, b) => rowTotals[a] - rowTotals[b])  // ascending for bottom-to-top display
      .slice(-10);  // take top 10 (last 10 after ascending sort)
    
    const filteredReceivers = sortedIndices.map(idx => api.receivers[idx]);
    const filteredMatrix = sortedIndices.map(idx => api.matrix[idx]);
    
    return {
      filteredReceivers,
      filteredMatrix,
      allReceivers: api.receivers,
    };
  }, [api]);

  /* ----- Filter receivers based on hidden state ----- */
  const visibleReceiverIndices = useMemo(() => {
    return filteredReceivers
      .map((receiver, index) => ({ receiver, index }))
      .filter(({ receiver }) => !hiddenReceivers.has(receiver))
      .map(({ index }) => index);
  }, [filteredReceivers, hiddenReceivers]);

  const visibleReceivers = visibleReceiverIndices.map(idx => filteredReceivers[idx]);

  /* ----- Toggle receiver visibility ----- */
  const toggleReceiver = (receiver: string) => {
    setHiddenReceivers(prev => {
      const newSet = new Set(prev);
      if (newSet.has(receiver)) {
        newSet.delete(receiver);
      } else {
        newSet.add(receiver);
      }
      return newSet;
    });
  };

  /* ----- Reset all filters ----- */
  const resetFilters = () => {
    setHiddenReceivers(new Set());
  };

  /* ----- chart.js datasets ----- */
  const datasets = useMemo(() => {
    if (!api) return [];
    
    return api.origins.map((origin, col) => ({
      label: origin,
      data: visibleReceiverIndices.map(rowIdx => filteredMatrix[rowIdx]?.[col] ?? 0),
      backgroundColor: colorForCountry(origin),
      borderWidth: 0,
    }));
  }, [api, filteredMatrix, visibleReceiverIndices]);

  const data = { labels: visibleReceivers, datasets };

  const options: any = {
    indexAxis: "y" as const,
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: { 
        stacked: true, 
        reverse: true,
        title: { display: true, text: "Total Patents Received" }, 
        ticks: { color: "#3B3C3D" } 
      },
      y: { 
        stacked: true, 
        position: "right",
        title: { display: true, text: "Receiving Country" }, 
        ticks: { color: "#3B3C3D", font: { weight: 600 } } 
      },
    },
    plugins: {
      tooltip: {
        mode: "nearest",
        intersect: false,
        callbacks: {
          label: (ctx: any) => ` ${ctx.dataset.label} → ${ctx.parsed.x}`,
        },
      },
      legend: { 
        position: "right", 
        labels: { 
          boxWidth: 14,
          filter: (legendItem: any) => {
            // Keep all legend items (origins) visible
            return true;
          }
        } 
      },
    },
  };

  if (loading) return <div>Loading international patent flow…</div>;
  if (error) return <div style={{ color: "#EA3C53" }}>{error}</div>;
  if (!api) return null;

  /* ----- Filter controls ----- */
  const FilterControls = () => (
    <div style={{
      margin: "12px 0",
      padding: "12px 16px",
      background: "#f8f9fa",
      borderRadius: "10px",
      border: "1px solid #e9ecef"
    }}>
      <div style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        marginBottom: "8px"
      }}>
        <div style={{ fontWeight: 600, fontSize: "14px", color: "#495057" }}>
          Filter Receiving Countries ({visibleReceivers.length} of {filteredReceivers.length} shown)
        </div>
        <button
          onClick={resetFilters}
          style={{
            padding: "4px 12px",
            fontSize: "12px",
            background: "#6c757d",
            color: "white",
            border: "none",
            borderRadius: "4px",
            cursor: "pointer"
          }}
        >
          Reset All
        </button>
      </div>
      <div style={{
        display: "flex",
        flexWrap: "wrap",
        gap: "8px"
      }}>
        {filteredReceivers.map(receiver => (
          <label
            key={receiver}
            style={{
              display: "flex",
              alignItems: "center",
              padding: "4px 8px",
              background: hiddenReceivers.has(receiver) ? "#e9ecef" : "#e7f4e4",
              borderRadius: "4px",
              border: `1px solid ${hiddenReceivers.has(receiver) ? "#dee2e6" : "#c3e6cb"}`,
              cursor: "pointer",
              fontSize: "13px",
              userSelect: "none"
            }}
          >
            <input
              type="checkbox"
              checked={!hiddenReceivers.has(receiver)}
              onChange={() => toggleReceiver(receiver)}
              style={{
                marginRight: "6px",
                cursor: "pointer"
              }}
            />
            {receiver}
            {hiddenReceivers.has(receiver) && (
              <span style={{
                marginLeft: "6px",
                fontSize: "11px",
                color: "#6c757d",
                fontStyle: "italic"
              }}>
                (hidden)
              </span>
            )}
          </label>
        ))}
      </div>
    </div>
  );

  /* ----- card wrapper ----- */
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
      <div style={{ textAlign: "center", marginTop: 4, fontSize: 13, color: "#666" }}>
        Where countries receive their patents from
      </div>

      <div style={{ 
        height: Math.max(400, visibleReceivers.length * 40 + 100), 
        marginTop: 10,
        transition: "height 0.3s ease"
      }}>
        <Bar data={data} options={options} />
      </div>

      <FilterControls />

      <div style={{
        marginTop: "16px",
        padding: "8px",
        fontSize: "12px",
        color: "#6c757d",
        textAlign: "center",
        borderTop: "1px solid #e9ecef"
      }}>
         
      </div>
    </div>
  );
};

export default InternationalPatentFlowChart;