import React, { useEffect, useState, useMemo, useCallback } from "react";
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
import LoadingSpinner from "./LoadingSpinner"; // Add this import

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
  const [showComment, setShowComment] = useState(false);
  const [commentPosition, setCommentPosition] = useState({ x: 0, y: 0 });

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

  /* ----- Analysis calculations ----- */
  const analysis = useMemo(() => {
    if (!api) return null;
    
    // Calculate total patents received by each country
    const receiverStats = api.receivers.map((receiver, idx) => {
      const totalReceived = api.matrix[idx]?.reduce((sum, val) => sum + val, 0) || 0;
      return { receiver, totalReceived };
    }).filter(stat => stat.totalReceived > 0);
    
    // Sort by total received
    const sortedReceivers = [...receiverStats].sort((a, b) => b.totalReceived - a.totalReceived);
    
    // Find top 3 receivers
    const topReceivers = sortedReceivers.slice(0, 3);
    
    // Calculate total patents in the system
    const totalPatents = api.matrix.flat().reduce((sum, val) => sum + val, 0);
    
    // Calculate concentration: percentage of patents received by top 3 receivers
    const top3Percentage = topReceivers.reduce((sum, stat) => sum + stat.totalReceived, 0) / totalPatents * 100;
    
    // Find strongest origin for each receiver
    const receiverDependencies = api.receivers.map((receiver, rowIdx) => {
      const row = api.matrix[rowIdx];
      if (!row) return null;
      
      let maxVal = 0;
      let maxOrigin = "";
      
      api.origins.forEach((origin, colIdx) => {
        if (row[colIdx] > maxVal) {
          maxVal = row[colIdx];
          maxOrigin = origin;
        }
      });
      
      return { receiver, mainOrigin: maxOrigin, mainPercentage: maxVal / (row.reduce((s, v) => s + v, 0) || 1) * 100 };
    }).filter(Boolean);
    
    // Find most dependent receiver
    const mostDependent = receiverDependencies.sort((a, b) => b!.mainPercentage - a!.mainPercentage)[0];
    
    return {
      totalOrigins: api.origins.length,
      totalReceivers: receiverStats.length,
      topReceivers,
      top3Percentage: top3Percentage.toFixed(1),
      mostDependent,
      totalPatents,
      insight: top3Percentage > 50 ? 
        "High concentration - Most patents flow to a few key markets" :
        top3Percentage > 30 ?
        "Moderate concentration - Patent distribution is relatively balanced" :
        "Low concentration - Patent filings are widely distributed across markets"
    };
  }, [api]);

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

  /* ----- Handle comment hover ----- */
  const handleCommentHover = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const commentWidth = 320;
    const commentHeight = 400;
    
    // Default: show to the right of the icon
    let x = rect.right + 10; // 10px offset from the icon
    let y = rect.top;
    
    // Check if going off right edge of screen
    if (x + commentWidth > window.innerWidth) {
      // Show to the left of the icon instead
      x = rect.left - commentWidth - 10;
    }
    
    // Check if going off bottom edge of screen
    if (y + commentHeight > window.innerHeight) {
      y = window.innerHeight - commentHeight - 10;
    }
    
    setCommentPosition({ x, y });
    setShowComment(true);
  }, []);

  // Use LoadingSpinner for loading state
  if (loading) return <LoadingSpinner text="Loading international patent flow…" />;
  
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
        position: "relative"
      }}
    >
      {/* Header with title and info icon */}
      <div style={{ 
        display: "flex", 
        justifyContent: "space-between", 
        alignItems: "center", 
        width: "100%",
        marginTop: 12,
        marginBottom: 8
      }}>
        <div>
          {/* Title placeholder if needed */}
        </div>
        
        {/* Info icon */}
        <div
          style={{
            width: 22,
            height: 22,
            borderRadius: "50%",
            backgroundColor: "#f0f0f0",
            border: "1px solid #ccc",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            cursor: "pointer",
            fontSize: "12px",
            fontWeight: "bold",
            color: "#666",
            zIndex: 10,
            marginRight: 10
          }}
          onMouseEnter={handleCommentHover}
          onMouseLeave={() => setShowComment(false)}
          role="button"
          aria-label="Show analysis information"
          tabIndex={0}
          onKeyDown={(e) => e.key === 'Enter' && setShowComment(!showComment)}
          onClick={() => setShowComment(!showComment)}
        >
          i
        </div>
      </div>

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

      {/* Stats summary - ABOVE FILTER */}
      {analysis && (
        <div style={{
          display: "flex",
          justifyContent: "space-between",
          marginTop: 20,
          padding: "12px",
          background: "#f5f7fa",
          borderRadius: "10px",
          fontSize: 13,
          boxShadow: "0 1px 3px rgba(0,0,0,0.05)"
        }}>
          <div style={{ textAlign: "center", flex: 1 }}>
            <div style={{ fontWeight: 700, color: "#2E86AB", fontSize: "15px" }}>
              {analysis.totalOrigins}
            </div>
            <div style={{ color: "#666" }}>Origin Countries</div>
          </div>
          <div style={{ textAlign: "center", flex: 1, borderLeft: "1px solid #ddd", borderRight: "1px solid #ddd" }}>
            <div style={{ fontWeight: 700, color: "#A23B72", fontSize: "15px" }}>
              {analysis.top3Percentage}%
            </div>
            <div style={{ color: "#666" }}>Top 3 Concentration</div>
          </div>
          <div style={{ textAlign: "center", flex: 1 }}>
            <div style={{ fontWeight: 700, color: "#73B769", fontSize: "15px" }}>
              {analysis.totalPatents}
            </div>
            <div style={{ color: "#666" }}>Total Patent Flows</div>
          </div>
        </div>
      )}

      {/* Filter controls - NOW BELOW STATS */}
      <FilterControls />

      {/* Comment Block (on hover) - FIXED POSITIONING */}
      {showComment && analysis && (
        <div
          style={{
            position: "fixed",
            left: commentPosition.x,
            top: commentPosition.y,
            width: 320,
            background: "#fff",
            border: "1px solid #ddd",
            borderRadius: "10px",
            padding: "18px",
            boxShadow: "0 6px 20px rgba(0,0,0,0.15)",
            zIndex: 10000,
            fontSize: "14px",
            lineHeight: "1.5"
          }}
          onMouseEnter={() => setShowComment(true)}
          onMouseLeave={() => setShowComment(false)}
        >
          <div style={{ fontWeight: "bold", marginBottom: "10px", color: "#333", fontSize: "16px" }}>
            📊 International Patent Flow Analysis
          </div>
          <div style={{ color: "#555" }}>
            <ul style={{ margin: "5px 0 10px 0", paddingLeft: "20px", fontSize: "13px" }}>
              {analysis.topReceivers.length > 0 && (
                <li style={{ marginBottom: "6px" }}>
                  <strong>Top Receivers:</strong> {analysis.topReceivers.map(r => r.receiver).join(", ")} receive{" "}
                  <strong>{analysis.top3Percentage}%</strong> of all patents
                </li>
              )}
              {analysis.mostDependent && (
                <li style={{ marginBottom: "6px" }}>
                  <strong>Most Dependent:</strong> {analysis.mostDependent.receiver} gets{" "}
                  <strong>{analysis.mostDependent.mainPercentage.toFixed(1)}%</strong> of its patents from {analysis.mostDependent.mainOrigin}
                </li>
              )}
            </ul>
            
            <div style={{ 
              marginTop: "12px", 
              padding: "10px", 
              background: "#f8f9fa",
              borderRadius: "6px",
              borderLeft: "3px solid #2E86AB",
              fontSize: "13px"
            }}>
              <strong>Market Concentration:</strong> {analysis.insight}
              <br />
              <div style={{ marginTop: "5px", color: "#666" }}>
                Analysis based on {analysis.totalPatents} patent flows across {analysis.totalOrigins} origin countries and {analysis.totalReceivers} receiving countries.
              </div>
            </div>

            <div style={{ 
              marginTop: "12px", 
              padding: "10px", 
              background: "#fff0f0",
              borderRadius: "6px",
              borderLeft: "3px solid #A23B72",
              fontSize: "12px"
            }}>
              <strong>Chart Interpretation:</strong>
              <br />
              Each horizontal bar shows where a receiving country gets its patents from.
              <br />
              <em>Stacked segments = different origin countries</em>
              <br />
              <em>Longer bars = more patents received</em>
            </div>
          </div>
          <div style={{ 
            marginTop: "12px", 
            fontSize: "12px", 
            color: "#888",
            fontStyle: "italic",
            borderTop: "1px solid #eee",
            paddingTop: "10px"
          }}>
            💡 Use the filter controls to focus on specific receiving countries of interest
          </div>
        </div>
      )}

      <div style={{
        marginTop: "16px",
        padding: "8px",
        fontSize: "12px",
        color: "#6c757d",
        textAlign: "center",
        borderTop: "1px solid #e9ecef"
      }}>
        {/* Footer placeholder */}
      </div>
    </div>
  );
};

export default InternationalPatentFlowChart;