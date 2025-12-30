import React, { useEffect, useState } from "react";
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

Chart.register(BarElement, CategoryScale, LinearScale, Tooltip, Legend);

interface ApiResponse {
  origins: string[];
  filings: string[];
  matrix: number[][];
}

interface DomesticData {
  country: string;
  domesticCount: number;
  totalFromOrigin: number;
  domesticPercentage: number;
  totalPercentage: number;
}

interface AnalysisResult {
  totalOrigins: number;
  topDomestic: DomesticData[];
  topTotal: DomesticData[];
  mostInternational: DomesticData | null;
  overallDomesticRate: string;
  insight: string;
  totalFilingsAnalyzed: number;
}

export const InternationalProtectionMatrixChart: React.FC<{ port?: number }> = ({
  port: overridePort,
}) => {
  const [api, setApi] = useState<ApiResponse | null>(null);
  const [firstFilingsTotal, setFirstFilingsTotal] = useState<number | null>(null);
  const [loading, setLoad] = useState(true);
  const [error, setErr] = useState<string | null>(null);
  const [showComment, setShowComment] = useState(false);
  const [commentPosition, setCommentPosition] = useState({ x: 0, y: 0 });

  useEffect(() => {
    let dead = false;
    (async () => {
      try {
        let port = overridePort;
        if (!port) {
          const txt = await fetch("/backend_port.txt")
            .then(r => r.text())
            .catch(() => "");
          const n = parseInt(txt.trim(), 10);
          port = Number.isFinite(n) ? n : 49473;
        }

        const res = await fetch(
          `http://localhost:${port}/api/international_protection_matrix`
        );
        if (!res.ok)
          throw new Error(`HTTP ${res.status} ${res.statusText}`);
        const json: ApiResponse = await res.json();
        if (!dead) setApi(json);

        const countRes = await fetch(
          `http://localhost:${port}/api/first_filings/count`
        );
        if (!countRes.ok)
          throw new Error(`HTTP ${countRes.status} ${countRes.statusText}`);
        const countJson: { count: number } = await countRes.json();
        if (!dead) setFirstFilingsTotal(countJson.count ?? 0);
      } catch (e: any) {
        if (!dead) setErr(e.message ?? String(e));
      } finally {
        if (!dead) setLoad(false);
      }
    })();
    return () => { dead = true; };
  }, [overridePort]);

  const analyzeMatrix = (data: ApiResponse, totalFirstFilings: number): AnalysisResult => {
    if (!data || !data.origins || !data.matrix) {
      return {
        totalOrigins: 0,
        topDomestic: [],
        topTotal: [],
        mostInternational: null,
        overallDomesticRate: "0.0",
        insight: "No data available for analysis.",
        totalFilingsAnalyzed: 0
      };
    }

    // Calculate domestic filing percentages
    const domesticData: DomesticData[] = data.origins.map((origin, rowIdx) => {
      const filingIndex = data.filings.indexOf(origin);
      const domesticCount = filingIndex >= 0 ? data.matrix[rowIdx]?.[filingIndex] || 0 : 0;
      const totalFromOrigin = data.matrix[rowIdx]?.reduce((sum, val) => sum + val, 0) || 0;
      const domesticPercentage = totalFromOrigin > 0 ? (domesticCount / totalFromOrigin * 100) : 0;
      const totalPercentage = totalFirstFilings > 0 ? (totalFromOrigin / totalFirstFilings * 100) : 0;
      
      return {
        country: origin,
        domesticCount,
        totalFromOrigin,
        domesticPercentage,
        totalPercentage
      };
    });

    // Sort by domestic percentage (highest first)
    const sortedByDomestic = [...domesticData]
      .filter(item => item.totalFromOrigin > 0)
      .sort((a, b) => b.domesticPercentage - a.domesticPercentage);

    // Sort by total filings (highest first)
    const sortedByTotal = [...domesticData]
      .filter(item => item.totalFromOrigin > 0)
      .sort((a, b) => b.totalFromOrigin - a.totalFromOrigin);

    // Find most international origin (lowest domestic percentage among active)
    const mostInternationalCandidates = [...domesticData]
      .filter(item => item.totalFromOrigin > 0)
      .sort((a, b) => a.domesticPercentage - b.domesticPercentage)
      .filter(item => item.domesticPercentage < 100);

    const mostInternational = mostInternationalCandidates.length > 0 ? mostInternationalCandidates[0] : null;

    // Calculate overall domestic rate
    const totalDomestic = domesticData.reduce((sum, item) => sum + item.domesticCount, 0);
    const totalAllFilings = domesticData.reduce((sum, item) => sum + item.totalFromOrigin, 0);
    const overallDomesticRate = totalAllFilings > 0 ? (totalDomestic / totalAllFilings * 100) : 0;

    let insight = "";
    if (overallDomesticRate > 70) {
      insight = "Strong domestic focus with most patents filed locally.";
    } else if (overallDomesticRate > 40) {
      insight = "Balanced strategy between domestic and international protection.";
    } else {
      insight = "Highly international strategy with most patents filed abroad.";
    }

    return {
      totalOrigins: data.origins.length,
      topDomestic: sortedByDomestic.slice(0, 3),
      topTotal: sortedByTotal.slice(0, 3),
      mostInternational,
      overallDomesticRate: overallDomesticRate.toFixed(1),
      insight,
      totalFilingsAnalyzed: totalAllFilings
    };
  };

  // Use LoadingSpinner for loading state
  if (loading) return <LoadingSpinner text="Loading patent-protection matrix…" />;
  
  if (error) return <div style={{ color: "#EA3C53" }}>{error}</div>;
  if (!api || firstFilingsTotal === null) return null;

  const analysis = analyzeMatrix(api, firstFilingsTotal);

  const rowTotals = api.origins.map((_, rowIdx) => 
    api.matrix[rowIdx]?.reduce((sum, val) => sum + val, 0) ?? 0
  );

  const sortedIndices = api.origins
    .map((_, idx) => idx)
    .filter(idx => rowTotals[idx] > 0)
    .sort((a, b) => rowTotals[a] - rowTotals[b])
    .slice(-10);

  const filteredOrigins = sortedIndices.map(idx => api.origins[idx]);
  const filteredMatrix = sortedIndices.map(idx => api.matrix[idx]);
  const filingIndexByCode: Record<string, number> = Object.fromEntries(
    api.filings.map((c, i) => [c, i])
  );

  const rdLegend = filteredOrigins.map((code, rowIdx) => {
    const colIdx = filingIndexByCode[code];
    const row = filteredMatrix[rowIdx] ?? [];
    const diag = colIdx !== undefined ? (row[colIdx] ?? 0) : 0;
    const pct = firstFilingsTotal > 0 ? (diag / firstFilingsTotal) * 100 : 0;
    return { code, pct };
  }).sort((a, b) => b.pct - a.pct);

  const datasets = api.filings.map((filing, colIdx) => ({
    label: filing,
    data: filteredOrigins.map((_, rowIdx) => filteredMatrix[rowIdx]?.[colIdx] ?? 0),
    backgroundColor: colorForCountry(filing),
    borderWidth: 0,
  }));

  const data = {
    labels: filteredOrigins,
    datasets,
  };

  const options: any = {
    indexAxis: "y" as const,
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      tooltip: {
        mode: "nearest",
        intersect: false,
        callbacks: {
          label: (ctx: any) =>
            ` ${ctx.dataset.label} → ${ctx.parsed.x}`,
        },
      },
      legend: {
        position: "left",
        labels: { boxWidth: 14 },
      },
    },
    scales: {
      x: { stacked: true, ticks: { color: "#3B3C3D" } },
      y: { stacked: true, ticks: { color: "#3B3C3D", font: { weight: 600 } } },
    },
  };

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
          {/* <div style={{ 
            padding: "8px 28px", 
            fontWeight: 700, 
            fontSize: 20, 
            background: "#232526", 
            color: "#fff", 
            borderRadius: 10, 
            alignSelf: "center", 
            boxShadow: "0 1px 8px #bdd24816",
            display: "inline-block"
          }}>
           
          </div> */}
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
          onMouseEnter={(e) => {
            const rect = e.currentTarget.getBoundingClientRect();
            setCommentPosition({ x: rect.right, y: rect.top });
            setShowComment(true);
          }}
          onMouseLeave={() => setShowComment(false)}
        >
          i
        </div>
      </div>

      <div
        style={{
          textAlign: "center",
          marginTop: 4,
          fontSize: 13,
          color: "#666",
          marginBottom: 10
        }}
      >
        Origin countries and their international filing patterns
      </div>

      <div style={{ height: 500, marginTop: 10 }}>
        <Bar data={data} options={options} />
      </div>

      {/* Statistics Summary */}
      <div style={{
        display: "flex",
        justifyContent: "space-between",
        marginTop: 20,
        padding: "12px",
        background: "#f5f7fa",
        borderRadius: "10px",
        fontSize: 13
      }}>
        <div style={{ textAlign: "center", flex: 1 }}>
          <div style={{ fontWeight: 700, color: "#2E86AB", fontSize: "15px" }}>
            {analysis.totalOrigins}
          </div>
          <div style={{ color: "#666" }}>Origin Countries</div>
        </div>
        <div style={{ textAlign: "center", flex: 1, borderLeft: "1px solid #ddd", borderRight: "1px solid #ddd" }}>
          <div style={{ fontWeight: 700, color: "#A23B72", fontSize: "15px" }}>
            {analysis.overallDomesticRate}%
          </div>
          <div style={{ color: "#666" }}>Domestic Rate</div>
        </div>
        <div style={{ textAlign: "center", flex: 1 }}>
          <div style={{ fontWeight: 700, color: "#73B769", fontSize: "15px" }}>
            {analysis.totalFilingsAnalyzed}
          </div>
          <div style={{ color: "#666" }}>Total Filings</div>
        </div>
      </div>

      <div
        style={{
          marginTop: 16,
          padding: 16,
          border: "2px solid #BDD248",
          borderRadius: 12,
          background: "#fafafa",
        }}
      >
        <div
          style={{
            textAlign: "center",
            fontWeight: 700,
            fontSize: 16,
            color: "#232526",
            marginBottom: 12,
          }}
        >
          R&D Investment 
        </div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 10, justifyContent: "center" }}>
          {rdLegend.map((item) => (
            <div
              key={item.code}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 6,
                padding: "4px 8px",
                borderRadius: 8,
                background: "#f6f6f6ff",
                boxShadow: "0 1px 2px rgba(0,0,0,0.05)",
              }}
            >
              <div style={{ width: 12, height: 12, borderRadius: 3, background: colorForCountry(item.code) }} />
              <span style={{ fontWeight: 600, color: "#232526" }}>{item.code}</span>
              <span style={{ color: "#888" }}>→</span>
              <span style={{ color: "#555" }}>{item.pct.toFixed(1)}%</span>
            </div>
          ))}
        </div>
      </div>

      {/* Comment Block (on hover) */}
      {showComment && analysis && (
        <div
          style={{
            position: "fixed",
            left: commentPosition.x - 320,
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
            🌍 International Protection Analysis
          </div>
          <div style={{ color: "#555" }}>
            {/* This matrix shows patent filing strategies across {analysis.totalOrigins} origin countries. */}
            {/* <br /><br /> */}
            
            {/* <div style={{ fontWeight: 600, marginBottom: "8px", color: "#232526" }}>
              Key Observations:
            </div> */}
            <ul style={{ margin: "5px 0 10px 0", paddingLeft: "20px", fontSize: "13px" }}>
              {/* {analysis.topDomestic.length > 0 && (
                <li style={{ marginBottom: "6px" }}>
                  <strong>Domestic Focus:</strong> {analysis.topDomestic[0].country} files{" "}
                  <strong>{analysis.topDomestic[0].domesticPercentage.toFixed(1)}%</strong> of its patents domestically
                </li>
              )} */}
              {analysis.topTotal.length > 0 && (
                <li style={{ marginBottom: "6px" }}>
                  <strong>Most Active:</strong> {analysis.topTotal[0].country} has the most total filings{" "}
                  ({analysis.topTotal[0].totalFromOrigin} patents)
                </li>
              )}
              {/* {analysis.mostInternational && (
                <li style={{ marginBottom: "6px" }}>
                  <strong>Most International:</strong> {analysis.mostInternational.country} files only{" "}
                  <strong>{analysis.mostInternational.domesticPercentage.toFixed(1)}%</strong> domestically
                </li>
              )} */}
            </ul>
            
            <div style={{ 
              marginTop: "12px", 
              padding: "10px", 
              background: "#f8f9fa",
              borderRadius: "6px",
              borderLeft: "3px solid #2E86AB",
              fontSize: "13px"
            }}>
              <strong>Overall Strategy:</strong> {analysis.insight}
              <br />
              <div style={{ marginTop: "5px", color: "#666" }}>
                <strong>{analysis.overallDomesticRate}%</strong> of patents are filed domestically across all countries.
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
              Each horizontal bar shows where inventors from a country file their patents.
              <br />
              <em>Stacked segments = different filing destinations</em>
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
            💡 The R&D Investment section shows domestic filing percentages for each origin country
          </div>
        </div>
      )}
    </div>
  );
};

export default InternationalProtectionMatrixChart;