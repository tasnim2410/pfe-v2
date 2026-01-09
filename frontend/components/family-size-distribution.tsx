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
import LoadingSpinner from "./LoadingSpinner"; 

// Register Chart.js parts once
Chart.register(BarElement, CategoryScale, LinearScale, Tooltip, Legend);

interface ApiResponse {
  datasets: { data: number[]; label: string }[];
  labels: (number | string)[];
}

export const FamilySizeDistributionChart: React.FC<{ port?: number; onHoverComment?: (text: string) => void }> = ({
  port: overridePort,
  onHoverComment,
}) => {
  const [chartData, setChartData] = useState<ApiResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showComment, setShowComment] = useState(false);
  const [commentPosition, setCommentPosition] = useState({ x: 0, y: 0 });

  /* ─── Fetch once ─── */
  useEffect(() => {
    let dead = false;

    (async () => {
      try {
        /* 1️⃣ Resolve backend port */
        let port = overridePort;
        if (!port) {
          const txt = await fetch("/backend_port.txt")
            .then((r) => r.text())
            .catch(() => "");
          const n = parseInt(txt.trim(), 10);
          port = Number.isFinite(n) ? n : 49473;
        }

        /* 2️⃣ Fetch data */
        const res = await fetch(
          `http://localhost:${port}/api/family_size_distribution`
        );
        if (!res.ok)
          throw new Error(`HTTP ${res.status} ${res.statusText} while fetching`);
        const json: ApiResponse = await res.json();

        if (!dead) setChartData(json);
      } catch (e: any) {
        if (!dead) setError(e.message ?? String(e));
      } finally {
        if (!dead) setLoading(false);
      }
    })();

    return () => {
      dead = true;
    };
  }, [overridePort]);

  /* ─── Analysis calculations ─── */
  const analysis = useMemo(() => {
    if (!chartData || !chartData.datasets[0] || !chartData.labels) return null;
    
    const dataPoints = chartData.datasets[0].data;
    const labels = chartData.labels;
    
    // Calculate total families
    const totalFamilies = dataPoints.reduce((sum, val) => sum + val, 0);
    
    // Calculate weighted sum for average family size
    let weightedSum = 0;
    for (let i = 0; i < Math.min(dataPoints.length, labels.length); i++) {
      const familySize = parseFloat(labels[i] as string) || 0;
      weightedSum += familySize * dataPoints[i];
    }
    
    const averageFamilySize = totalFamilies > 0 ? (weightedSum / totalFamilies) : 0;
    
    // Find maximum count and corresponding family size
    let maxCount = 0;
    let maxFamilySize = 0;
    dataPoints.forEach((count, idx) => {
      if (count > maxCount) {
        maxCount = count;
        maxFamilySize = parseFloat(labels[idx] as string) || 0;
      }
    });
    
    // Find median family size
    let cumulative = 0;
    let medianFamilySize = 0;
    const medianThreshold = totalFamilies / 2;
    for (let i = 0; i < dataPoints.length; i++) {
      cumulative += dataPoints[i];
      if (cumulative >= medianThreshold) {
        medianFamilySize = parseFloat(labels[i] as string) || 0;
        break;
      }
    }
    
    // Calculate concentration: percentage of families in the most common size
    const concentrationPercentage = (maxCount / totalFamilies) * 100;
    
    // Generate insight based on average family size
    let insight = "";
    if (averageFamilySize < 2) {
      insight = "Very small families - Most inventions have little international protection.";
    } else if (averageFamilySize < 5) {
      insight = "Moderate family sizes - Balanced international filing strategy.";
    } else {
      insight = "Large families - Strong international protection strategy with extensive patent families.";
    }
    
    return {
      totalFamilies,
      averageFamilySize: averageFamilySize.toFixed(1),
      mostCommonSize: maxFamilySize,
      mostCommonCount: maxCount,
      medianFamilySize,
      concentrationPercentage: concentrationPercentage.toFixed(1),
      insight
    };
  }, [chartData]);

  const analysisText = useMemo(() => {
    if (!analysis) return "No data available for analysis.";
    const lines: string[] = [];
    lines.push(`Average family size: ${analysis.averageFamilySize}`);
    lines.push(`Most common size: ${analysis.mostCommonSize} (${analysis.mostCommonCount} families)`);
    lines.push(`Concentration: ${analysis.concentrationPercentage}%`);
    if (analysis.medianFamilySize > 0) {
      lines.push(`Median family size: ${analysis.medianFamilySize}`);
    }
    lines.push("");
    lines.push(`Insight: ${analysis.insight}`);
    lines.push(`Total families analyzed: ${analysis.totalFamilies}`);
    return lines.join("\n").trim();
  }, [analysis]);

  /* ─── Handle comment hover ─── */
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

  if (loading) return <LoadingSpinner text="Loading family-size distribution…" />;
  if (error) return <div style={{ color: "#EA3C53" }}>{error}</div>;
  if (!chartData) return null;

  /* ─── Build chart.js props ─── */
  const data = {
    labels: chartData.labels.map(String), // ensure strings for category axis
    datasets: [
      {
        label: chartData.datasets[0].label,
        data: chartData.datasets[0].data,
        backgroundColor: "#F2D15F",
        borderColor: "#CBAA40",
        borderWidth: 1,
      },
    ],
  };

  const options: any = {
    responsive: true,
    maintainAspectRatio: false,
    onHover: (_event: any, activeElements: any[]) => {
      if (!onHoverComment) return;
      if (!activeElements || activeElements.length === 0) return;

      const el = activeElements[0];
      const idx = el.index;
      const label = data.labels?.[idx];
      const count = data.datasets?.[0]?.data?.[idx];
      if (label === undefined || count === undefined) return;

      const pointText = `Family size ${label}: ${count} families`;
      const fullText = analysisText ? `${pointText}\n\n${analysisText}` : pointText;
      onHoverComment(fullText);
    },
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: (ctx: any) => ` ${ctx.parsed.y} patents`,
        },
      },
    },
    scales: {
      x: {
        title: { text: "Family Size", display: true, color: "#3B3C3D" },
        grid: { display: false },
        ticks: { color: "#3B3C3D", font: { weight: 600 } },
      },
      y: {
        beginAtZero: true,
        title: { text: "Count", display: true, color: "#3B3C3D" },
        ticks: { color: "#3B3C3D" },
      },
    },
  };

  /* ─── Card wrapper consistent with other widgets ─── */
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
          {/* Title placeholder - optional */}
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

      {/* Subtitle */}
      <div style={{ 
        textAlign: "center", 
        marginTop: 4, 
        fontSize: 13, 
        color: "#666",
        marginBottom: 10 
      }}>
        Distribution of patent family sizes
      </div>

      <div style={{ height: 280, marginTop: 10 }}>
        <Bar data={data} options={options} />
      </div>

      {/* Stats summary */}
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
          <div style={{ textAlign: "center", flex: 1, borderRight: "1px solid #ddd" }}>
            <div style={{ fontWeight: 700, color: "#A23B72", fontSize: "15px" }}>
              {analysis.mostCommonSize}
            </div>
            <div style={{ color: "#666" }}>Most Common Size</div>
          </div>
          <div style={{ textAlign: "center", flex: 1 }}>
            <div style={{ fontWeight: 700, color: "#73B769", fontSize: "15px" }}>
              {analysis.averageFamilySize}
            </div>
            <div style={{ color: "#666" }}>Avg. Family Size</div>
          </div>
        </div>
      )}

      {/* Comment Block (on hover) */}
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
            👨‍👩‍👧‍👦 Patent Family Size Analysis
          </div>
          <div style={{ color: "#555" }}>
            <ul style={{ margin: "5px 0 10px 0", paddingLeft: "20px", fontSize: "13px" }}>
              <li style={{ marginBottom: "6px" }}>
                <strong>Average Family Size:</strong> {analysis.averageFamilySize} patents per family
              </li>
              <li style={{ marginBottom: "6px" }}>
                <strong>Most Common:</strong> {analysis.mostCommonCount} families have {analysis.mostCommonSize} members
              </li>
              <li style={{ marginBottom: "6px" }}>
                <strong>Concentration:</strong> {analysis.concentrationPercentage}% of families have {analysis.mostCommonSize} members
              </li>
              {analysis.medianFamilySize > 0 && (
                <li style={{ marginBottom: "6px" }}>
                  <strong>Median Family Size:</strong> {analysis.medianFamilySize} patents
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
              <strong>Strategy Insight:</strong> {analysis.insight}
              <br />
              <div style={{ marginTop: "5px", color: "#666" }}>
                Analysis based on {analysis.totalFamilies} patent families.
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
              Each bar shows how many patent families have a specific number of members.
              <br />
              <em>Taller bars = more families of that size</em>
              <br />
              <em>Wider families = more international protection</em>
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
            💡 Larger families indicate stronger international protection strategies
          </div>
        </div>
      )}
    </div>
  );
};

export default FamilySizeDistributionChart;