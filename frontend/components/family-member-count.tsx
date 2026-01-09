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
// register chart.js pieces only once
Chart.register(BarElement, CategoryScale, LinearScale, Tooltip, Legend);

interface ApiResponse {
  datasets: { data: number[]; label: string }[];
  labels: string[];
}

// Define proper type for Chart.js tick callback
interface TickContext {
  value: number;
  index: number;
  label: string;
}

export const FamilyMemberCountChart: React.FC<{ port?: number; onHoverComment?: (text: string) => void }> = ({
  port: overridePort,
  onHoverComment,
}) => {
  const [chartData, setChartData] = useState<ApiResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [showComment, setShowComment] = useState(false);
  const [commentPosition, setCommentPosition] = useState({ x: 0, y: 0 });

  /* ─── fetch once ─── */
  useEffect(() => {
    let dead = false;

    (async () => {
      try {
        /* 1️⃣  Resolve backend port */
        let port = overridePort;
        if (!port) {
          const txt = await fetch("/backend_port.txt")
            .then((r) => r.text())
            .catch(() => "");
          const n = parseInt(txt.trim(), 10);
          port = Number.isFinite(n) ? n : 49473;
        }

        /* 2️⃣  Fetch data */
        const res = await fetch(
          `http://localhost:${port}/api/family_member_counts`
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
    
    // Calculate total family members
    const totalFamilyMembers = dataPoints.reduce((sum, val) => sum + val, 0);
    
    // Find top countries
    const countryStats = labels.map((label, index) => ({
      country: label,
      count: dataPoints[index],
      percentage: (dataPoints[index] / totalFamilyMembers) * 100
    }));
    
    // Sort by count descending
    const sortedCountries = [...countryStats].sort((a, b) => b.count - a.count);
    
    // Top 3 countries
    const topCountries = sortedCountries.slice(0, 3);
    
    // Average family members per country
    const averagePerCountry = totalFamilyMembers / labels.length;
    
    // Calculate concentration: percentage by top 3 countries
    const top3Concentration = topCountries.reduce((sum, country) => sum + country.percentage, 0);
    
    // Number of countries with family members
    const countriesWithData = dataPoints.filter(count => count > 0).length;
    
    // Generate insight based on concentration
    let insight = "";
    if (top3Concentration > 70) {
      insight = "High concentration - Most family members are concentrated in a few key countries.";
    } else if (top3Concentration > 50) {
      insight = "Moderate concentration - Family members are somewhat concentrated in key countries.";
    } else {
      insight = "Low concentration - Family members are widely distributed across many countries.";
    }
    
    return {
      totalFamilyMembers,
      totalCountries: labels.length,
      countriesWithData,
      averagePerCountry: averagePerCountry.toFixed(1),
      topCountries,
      top3Concentration: top3Concentration.toFixed(1),
      insight
    };
  }, [chartData]);

  const analysisText = useMemo(() => {
    if (!analysis) return "No data available for analysis.";
    const lines: string[] = [];
    lines.push(`Total family members: ${analysis.totalFamilyMembers}`);
    lines.push(`Countries covered: ${analysis.countriesWithData}/${analysis.totalCountries}`);
    lines.push(`Average per country: ${analysis.averagePerCountry}`);
    if (analysis.topCountries?.length) {
      lines.push("");
      lines.push(
        `Top countries: ${analysis.topCountries
          .map(c => `${c.country} (${c.count})`)
          .join(", ")}`
      );
      lines.push(`Top 3 concentration: ${analysis.top3Concentration}%`);
    }
    lines.push("");
    lines.push(`Insight: ${analysis.insight}`);
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

  if (loading) return <LoadingSpinner text="Loading family-member counts…" />;
  if (error) return <div style={{ color: "#EA3C53" }}>{error}</div>;
  if (!chartData) return null;

  /* ─── build chart.js props ─── */
  const data = {
    labels: chartData.labels,
    datasets: [
      {
        label: chartData.datasets[0].label,
        data: chartData.datasets[0].data,
        backgroundColor: "#BDD248",
        borderColor: "#8AA73A",
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
      const label = chartData.labels?.[idx];
      const count = chartData.datasets?.[0]?.data?.[idx];
      if (label === undefined || count === undefined) return;

      const pointText = `${label}: ${count} family members`;
      const fullText = analysisText ? `${pointText}\n\n${analysisText}` : pointText;
      onHoverComment(fullText);
    },
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        callbacks: {
          label: (ctx: any) => ` ${ctx.parsed.y} family members`,
        },
      },
    },
    scales: {
      x: {
        grid: { display: false },
        ticks: { 
          color: "#3B3C3D", 
          font: { weight: 600 },
          maxTicksLimit: 10, // Limit number of labels shown
          callback: function(value: number | string, index: number, ticks: any[]): string {
            // Get the label using the index
            const label = chartData.labels[index];
            if (typeof label === 'string') {
              if (label.length > 8) {
                return label.substring(0, 8) + '...';
              }
              return label;
            }
            return String(label || '');
          }
        },
      },
      y: {
        beginAtZero: true,
        ticks: { color: "#3B3C3D" },
        title: {
          display: true,
          text: "Family Members",
          color: "#3B3C3D"
        }
      },
    },
  };

  /* ─── card wrapper matches your other widgets ─── */
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
        Number of family members per country
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
          <div style={{ textAlign: "center", flex: 1 }}>
            <div style={{ fontWeight: 700, color: "#2E86AB", fontSize: "15px" }}>
              {analysis.totalFamilyMembers}
            </div>
            <div style={{ color: "#666" }}>Total Members</div>
          </div>
          <div style={{ textAlign: "center", flex: 1, borderLeft: "1px solid #ddd", borderRight: "1px solid #ddd" }}>
            <div style={{ fontWeight: 700, color: "#A23B72", fontSize: "15px" }}>
              {analysis.countriesWithData}
            </div>
            <div style={{ color: "#666" }}>Active Countries</div>
          </div>
          <div style={{ textAlign: "center", flex: 1 }}>
            <div style={{ fontWeight: 700, color: "#73B769", fontSize: "15px" }}>
              {analysis.averagePerCountry}
            </div>
            <div style={{ color: "#666" }}>Avg. per Country</div>
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
            🌍 Family Member Distribution
          </div>
          <div style={{ color: "#555" }}>
            <ul style={{ margin: "5px 0 10px 0", paddingLeft: "20px", fontSize: "13px" }}>
              {analysis.topCountries.length > 0 && (
                <>
                  <li style={{ marginBottom: "6px" }}>
                    <strong>Top Countries:</strong> {analysis.topCountries.map(c => c.country).join(", ")}
                  </li>
                  <li style={{ marginBottom: "6px" }}>
                    <strong>Top 3 Concentration:</strong> {analysis.top3Concentration}% of family members
                  </li>
                </>
              )}
              {/* <li style={{ marginBottom: "6px" }}>
                <strong>Coverage:</strong> {analysis.countriesWithData} of {analysis.totalCountries} countries have family members
              </li> */}
              <li style={{ marginBottom: "6px" }}>
                <strong>Distribution:</strong> Average of {analysis.averagePerCountry} members per country
              </li>
            </ul>
            
            <div style={{ 
              marginTop: "12px", 
              padding: "10px", 
              background: "#f8f9fa",
              borderRadius: "6px",
              borderLeft: "3px solid #2E86AB",
              fontSize: "13px"
            }}>
              <strong>Geographic Strategy:</strong> {analysis.insight}
              <br />
              <div style={{ marginTop: "5px", color: "#666" }}>
                Analysis based on {analysis.totalFamilyMembers} family members across {analysis.countriesWithData} countries.
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
              Each bar shows the number of patent family members in a specific country.
              <br />
              <em>Taller bars = more family members in that country</em>
              <br />
              <em>Country codes = ISO country codes (e.g., US, CN, JP)</em>
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
            💡 Family members = individual patents within patent families
          </div>
        </div>
      )}
    </div>
  );
};

export default FamilyMemberCountChart;