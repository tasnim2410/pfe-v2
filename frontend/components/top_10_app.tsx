import React, { useEffect, useState } from "react";
import LoadingSpinner from "./LoadingSpinner";
import { Bar } from "react-chartjs-2";
import { Chart, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from "chart.js";

// Register necessary Chart.js components
Chart.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

export const Top10Applicants: React.FC<{ onHoverComment?: (text: string) => void }> = ({ onHoverComment }) => {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  
  // Comment section state
  const [showGeneralComment, setShowGeneralComment] = useState(false);
  const [commentPosition, setCommentPosition] = useState({ x: 0, y: 0 });

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

  const analysisText = (() => {
    if (!data || !data.labels || !data.datasets || !data.datasets[0]?.data) {
      return "No applicant data available for analysis.";
    }

    const labels = data.labels as string[];
    const patentCounts = data.datasets[0].data as number[];
    const top3Count = Math.min(3, labels.length);

    const topApplicants: { name: string; count: number; rank: number }[] = [];
    for (let i = 0; i < top3Count; i++) {
      topApplicants.push({ name: labels[i], count: patentCounts[i], rank: i + 1 });
    }

    const top1Percentage = data.top10_total > 0 ? (topApplicants[0].count / data.top10_total * 100).toFixed(1) : "0";

    const lines: string[] = [];
    lines.push("The top applicants in this technology are:");
    lines.push("");
    for (const a of topApplicants) {
      lines.push(`${a.rank}. ${a.name}: ${a.count} patents`);
    }
    lines.push("");
    lines.push(`Together, these top 3 applicants hold ${top1Percentage}% of patents among the top 10.`);
    lines.push("");
    if (data.percentage !== undefined) {
      const pct = String(data.percentage);
      const level = parseFloat(pct) > 50 ? "strong" : parseFloat(pct) > 30 ? "moderate" : "some";
      lines.push(`The top 10 applicants represent ${pct}% of the top 100 applicants, showing ${level} concentration of patent ownership.`);
    }
    return lines.join("\n").trim();
  })();

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
    onHover: (_event: any, activeElements: any[]) => {
      if (!onHoverComment) return;
      if (!activeElements || activeElements.length === 0) return;

      const el = activeElements[0];
      const idx = el.index;

      const label = data?.labels?.[idx];
      const count = data?.datasets?.[0]?.data?.[idx];
      if (label === undefined || count === undefined) return;

      const pointText = `${label}: ${count} patents`;
      const fullText = analysisText ? `${pointText}\n\n${analysisText}` : pointText;
      onHoverComment(fullText);
    },
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
      margin: "0 auto",
      position: "relative" // Added for positioning the info icon
    }}>
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
            🏆 Top Applicants Analysis
          </div>
          <div style={{ color: "#555" }}>
            {(() => {
              if (!data || !data.labels || !data.datasets || !data.datasets[0]?.data) {
                return "No applicant data available for analysis.";
              }
              
              const labels = data.labels;
              const patentCounts = data.datasets[0].data;
              const top3Count = Math.min(3, labels.length);
              
              // Get top 3 applicants (already sorted by API)
              const topApplicants = [];
              for (let i = 0; i < top3Count; i++) {
                topApplicants.push({
                  name: labels[i],
                  count: patentCounts[i],
                  rank: i + 1
                });
              }
              
              // Calculate dominance percentage
              const top1Percentage = data.top10_total > 0 ? 
                (topApplicants[0].count / data.top10_total * 100).toFixed(1) : "0";
              
              return (
                <>
                  The top applicants in this technology are:
                  <br /><br />
                  {topApplicants.map((applicant, index) => (
                    <div key={index} style={{ marginBottom: "4px" }}>
                      <strong>{applicant.rank}. {applicant.name}</strong>: {applicant.count} patents
                    </div>
                  ))}
                  <br />
                  Together, these top 3 applicants hold{" "}
                  <strong>{top1Percentage}%</strong> of patents among the top 10.
                  <br /><br />
                  The top 10 applicants represent{" "}
                  <strong>{data.percentage}%</strong> of the top 100 applicants, 
                  showing {parseFloat(data.percentage) > 50 ? "strong" : parseFloat(data.percentage) > 30 ? "moderate" : "some"} 
                  concentration of patent ownership.
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
            💡 <strong>Tip:</strong> The data is sorted by patent count (highest to lowest)
          </div>
        </div>
      )}
    </div>
  );
};

export default Top10Applicants;