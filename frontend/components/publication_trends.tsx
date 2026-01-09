import React, { useEffect, useState } from "react";
import LoadingSpinner from "./LoadingSpinner";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  LineElement,
  PointElement,
  CategoryScale,
  LinearScale,
  Tooltip,
  Legend,
  Title,
} from "chart.js";

// Register Chart.js modules
ChartJS.register(LineElement, PointElement, CategoryScale, LinearScale, Tooltip, Legend, Title);

const CHART_BG = "#fff";
const ACCENT = "#232526";
const HIGHLIGHT = "#BDD248";

type ChartProps = { width?: number; height?: number; onHoverComment?: (text: string) => void };
const PublicationTrends: React.FC<ChartProps> = ({ width, height, onHoverComment }) => {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [showComment, setShowComment] = useState(false);
  const [commentPosition, setCommentPosition] = useState({ x: 0, y: 0 });

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setErr(null);

    fetch("/backend_port.txt")
      .then((res) => res.text())
      .then((port) =>
        fetch(`http://localhost:${port.trim()}/api/patents/first_filing_years`)
      )
      .then((res) => res.json())
      .then((json) => {
        if (!cancelled) setData(json);
        setLoading(false);
      })
      .catch(() => {
        if (!cancelled) setErr("Failed to fetch publication trend data.");
        setLoading(false);
      });

    return () => { cancelled = true; };
  }, []);

  // Function to analyze publication trends and find significant rises
  const analyzeTrends = (labels: string[], dataset: any) => {
    if (!labels || !dataset || !dataset.data) {
      return { overallTrend: "No data available", significantRises: [] };
    }

    const dataPoints = dataset.data;
    const years = labels;
    
    if (dataPoints.length < 2) {
      return { 
        overallTrend: "Insufficient data for trend analysis", 
        significantRises: [] 
      };
    }

    // Calculate overall trend
    const firstValue = dataPoints[0];
    const lastValue = dataPoints[dataPoints.length - 1];
    const overallChange = lastValue - firstValue;
    const overallPercentage = firstValue > 0 ? (overallChange / firstValue) * 100 : 0;
    
    let overallTrend = "";
    if (overallChange > 0) {
      overallTrend = `an overall increase of ${overallChange} patents (${overallPercentage.toFixed(1)}%)`;
    } else if (overallChange < 0) {
      overallTrend = `a decrease of ${Math.abs(overallChange)} patents (${Math.abs(overallPercentage).toFixed(1)}%)`;
    } else {
      overallTrend = "stable publication numbers";
    }

    // Find significant year-over-year rises
    const significantRises = [];
    for (let i = 1; i < dataPoints.length; i++) {
      const prevValue = dataPoints[i - 1];
      const currValue = dataPoints[i];
      const change = currValue - prevValue;
      
      // Only consider positive changes
      if (change > 0) {
        const percentageChange = prevValue > 0 ? (change / prevValue) * 100 : 100;
        
        // Define what constitutes a "significant" rise
        const isSignificant = change >= 10 || percentageChange >= 25;
        
        if (isSignificant) {
          significantRises.push({
            year: years[i],
            prevYear: years[i - 1],
            change,
            percentage: percentageChange.toFixed(1),
            currentValue: currValue
          });
        }
      }
    }

    // Sort by biggest increases (absolute change)
    significantRises.sort((a, b) => b.change - a.change);

    return {
      overallTrend,
      significantRises: significantRises.slice(0, 3), // Top 3 rises
      peakYear: years[dataPoints.indexOf(Math.max(...dataPoints))],
      peakValue: Math.max(...dataPoints),
      totalYears: years.length
    };
  };

  if (loading) return <LoadingSpinner text="Loading Publication Trends..." />;
  if (err) return <div style={{ color: "#EA3C53", textAlign: "center" }}>{err}</div>;
  if (!data) return null;

  // Analyze trends for the comment
  const analysis = analyzeTrends(data.labels, data.datasets[0]);

  const analysisText = (() => {
    const lines: string[] = [];
    lines.push(`Over ${analysis.totalYears} years of data, the trend shows ${analysis.overallTrend}.`);
    if (analysis.peakYear) {
      lines.push("");
      lines.push(`The peak year was ${analysis.peakYear} with ${analysis.peakValue} patents.`);
    }
    lines.push("");
    if (analysis.significantRises?.length) {
      lines.push("Significant Publication Rises:");
      for (const rise of analysis.significantRises) {
        lines.push(`${rise.year}: Increased by ${rise.change} patents (${rise.percentage}%) from ${rise.prevYear}`);
      }
    } else {
      lines.push("No significant year-over-year rises detected (threshold: ≥10 patents or ≥25% increase).");
    }
    return lines.join("\n").trim();
  })();

  // Chart.js data
  const chartData = {
    labels: data.labels,
    datasets: data.datasets.map((ds: any) => ({
      ...ds,
      fill: false,
      borderColor: HIGHLIGHT,
      backgroundColor: HIGHLIGHT,
      pointBackgroundColor: ACCENT,
      pointRadius: 4,
      pointHoverRadius: 7,
      tension: 0.2,
    })),
  };

  // Chart.js options
  const options = {
    responsive: true,
    onHover: (_event: any, activeElements: any[]) => {
      if (!onHoverComment) return;
      if (!activeElements || activeElements.length === 0) return;

      const el = activeElements[0];
      const idx = el.index;
      const datasetIndex = el.datasetIndex ?? 0;

      const year = data?.labels?.[idx];
      const value = data?.datasets?.[datasetIndex]?.data?.[idx];
      if (year === undefined || value === undefined) return;

      const pointText = `${year}: ${value} patents`;
      const fullText = analysisText ? `${pointText}\n\n${analysisText}` : pointText;
      onHoverComment(fullText);
    },
    plugins: {
      legend: { display: false },
      title: {
        display: true,
        text: "Publication Trend by First Filing Year",
        color: ACCENT,
        font: { size: 22, weight: 700 },
        padding: { top: 15, bottom: 30 },
      },
      tooltip: {
        callbacks: {
          label: (ctx: any) => ` ${ctx.parsed.y} patents`,
        },
        backgroundColor: "#232526",
        titleColor: "#fff",
        bodyColor: "#BDD248",
      },
    },
    scales: {
      x: {
        title: {
          display: true,
          text: "Year",
          color: ACCENT,
          font: { size: 16, weight: 700 }
        },
        ticks: { color: ACCENT, font: { size: 13, weight: 600 } },
        grid: { color: "#eee" }
      },
      y: {
        title: {
          display: true,
          text: "Number of Patents",
          color: ACCENT,
          font: { size: 16, weight: 700 }
        },
        beginAtZero: true,
        ticks: { color: ACCENT, font: { size: 13, weight: 600 } },
        grid: { color: "#eee" }
      }
    }
  };

  return (
    <div style={{
      background: CHART_BG,
      borderRadius: 18,
      boxShadow: "0 2px 18px #B2DBA422",
      padding: 30,
      width: 800,
      maxWidth: "100%",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      minHeight: 440,
      margin: "0 auto",
      position: "relative"
    }}>
      {/* Info icon at top-right corner */}
      <div
        style={{
          position: "absolute",
          top: 10,
          right: 10,
          width: 20,
          height: 20,
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
          zIndex: 10
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

      <div style={{ width: "100%", maxWidth: 740, height: 360 }}>
        <Line data={chartData} options={options} />
      </div>

      {/* Comment Block (on hover) */}
      {showComment && (
        <div
          style={{
            position: "fixed",
            left: commentPosition.x - 280,
            top: commentPosition.y,
            width: 300,
            background: "#fff",
            border: "1px solid #ddd",
            borderRadius: "8px",
            padding: "15px",
            boxShadow: "0 4px 12px rgba(0,0,0,0.15)",
            zIndex: 10000,
            fontSize: "14px",
            lineHeight: "1.4"
          }}
          onMouseEnter={() => setShowComment(true)}
          onMouseLeave={() => setShowComment(false)}
        >
          <div style={{ fontWeight: "bold", marginBottom: "8px", color: "#333", fontSize: "15px" }}>
            📊 Publication Trend Analysis
          </div>
          <div style={{ color: "#555" }}>
            Over <strong>{analysis.totalYears} years</strong> of data, the trend shows{" "}
            <strong>{analysis.overallTrend}</strong>.
            <br /><br />
            
            {analysis.peakYear && (
              <>
                The peak year was <strong>{analysis.peakYear}</strong> with{" "}
                <strong>{analysis.peakValue} patents</strong>.
                <br /><br />
              </>
            )}

            {analysis.significantRises.length > 0 ? (
              <>
                <div style={{ fontWeight: 600, marginBottom: "5px", color: ACCENT }}>
                  📈 Significant Publication Rises:
                </div>
                <ul style={{ margin: "5px 0 10px 0", paddingLeft: "20px" }}>
                  {analysis.significantRises.map((rise, index) => (
                    <li key={index} style={{ marginBottom: "5px" }}>
                      <strong>{rise.year}</strong>: Increased by{" "}
                      <strong style={{ color: HIGHLIGHT }}>
                        {rise.change} patents ({rise.percentage}%)
                      </strong>{" "}
                      from {rise.prevYear}
                    </li>
                  ))}
                </ul>
              </>
            ) : (
              <div style={{ fontStyle: "italic", color: "#777" }}>
                No significant year-over-year rises detected (threshold: ≥10 patents or ≥25% increase).
              </div>
            )}
          </div>
          <div style={{ 
            marginTop: "10px", 
            fontSize: "12px", 
            color: "#888",
            fontStyle: "italic",
            borderTop: "1px solid #eee",
            paddingTop: "8px"
          }}>
            💡 <strong>Tip:</strong> Hover over data points for exact patent counts
          </div>
        </div>
      )}
    </div>
  );
};

export default PublicationTrends;