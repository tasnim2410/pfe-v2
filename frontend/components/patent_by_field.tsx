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

ChartJS.register(LineElement, PointElement, CategoryScale, LinearScale, Tooltip, Legend, Title);

const CHART_BG = "#fff";
const ACCENT = "#232526";
const COLORS = [
  "#3366CC", // Blue
  "#DC3912", // Red
  "#FF9900", // Orange
  "#109618", // Green
  "#990099", // Purple
  "#3B3EAC", // Indigo
  "#0099C6", // Cyan
  "#DD4477", // Pink
  "#66AA00", // Lime
  "#B82E2E", // Dark Red
  "#316395", // Steel Blue
  "#994499", // Violet
  "#22AA99", // Teal
  "#AAAA11", // Olive
  "#6633CC", // Deep Purple
  "#E67300", // Amber
  "#8B0707", // Maroon
  "#329262", // Emerald
  "#5574A6", // Slate Blue
  "#3B3EAC"  // Indigo
];

const FieldLegend = ({ fields }: { fields: string[] }) => (
  <div style={{
    display: "flex", gap: 22, margin: "18px 0 0 0", flexWrap: "wrap",
    justifyContent: "center"
  }}>
    {fields.map((f, i) => (
      <span key={f} style={{
        display: "flex", alignItems: "center", fontWeight: 600, fontSize: 16,
        color: ACCENT, gap: 8
      }}>
        <span style={{
          width: 16, height: 16, borderRadius: "50%",
          background: COLORS[i % COLORS.length], display: "inline-block", marginRight: 5,
          border: "1.5px solid #d8d8d8"
        }} />{f}
      </span>
    ))}
  </div>
);

const PatentFieldTrends: React.FC = () => {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [classifying, setClassifying] = useState(false);
  const [showComment, setShowComment] = useState(false);
  const [commentPosition, setCommentPosition] = useState({ x: 0, y: 0 });

  const loadData = async () => {
    setLoading(true);
    setErr(null);
    let cancelled = false;
    try {
      const portRes = await fetch("/backend_port.txt");
      const port = (await portRes.text()).trim();
      const apiRes = await fetch(`http://localhost:${port}/api/patent_field_trends`);
      if (!apiRes.ok) throw new Error();
      const json = await apiRes.json();
      if (!cancelled) setData(json);
      setLoading(false);
    } catch (e) {
      setErr("Failed to fetch patent field trends. You may need to classify the patents database.");
      setLoading(false);
    }
    return () => { cancelled = true; };
  };

  useEffect(() => { loadData(); }, []);

  const classify = async () => {
    setClassifying(true);
    setErr(null);
    try {
      const portRes = await fetch("/backend_port.txt");
      const port = (await portRes.text()).trim();
      const resp = await fetch(`http://localhost:${port}/classify_patents`, { method: "POST" });
      if (!resp.ok) throw new Error();
      await new Promise((res) => setTimeout(res, 1500));
      loadData();
    } catch {
      setErr("Failed to classify patents. Is the backend running?");
    }
    setClassifying(false);
  };

  // Function to analyze fields with highest counts
  const analyzeTopFields = (labels: string[], datasets: any[]) => {
    if (!labels || !datasets || datasets.length === 0) {
      return { 
        topFields: [], 
        technologyTrend: "No data available for analysis.",
        dominantAreas: "Insufficient data"
      };
    }

    // Get the most recent year's data
    const recentYearIndex = labels.length - 1;
    
    // Create array of fields with their recent patent counts
    const fieldData = datasets.map((dataset, index) => {
      const recentCount = dataset.data[recentYearIndex] || 0;
      const totalCount = dataset.data.reduce((sum: number, val: number) => sum + val, 0);
      
      return {
        field: dataset.label,
        color: COLORS[index % COLORS.length],
        recentCount,
        totalCount,
        dataPoints: dataset.data
      };
    });

    // Sort by recent count (highest first)
    const sortedByRecent = [...fieldData].sort((a, b) => b.recentCount - a.recentCount);
    const top3Recent = sortedByRecent.slice(0, 3);

    // Sort by total count (highest first)
    const sortedByTotal = [...fieldData].sort((a, b) => b.totalCount - a.totalCount);
    const top3Total = sortedByTotal.slice(0, 3);

    // Determine technology trend based on top fields
    const getTechnologyTrend = (topFields: typeof top3Recent) => {
      const fieldNames = topFields.map(f => f.field);
      
      // Check for specific technology areas
      const hasEngineering = fieldNames.some(f => f.toLowerCase().includes('engineering'));
      const hasComputerScience = fieldNames.some(f => f.toLowerCase().includes('computer'));
      const hasMedicine = fieldNames.some(f => f.toLowerCase().includes('medicine'));
      const hasMaterials = fieldNames.some(f => f.toLowerCase().includes('materials'));
      const hasEnergy = fieldNames.some(f => f.toLowerCase().includes('energy'));
      const hasCommunications = fieldNames.some(f => f.toLowerCase().includes('communication'));
      
      // Determine trend based on field composition
      if (hasComputerScience && hasEngineering && hasCommunications) {
        return "This indicates strong focus on digital transformation, connectivity, and smart systems, reflecting the ongoing digital revolution.";
      } else if (hasMedicine && hasComputerScience) {
        return "The dominance of medicine and computer science suggests significant growth in healthtech, digital health, and medical AI applications.";
      } else if (hasEngineering && hasMaterials && hasEnergy) {
        return "This reflects focus on sustainable technologies, advanced manufacturing, and clean energy solutions.";
      } else if (hasComputerScience) {
        return "The strong computer science presence indicates continued innovation in software, AI, and digital technologies.";
      } else if (hasEngineering) {
        return "Engineering dominance suggests focus on infrastructure, manufacturing, and practical technological applications.";
      } else {
        return "The patent distribution indicates balanced technological development across multiple domains.";
      }
    };

    // Determine dominant application areas
    const getDominantAreas = (topFields: typeof top3Recent) => {
      const areas: string[] = [];
      topFields.forEach(field => {
        const fieldName = field.field.toLowerCase();
        
        if (fieldName.includes('engineering')) areas.push("Infrastructure, manufacturing, and systems design");
        if (fieldName.includes('computer')) areas.push("Software development, AI, and digital solutions");
        if (fieldName.includes('medicine')) areas.push("Healthcare technologies and medical devices");
        if (fieldName.includes('materials')) areas.push("Advanced materials and nanotechnology");
        if (fieldName.includes('energy')) areas.push("Renewable energy and power systems");
        if (fieldName.includes('communication')) areas.push("Telecommunications and networking");
        if (fieldName.includes('transportation')) areas.push("Mobility and logistics");
        if (fieldName.includes('chemistry')) areas.push("Chemical processes and materials science");
        if (fieldName.includes('physics')) areas.push("Fundamental research and applied physics");
        if (fieldName.includes('design')) areas.push("Product design and user experience");
        if (fieldName.includes('manufacturing')) areas.push("Production technologies and automation");
        if (fieldName.includes('art')) areas.push("Creative technologies and digital media");
      });

      // Remove duplicates and limit to top areas
      const uniqueAreas = [...new Set(areas)];
      return uniqueAreas.slice(0, 3).join(", ");
    };

    const technologyTrend = getTechnologyTrend(top3Recent);
    const dominantAreas = getDominantAreas(top3Recent);

    return {
      topFields: top3Recent,
      totalFieldsCount: datasets.length,
      technologyTrend,
      dominantAreas,
      recentYear: labels[recentYearIndex] || "current year"
    };
  };

  if (loading) return <LoadingSpinner text="Loading Patent Field Trends..." />;
  if (err) return (
    <div style={{ color: "#EA3C53", textAlign: "center", margin: 22 }}>
      {err}
      <br />
      <button
        onClick={classify}
        disabled={classifying}
        style={{
          marginTop: 12, padding: "7px 26px", borderRadius: 9,
          background: "#BDD248", color: "#232526", fontWeight: 800,
          border: "none", fontSize: 17, cursor: "pointer",
          opacity: classifying ? 0.6 : 1,
          transition: "opacity .2s"
        }}
      >
        {classifying ? "Classifying..." : "Classify Now"}
      </button>
    </div>
  );
  if (!data) return null;

  // Analyze trends for the comment
  const analysis = analyzeTopFields(data.labels, data.datasets);

  const chartData = {
    labels: data.labels,
    datasets: data.datasets.map((ds: any, i: number) => ({
      ...ds,
      borderColor: COLORS[i % COLORS.length],
      backgroundColor: COLORS[i % COLORS.length],
      pointBackgroundColor: "#fff",
      pointBorderColor: COLORS[i % COLORS.length],
      pointRadius: 0,          // Points hidden by default
      pointHoverRadius: 7,     // Appear on hover
      borderWidth: 3,
      fill: false,
      tension: 0.20,
    })),
  };
  
  const options = {
    responsive: true,
    plugins: {
      legend: { display: false },
      title: {
        display: true,
        text: "Patent Trends by Field",
        color: ACCENT,
        font: { size: 22, weight:  700 },
        padding: { top: 12, bottom: 28 },
      },
      tooltip: {
        mode: 'index' as const,
        intersect: false,
        callbacks: {
          label: (ctx: any) =>
            `${ctx.dataset.label}: ${ctx.parsed.y} patents in ${ctx.label}`,
        },
        backgroundColor: "#232526",
        titleColor: "#fff",
        bodyColor: "#BDD248",
        borderColor: "#EA3C53",
        borderWidth: 1.2
      },
    },
    hover: {
      mode: 'index' as const,
      intersect: false,
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
      width: 860,
      maxWidth: "100%",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      minHeight: 480,
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

      <div style={{ width: "100%", maxWidth: 790, height: 400 }}>
        <Line data={chartData} options={options} />
      </div>
      <FieldLegend fields={data.datasets.map((d: any) => d.label)} />

      {/* Comment Block (on hover) */}
      {showComment && (
        <div
          style={{
            position: "fixed",
            left: commentPosition.x - 320,
            top: commentPosition.y,
            width: 320,
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
            📊 Technology Field Analysis
          </div>
          <div style={{ color: "#555" }}>
            Based on {analysis.totalFieldsCount} technological fields, the most active areas in {analysis.recentYear} are:
            <br /><br />
            
            {analysis.topFields.length > 0 && (
              <ul style={{ margin: "5px 0 10px 0", paddingLeft: "20px" }}>
                {analysis.topFields.map((field, index) => (
                  <li key={index} style={{ marginBottom: "8px" }}>
                    <span style={{ 
                      display: "inline-block", 
                      width: "12px", 
                      height: "12px", 
                      borderRadius: "50%", 
                      backgroundColor: field.color, 
                      marginRight: "8px",
                      verticalAlign: "middle"
                    }}></span>
                    <strong>{field.field}</strong>: <strong>{field.recentCount}</strong> patents
                  </li>
                ))}
              </ul>
            )}

            <div style={{ 
              marginTop: "12px", 
              padding: "10px", 
              background: "#f8f9fa",
              borderRadius: "6px",
              borderLeft: "3px solid #3366CC"
            }}>
              <div style={{ fontWeight: 600, marginBottom: "5px", color: ACCENT }}>
                Technology Trend:
              </div>
              {analysis.technologyTrend}
              <br /><br />
              {/* <div style={{ fontWeight: 600, marginBottom: "5px", color: ACCENT }}>
                Dominant Application Areas:
              </div>
              {analysis.dominantAreas} */}
            </div>
          </div>
          <div style={{ 
            marginTop: "10px", 
            fontSize: "12px", 
            color: "#888",
            fontStyle: "italic",
            borderTop: "1px solid #eee",
            paddingTop: "8px"
          }}>
            💡 <strong>Tip:</strong> The color-coded lines show patent trends across different technology fields
          </div>
        </div>
      )}
    </div>
  );
};

export default PatentFieldTrends;