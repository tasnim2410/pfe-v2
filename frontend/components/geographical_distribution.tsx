// components/geographical_distribution.tsx
"use client";

import React, { useEffect, useState } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";
import LoadingSpinner from "./LoadingSpinner";

// Country code to name mapping
const COUNTRY_NAMES: { [key: string]: string } = {
  US: "United States",
  CN: "China",
  JP: "Japan",
  DE: "Germany",
  KR: "South Korea",
  FR: "France",
  GB: "United Kingdom",
  TW: "Taiwan",
  CH: "Switzerland",
  NL: "Netherlands",
  CA: "Canada",
  IT: "Italy",
  SE: "Sweden",
  IN: "India",
  FI: "Finland",
  RU: "Russia",
  AU: "Australia",
  ES: "Spain",
  BE: "Belgium",
  AT: "Austria",
  DK: "Denmark",
  NO: "Norway",
  IL: "Israel",
  SG: "Singapore",
  BR: "Brazil",
  MX: "Mexico"
};

type ApiResp = { labels: string[]; datasets: { data: number[] }[] };

type CountryData = {
  iso: string;
  count: number;
  name: string;
};

export default function GeographicalDistribution() {
  const [rows, setRows] = useState<CountryData[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [showComment, setShowComment] = useState(false);
  const [commentPosition, setCommentPosition] = useState({ x: 0, y: 0 });
  const [hoveredBar, setHoveredBar] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const port = (await (await fetch("/backend_port.txt")).text()).trim();
        const json: ApiResp = await fetch(
          `http://localhost:${port}/api/geographical_distribution`
        ).then(r => r.json());

        const data = json.labels.map((iso, i) => ({
          iso,
          count: json.datasets[0].data[i] ?? 0,
          name: COUNTRY_NAMES[iso] || iso
        }))
        .sort((a, b) => b.count - a.count)
        .slice(0, 10); // Show top 10 countries

        setRows(data);
      } catch {
        setErr("Error loading geographical data.");
      }
    })();
  }, []);

  const analyzeData = (data: CountryData[]) => {
    if (!data || data.length === 0) {
      return {
        topCountries: [],
        totalPatents: 0,
        dominanceScore: 0,
        insights: "No geographical data available."
      };
    }

    const totalPatents = data.reduce((sum, item) => sum + item.count, 0);
    const top3 = data.slice(0, 3);
    const top3Total = top3.reduce((sum, item) => sum + item.count, 0);
    const dominanceScore = totalPatents > 0 ? (top3Total / totalPatents) * 100 : 0;

    let insight = "";
    if (dominanceScore > 70) {
      insight = "Highly concentrated in a few key countries.";
    } else if (dominanceScore > 40) {
      insight = "Moderately concentrated with clear leaders.";
    } else {
      insight = "Evenly distributed across multiple countries.";
    }

    return {
      topCountries: top3,
      totalPatents,
      dominanceScore: dominanceScore.toFixed(1),
      insight,
      countryCount: data.length
    };
  };

  const analysis = rows ? analyzeData(rows) : null;

  // Color gradient based on count
  const getBarColor = (count: number, maxCount: number) => {
    const intensity = count / maxCount;
    if (intensity > 0.7) return "#2E86AB"; // Blue for high
    if (intensity > 0.4) return "#73B769"; // Green for medium
    return "#FFD166"; // Yellow for low
  };

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const countryName = COUNTRY_NAMES[label] || label;
      return (
        <div style={{
          backgroundColor: 'rgba(35, 37, 38, 0.95)',
          color: '#fff',
          padding: '12px',
          borderRadius: '8px',
          border: '1px solid #333',
          boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
          fontSize: '14px',
          lineHeight: '1.4'
        }}>
          <div style={{ fontWeight: 600, marginBottom: '4px' }}>
            {countryName}
          </div>
          <div style={{ color: '#BDD248', fontWeight: 700 }}>
            {payload[0].value} patents
          </div>
          {analysis && (
            <div style={{ fontSize: '12px', color: '#ccc', marginTop: '6px' }}>
              {((payload[0].value / analysis.totalPatents) * 100).toFixed(1)}% of total
            </div>
          )}
        </div>
      );
    }
    return null;
  };

  if (err) return <div className="text-red-500">{err}</div>;
  if (!rows) return <LoadingSpinner text="Loading chart…" size={28} />;

  const maxCount = Math.max(...rows.map(r => r.count));

  return (
    <div style={{
      background: "#fff",
      borderRadius: "18px",
      boxShadow: "0 2px 18px #B2DBA422",
      padding: "25px",
      width: "100%",
      minHeight: "400px",
      position: "relative"
    }}>
      {/* Header with info icon */}
      <div style={{ 
        display: "flex", 
        justifyContent: "space-between", 
        alignItems: "center", 
        marginBottom: "20px" 
      }}>
        <div>
          <div style={{ fontWeight: 800, fontSize: "22px", color: "#232526", marginBottom: "4px" }}>
            Geographical Distribution
          </div>
          <div style={{ fontSize: "14px", color: "#666" }}>
            Top 10 countries by patent count
          </div>
        </div>
        
        {/* Info icon */}
        <div
          style={{
            width: "22px",
            height: "22px",
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
      </div>

      {/* Bar Chart */}
      <div style={{ height: "280px", width: "100%" }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart 
            data={rows} 
            margin={{ top: 5, right: 30, left: 20, bottom: 20 }}
            onMouseEnter={(data: any) => {
              if (data && data.activeLabel) {
                setHoveredBar(data.activeLabel);
              }
            }}
            onMouseLeave={() => setHoveredBar(null)}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" vertical={false} />
            <XAxis 
              dataKey="iso" 
              axisLine={false}
              tickLine={false}
              tick={{ fill: '#666', fontSize: 12, fontWeight: 600 }}
              label={{ 
                value: 'Country Code', 
                position: 'insideBottom', 
                offset: -10,
                style: { fill: '#232526', fontSize: 13, fontWeight: 700 }
              }}
            />
            <YAxis 
              allowDecimals={false}
              axisLine={false}
              tickLine={false}
              tick={{ fill: '#666', fontSize: 12, fontWeight: 600 }}
              label={{ 
                value: 'Number of Patents', 
                angle: -90, 
                position: 'insideLeft',
                style: { fill: '#232526', fontSize: 13, fontWeight: 700 }
              }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Bar 
              dataKey="count" 
              radius={[4, 4, 0, 0]}
            >
              {rows.map((entry, index) => (
                <Cell 
                  key={`cell-${index}`}
                  fill={getBarColor(entry.count, maxCount)}
                  stroke={hoveredBar === entry.iso ? "#232526" : "none"}
                  strokeWidth={hoveredBar === entry.iso ? 2 : 0}
                  style={{ 
                    transition: "all 0.2s",
                    opacity: hoveredBar && hoveredBar !== entry.iso ? 0.4 : 1 
                  }}
                  onMouseEnter={() => setHoveredBar(entry.iso)}
                  onMouseLeave={() => setHoveredBar(null)}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Country Legend */}
      <div style={{ 
        display: "flex", 
        flexWrap: "wrap", 
        gap: "8px", 
        marginTop: "15px",
        justifyContent: "center"
      }}>
        {rows.slice(0, 5).map((item, index) => (
          <div 
            key={item.iso}
            style={{
              display: "flex",
              alignItems: "center",
              padding: "4px 10px",
              background: "#f8f9fa",
              borderRadius: "16px",
              fontSize: "12px",
              fontWeight: 500,
              color: "#232526",
              border: "1px solid #eaeaea"
            }}
            onMouseEnter={() => setHoveredBar(item.iso)}
            onMouseLeave={() => setHoveredBar(null)}
          >
            <div style={{
              width: "10px",
              height: "10px",
              borderRadius: "50%",
              backgroundColor: getBarColor(item.count, maxCount),
              marginRight: "6px"
            }} />
            <span style={{ fontWeight: 600 }}>{item.iso}</span>
            <span style={{ marginLeft: "4px", color: "#666" }}>: {item.count}</span>
          </div>
        ))}
      </div>

      {/* Statistics Summary */}
      {analysis && (
        <div style={{
          display: "flex",
          justifyContent: "space-between",
          marginTop: "20px",
          padding: "12px",
          background: "#f5f7fa",
          borderRadius: "10px",
          fontSize: "13px"
        }}>
          <div style={{ textAlign: "center", flex: 1 }}>
            <div style={{ fontWeight: 700, color: "#2E86AB", fontSize: "15px" }}>
              {analysis.topCountries.length > 0 ? analysis.topCountries[0].iso : "N/A"}
            </div>
            <div style={{ color: "#666" }}>Top Country</div>
          </div>
          <div style={{ textAlign: "center", flex: 1, borderLeft: "1px solid #ddd", borderRight: "1px solid #ddd" }}>
            <div style={{ fontWeight: 700, color: "#A23B72", fontSize: "15px" }}>
              {analysis.totalPatents}
            </div>
            <div style={{ color: "#666" }}>Total Patents</div>
          </div>
          <div style={{ textAlign: "center", flex: 1 }}>
            <div style={{ fontWeight: 700, color: "#73B769", fontSize: "15px" }}>
              {analysis.countryCount}
            </div>
            <div style={{ color: "#666" }}>Countries</div>
          </div>
        </div>
      )}

      {/* Comment Block (on hover) */}
      {showComment && analysis && (
        <div
          style={{
            position: "fixed",
            left: commentPosition.x - 280,
            top: commentPosition.y,
            width: 280,
            background: "#fff",
            border: "1px solid #ddd",
            borderRadius: "10px",
            padding: "15px",
            boxShadow: "0 4px 20px rgba(0,0,0,0.15)",
            zIndex: 10000,
            fontSize: "13px",
            lineHeight: "1.4"
          }}
          onMouseEnter={() => setShowComment(true)}
          onMouseLeave={() => setShowComment(false)}
        >
          <div style={{ fontWeight: "bold", marginBottom: "8px", color: "#333", fontSize: "14px" }}>
            🌍 Geographic Analysis
          </div>
          <div style={{ color: "#555" }}>
            Patent activity is concentrated in {analysis.countryCount} countries.
            
            {analysis.topCountries.length > 0 && (
              <>
                <br /><br />
                <div style={{ fontWeight: 600, marginBottom: "6px", color: "#232526" }}>
                  Top 3 Countries:
                </div>
                <ul style={{ margin: "5px 0 10px 0", paddingLeft: "18px" }}>
                  {analysis.topCountries.map((country, index) => (
                    <li key={index} style={{ marginBottom: "5px" }}>
                      <strong>{country.name || country.iso}</strong>: {country.count} patents
                      <span style={{ color: "#666", fontSize: "12px", marginLeft: "6px" }}>
                        ({((country.count / analysis.totalPatents) * 100).toFixed(1)}%)
                      </span>
                    </li>
                  ))}
                </ul>
              </>
            )}
            
            <div style={{ 
              marginTop: "10px", 
              padding: "10px", 
              background: "#f8f9fa",
              borderRadius: "6px",
              borderLeft: "3px solid #2E86AB",
              fontSize: "12px"
            }}>
              <strong>Insight:</strong> {analysis.insight}
              <br />
              <div style={{ marginTop: "5px", color: "#666" }}>
                Top 3 countries account for {analysis.dominanceScore}% of patents.
              </div>
            </div>
          </div>
          <div style={{ 
            marginTop: "10px", 
            fontSize: "11px", 
            color: "#888",
            fontStyle: "italic",
            borderTop: "1px solid #eee",
            paddingTop: "8px"
          }}>
            💡 Higher bars indicate more patent activity in that country
          </div>
        </div>
      )}
    </div>
  );
}