import React, { useEffect, useState } from "react";
import LoadingSpinner from "./LoadingSpinner";

const PIE_COLORS = [
  "#c8d7d1", "#99c8b3", "#679e5f",
  "#FFD166", "#FFE8A3", "#EF476F",
  "#F28D8D", "#A29EB6"
];

type ApplicantSummary = {
  details: { applicant_name: string; applicant_type: string }[];
  labels: string[];
  percentages: number[];
};

type CoApplicantInfo = {
  coapplicant_count: number;
  coapplicant_rate: number;
  total_applications: number;
};

export const ApplicantTypePie: React.FC = () => {
  const [summary, setSummary] = useState<ApplicantSummary | null>(null);
  const [coapplicant, setCoapplicant] = useState<CoApplicantInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; label: string; perc: number } | null>(null);
  const [showGeneralComment, setShowGeneralComment] = useState(false);
  const [commentPosition, setCommentPosition] = useState({ x: 0, y: 0 });

  /** ───────────────────────── data fetch ─────────────────────────── */
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const port = (await (await fetch("/backend_port.txt")).text()).trim();
        await fetch(`http://localhost:${port}/api/analyze_applicants`, { method: "POST" });

        const [sum, co] = await Promise.all([
          fetch(`http://localhost:${port}/api/applicant_type_summary`).then(r => r.json()),
          fetch(`http://localhost:${port}/api/coapplicant_rate`).then(r => r.json())
        ]);
        if (!cancelled) {
          setSummary(sum);
          setCoapplicant(co);
          setLoading(false);
        }
      } catch {
        if (!cancelled) {
          setErr("Failed to fetch applicant data.");
          setLoading(false);
        }
      }
    })();
    return () => { cancelled = true; };
  }, []);

  /** ────────────────────── sub-component (pie) ───────────────────── */
  function PieChart({ percentages, labels }: { percentages: number[]; labels: string[] }) {
    const radius = 100, cx = 110, cy = 110, total = 100;
    let startAngle = 0;

    const coord = (angle: number, r: number) => [
      cx + r * Math.cos((Math.PI / 180) * (angle - 90)),
      cy + r * Math.sin((Math.PI / 180) * (angle - 90))
    ];

    const arcs = percentages.map((perc, i) => {
      const angle = (perc / total) * 360;
      const endAngle = startAngle + angle;
      const [x1, y1] = coord(startAngle, radius);
      const [x2, y2] = coord(endAngle, radius);
      const large = angle > 180 ? 1 : 0;
      const d = `M${cx},${cy} L${x1},${y1} A${radius},${radius},0,${large},1,${x2},${y2} Z`;
      const midAngle = startAngle + angle / 2;
      const [tx, ty] = coord(midAngle, radius * 0.72);
      startAngle += angle;

      const over = (e: React.MouseEvent) => {
        setHoverIdx(i);
        const bbox = (e.target as SVGPathElement).ownerSVGElement?.getBoundingClientRect();
        if (bbox) setTooltip({ x: tx + bbox.left, y: ty + bbox.top, label: labels[i], perc });
      };

      return (
        <path
          key={i}
          d={d}
          fill={PIE_COLORS[i % PIE_COLORS.length]}
          stroke={hoverIdx === i ? "#333" : "#fff"}
          strokeWidth={hoverIdx === i ? 3 : 2}
          style={{ filter: hoverIdx === i ? "brightness(1.12)" : undefined, transition: "all .18s", cursor: "pointer" }}
          onMouseEnter={over}
          onMouseMove={over}
        />
      );
    });

    return (
      <div
        style={{ position: "relative", width: 220, height: 220 }}
        onMouseLeave={() => { setHoverIdx(null); setTooltip(null); }}   
      >
        <svg width={220} height={220} viewBox="0 0 220 220">{arcs}</svg>
        
        {/* Info icon for general comment */}
        <div
          style={{
            position: "absolute",
            top: 8,
            right: 8,
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
            setCommentPosition({ x: rect.left, y: rect.bottom });
            setShowGeneralComment(true);
          }}
          onMouseLeave={() => setShowGeneralComment(false)}
        >
          i
        </div>

        {tooltip && (
          <div style={{
            position: "fixed",
            left: tooltip.x + 12,
            top: tooltip.y - 12,
            background: "#232526",
            color: "#fff",
            fontSize: 15,
            padding: "7px 17px",
            borderRadius: 9,
            boxShadow: "0 2px 12px #0003",
            pointerEvents: "none",
            zIndex: 99999,
            fontWeight: 600,
            whiteSpace: "nowrap"
          }}>
            {tooltip.label}: <span style={{ color: "#BDD248", fontWeight: 800 }}>{tooltip.perc.toFixed(1)}%</span>
          </div>
        )}
      </div>
    );
  }

  /** ────────────────────── UI state handling ─────────────────────── */
  if (loading) return <LoadingSpinner text="Loading applicant type pie..." />;
  if (err) return <div style={{ color: "red" }}>{err}</div>;
  if (!summary) return <div>No data available.</div>;

  /** ────────────────────────── render  ───────────────────────────── */
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8, position: "relative" }}>
      {/* Pie + legend row */}
      <div style={{ display: "flex", alignItems: "center", gap: 24 }}>
        <PieChart percentages={summary.percentages} labels={summary.labels} />

        {/* Legend */}
        <div style={{ display: "flex", flexDirection: "column", flexWrap: "wrap", maxHeight: 180 }}>
          {summary.labels.map((lab, i) => (
            <div key={i} style={{ display: "flex", alignItems: "center", marginBottom: 6, width: 140 }}>
              <span style={{
                width: 12, height: 12, borderRadius: "50%",
                background: PIE_COLORS[i % PIE_COLORS.length],
                marginRight: 6,
                border: hoverIdx === i ? "2px solid #333" : "1px solid #ccc"
              }} />
              <span style={{
                fontSize: 14, color: "#232526",
                whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis"
              }}>
                {lab}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Co-applicant rate */}
      {coapplicant && (
        <div style={{ fontSize: 15 }}>
          Co-applicant rate:&nbsp;
          <strong>{coapplicant.coapplicant_rate.toFixed(2)}%</strong>&nbsp;
          ({coapplicant.coapplicant_count} / {coapplicant.total_applications})
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
      📊 Analysis Comment
    </div>
    <div style={{ color: "#555" }}>
      {(() => {
        if (!summary || !summary.labels.length) {
          return "No applicant data available for analysis.";
        }
        
        // Create array of label-percentage pairs
        const data = summary.labels.map((label, index) => ({
          label,
          percentage: summary.percentages[index] || 0
        }));
        
        // Sort by percentage descending
        const sortedData = [...data].sort((a, b) => b.percentage - a.percentage);
        
        // Get top 3
        const top3 = sortedData.slice(0, 3);
        
        // Calculate total of top 3
        const top3Total = top3.reduce((sum, item) => sum + item.percentage, 0);
        
        // Check for co-applicant data
        const hasCoapplicant = coapplicant && coapplicant.coapplicant_rate > 0;
        
        return (
          <>
            The applicant distribution shows{" "}
            <strong>{top3[0].label}</strong> as the largest group at{" "}
            <strong>{top3[0].percentage.toFixed(1)}%</strong>, followed by{" "}
            <strong>{top3[1].label}</strong> ({top3[1].percentage.toFixed(1)}%) and{" "}
            <strong>{top3[2].label}</strong> ({top3[2].percentage.toFixed(1)}%).
            <br /><br />
            These top 3 applicant types account for{" "}
            <strong>{top3Total.toFixed(1)}%</strong> of all applications.
            {hasCoapplicant && (
              <>
                {" "}The co-applicant rate of{" "}
                <strong>{coapplicant.coapplicant_rate.toFixed(2)}%</strong> suggests{" "}
                {coapplicant.coapplicant_rate > 30 
                  ? "frequent multi-party applications."
                  : coapplicant.coapplicant_rate > 15 
                  ? "moderate collaboration in applications."
                  : "mostly single-party applications."
                }
              </>
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
      💡 <strong>Tip:</strong> Hover over pie segments for detailed percentages
    </div>
  </div>
)}
    </div>
  );
};

export default ApplicantTypePie;