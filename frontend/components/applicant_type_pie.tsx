import React, { useEffect, useState } from "react";

const PIE_COLORS = [
  "#E6F4EA", // Soft Mint Green
  "#B9E6C9", // Fresh Light Green
  "#70C36A", // Leafy Vibrant Green
  "#FFD166", // Warm Amber/Yellow
  "#FFE8A3", // Pale Butter Yellow
  "#EF476F", // Coral Red
  "#F28D8D", // Soft Warm Pink
  "#A29EB6"  // Muted Lavender Gray (for contrast / grounding)
];


type ApplicantSummary = {
  details: { applicant_name: string; applicant_type: string }[];
  labels: string[];
  percentages: number[];
};

export const ApplicantTypePie: React.FC = () => {
  const [summary, setSummary] = useState<ApplicantSummary | null>(null);
  const [coapplicant, setCoapplicant] = useState<{coapplicant_count: number, coapplicant_rate: number, total_applications: number} | null>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);
  const [tooltip, setTooltip] = useState<{x: number, y: number, label: string, perc: number} | null>(null);

  useEffect(() => {
    let cancelled = false;
    const run = async () => {
      setLoading(true);
      setErr(null);
      let port = "";
      try {
        const res = await fetch("/backend_port.txt");
        port = (await res.text()).trim();
      } catch {
        setErr("Could not fetch backend port");
        setLoading(false);
        return;
      }
      // 1. Analyze step
      try {
        const analyze = await fetch(`http://localhost:${port}/api/analyze_applicants`, { method: "POST" });
        const json = await analyze.json();
        if (!json.status || json.status !== "success") throw new Error("Analysis failed");
      } catch (e) {
        setErr("Failed to analyze applicants.");
        setLoading(false);
        return;
      }
      // 2. Fetch summary
      try {
        const res = await fetch(`http://localhost:${port}/api/applicant_type_summary`);
        const data = await res.json();
        if (cancelled) return;
        setSummary(data);
      } catch (e) {
        setErr("Could not fetch applicant type summary.");
        setLoading(false);
        return;
      }
      // 3. Fetch coapplicant rate
      try {
        const res = await fetch(`http://localhost:${port}/api/coapplicant_rate`);
        const data = await res.json();
        if (cancelled) return;
        setCoapplicant(data);
        setLoading(false);
      } catch (e) {
        setErr("Could not fetch coapplicant rate.");
        setLoading(false);
      }
    };
    run();
    return () => { cancelled = true; };
  }, []);

  // Interactive PieChart
  function PieChart({ percentages, labels }: { percentages: number[], labels: string[] }) {
    const radius = 100;
    const cx = 110, cy = 110;
    const total = 100;
    let startAngle = 0;

    const getCoord = (angle: number, r: number) => [
      cx + r * Math.cos((Math.PI / 180) * (angle - 90)),
      cy + r * Math.sin((Math.PI / 180) * (angle - 90))
    ];

    const paths = percentages.map((perc, i) => {
      const angle = (perc / total) * 360;
      const endAngle = startAngle + angle;
      const [x1, y1] = getCoord(startAngle, radius);
      const [x2, y2] = getCoord(endAngle, radius);
      const large = angle > 180 ? 1 : 0;
      const d = [
        `M${cx},${cy}`,
        `L${x1},${y1}`,
        `A${radius},${radius},0,${large},1,${x2},${y2}`,
        'Z'
      ].join(' ');
      // For tooltip, get the middle angle for this segment
      const midAngle = startAngle + angle / 2;
      const [tx, ty] = getCoord(midAngle, radius * 0.72);

      const handleMouseOver = (e: React.MouseEvent) => {
        setHoverIdx(i);
        const bbox = (e.target as SVGPathElement).ownerSVGElement?.getBoundingClientRect();
        if (bbox) {
          setTooltip({
            x: tx + bbox.left,
            y: ty + bbox.top,
            label: labels[i],
            perc: perc
          });
        }
      };
      const handleMouseMove = (e: React.MouseEvent) => {
        const bbox = (e.target as SVGPathElement).ownerSVGElement?.getBoundingClientRect();
        if (bbox) {
          setTooltip({
            x: tx + bbox.left,
            y: ty + bbox.top,
            label: labels[i],
            perc: perc
          });
        }
      };
      const handleMouseOut = () => {
        setHoverIdx(null);
        setTooltip(null);
      };
      startAngle += angle;
      return (
        <path
          key={i}
          d={d}
          fill={PIE_COLORS[i % PIE_COLORS.length]}
          stroke={hoverIdx === i ? "#333" : "#fff"}
          strokeWidth={hoverIdx === i ? 3 : 2}
          style={{ filter: hoverIdx === i ? "brightness(1.12)" : undefined, cursor: "pointer", transition: "all .18s" }}
          onMouseOver={handleMouseOver}
          onMouseMove={handleMouseMove}
          onMouseOut={handleMouseOut}
        />
      );
    });

    return (
      <div
        style={{ position: "relative", width: 220, height: 220 }}
        onMouseLeave={() => {
          setHoverIdx(null);
          setTooltip(null);
        }}
      >
        <svg width={220} height={220} viewBox="0 0 220 220">
          {paths}
        </svg>
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

  return (
    <div style={{
      background: "#fff",
      borderRadius: 18,
      boxShadow: "0 2px 18px #B2DBA422",
      padding: 20,
      width: 410,
      maxWidth: "100%",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      minHeight: 340,
      margin: "0 auto"
    }}>

      {loading && (
        <div style={{ marginTop: 40 }}>
          {/* Spinner */}
          <div style={{
            width: 38, height: 38, border: "5px solid #BDD248", borderTop: "5px solid #232526",
            borderRadius: "50%", animation: "spin 1s linear infinite"
          }} />
          <style>{`
            @keyframes spin { 0%{transform:rotate(0deg)} 100%{transform:rotate(360deg)} }
          `}</style>
          <div style={{ color: "#232526", fontSize: 16, marginTop: 18, fontWeight: 500 }}>
            Loading applicant types...
          </div>
        </div>
      )}
      {err && (
        <div style={{ color: "#EA3C53", fontWeight: 700, marginTop: 36 }}>
          {err}
        </div>
      )}
      {!loading && summary && (
        <>
          <PieChart percentages={summary.percentages} labels={summary.labels} />
          {coapplicant && (
            <div style={{
              marginTop: 24,
              width: "100%",
              display: "flex",
              flexDirection: "column",
              alignItems: "center"
            }}>
              <div style={{
                fontWeight: 700,
                fontSize: 18,
                color: "#232526",
                marginBottom: 2
              }}>
                Co-applicant Rate
              </div>
              <div style={{
                fontSize: 16,
                color: "#36B37E",
                fontWeight: 800
              }}>
                {coapplicant.coapplicant_rate.toFixed(1)}%
              </div>
              <div style={{
                fontSize: 14,
                color: "#888",
                marginTop: 1
              }}>
                {coapplicant.coapplicant_count} out of {coapplicant.total_applications} applications
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default ApplicantTypePie;
