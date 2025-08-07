// import React, { useEffect, useState } from "react";

// const PIE_COLORS = [
//   "#E6F4EA", // Soft Mint Green
//   "#B9E6C9", // Fresh Light Green
//   "#70C36A", // Leafy Vibrant Green
//   "#FFD166", // Warm Amber/Yellow
//   "#FFE8A3", // Pale Butter Yellow
//   "#EF476F", // Coral Red
//   "#F28D8D", // Soft Warm Pink
//   "#A29EB6"  // Muted Lavender Gray (for contrast / grounding)
// ];


// type ApplicantSummary = {
//   details: { applicant_name: string; applicant_type: string }[];
//   labels: string[];
//   percentages: number[];
// };

// type ChartProps = { width?: number; height?: number };
// export const ApplicantTypePie: React.FC<ChartProps> = ({ width, height }) => {
//   const [summary, setSummary] = useState<ApplicantSummary | null>(null);
//   const [coapplicant, setCoapplicant] = useState<{coapplicant_count: number, coapplicant_rate: number, total_applications: number} | null>(null);
//   const [loading, setLoading] = useState(true);
//   const [err, setErr] = useState<string | null>(null);
//   const [hoverIdx, setHoverIdx] = useState<number | null>(null);
//   const [tooltip, setTooltip] = useState<{x: number, y: number, label: string, perc: number} | null>(null);

//   useEffect(() => {
//     let cancelled = false;
//     const run = async () => {
//       setLoading(true);
//       setErr(null);
//       let port = "";
//       try {
//         const res = await fetch("/backend_port.txt");
//         port = (await res.text()).trim();
//       } catch {
//         setErr("Could not fetch backend port");
//         setLoading(false);
//         return;
//       }
//       // 1. Analyze step
//       try {
//         const analyze = await fetch(`http://localhost:${port}/api/analyze_applicants`, { method: "POST" });
//         const json = await analyze.json();
//         if (!json.status || json.status !== "success") throw new Error("Analysis failed");
//       } catch (e) {
//         setErr("Failed to analyze applicants.");
//         setLoading(false);
//         return;
//       }
//       // 2. Fetch summary
//       try {
//         const res = await fetch(`http://localhost:${port}/api/applicant_type_summary`);
//         const data = await res.json();
//         if (cancelled) return;
//         setSummary(data);
//       } catch (e) {
//         setErr("Could not fetch applicant type summary.");
//         setLoading(false);
//         return;
//       }
//       // 3. Fetch coapplicant rate
//       try {
//         const res = await fetch(`http://localhost:${port}/api/coapplicant_rate`);
//         const data = await res.json();
//         if (cancelled) return;
//         setCoapplicant(data);
//         setLoading(false);
//       } catch (e) {
//         setErr("Could not fetch coapplicant rate.");
//         setLoading(false);
//       }
//     };
//     run();
//     return () => { cancelled = true; };
//   }, []);

//   // Interactive PieChart
//   function PieChart({ percentages, labels }: { percentages: number[], labels: string[] }) {
//     const radius = 100;
//     const cx = 110, cy = 110;
//     const total = 100;
//     let startAngle = 0;

//     const getCoord = (angle: number, r: number) => [
//       cx + r * Math.cos((Math.PI / 180) * (angle - 90)),
//       cy + r * Math.sin((Math.PI / 180) * (angle - 90))
//     ];

//     const paths = percentages.map((perc, i) => {
//       const angle = (perc / total) * 360;
//       const endAngle = startAngle + angle;
//       const [x1, y1] = getCoord(startAngle, radius);
//       const [x2, y2] = getCoord(endAngle, radius);
//       const large = angle > 180 ? 1 : 0;
//       const d = [
//         `M${cx},${cy}`,
//         `L${x1},${y1}`,
//         `A${radius},${radius},0,${large},1,${x2},${y2}`,
//         'Z'
//       ].join(' ');
//       // For tooltip, get the middle angle for this segment
//       const midAngle = startAngle + angle / 2;
//       const [tx, ty] = getCoord(midAngle, radius * 0.72);

//       const handleMouseOver = (e: React.MouseEvent) => {
//         setHoverIdx(i);
//         const bbox = (e.target as SVGPathElement).ownerSVGElement?.getBoundingClientRect();
//         if (bbox) {
//           setTooltip({
//             x: tx + bbox.left,
//             y: ty + bbox.top,
//             label: labels[i],
//             perc: perc
//           });
//         }
//       };
//       const handleMouseMove = (e: React.MouseEvent) => {
//         const bbox = (e.target as SVGPathElement).ownerSVGElement?.getBoundingClientRect();
//         if (bbox) {
//           setTooltip({
//             x: tx + bbox.left,
//             y: ty + bbox.top,
//             label: labels[i],
//             perc: perc
//           });
//         }
//       };
//       const handleMouseOut = () => {
//         setHoverIdx(null);
//         setTooltip(null);
//       };
//       startAngle += angle;
//       return (
//         <path
//           key={i}
//           d={d}
//           fill={PIE_COLORS[i % PIE_COLORS.length]}
//           stroke={hoverIdx === i ? "#333" : "#fff"}
//           strokeWidth={hoverIdx === i ? 3 : 2}
//           style={{ filter: hoverIdx === i ? "brightness(1.12)" : undefined, cursor: "pointer", transition: "all .18s" }}
//           onMouseOver={handleMouseOver}
//           onMouseMove={handleMouseMove}
//           onMouseOut={handleMouseOut}
//         />
//       );
//     });

//     return (
//       <div
//         style={{ position: "relative", width: 220, height: 220 }}
//         onMouseLeave={() => {
//           setHoverIdx(null);
//           setTooltip(null);
//         }}
//       >
//         <svg width={220} height={220} viewBox="0 0 220 220">
//           {paths}
//         </svg>
//         {tooltip && (
//           <div style={{
//             position: "fixed",
//             left: tooltip.x + 12,
//             top: tooltip.y - 12,
//             background: "#232526",
//             color: "#fff",
//             fontSize: 15,
//             padding: "7px 17px",
//             borderRadius: 9,
//             boxShadow: "0 2px 12px #0003",
//             pointerEvents: "none",
//             zIndex: 99999,
//             fontWeight: 600,
//             whiteSpace: "nowrap"
//           }}>
//             {tooltip.label}: <span style={{ color: "#BDD248", fontWeight: 800 }}>{tooltip.perc.toFixed(1)}%</span>
//           </div>
//         )}
//       </div>
//     );
//   }

//   return (
//     <div style={{
//       background: "#fff",
//       borderRadius: 18,
//       boxShadow: "0 2px 18px #B2DBA422",
//       padding: 20,
//       width: 410,
//       maxWidth: "100%",
//       display: "flex",
//       flexDirection: "column",
//       alignItems: "center",
//       minHeight: 340,
//       margin: "0 auto"
//     }}>

//       {loading && (
//         <div style={{ marginTop: 40 }}>
//           {/* Spinner */}
//           <div style={{
//             width: 38, height: 38, border: "5px solid #BDD248", borderTop: "5px solid #232526",
//             borderRadius: "50%", animation: "spin 1s linear infinite"
//           }} />
//           <style>{`
//             @keyframes spin { 0%{transform:rotate(0deg)} 100%{transform:rotate(360deg)} }
//           `}</style>
//           <div style={{ color: "#232526", fontSize: 16, marginTop: 18, fontWeight: 500 }}>
//             Loading applicant types...
//           </div>
//         </div>
//       )}
//       {err && (
//         <div style={{ color: "#EA3C53", fontWeight: 700, marginTop: 36 }}>
//           {err}
//         </div>
//       )}
//       {!loading && summary && (
//         <>
//           <PieChart percentages={summary.percentages} labels={summary.labels} />
//           {coapplicant && (
//             <div style={{
//               marginTop: 24,
//               width: "100%",
//               display: "flex",
//               flexDirection: "column",
//               alignItems: "center"
//             }}>
//               <div style={{
//                 fontWeight: 700,
//                 fontSize: 18,
//                 color: "#232526",
//                 marginBottom: 2
//               }}>
//                 Co-applicant Rate
//               </div>
//               <div style={{
//                 fontSize: 16,
//                 color: "#36B37E",
//                 fontWeight: 800
//               }}>
//                 {coapplicant.coapplicant_rate.toFixed(1)}%
//               </div>
//               <div style={{
//                 fontSize: 14,
//                 color: "#888",
//                 marginTop: 1
//               }}>
//                 {coapplicant.coapplicant_count} out of {coapplicant.total_applications} applications
//               </div>
//             </div>
//           )}
//         </>
//       )}
//     </div>
//   );
// };

// export default ApplicantTypePie;




// import React, { useEffect, useState } from "react";

// const PIE_COLORS = [
//   "#E6F4EA", // Soft Mint Green
//   "#B9E6C9", // Fresh Light Green
//   "#70C36A", // Leafy Vibrant Green
//   "#FFD166", // Warm Amber/Yellow
//   "#FFE8A3", // Pale Butter Yellow
//   "#EF476F", // Coral Red
//   "#F28D8D", // Soft Warm Pink
//   "#A29EB6"  // Muted Lavender Gray
// ];

// type ApplicantSummary = {
//   details: { applicant_name: string; applicant_type: string }[];
//   labels: string[];
//   percentages: number[];
// };

// type CoApplicantInfo = {
//   coapplicant_count: number;
//   coapplicant_rate: number;
//   total_applications: number;
// };

// type ChartProps = { width?: number; height?: number };

// export const ApplicantTypePie: React.FC<ChartProps> = () => {
//   const [summary, setSummary] = useState<ApplicantSummary | null>(null);
//   const [coapplicant, setCoapplicant] = useState<CoApplicantInfo | null>(null);
//   const [loading, setLoading] = useState(true);
//   const [err, setErr] = useState<string | null>(null);
//   const [hoverIdx, setHoverIdx] = useState<number | null>(null);
//   const [tooltip, setTooltip] = useState<{ x: number; y: number; label: string; perc: number } | null>(null);

//   /** ------------------------------------------------------------------
//    *  Data fetch
//    *  -----------------------------------------------------------------*/
//   useEffect(() => {
//     let cancelled = false;

//     const run = async () => {
//       setLoading(true);
//       setErr(null);

//       try {
//         // 1. Get backend port
//         const portTxt = await (await fetch("/backend_port.txt")).text();
//         const port = portTxt.trim();

//         // 2. Tell backend to analyse
//         const resAnalyse = await fetch(`http://localhost:${port}/api/analyze_applicants`, { method: "POST" });
//         if (!((await resAnalyse.json()).status === "success")) throw new Error();

//         // 3. Summary
//         const resSum = await fetch(`http://localhost:${port}/api/applicant_type_summary`);
//         const sumJson = await resSum.json();
//         if (!cancelled) setSummary(sumJson);

//         // 4. Co-applicant rate
//         const resCo = await fetch(`http://localhost:${port}/api/coapplicant_rate`);
//         const coJson = await resCo.json();
//         if (!cancelled) setCoapplicant(coJson);

//         setLoading(false);
//       } catch {
//         if (!cancelled) {
//           setErr("Failed to fetch applicant data.");
//           setLoading(false);
//         }
//       }
//     };

//     run();
//     return () => { cancelled = true; };
//   }, []);

//   /** ------------------------------------------------------------------
//    *  PieChart sub-component
//    *  -----------------------------------------------------------------*/
//   function PieChart({ percentages, labels }: { percentages: number[]; labels: string[] }) {
//     const radius = 100;
//     const cx = 110, cy = 110;
//     const total = 100;
//     let startAngle = 0;

//     const getCoord = (angle: number, r: number) => [
//       cx + r * Math.cos((Math.PI / 180) * (angle - 90)),
//       cy + r * Math.sin((Math.PI / 180) * (angle - 90))
//     ];

//     const arcs = percentages.map((perc, i) => {
//       const angle = (perc / total) * 360;
//       const endAngle = startAngle + angle;
//       const [x1, y1] = getCoord(startAngle, radius);
//       const [x2, y2] = getCoord(endAngle, radius);
//       const large = angle > 180 ? 1 : 0;
//       const d = `M${cx},${cy} L${x1},${y1} A${radius},${radius},0,${large},1,${x2},${y2} Z`;
//       const midAngle = startAngle + angle / 2;
//       const [tx, ty] = getCoord(midAngle, radius * 0.72);
//       startAngle += angle;

//       const handleOver = (e: React.MouseEvent) => {
//         setHoverIdx(i);
//         const bbox = (e.target as SVGPathElement).ownerSVGElement?.getBoundingClientRect();
//         if (bbox) {
//           setTooltip({ x: tx + bbox.left, y: ty + bbox.top, label: labels[i], perc });
//         }
//       };

//       return (
//         <path
//           key={i}
//           d={d}
//           fill={PIE_COLORS[i % PIE_COLORS.length]}
//           stroke={hoverIdx === i ? "#333" : "#fff"}
//           strokeWidth={hoverIdx === i ? 3 : 2}
//           style={{ filter: hoverIdx === i ? "brightness(1.12)" : undefined, cursor: "pointer", transition: "all .18s" }}
//           onMouseEnter={handleOver}
//           onMouseMove={handleOver}
//           onMouseLeave={() => { setHoverIdx(null); setTooltip(null); }}
//         />
//       );
//     });

//     return (
//       <div style={{ position: "relative", width: 220, height: 220 }}>
//         <svg width={220} height={220} viewBox="0 0 220 220">{arcs}</svg>
//         {tooltip && (
//           <div style={{
//             position: "fixed",
//             left: tooltip.x + 12,
//             top: tooltip.y - 12,
//             background: "#232526",
//             color: "#fff",
//             fontSize: 15,
//             padding: "7px 17px",
//             borderRadius: 9,
//             boxShadow: "0 2px 12px #0003",
//             pointerEvents: "none",
//             zIndex: 99999,
//             fontWeight: 600,
//             whiteSpace: "nowrap"
//           }}>
//             {tooltip.label}:{" "}
//             <span style={{ color: "#BDD248", fontWeight: 800 }}>
//               {tooltip.perc.toFixed(1)}%
//             </span>
//           </div>
//         )}
//       </div>
//     );
//   }

//   /** ------------------------------------------------------------------
//    *  Conditional UI states
//    *  -----------------------------------------------------------------*/
//   if (loading) return <LoadingSpinner text="Loading applicant type pie..." />;
//   if (err) return <div style={{ color: "red" }}>{err}</div>;
//   if (!summary) return <div>No data available.</div>;

//   /** ------------------------------------------------------------------
//    *  Main render
//    *  -----------------------------------------------------------------*/
//   return (
//     <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
//       {/* Row: pie + legend */}
//       <div style={{ display: "flex", alignItems: "center", gap: 24 }}>
//         <PieChart percentages={summary.percentages} labels={summary.labels} />

//         {/* Legend */}
//         <div style={{
//           display: "flex",
//           flexDirection: "column",
//           flexWrap: "wrap",
//           maxHeight: 180
//         }}>
//           {summary.labels.map((lab, i) => (
//             <div key={i} style={{
//               display: "flex",
//               alignItems: "center",
//               marginBottom: 6,
//               width: 140
//             }}>
//               <span style={{
//                 width: 12,
//                 height: 12,
//                 borderRadius: "50%",
//                 background: PIE_COLORS[i % PIE_COLORS.length],
//                 marginRight: 6,
//                 border: hoverIdx === i ? "2px solid #333" : "1px solid #ccc"
//               }} />
//               <span style={{
//                 fontSize: 14,
//                 color: "#232526",
//                 whiteSpace: "nowrap",
//                 overflow: "hidden",
//                 textOverflow: "ellipsis"
//               }}>
//                 {lab}
//               </span>
//             </div>
//           ))}
//         </div>
//       </div>

//       {/* Co-applicant rate */}
//       {coapplicant && (
//         <div style={{ fontSize: 15 }}>
//           Co-applicant rate:&nbsp;
//           <strong>{coapplicant.coapplicant_rate.toFixed(2)}%</strong>&nbsp;
//           ({coapplicant.coapplicant_count} / {coapplicant.total_applications})
//         </div>
//       )}
//     </div>
//   );
// };

// export default ApplicantTypePie;







import React, { useEffect, useState } from "react";
import LoadingSpinner from "./LoadingSpinner";

const PIE_COLORS = [
  "#E6F4EA", "#B9E6C9", "#70C36A",
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
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
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
    </div>
  );
};

export default ApplicantTypePie;
