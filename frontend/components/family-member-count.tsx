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

// register chart.js pieces only once
Chart.register(BarElement, CategoryScale, LinearScale, Tooltip, Legend);

interface ApiResponse {
  datasets: { data: number[]; label: string }[];
  labels: string[];
}

export const FamilyMemberCountChart: React.FC<{ port?: number }> = ({
  port: overridePort,
}) => {
  const [chartData, setChartData] = useState<ApiResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

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

  if (loading) return <div>Loading family-member counts…</div>;
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
        ticks: { color: "#3B3C3D", font: { weight: 600 } },
      },
      y: {
        beginAtZero: true,
        ticks: { color: "#3B3C3D" },
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
      }}
    >
      <div
        style={{
          marginTop: 12,
          padding: "8px 28px",
          fontWeight: 700,
          fontSize: 20,
          background: "#232526",
          color: "#fff",
          borderRadius: 10,
          alignSelf: "center",
          boxShadow: "0 1px 8px #bdd24816",
        }}
      >
        Family Member Count / Country
      </div>

      <div style={{ height: 280, marginTop: 10 }}>
        <Bar data={data} options={options} />
      </div>
    </div>
  );
};

export default FamilyMemberCountChart;


// import React, { useEffect, useMemo, useState } from "react";
// import { ComposableMap, Geographies, Geography } from "react-simple-maps";
// import { scaleLinear } from "d3-scale";
// import { Tooltip as ReactTooltip } from 'react-tooltip';

// interface ApiResponse {
//   datasets: { data: number[]; label: string }[];
//   labels: string[]; // ISO-3166 alpha-2 codes (JP, CN, …)
// }

// /* topojson world polygons */
// const geoUrl =
//   "https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json";

// /* quick helper for title-case country names in tooltip */
// const niceName = (raw?: string) =>
//   raw ? raw.replace(/(^|\s)\S/g, (c) => c.toUpperCase()) : "Unknown";

// const FamilyMemberCountMap: React.FC<{ port?: number }> = ({ port: override }) => {
//   const [data, setData] = useState<ApiResponse | null>(null);
//   const [error, setError] = useState<string | null>(null);
//   const [loading, setLoading] = useState(true);
//   const [tooltipContent, setTooltipContent] = useState("");
//   const [hoveredCountry, setHoveredCountry] = useState<string | null>(null);

//   /* ─── fetch once ─── */
//   useEffect(() => {
//     let dead = false;
    
//     (async () => {
//       try {
//         let port = override;
//         if (!port) {
//           const txt = await fetch("/backend_port.txt").then(r => r.text()).catch(() => "");
//           const n = parseInt(txt.trim(), 10);
//           port = Number.isFinite(n) ? n : 49473;
//         }

//         console.log(`Fetching from: http://localhost:${port}/api/family_member_counts`);
//         const res = await fetch(`http://localhost:${port}/api/family_member_counts`);
//         if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`);
//         const json: ApiResponse = await res.json();
//         console.log("API Response:", json);
//         if (!dead) setData(json);
//       } catch (e: any) {
//         console.error("API Error:", e);
//         if (!dead) setError(e.message ?? String(e));
//       } finally {
//         if (!dead) setLoading(false);
//       }
//     })();
    
//     return () => { dead = true; };
//   }, [override]);

//   /* ─── build lookup + colour-scale ─── */
//   const { countByIso, colour, maxCount } = useMemo(() => {
//     if (!data) return { countByIso: {}, colour: (n: number) => "#EEE", maxCount: 0 };

//     const counts: Record<string, number> = {};
//     data.labels.forEach((iso, i) => { counts[iso] = data.datasets[0].data[i]; });

//     const max = Math.max(...Object.values(counts));
    
//     /* Enhanced color scale - more vibrant colors */
//     const scale = scaleLinear<string>()
//       .domain([0, max])
//       .range(["#C8E6C9", "#1B5E20"]); // Light green to dark green

//     return { 
//       countByIso: counts, 
//       colour: (n: number) => n > 0 ? scale(n) : "#F5F5F5",
//       maxCount: max
//     };
//   }, [data]);

//   /* ─── render states ─── */
//   if (loading) return (
//     <div style={{ 
//       display: 'flex', 
//       justifyContent: 'center', 
//       alignItems: 'center', 
//       height: '400px',
//       fontSize: '18px',
//       color: '#666'
//     }}>
//       Loading family member counts…
//     </div>
//   );
  
//   if (error) return (
//     <div style={{ 
//       color: "#EA3C53", 
//       textAlign: 'center', 
//       padding: '20px',
//       fontSize: '16px'
//     }}>
//       <div>Error loading data: {error}</div>
//       <div style={{ fontSize: '14px', marginTop: '10px', color: '#666' }}>
//         Make sure your backend is running on the correct port
//       </div>
//     </div>
//   );
  
//   if (!data) return null;

//   return (
//     <>
//       {/* Enhanced Tooltip */}
//       <ReactTooltip 
//         id="country-tooltip"
//         place="top"
//         style={{
//           backgroundColor: '#333',
//           color: '#fff',
//           borderRadius: '8px',
//           padding: '8px 12px',
//           fontSize: '14px',
//           fontWeight: '500',
//           boxShadow: '0 4px 8px rgba(0,0,0,0.3)'
//         }}
//       >
//         {tooltipContent}
//       </ReactTooltip>

//       <div
//         style={{
//           background: "#fff",
//           borderRadius: 18,
//           boxShadow: "0 4px 20px rgba(0,0,0,0.1)",
//           padding: "0 16px 16px 16px",
//           width: "100%",
//           display: "flex",
//           flexDirection: "column",
//           maxWidth: "1200px",
//           margin: "0 auto"
//         }}
//       >
//         {/* Header */}
//         <div
//           style={{
//             marginTop: 12,
//             padding: "12px 28px",
//             fontWeight: 700,
//             fontSize: 20,
//             background: "linear-gradient(135deg, #2E7D32, #1B5E20)",
//             color: "#fff",
//             borderRadius: 10,
//             alignSelf: "center",
//             boxShadow: "0 2px 10px rgba(0,0,0,0.2)",
//             textAlign: "center"
//           }}
//         >
//           Family Member Count by Country
//         </div>

//         {/* Legend */}
//         <div style={{
//           display: 'flex',
//           alignItems: 'center',
//           justifyContent: 'center',
//           marginTop: 15,
//           marginBottom: 10,
//           fontSize: '14px',
//           color: '#666'
//         }}>
//           <span style={{ marginRight: 10 }}>Fewer families</span>
//           <div style={{
//             width: 100,
//             height: 10,
//             background: 'linear-gradient(to right, #C8E6C9, #1B5E20)',
//             borderRadius: 5,
//             border: '1px solid #ddd'
//           }}></div>
//           <span style={{ marginLeft: 10 }}>More families</span>
//           <span style={{ marginLeft: 15, fontWeight: 600 }}>
//             Max: {maxCount}
//           </span>
//         </div>

//         {/* Map */}
//         <div style={{ height: 400, marginTop: 10 }}>
//           <ComposableMap
//             projection="geoEqualEarth"
//             projectionConfig={{ scale: 140 }}
//             style={{ width: "100%", height: "100%" }}
//           >
//             <Geographies geography={geoUrl}>
//               {({ geographies }: { geographies: any[] }) =>
//                 geographies.map((geo: any) => {
//                   const iso2 = geo.properties.ISO_A2 as string;
//                   const val = countByIso[iso2] ?? 0;
//                   const isHovered = hoveredCountry === iso2;
                  
//                   return (
//                     <Geography
//                       key={geo.rsmKey}
//                       geography={geo}
//                       fill={colour(val)}
//                       stroke={isHovered ? "#2E7D32" : "#CCC"}
//                       strokeWidth={isHovered ? 1.5 : 0.5}
//                       data-tooltip-id="country-tooltip"
//                       onMouseEnter={() => {
//                         const name =
//                           niceName(
//                             geo.properties.NAME ||
//                             geo.properties.ADMIN ||
//                             geo.properties.NAME_LONG
//                           ) || iso2;
//                         setTooltipContent(`${name}: ${val} family members`);
//                         setHoveredCountry(iso2);
//                       }}
//                       onMouseLeave={() => {
//                         setTooltipContent("");
//                         setHoveredCountry(null);
//                       }}
//                       style={{
//                         default: { 
//                           outline: "none",
//                           cursor: val ? "pointer" : "default"
//                         },
//                         hover: { 
//                           outline: "none", 
//                           opacity: 0.8,
//                           filter: "brightness(1.1)"
//                         },
//                         pressed: { 
//                           outline: "none",
//                           opacity: 0.9
//                         },
//                       }}
//                     />
//                   );
//                 })
//               }
//             </Geographies>
//           </ComposableMap>
//         </div>

//         {/* Stats */}
//         <div style={{
//           marginTop: 15,
//           padding: 15,
//           background: '#F8F9FA',
//           borderRadius: 10,
//           textAlign: 'center'
//         }}>
//           <div style={{ fontSize: '14px', color: '#666' }}>
//             Total Countries: {Object.keys(countByIso).length} | 
//             Total Family Members: {Object.values(countByIso).reduce((a, b) => a + b, 0)}
//           </div>
//         </div>
//       </div>
//     </>
//   );
// };

// export default FamilyMemberCountMap;