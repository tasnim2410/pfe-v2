import React, { useEffect, useState } from "react";

// Color palette for distinct applicant types
const COLOR_PALETTE = [
  "#3877b3", // Blue
  "#50c878", // Emerald Green
  "#f2994a", // Orange
  "#ca4e79", // Rose
  "#7a6ff0", // Soft Purple
  "#ffd166", // Pastel Yellow
  "#43b5a0", // Teal
  "#e46b70", // Coral Red
  "#6ec6ca", // Aqua
  "#f9c846", // Sunflower
  "#9f86c0", // Lavender
  "#828282", // Neutral Gray
  "#4f4f4f"  // Deep Charcoal (for overflow)
];

// Assign a color for each unique applicant type
function getTypeColorMap(types: string[]) {
  const colorMap: { [type: string]: string } = {};
  // Optionally sort for stable assignments across reloads
  types.sort();
  types.forEach((type, i) => {
    colorMap[type] = COLOR_PALETTE[i % COLOR_PALETTE.length];
  });
  return colorMap;
}

function getNodePositions(types: string[]): { [type: string]: [number, number] } {
  const n = types.length;
  const cx = 250, cy = 200, r = 140;
  const mapping: { [type: string]: [number, number] } = {};
  types.forEach((t, i) => {
    const angle = (2 * Math.PI * i) / n - Math.PI / 2;
    mapping[t] = [
      Math.round(cx + r * Math.cos(angle)),
      Math.round(cy + r * Math.sin(angle))
    ];
  });
  return mapping;
}

const Tooltip: React.FC<{ x: number, y: number, children: React.ReactNode }> = ({ x, y, children }) => (
  <div
    style={{
      position: "fixed",
      left: x + 12,
      top: y + 6,
      zIndex: 10000,
      pointerEvents: "none",
      background: "#232526",
      color: "#fff",
      borderRadius: 7,
      padding: "7px 13px",
      fontSize: 15,
      fontWeight: 600,
      boxShadow: "0 2px 14px #23252636",
      whiteSpace: "nowrap",
      userSelect: "none"
    }}
  >
    {children}
  </div>
);

const cardStyle: React.CSSProperties = {
  background: "#fff",
  borderRadius: 18,
  boxShadow: "0 2px 18px #B2DBA422",
  padding: 30,
  width: 550,
  maxWidth: "100%",
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  minHeight: 350,
  margin: "0 auto",
  position: "relative"
};

const legendStyle: React.CSSProperties = {
  display: "flex",
  flexWrap: "wrap",
  gap: 14,
  margin: "16px 0 8px 0",
  fontSize: 15,
  alignItems: "center",
  justifyContent: "center"
};

function getDominantNetworkType(typeCounts: { [k: string]: number }) {
  // Helper functions to categorize types
  const isCompany = (type: string) => {
    const lowerType = type.toLowerCase();
    return lowerType.includes('company') || 
           lowerType.includes('corp') || 
           lowerType.includes('inc') || 
           lowerType.includes('ltd') || 
           lowerType.includes('co.') || 
           lowerType.includes('llc') || 
           lowerType.includes('ag') || 
           lowerType.includes('gmbh') || 
           lowerType.includes('co') || 
           lowerType.includes('holdings') || 
           lowerType.includes('ventures') ||
           // Include all specific company types from Python function
           lowerType.includes('energy company') ||
           lowerType.includes('technology company') ||
           lowerType.includes('material science/nanotechnology company') ||
           lowerType.includes('environmental protection company');
  };

  const isInventor = (type: string) => type.toLowerCase().includes('inventor');
  
  const isAutomotive = (type: string) => type.toLowerCase().includes('automotive');
  
  const isUniversity = (type: string) => 
    type.toLowerCase().includes('university') || 
    type.toLowerCase().includes('research institution');
  
  const isTechnicalUniversity = (type: string) => 
    type.toLowerCase().includes('technical university');
  
  const isResearchLab = (type: string) => 
    type.toLowerCase().includes('research laboratory');
  
  const isGovernment = (type: string) => 
    type.toLowerCase().includes('government') || 
    type.toLowerCase().includes('public institution');

  // Initialize counters for approved network types
  const counters = {
    "company-company": 0,
    "company-individual inventor": 0,
    "company-automotive manufacturer": 0,
    "company-university/research institution": 0,
    "company-technical university": 0,
    "company-research laboratory": 0,
    "company-government/public institution": 0
  };

  // Process each edge type pair
  Object.entries(typeCounts).forEach(([pair, count]) => {
    const [a, b] = pair.split("-");
    
    // Determine the relationship type
    let relationshipType = null;
    
    if (isCompany(a) && isCompany(b)) {
      relationshipType = "company-company";
    } else if ((isCompany(a) && isInventor(b)) || (isCompany(b) && isInventor(a))) {
      relationshipType = "company-individual inventor";
    } else if ((isCompany(a) && isAutomotive(b)) || (isCompany(b) && isAutomotive(a))) {
      relationshipType = "company-automotive manufacturer";
    } else if ((isCompany(a) && isUniversity(b)) || (isCompany(b) && isUniversity(a))) {
      relationshipType = "company-university/research institution";
    } else if ((isCompany(a) && isTechnicalUniversity(b)) || (isCompany(b) && isTechnicalUniversity(a))) {
      relationshipType = "company-technical university";
    } else if ((isCompany(a) && isResearchLab(b)) || (isCompany(b) && isResearchLab(a))) {
      relationshipType = "company-research laboratory";
    } else if ((isCompany(a) && isGovernment(b)) || (isCompany(b) && isGovernment(a))) {
      relationshipType = "company-government/public institution";
    }
    
    if (relationshipType) {
      counters[relationshipType as keyof typeof counters] += count;
    }
  });

  // Find the dominant network type
  const entries = Object.entries(counters).filter(([, count]) => count > 0);
  
  if (entries.length === 0) {
    return { label: "company", count: 0 };
  }

  // Sort by count in descending order and return the highest
  const dominant = entries.sort(([, a], [, b]) => b - a)[0];
  return { label: dominant[0], count: dominant[1] };
}

const ApplicantCollaborationNetwork: React.FC = () => {
  const [data, setData] = useState<any | null>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  // Tooltip state
  const [hoverEdge, setHoverEdge] = useState<null | { source: string, target: string, weight: number, x: number, y: number }>(null);
  const [hoverNode, setHoverNode] = useState<null | { type: string, count: number, x: number, y: number }>(null);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      setLoading(true);
      setErr(null);
      try {
        // Get the port ONCE, use for both requests
        const portRes = await fetch("/backend_port.txt");
        const port = (await portRes.text()).trim();

        // 1. POST to update applicants (required before getting network)
        await fetch(`http://localhost:${port}/api/analyze_applicants`, {
          method: "POST"
        });

        // 2. GET the applicant collaboration network
        const apiRes = await fetch(`http://localhost:${port}/api/applicant_collaboration_network`);
        if (!apiRes.ok) throw new Error();
        const json = await apiRes.json();
        if (!cancelled) setData(json);
        setLoading(false);
      } catch {
        setErr("Failed to fetch applicant collaboration network.");
        setLoading(false);
      }
    };
    load();
    return () => { cancelled = true; };
  }, []);

  if (loading) return <div style={{ textAlign: "center", marginTop: 32 }}>Loading Applicant Collaboration Network...</div>;
  if (err) return (
    <div style={{ color: "#EA3C53", textAlign: "center", margin: 22 }}>
      {err}
    </div>
  );
  if (!data) return null;

  const nodeTypes = Array.from(
    new Set(data.edges.flatMap((e: any) => [e.source, e.target]))
  ) as string[];

  // Assign unique color to each node type
  const typeColorMap = getTypeColorMap([...nodeTypes]);
  const getNodeColor = (type: string) =>
    typeColorMap[type] || "#BABEC6";

  const nodePos = getNodePositions(nodeTypes);
  const nodes = nodeTypes.map((type: string) => ({
    type,
    pos: nodePos[type]
  }));

  const W = 500, H = 400, R = 33;

  const nodeConnectionTotals: { [type: string]: number } = {};
  data.edges.forEach((edge: any) => {
    nodeConnectionTotals[edge.source] = (nodeConnectionTotals[edge.source] || 0) + edge.weight;
    nodeConnectionTotals[edge.target] = (nodeConnectionTotals[edge.target] || 0) + edge.weight;
  });

  const getSVGCoords = (evt: React.MouseEvent) => {
    return { x: evt.clientX, y: evt.clientY };
  };

  const summary = getDominantNetworkType(data.type_counts);

  return (
    <div style={cardStyle}>
      <div style={{ fontWeight: 800, fontSize: 22, color: "#232526", marginBottom: 12 }}>
        Applicant Collaboration Network
      </div>
      {/* SVG Network */}
      <svg width={W} height={H} style={{ display: "block", margin: "0 auto" }}>
        {/* Edges */}
        {data.edges.map((edge: any, i: number) => {
          const from = nodes.find(n => n.type === edge.source)?.pos;
          const to = nodes.find(n => n.type === edge.target)?.pos;
          if (!from || !to) return null;
          return (
            <g
              key={i}
              onMouseMove={e => {
                const { x, y } = getSVGCoords(e);
                setHoverEdge({
                  source: edge.source,
                  target: edge.target,
                  weight: edge.weight,
                  x,
                  y
                });
              }}
              onMouseLeave={() => setHoverEdge(null)}
              style={{ cursor: "pointer" }}
            >
              <line
                x1={from[0]} y1={from[1]} x2={to[0]} y2={to[1]}
                stroke="#B2BFC3"
                strokeWidth={2 + Math.log(edge.weight + 1) * 2}
                opacity={0.80}
                markerEnd="url(#arrowhead)"
              />
            </g>
          );
        })}
        {/* Arrow marker */}
        <defs>
          <marker id="arrowhead" markerWidth="7" markerHeight="7" refX="7" refY="3.5" orient="auto" markerUnits="strokeWidth">
            <polygon points="0 0, 7 3.5, 0 7" fill="#B2BFC3" />
          </marker>
        </defs>
        {/* Nodes */}
        {nodes.map((node, i) => (
          <g
            key={node.type}
            onMouseMove={e => {
              const { x, y } = getSVGCoords(e);
              setHoverNode({
                type: node.type,
                count: nodeConnectionTotals[node.type] || 0,
                x,
                y
              });
            }}
            onMouseLeave={() => setHoverNode(null)}
            style={{ cursor: "pointer" }}
          >
            <circle
              cx={node.pos[0]}
              cy={node.pos[1]}
              r={R}
              fill={getNodeColor(node.type)}
              stroke="#232526"
              strokeWidth={2}
            />
            <text
              x={node.pos[0]}
              y={node.pos[1]}
              textAnchor="middle"
              alignmentBaseline="middle"
              style={{
                fill: "#232526",
                fontWeight: 800,
                fontSize: 13,
                pointerEvents: "none"
              }}
            >
              {node.type.length > 20
                ? node.type.replace("university/research institution", "university")
                : node.type}
            </text>
          </g>
        ))}
      </svg>
      {/* Node legend */}
      <div style={legendStyle}>
        {nodeTypes.map(type => (
          <span key={type} style={{ display: "flex", alignItems: "center", gap: 5 }}>
            <span style={{
              width: 14, height: 14, borderRadius: "50%",
              background: getNodeColor(type), display: "inline-block",
              border: "1.5px solid #bbb"
            }} />
            {type.length > 20
              ? type.replace("university/research institution", "university")
              : type}
          </span>
        ))}
      </div>
      {/* Dominant network summary */}
      <div style={{
        background: "#858181",
        color: "#fff",
        borderRadius: 8,
        padding: "12px 16px",
        marginTop: 19,
        fontWeight: 700,
        fontSize: 17,
        letterSpacing: 0.2,
        textAlign: "center",
        boxShadow: "0 2px 8px #EA3C5333"
      }}>
        Dominant Network Type:{" "}
        <span style={{ color: "#BDD248", fontWeight: 800, fontSize: 18, marginLeft: 6 }}>
          {summary.label} <span style={{ fontSize: 15, color: "#fff", fontWeight: 600 }}></span>
        </span>
      </div>
      {/* Tooltips */}
      {hoverEdge && (
        <Tooltip x={hoverEdge.x} y={hoverEdge.y}>
          {hoverEdge.source.replace("university/research institution", "university")} —{" "}
          {hoverEdge.target.replace("university/research institution", "university")}
          :{" "}
          <span style={{ color: "#BDD248", fontWeight: 700 }}>{hoverEdge.weight}</span>{" "}
          connection{hoverEdge.weight > 1 ? "s" : ""}
        </Tooltip>
      )}
      {hoverNode && (
        <Tooltip x={hoverNode.x} y={hoverNode.y}>
          {hoverNode.type.replace("university/research institution", "university")}
          :{" "}
          <span style={{ color: "#BDD248", fontWeight: 700 }}>{hoverNode.count}</span>{" "}
          total collaboration{hoverNode.count !== 1 ? "s" : ""}
        </Tooltip>
      )}
    </div>
  );
};

export default ApplicantCollaborationNetwork;