import React, { useEffect, useState } from "react";

// Enhanced color palette with better contrast
const COLOR_PALETTE = [
  "#2E86AB", // Vibrant Blue
  "#A23B72", // Magenta
  "#F18F01", // Orange
  "#73B769", // Green
  "#7B6CF0", // Purple
  "#FFD166", // Yellow
  "#06D6A0", // Teal
  "#EF476F", // Pink
  "#118AB2", // Ocean Blue
  "#FF9E6D", // Coral
  "#9B5DE5", // Violet
  "#00BBF9", // Sky Blue
  "#F15BB5"  // Hot Pink
];

// Assign a color for each unique applicant type
function getTypeColorMap(types: string[]) {
  const colorMap: { [type: string]: string } = {};
  types.forEach((type, i) => {
    colorMap[type] = COLOR_PALETTE[i % COLOR_PALETTE.length];
  });
  return colorMap;
}

function getNodePositions(types: string[]): { [type: string]: [number, number] } {
  const n = types.length;
  const cx = 250, cy = 200, r = types.length > 5 ? 140 : 100;
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
      background: "rgba(35, 37, 38, 0.95)",
      color: "#fff",
      borderRadius: 8,
      padding: "10px 15px",
      fontSize: 14,
      fontWeight: 500,
      boxShadow: "0 4px 20px rgba(0,0,0,0.3)",
      whiteSpace: "nowrap",
      userSelect: "none",
      backdropFilter: "blur(4px)",
      maxWidth: 300,
      lineHeight: 1.4
    }}
  >
    {children}
  </div>
);

const cardStyle: React.CSSProperties = {
  background: "#fff",
  borderRadius: 20,
  boxShadow: "0 4px 25px rgba(178, 219, 164, 0.15)",
  padding: 30,
  width: 600,
  maxWidth: "100%",
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  minHeight: 500,
  margin: "0 auto",
  position: "relative",
  border: "1px solid #f0f0f0"
};

const legendStyle: React.CSSProperties = {
  display: "flex",
  flexWrap: "wrap",
  gap: 10,
  margin: "20px 0 15px 0",
  fontSize: 13,
  alignItems: "center",
  justifyContent: "center",
  padding: "10px",
  background: "#f9f9f9",
  borderRadius: 10,
  maxWidth: "90%"
};

function getNetworkAnalysis(edges: any[]) {
  if (!edges || edges.length === 0) {
    return { 
      dominantType: "No Data",
      dominantCount: 0,
      strongestConnection: null,
      companyInventorCount: 0,
      companyCompanyCount: 0,
      companyUniversityCount: 0
    };
  }

  // Helper functions to categorize entities
  const isCompany = (type: string) => {
    const lowerType = type.toLowerCase();
    return lowerType.includes('company');
  };

  const isInventor = (type: string) => type.toLowerCase().includes('inventor');
  const isUniversity = (type: string) => 
    type.toLowerCase().includes('university') || 
    type.toLowerCase().includes('research institution');

  // Calculate connection counts
  let companyInventorCount = 0;
  let companyCompanyCount = 0;
  let companyUniversityCount = 0;

  edges.forEach((edge: any) => {
    const { source, target, weight } = edge;
    
    if ((isCompany(source) && isInventor(target)) || (isCompany(target) && isInventor(source))) {
      companyInventorCount += weight;
    } else if (isCompany(source) && isCompany(target)) {
      companyCompanyCount += weight;
    } else if ((isCompany(source) && isUniversity(target)) || (isCompany(target) && isUniversity(source))) {
      companyUniversityCount += weight;
    }
  });

  // Find the dominant network type
  const connectionCounts = {
    "company-individual inventor": companyInventorCount,
    "company-company": companyCompanyCount,
    "company-university/research institution": companyUniversityCount
  };

  const entries = Object.entries(connectionCounts);
  const [dominantType, dominantCount] = entries.reduce(
    (max, entry) => (entry[1] > max[1] ? entry : max),
    ["", 0]
  );

  // Find strongest connection
  const maxWeight = Math.max(...edges.map((e: any) => e.weight));
  const strongestConnection = edges.find((e: any) => e.weight === maxWeight);

  // Calculate percentages
  const totalConnections = edges.reduce((sum, edge) => sum + edge.weight, 0);
  const percentage = totalConnections > 0 ? (dominantCount / totalConnections * 100).toFixed(1) : "0";

  return {
    dominantType,
    dominantCount,
    percentage,
    strongestConnection,
    companyInventorCount,
    companyCompanyCount,
    companyUniversityCount,
    totalConnections
  };
}

const ApplicantCollaborationNetwork: React.FC = () => {
  const [data, setData] = useState<any | null>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [showComment, setShowComment] = useState(false);
  const [commentPosition, setCommentPosition] = useState({ x: 0, y: 0 });

  // Tooltip state
  const [hoverEdge, setHoverEdge] = useState<null | { source: string, target: string, weight: number, x: number, y: number }>(null);
  const [hoverNode, setHoverNode] = useState<null | { type: string, connections: number, x: number, y: number }>(null);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      setLoading(true);
      setErr(null);
      try {
        const portRes = await fetch("/backend_port.txt");
        const port = (await portRes.text()).trim();

        await fetch(`http://localhost:${port}/api/analyze_applicants`, {
          method: "POST"
        });

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

  if (loading) return <div style={{ textAlign: "center", marginTop: 32, fontSize: 16, color: "#666" }}>Loading Collaboration Network...</div>;
  if (err) return (
    <div style={{ color: "#EA3C53", textAlign: "center", margin: 22, fontSize: 15 }}>
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

  const W = 500, H = 400, R = 35;

  // Calculate node connection counts
  const nodeConnections: { [type: string]: number } = {};
  data.edges.forEach((edge: any) => {
    nodeConnections[edge.source] = (nodeConnections[edge.source] || 0) + edge.weight;
    nodeConnections[edge.target] = (nodeConnections[edge.target] || 0) + edge.weight;
  });

  // Find strongest connection for visual emphasis
  const maxWeight = Math.max(...data.edges.map((e: any) => e.weight));
  const strongestEdge = data.edges.find((e: any) => e.weight === maxWeight);

  const getSVGCoords = (evt: React.MouseEvent) => {
    return { x: evt.clientX, y: evt.clientY };
  };

  const analysis = getNetworkAnalysis(data.edges);

  // Format edge weight for display
  const getEdgeStrokeWidth = (weight: number) => {
    return 2 + (weight / maxWeight) * 8;
  };

  // Get a simple description for the dominant type
  const getDominantTypeDescription = (type: string) => {
    switch(type) {
      case "company-individual inventor":
        return "Companies collaborating with independent inventors";
      case "company-company":
        return "Business-to-business partnerships";
      case "company-university/research institution":
        return "Industry-academia research collaborations";
      default:
        return "Collaboration network";
    }
  };

  return (
    <div style={cardStyle}>
      {/* Header with title and info icon */}
      <div style={{ 
        display: "flex", 
        justifyContent: "space-between", 
        alignItems: "center", 
        width: "100%",
        marginBottom: 20,
        position: "relative"
      }}>
        <div style={{ fontWeight: 800, fontSize: 24, color: "#232526", letterSpacing: 0.5 }}>
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

      {/* Subtitle */}
      <div style={{ 
        fontSize: 14, 
        color: "#666", 
        marginBottom: 25, 
        textAlign: "center",
        maxWidth: "90%"
      }}>
        Visualizing collaboration patterns between different applicant types
      </div>

      {/* SVG Network */}
      <div style={{ 
        position: "relative", 
        width: W, 
        height: H, 
        marginBottom: 10,
        background: "#f9f9f9",
        borderRadius: 12,
        overflow: "hidden",
        border: "1px solid #eaeaea"
      }}>
        <svg width={W} height={H} style={{ display: "block" }}>
          {/* Edges */}
          {data.edges.map((edge: any, i: number) => {
            const from = nodes.find(n => n.type === edge.source)?.pos;
            const to = nodes.find(n => n.type === edge.target)?.pos;
            if (!from || !to) return null;
            
            const isStrongest = edge.weight === maxWeight;
            
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
                  stroke={isStrongest ? "#FF6B6B" : "#8FA4B8"}
                  strokeWidth={getEdgeStrokeWidth(edge.weight)}
                  opacity={isStrongest ? 0.9 : 0.7}
                  strokeLinecap="round"
                />
                {/* Edge weight label */}
                <text
                  x={(from[0] + to[0]) / 2}
                  y={(from[1] + to[1]) / 2 - 5}
                  textAnchor="middle"
                  style={{
                    fill: isStrongest ? "#FF6B6B" : "#5A6C7D",
                    fontWeight: 700,
                    fontSize: 12,
                    pointerEvents: "none",
                    paintOrder: "stroke",
                    stroke: "#fff",
                    strokeWidth: 3,
                    strokeLinecap: "round",
                    strokeLinejoin: "round"
                  }}
                >
                  {edge.weight}
                </text>
              </g>
            );
          })}
          
          {/* Nodes */}
          {nodes.map((node, i) => {
            const connections = nodeConnections[node.type] || 0;
            return (
              <g
                key={node.type}
                onMouseMove={e => {
                  const { x, y } = getSVGCoords(e);
                  setHoverNode({
                    type: node.type,
                    connections,
                    x,
                    y
                  });
                }}
                onMouseLeave={() => setHoverNode(null)}
                style={{ cursor: "pointer" }}
              >
                {/* Node outer glow for highly connected nodes */}
                {connections > 10 && (
                  <circle
                    cx={node.pos[0]}
                    cy={node.pos[1]}
                    r={R + 5}
                    fill="none"
                    stroke="#FFD166"
                    strokeWidth={2}
                    opacity={0.4}
                  />
                )}
                
                <circle
                  cx={node.pos[0]}
                  cy={node.pos[1]}
                  r={R}
                  fill={getNodeColor(node.type)}
                  stroke="#232526"
                  strokeWidth={2.5}
                  filter="drop-shadow(0 2px 4px rgba(0,0,0,0.1))"
                />
                
                {/* Connection count badge */}
                <circle
                  cx={node.pos[0] + R - 10}
                  cy={node.pos[1] - R + 10}
                  r={12}
                  fill="#232526"
                />
                <text
                  x={node.pos[0] + R - 10}
                  y={node.pos[1] - R + 13}
                  textAnchor="middle"
                  style={{
                    fill: "#fff",
                    fontWeight: 700,
                    fontSize: 10,
                    pointerEvents: "none"
                  }}
                >
                  {connections}
                </text>
                
                <text
                  x={node.pos[0]}
                  y={node.pos[1] + 5}
                  textAnchor="middle"
                  style={{
                    fill: "#232526",
                    fontWeight: 600,
                    fontSize: 11,
                    pointerEvents: "none",
                    textShadow: "0 1px 2px rgba(255,255,255,0.8)"
                  }}
                >
                  {node.type.replace("university/research institution", "University").replace("company - ", "")}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      {/* Network Statistics - IMPROVED */}
      <div style={{
        display: "flex",
        justifyContent: "space-between",
        width: "90%",
        margin: "15px 0",
        padding: "12px",
        background: "#f5f7fa",
        borderRadius: 10,
        fontSize: 13
      }}>
        <div style={{ textAlign: "center", flex: 1 }}>
          <div style={{ fontWeight: 700, color: "#2E86AB", fontSize: 16 }}>
            {nodeTypes.length}
          </div>
          <div style={{ color: "#666" }}>Applicant Types</div>
        </div>
        <div style={{ textAlign: "center", flex: 1, borderLeft: "1px solid #ddd", borderRight: "1px solid #ddd" }}>
          <div style={{ fontWeight: 700, color: "#A23B72", fontSize: 16 }}>
            {data.edges.length}
          </div>
          <div style={{ color: "#666" }}>Connections</div>
        </div>
        <div style={{ textAlign: "center", flex: 1 }}>
          <div style={{ fontWeight: 700, color: "#F18F01", fontSize: 16 }}>
            {maxWeight}
          </div>
          <div style={{ color: "#666" }}>Strongest Link</div>
        </div>
      </div>

      {/* Dominant Network Type - IMPROVED DISPLAY */}
      <div style={{
        background: "linear-gradient(135deg, #232526 0%, #414345 100%)",
        color: "#fff",
        borderRadius: 12,
        padding: "18px 20px",
        marginTop: 10,
        fontWeight: 700,
        fontSize: 16,
        letterSpacing: 0.2,
        textAlign: "center",
        boxShadow: "0 4px 15px rgba(35, 37, 38, 0.2)",
        width: "90%",
        border: "1px solid #333"
      }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "center", marginBottom: 8 }}>
          <div style={{ 
            width: 12, 
            height: 12, 
            borderRadius: "50%", 
            backgroundColor: "#BDD248", 
            marginRight: 10 
          }}></div>
          <div>Network Pattern:</div>
        </div>
        <div style={{ 
          color: "#BDD248", 
          fontWeight: 800, 
          fontSize: 18, 
          marginBottom: 6
        }}>
          {getDominantTypeDescription(analysis.dominantType)}
        </div>
        <div style={{ 
          fontSize: 13, 
          fontWeight: 500, 
          color: "#ccc",
          lineHeight: 1.4,
          opacity: 0.9
        }}>
          This pattern represents <strong>{analysis.percentage}%</strong> of all collaborations 
          ({analysis.dominantCount} of {analysis.totalConnections} connections)
        </div>
        
        {/* Strongest Connection Highlight */}
        {analysis.strongestConnection && (
          <div style={{
            marginTop: 10,
            padding: "8px",
            background: "rgba(255, 107, 107, 0.1)",
            borderRadius: 6,
            borderLeft: "3px solid #FF6B6B"
          }}>
            <div style={{ fontSize: 12, color: "#FFD166", fontWeight: 600 }}>
              Strongest Connection:
            </div>
            <div style={{ fontSize: 13, color: "#fff" }}>
              {analysis.strongestConnection.source.replace("company - ", "").replace("university/research institution", "University")} ↔{" "}
              {analysis.strongestConnection.target.replace("company - ", "").replace("university/research institution", "University")}
              <span style={{ marginLeft: 8, color: "#FF6B6B", fontWeight: 700 }}>
                ({analysis.strongestConnection.weight} collaborations)
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Comment Block (on hover) - SIMPLIFIED */}
      {showComment && (
        <div
          style={{
            position: "fixed",
            left: commentPosition.x - 280,
            top: commentPosition.y,
            width: 260,
            background: "#fff",
            border: "1px solid #ddd",
            borderRadius: "8px",
            padding: "15px",
            boxShadow: "0 4px 12px rgba(0,0,0,0.15)",
            zIndex: 10000,
            fontSize: "13px",
            lineHeight: "1.4"
          }}
          onMouseEnter={() => setShowComment(true)}
          onMouseLeave={() => setShowComment(false)}
        >
          <div style={{ fontWeight: "bold", marginBottom: "8px", color: "#333", fontSize: "14px" }}>
            🔗 Network Overview
          </div>
          <div style={{ color: "#555" }}>
            <div style={{ marginBottom: "10px" }}>
              This network shows how different applicant types collaborate on patents.
            </div>
            
            <div style={{ 
              padding: "8px", 
              background: "#f8f9fa",
              borderRadius: "6px",
              borderLeft: "3px solid #2E86AB",
              fontSize: "12px"
            }}>
              <strong>Key Insight:</strong>
              <br />
              The network is dominated by <strong>{getDominantTypeDescription(analysis.dominantType)}</strong>.
            </div>

            {analysis.companyInventorCount > 0 && (
              <div style={{ marginTop: "10px", fontSize: "12px" }}>
                <strong>Collaboration Breakdown:</strong>
                <div style={{ marginTop: "5px" }}>
                  • Company-Inventor: {analysis.companyInventorCount} connections
                </div>
                {analysis.companyCompanyCount > 0 && (
                  <div>• Company-Company: {analysis.companyCompanyCount} connections</div>
                )}
                {analysis.companyUniversityCount > 0 && (
                  <div>• Company-University: {analysis.companyUniversityCount} connections</div>
                )}
              </div>
            )}
          </div>
          <div style={{ 
            marginTop: "10px", 
            fontSize: "11px", 
            color: "#888",
            fontStyle: "italic",
            borderTop: "1px solid #eee",
            paddingTop: "8px"
          }}>
            💡 Thicker lines indicate stronger collaborations
          </div>
        </div>
      )}

      {/* Tooltips */}
      {hoverEdge && (
        <Tooltip x={hoverEdge.x} y={hoverEdge.y}>
          <div style={{ fontWeight: 600, marginBottom: 4 }}>
            {hoverEdge.source.replace("company - ", "").replace("university/research institution", "University")} ↔{" "}
            {hoverEdge.target.replace("company - ", "").replace("university/research institution", "University")}
          </div>
          <div style={{ color: "#BDD248", fontWeight: 700 }}>
            {hoverEdge.weight} collaboration{hoverEdge.weight > 1 ? "s" : ""}
          </div>
        </Tooltip>
      )}
      
      {hoverNode && (
        <Tooltip x={hoverNode.x} y={hoverNode.y}>
          <div style={{ fontWeight: 600, marginBottom: 4 }}>
            {hoverNode.type.replace("company - ", "").replace("university/research institution", "University")}
          </div>
          <div style={{ color: "#BDD248", fontWeight: 700 }}>
            {hoverNode.connections} total connection{hoverNode.connections !== 1 ? "s" : ""}
          </div>
        </Tooltip>
      )}
    </div>
  );
};

export default ApplicantCollaborationNetwork;