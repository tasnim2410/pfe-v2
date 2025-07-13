

import React, { useEffect, useState } from "react";

const STAGES = [
  {
    label: "Emerging",
    color: "#D8EDC0",
    range: [-Infinity, 10],
  },
  {
    label: "Beginning",
    color: "#C7E6A0",
    range: [10, 20],
  },
  {
    label: "Ongoing",
    color: "#92CF4D",
    range: [20, 30],
  },
  {
    label: "Slowing",
    color: "#FFC001",
    range: [30, 50],
  },
  {
    label: "Ending",
    color: "#FE0000",
    range: [50, Infinity],
  },
];

function getStage(value: number) {
  for (const stage of STAGES) {
    if (value >= stage.range[0] && value < stage.range[1]) {
      return stage;
    }
  }
  return STAGES[0];
}

export const InnovationCycle: React.FC = () => {
  const [percentage, setPercentage] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;
    // 1. POST to analyze_applicants to update database (hardcoded port as user requested)
    fetch("/backend_port.txt")
      .then((res) => res.text())
      .then((port) =>
        fetch(`http://localhost:${port.trim()}/api/analyze_applicants`, { method: "POST" })
      )
      .catch(() => {
        setError("Failed to update data");
        return { status: "success" };
      })
      .finally(() => {
        // 2. Fetch backend port for the innovation cycle API
        fetch("/backend_port.txt")
          .then((res) => res.text())
          .then((port) => {
            if (!isMounted) return;
            return fetch(`http://localhost:${port.trim()}/api/innovation_cycle`)
              .then(async (res) => {
                let raw = await res.text();
                let value: number | null = null;
                try {
                  const json = JSON.parse(raw);
                  if (typeof json === "number") value = json;
                  else if (typeof json === "string" && !isNaN(Number(json))) value = Number(json);
                  else if (json && typeof json.value === "number") value = json.value;
                } catch {
                  if (!isNaN(Number(raw))) value = Number(raw);
                }
                if (isMounted) {
                  if (typeof value === "number" && !isNaN(value)) setPercentage(value);
                  else setError("Invalid innovation cycle response");
                }
              })
              .catch(() => {
                if (isMounted) setError("Could not fetch innovation cycle");
              });
          })
          .catch(() => {
            if (isMounted) setError("Could not fetch backend port");
          });
      });
    return () => {
      isMounted = false;
    };
  }, []);

  if (error) return <div style={{ color: "red" }}>{error}</div>;
  if (percentage === null) return <div>Loading innovation cycle...</div>;

  const activeStage = getStage(percentage);
  const activeIndex = STAGES.findIndex((s) => s.label === activeStage.label);

  // --- Bigger SVG, more center space, but overall smaller card ---
  const size = 245;
  const thickness = 37;
  const rOuter = size / 2 - 8;
  const rInner = rOuter - thickness;
  const cx = size / 2;
  const cy = size / 2;
  const anglePerStage = 360 / STAGES.length;

  // Arrow now sits inside, pointing ~75% from center
  const arrowAngle = anglePerStage * activeIndex + anglePerStage / 2 - 90;
  const arrowTipRadius = rInner - 5; // a bit before the color, not touching
  const arrowBaseRadius = arrowTipRadius - 22; // arrow length
  const arrowWidth = 14;

  const arrowTipX = cx + arrowTipRadius * Math.cos((arrowAngle) * Math.PI / 180);
  const arrowTipY = cy + arrowTipRadius * Math.sin((arrowAngle) * Math.PI / 180);

  const baseAngle1 = arrowAngle - arrowWidth / 2;
  const baseAngle2 = arrowAngle + arrowWidth / 2;

  const base1X = cx + arrowBaseRadius * Math.cos((baseAngle1) * Math.PI / 180);
  const base1Y = cy + arrowBaseRadius * Math.sin((baseAngle1) * Math.PI / 180);

  const base2X = cx + arrowBaseRadius * Math.cos((baseAngle2) * Math.PI / 180);
  const base2Y = cy + arrowBaseRadius * Math.sin((baseAngle2) * Math.PI / 180);

  function getArcPath(startAngle: number, endAngle: number) {
    const start = (Math.PI / 180) * startAngle;
    const end = (Math.PI / 180) * endAngle;
    const x1 = cx + rOuter * Math.cos(start);
    const y1 = cy + rOuter * Math.sin(start);
    const x2 = cx + rOuter * Math.cos(end);
    const y2 = cy + rOuter * Math.sin(end);
    const x3 = cx + rInner * Math.cos(end);
    const y3 = cy + rInner * Math.sin(end);
    const x4 = cx + rInner * Math.cos(start);
    const y4 = cy + rInner * Math.sin(start);

    const largeArc = endAngle - startAngle > 180 ? 1 : 0;

    return `
      M ${x1} ${y1}
      A ${rOuter} ${rOuter} 0 ${largeArc} 1 ${x2} ${y2}
      L ${x3} ${y3}
      A ${rInner} ${rInner} 0 ${largeArc} 0 ${x4} ${y4}
      Z
    `;
  }

  return (
    <div style={{
      display: "flex", flexDirection: "column", alignItems: "center", gap: 12,
      padding: "17px 0", background: "#fff", borderRadius: 22,
      boxShadow: "0 2px 18px #B2DBA422", maxWidth: 295, minWidth: 220,
    }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        {STAGES.map((stage, i) => {
          const start = i * anglePerStage - 90;
          const end = (i + 1) * anglePerStage - 90;
          return (
            <path
              key={stage.label}
              d={getArcPath(start, end)}
              fill={stage.color}
              opacity={activeStage.label === stage.label ? 1 : 0.42}
              stroke="#fff"
              strokeWidth="2"
              style={{
                filter: activeStage.label === stage.label ? "drop-shadow(0px 0px 7px #bbb)" : undefined,
                transition: "opacity 0.3s, filter 0.3s"
              }}
            />
          );
        })}

        {/* Center label */}
        <text
          x={cx}
          y={cy - 16}
          textAnchor="middle"
          fontWeight={700}
          fontSize="21"
          fill="#232526"
          style={{ userSelect: "none", letterSpacing: "1.2px" }}
        >
          INNOVATION
        </text>
        <text
          x={cx}
          y={cy + 8}
          textAnchor="middle"
          fontWeight={600}
          fontSize="17"
          fill="#232526"
          style={{ userSelect: "none", letterSpacing: "1.1px" }}
        >
          CYCLE
        </text>
        {/* Percentage under text, in the circle */}
        <text
          x={cx}
          y={cy + 34}
          textAnchor="middle"
          fontWeight={600}
          fontSize="17"
          fill="#BDD248"
          style={{ userSelect: "none" }}
        >
          {percentage.toFixed(2)}%
        </text>

        {/* Arrowhead, further back from the colored edge */}
        <polygon
          points={`
            ${arrowTipX},${arrowTipY}
            ${base1X},${base1Y}
            ${base2X},${base2Y}
          `}
          fill="#232526"
          style={{ filter: "drop-shadow(0 1px 2px #bbb)" }}
        />
      </svg>
      {/* Legend */}
      <div style={{
        display: "flex", flexDirection: "row", gap: 12, marginTop: 3, flexWrap: "wrap", justifyContent: "center",
      }}>
        {STAGES.map((stage) => (
          <div key={stage.label} style={{ display: "flex", alignItems: "center", gap: 5, margin: "2px 0" }}>
            <div style={{
              width: 17, height: 8,
              background: stage.color,
              borderRadius: 5,
              opacity: 0.99,
              border: "1.1px solid #eee",
              boxShadow: "0 1px 2px #eee"
            }} />
            <span style={{
              color: "#232526",
              fontSize: 13.5,
              fontWeight: 600,
              textShadow: "0 1px 1px #fff7"
            }}>{stage.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

