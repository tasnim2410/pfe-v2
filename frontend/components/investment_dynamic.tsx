import React, { useEffect, useRef, useState } from "react";
import LoadingSpinner from "./LoadingSpinner";

// Stages: Label, color, logic min/max
const STAGES = [
  {
    label: "Declining",
    color: "#F14A37",
    check: (gr: number) => gr < 0,
  },
  {
    label: "Steady",
    color: "#F2D15F",
    check: (gr: number) => gr >= 0 && gr < 10,
  },
  {
    label: "Quite Trending",
    color: "#BDD248",
    check: (gr: number) => gr >= 10 && gr < 20,
  },
  {
    label: "Trending",
    color: "#B2DBA4",
    check: (gr: number) => gr >= 20 && gr < 50,
  },
  {
    label: "Booming",
    color: "#D1EDCE",
    check: (gr: number) => gr >= 50,
  },
];
const arrowBoxHeight = 24;
const arrowHeight = 15;

function getStage(growthRate: number) {
  for (let i = 0; i < STAGES.length; i++) {
    if (STAGES[i].check(growthRate)) return { ...STAGES[i], index: i };
  }
  return { ...STAGES[STAGES.length - 1], index: STAGES.length - 1 };
}

export const InvestmentDynamic: React.FC<{ onHoverComment?: (text: string) => void }> = ({ onHoverComment }) => {
  const [growthRate, setGrowthRate] = useState<number | null>(null);
  const [years, setYears] = useState<[number, number] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showComment, setShowComment] = useState(false);
  const [commentPosition, setCommentPosition] = useState({ x: 0, y: 0 });

  const rowRef = useRef<HTMLDivElement>(null);
  const arrowRef = useRef<HTMLDivElement>(null);
  const [arrowLeft, setArrowLeft] = useState<number>(0);

  // Fetch logic (unchanged)
  useEffect(() => {
    let isMounted = true;
    fetch("/backend_port.txt")
      .then((res) => res.text())
      .then((port) => {
        if (!isMounted) return;
        return fetch(`http://localhost:${port.trim()}/api/growth_rate`)
          .then((res) => res.json())
          .then((json) => {
            if (!isMounted) return;
            if (Array.isArray(json.growth_rate)) {
              setGrowthRate(json.growth_rate[0]);
              setYears([json.growth_rate[1], json.growth_rate[2]]);
            } else {
              setError("Unexpected API response");
            }
          })
          .catch(() => {
            if (isMounted) setError("Could not fetch growth rate");
          });
      })
      .catch(() => {
        if (isMounted) setError("Could not fetch backend port");
      });
    return () => { isMounted = false; };
  }, []);

  const getInterpretation = (rate: number, stageLabel: string) => {
    if (rate < 0) {
      return "This indicates a decrease in investment activity, which may reflect market contraction, reduced investor confidence, or external economic pressures.";
    } else if (rate >= 0 && rate < 10) {
      return "This suggests stable but slow growth, typical of mature markets or industries with limited expansion opportunities.";
    } else if (rate >= 10 && rate < 20) {
      return "This shows healthy growth, indicating increasing market interest and potential expansion opportunities.";
    } else if (rate >= 20 && rate < 50) {
      return "This represents strong growth, signaling high market demand and significant investment opportunities.";
    } else {
      return "This indicates explosive growth, often seen in emerging markets, innovative sectors, or during investment booms.";
    }
  };

  if (error) return <div style={{ color: "#EA3C53" }}>{error}</div>;
  if (growthRate === null)
    return <LoadingSpinner text="Loading investment dynamic..." />;

  const stage = getStage(growthRate);

  const analysisText = (() => {
    const lines: string[] = [];
    lines.push(`Investment growth rate: ${growthRate.toFixed(2)}%`);
    if (years) lines.push(`Period: ${years[0]} - ${years[1]}`);
    lines.push(`Stage: ${stage.label}`);
    lines.push("");
    lines.push(getInterpretation(growthRate, stage.label));
    return lines.join("\n").trim();
  })();

  // Arrow position calculation
  useEffect(() => {
    if (growthRate === null) return;
    if (!rowRef.current) return;
    const stage = getStage(growthRate);

    const rowWidth = rowRef.current.offsetWidth;
    const cellWidth = rowWidth / STAGES.length;
    setArrowLeft(cellWidth * stage.index + cellWidth / 2);
  }, [growthRate]);

  return (
    <div style={{
      background: "#fff",
      borderRadius: 18,
      boxShadow: "0 2px 18px #B2DBA422",
      padding: "0 10px 13px 10px",
      minWidth: 0,
      maxWidth: "100%",
      width: "100%",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      position: "relative"
    }}>
      {/* Info icon at top-right corner of the card */}
      <div
        style={{
          position: "absolute",
          top: 5,
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

          if (onHoverComment) {
            const pointText = `Investment Dynamic: ${growthRate.toFixed(2)}% (${stage.label})`;
            const fullText = analysisText ? `${pointText}\n\n${analysisText}` : pointText;
            onHoverComment(fullText);
          }
        }}
        onMouseLeave={() => setShowComment(false)}
      >
        i
      </div>

      {/* Title */}
      <div style={{
        marginTop: 12,
        marginBottom: 0,
        padding: "8px 34px",
        fontWeight: 700,
        fontSize: 20,
        letterSpacing: "0.6px",
        background: "#232526",
        color: "#fff",
        borderRadius: 10,
        boxShadow: "0 1px 8px #bdd24816"
      }}>
        Investment Dynamic
      </div>

      {/* Arrow pointing to active stage */}
      <div style={{
        height: arrowBoxHeight,
        width: "100%",
        display: "flex",
        alignItems: "flex-end",
        justifyContent: "flex-start",
        position: "relative",
        minHeight: arrowBoxHeight,
        marginBottom: "-1px"
      }}>
        <svg
          width="100%"
          height={arrowBoxHeight}
          style={{
            position: "absolute",
            left: 0,
            top: 0,
            pointerEvents: "none",
            zIndex: 3
          }}
        >
          <polygon
            points={
              arrowLeft
                ? `
              ${arrowLeft - 10},${arrowBoxHeight - arrowHeight}
              ${arrowLeft + 10},${arrowBoxHeight - arrowHeight}
              ${arrowLeft},${arrowBoxHeight - 2}
              `
                : ""
            }
            fill="#232526"
            style={{
              filter: "drop-shadow(0 2px 2px #B2DBA4AA)"
            }}
          />
        </svg>
      </div>

      {/* Stage boxes */}
      <div
        ref={rowRef}
        style={{
          display: "flex",
          flexDirection: "row",
          marginTop: 2,
          marginBottom: 4,
          borderRadius: 12,
          overflow: "hidden",
          boxShadow: "0 1px 6px #bbb5",
          width: "100%"
        }}
      >
        {STAGES.map((stg, i) => (
          <div
            key={stg.label}
            style={{
              background: stg.color,
              color: i === stage.index ? "#232526" : "#3B3C3D",
              fontWeight: i === stage.index ? 800 : 500,
              flex: 1,
              padding: "11px 0px 10px 0px",
              textAlign: "center",
              fontSize: 15.7,
              borderRight: i < STAGES.length - 1 ? "2px solid #fff" : "none",
              opacity: i === stage.index ? 1 : 0.52,
              transition: "all 0.3s",
              position: "relative",
              zIndex: 1,
              minWidth: 0,
              maxWidth: "100%",
              wordBreak: "break-word"
            }}
          >
            {stg.label}
          </div>
        ))}
      </div>

      {/* Growth Rate and Years */}
      <div style={{
        color: "#232526",
        fontSize: 15,
        fontWeight: 500,
        marginTop: 6,
        marginBottom: 0,
        textAlign: "center",
        letterSpacing: "0.02em"
      }}>
        {`Growth rate: `}
        <span style={{ color: "#BDD248", fontWeight: 700 }}>
          {growthRate?.toFixed(2)}%
        </span>
        {years && (
          <span style={{ color: "#666", marginLeft: 8 }}>
            ({years[0]} - {years[1]})
          </span>
        )}
      </div>

      {/* Comment Block (on hover) */}
      {showComment && (
       <div
  style={{
    position: "fixed",
    left: commentPosition.x - 250,
    top: commentPosition.y,
    width: 280,
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
    📈 Growth Analysis
  </div>
  <div style={{ color: "#555" }}>
    The investment growth rate of <strong>{growthRate?.toFixed(2)}%</strong> 
    {years && ` from ${years[0]} to ${years[1]}`} places it in the{" "}
    <strong style={{ color: stage.color }}>{stage.label}</strong> category.
    <br /><br />
    {getInterpretation(growthRate, stage.label)}
  </div>
  <div style={{ 
    marginTop: "10px", 
    fontSize: "12px", 
    color: "#888",
    fontStyle: "italic",
    borderTop: "1px solid #eee",
    paddingTop: "8px"
  }}>
    💡 <strong>Tip:</strong> The arrow indicates the current growth stage
  </div>
</div>
      )}
    </div>
  );
};

export default InvestmentDynamic;