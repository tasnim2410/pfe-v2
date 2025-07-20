// MarketStrategyCard.tsx
import React, { useEffect, useRef, useState } from "react";

/* ── STAGES ────────────────────────────────────────────────────────────── */
type Level = "local" | "regional" | "global";

const STAGES: { label: string; color: string; key: Level }[] = [
  { key: "local",    label: "Local",    color: "#F14A37" }, // red
  { key: "regional", label: "Regional", color: "#F2D15F" }, // yellow
  { key: "global",   label: "Global",   color: "#BDD248" }, // green
];

const arrowBoxHeight = 24;
const arrowHeight    = 15;

/* ── COMPONENT ─────────────────────────────────────────────────────────── */
interface Props {
  /** `local` | `regional` | `global` — defaults to `"regional"` */
  level?: Level;
}

export const MarketStrategyCard: React.FC<Props> = ({ level = "regional" }) => {
  /* arrow centre pos */
  const [arrowLeft, setLeft] = useState(0);

  /* refs to measure exact cell width */
  const rowRef   = useRef<HTMLDivElement>(null);
  const cellRefs = useRef<HTMLDivElement[]>([]);

  /* calculate arrow whenever level or layout changes */
  useEffect(() => {
    const idx = STAGES.findIndex((s) => s.key === level);
    const el  = cellRefs.current[idx];
    if (el && rowRef.current) {
      const { offsetLeft, offsetWidth } = el;
      setLeft(offsetLeft + offsetWidth / 2);
    }
  }, [level]);

  return (
    <div
      style={{
        background: "#fff",
        borderRadius: 18,
        boxShadow: "0 2px 18px #B2DBA422",
        padding: "0 10px 13px 10px",
        width: "100%",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
      }}
    >
      {/* Title */}
      <div
        style={{
          marginTop: 12,
          padding: "8px 34px",
          fontWeight: 700,
          fontSize: 20,
          letterSpacing: "0.6px",
          background: "#232526",
          color: "#fff",
          borderRadius: 10,
          boxShadow: "0 1px 8px #bdd24816",
        }}
      >
        Market Strategy
      </div>

      {/* Arrow */}
      <div
        style={{
          height: arrowBoxHeight,
          width: "100%",
          position: "relative",
          display: "flex",
          alignItems: "flex-end",
          marginBottom: "-1px",
        }}
      >
        <svg
          width="100%"
          height={arrowBoxHeight}
          style={{
            position: "absolute",
            left: 0,
            top: 0,
            pointerEvents: "none",
            zIndex: 3,
          }}
        >
          <polygon
            points={`${arrowLeft - 10},${arrowBoxHeight - arrowHeight}
                     ${arrowLeft + 10},${arrowBoxHeight - arrowHeight}
                     ${arrowLeft},${arrowBoxHeight - 2}`}
            fill="#232526"
            style={{ filter: "drop-shadow(0 2px 2px #B2DBA4AA)" }}
          />
        </svg>
      </div>

      {/* Stage boxes */}
      <div
        ref={rowRef}
        style={{
          display: "flex",
          marginTop: 2,
          marginBottom: 4,
          borderRadius: 12,
          overflow: "hidden",
          boxShadow: "0 1px 6px #bbb5",
          width: "100%",
        }}
      >
        {STAGES.map((s, i) => (
          <div
            key={s.key}
            ref={(el) => {
              if (el) cellRefs.current[i] = el;
            }}
            style={{
              background: s.color,
              color: s.key === level ? "#232526" : "#3B3C3D",
              fontWeight: s.key === level ? 800 : 500,
              flex: 1,
              padding: "11px 0 10px 0",
              textAlign: "center",
              fontSize: 15.7,
              borderRight: i < STAGES.length - 1 ? "2px solid #fff" : "none",
              opacity: s.key === level ? 1 : 0.25,
              filter:
                s.key === level ? "brightness(1.1) saturate(1.35)" : "none",
              transition: "all 0.3s",
              wordBreak: "break-word",
            }}
          >
            {s.label}
          </div>
        ))}
      </div>

      {/* Subtitle (static) */}
      <div
        style={{
          color: "#232526",
          fontSize: 15,
          fontWeight: 500,
          marginTop: 6,
          textAlign: "center",
        }}
      >
        Current level:&nbsp;
        <span style={{ color: "#BDD248", fontWeight: 700 }}>
          {level.toUpperCase()}
        </span>
      </div>
    </div>
  );
};

export default MarketStrategyCard;
