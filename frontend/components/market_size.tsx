// MarketSizeCard.tsx
import React, { useEffect, useRef, useState } from "react";

/* ── STAGES ───────────────────────────────────────── */
type Size = "small" | "medium" | "big";

const STAGES: { key: Size; label: string; color: string }[] = [
  { key: "small",  label: "Small",  color: "#F14A37" }, // red
  { key: "medium", label: "Medium", color: "#F2D15F" }, // yellow
  { key: "big",    label: "Big",    color: "#BDD248" }, // green
];

const arrowBoxHeight = 24;
const arrowHeight    = 15;

/* ── COMPONENT ────────────────────────────────────── */
export const MarketSizeCard: React.FC = () => {
  const rowRef   = useRef<HTMLDivElement>(null);
  const cellRefs = useRef<HTMLDivElement[]>([]);
  const [arrowLeft, setArrowLeft] = useState(0);
  const [sizeState, setSizeState] = useState<Size | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const hasRunRef = useRef(false);

  // Fetch market metrics and classify market size
  useEffect(() => {
    // Prevent duplicate calls (React StrictMode in dev mode calls effects twice)
    if (hasRunRef.current) return;
    hasRunRef.current = true;

    const run = async () => {
      try {
        setIsLoading(true);
        setError(null);
        // Resolve backend port from public file (same pattern as IPStat/PublicationsByYear)
        const portRes = await fetch("/backend_port.txt");
        if (!portRes.ok) throw new Error(`Port file HTTP ${portRes.status}`);
        const portStr = (await portRes.text()).trim();

        // Call legal_status/ops endpoint to compute market strategy index
        const opsRes = await fetch(`http://localhost:${portStr}/api/legal_status/ops`, {
          method: 'POST'
        });
        if (!opsRes.ok) throw new Error(`Legal status OPS HTTP ${opsRes.status}`);
        const opsData = await opsRes.json();
        console.log("Legal status OPS response:", opsData);

        // Call market metrics endpoint to get total market value
        const metricsRes = await fetch(`http://localhost:${portStr}/api/market_metrics`);
        if (!metricsRes.ok) throw new Error(`Market metrics HTTP ${metricsRes.status}`);
        const metrics = await metricsRes.json();
        console.log("Market metrics response:", metrics);
        
        const mv = Number(metrics?.market_value ?? 0);
        console.log("Market value:", mv);

        // Classify by fixed thresholds:
        // < 10M = small, 10M-100M = medium, > 100M = big
        const SMALL_THRESHOLD = 10_000_000;
        const MEDIUM_THRESHOLD = 100_000_000;
        console.log("Thresholds - Small:", SMALL_THRESHOLD, "Medium:", MEDIUM_THRESHOLD);
        
        let next: Size = "medium";
        if (mv < SMALL_THRESHOLD) next = "small";
        else if (mv <= MEDIUM_THRESHOLD) next = "medium";
        else next = "big";

        setSizeState(next);
        setIsLoading(false);
      } catch (e) {
        console.error("MarketSizeCard: failed to load market metrics", e);
        setError(e instanceof Error ? e.message : "Failed to load market size data");
        setIsLoading(false);
      }
    };
    run();
  }, []);

  /* centre arrow on active stage */
  useEffect(() => {
    if (!sizeState) return;
    const idx = STAGES.findIndex((s) => s.key === sizeState);
    const el  = cellRefs.current[idx];
    if (el) {
      const { offsetLeft, offsetWidth } = el;
      setArrowLeft(offsetLeft + offsetWidth / 2);
    }
  }, [sizeState]);

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
        Market Size
      </div>

      {isLoading ? (
        /* Loading State */
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            minHeight: 120,
            padding: "20px",
          }}
        >
          <div
            style={{
              width: 40,
              height: 40,
              border: "4px solid #f3f3f3",
              borderTop: "4px solid #BDD248",
              borderRadius: "50%",
              animation: "spin 1s linear infinite",
            }}
          />
          <style>{`
            @keyframes spin {
              0% { transform: rotate(0deg); }
              100% { transform: rotate(360deg); }
            }
          `}</style>
          <div
            style={{
              marginTop: 12,
              color: "#232526",
              fontSize: 14,
              fontWeight: 500,
            }}
          >
            Loading market size data...
          </div>
        </div>
      ) : error ? (
        /* Error State */
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            minHeight: 120,
            padding: "20px",
            color: "#F14A37",
            fontSize: 14,
            fontWeight: 500,
            textAlign: "center",
          }}
        >
          <div style={{ marginBottom: 8, fontSize: 24 }}>⚠️</div>
          <div>Error: {error}</div>
        </div>
      ) : sizeState ? (
        <>
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
              style={{ position: "absolute", left: 0, top: 0, pointerEvents: "none", zIndex: 3 }}
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
                ref={(el) => { if (el) cellRefs.current[i] = el; }}
                style={{
                  background: s.color,
                  color: s.key === sizeState ? "#232526" : "#3B3C3D",
                  fontWeight: s.key === sizeState ? 800 : 500,
                  flex: 1,
                  padding: "11px 0 10px 0",
                  textAlign: "center",
                  fontSize: 15.7,
                  borderRight: i < STAGES.length - 1 ? "2px solid #fff" : "none",
                  opacity: s.key === sizeState ? 1 : 0.25,
                  filter: s.key === sizeState ? "brightness(1.1) saturate(1.35)" : "none",
                  transition: "all 0.3s",
                  wordBreak: "break-word",
                }}
              >
                {s.label}
              </div>
            ))}
          </div>

          {/* Subtitle */}
          <div
            style={{
              color: "#232526",
              fontSize: 15,
              fontWeight: 500,
              marginTop: 6,
              textAlign: "center",
            }}
          >
            Current size:&nbsp;
            <span style={{ color: "#BDD248", fontWeight: 700 }}>
              {sizeState.toUpperCase()}
            </span>
          </div>
        </>
      ) : null}
    </div>
  );
};

export default MarketSizeCard;
