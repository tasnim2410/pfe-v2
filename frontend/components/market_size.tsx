// MarketSizeCard.tsx
import React, { useEffect, useRef, useState } from "react";
import LoadingSpinner from "./LoadingSpinner"; // Add this import

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
  const [marketValue, setMarketValue] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showComment, setShowComment] = useState(false);
  const [commentPosition, setCommentPosition] = useState({ x: 0, y: 0 });
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
        const opsRes = await fetch(`http://localhost:${portStr}/api/legal_status/fetch_xml`, {
          method: 'POST'
        });
        if (!opsRes.ok) throw new Error(`Legal status OPS HTTP ${opsRes.status}`);
        const opsData = await opsRes.json();
        console.log("Legal status OPS response:", opsData);
        
        // Call family_members/ops endpoint to update DB
        const familyRes = await fetch(`http://localhost:${portStr}/api/family_members/ops`, {
          method: 'POST'
        });
        if (!familyRes.ok) throw new Error(`Family members OPS HTTP ${familyRes.status}`);
        const familyData = await familyRes.json();
        console.log("Family members OPS response:", familyData);

        // Load market strategy from legal XML
        const loadRes = await fetch(`http://localhost:${portStr}/api/market_strategy/load_from_legal_xml`, {
          method: 'POST'
        });
        if (!loadRes.ok) throw new Error(`Market strategy load HTTP ${loadRes.status}`);
        const loadData = await loadRes.json();
        console.log("Market strategy load response:", loadData);

        // Call market metrics endpoint to get total market value
        const metricsRes = await fetch(`http://localhost:${portStr}/api/market_metrics`);
        if (!metricsRes.ok) throw new Error(`Market metrics HTTP ${metricsRes.status}`);
        const metrics = await metricsRes.json();
        console.log("Market metrics response:", metrics);
        
        const mv = Number(metrics?.market_value ?? 0);
        console.log("Market value:", mv);
        
        setMarketValue(mv);

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

  const formatMoney = (value: number): string => {
    if (value >= 1_000_000) {
      return `${(value / 1_000_000).toLocaleString(undefined, { maximumFractionDigits: 2, minimumFractionDigits: 2 })}M$`;
    } else if (value >= 1_000) {
      return `${(value / 1_000).toLocaleString(undefined, { maximumFractionDigits: 1, minimumFractionDigits: 1 })}K$`;
    }
    return `${value.toLocaleString()}$`;
  };

  const getInterpretation = (size: Size, value: number): string => {
    if (size === "small") {
      return "This indicates a niche or emerging market with limited patent investment activity. Small markets typically represent early-stage technologies, specialized applications, or domains with limited commercial interest. They may offer opportunities for early entry but come with higher uncertainty.";
    } else if (size === "medium") {
      return "This represents a maturing market with significant but not overwhelming investment. Medium-sized markets suggest growing industry interest, increasing competition, and established use cases. They often indicate technologies transitioning from research to commercialization.";
    } else {
      return "This signifies a large, established market with substantial patent investment. Big markets indicate high technological intensity, significant commercial value, and strong competitive dynamics. They typically represent mature technologies with broad applications and high market expectations.";
    }
  };

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
        position: "relative"
      }}
    >
      {/* Info icon at top-right corner */}
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
        }}
        onMouseLeave={() => setShowComment(false)}
      >
        i
      </div>

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
        /* Use LoadingSpinner component */
        <div style={{ 
          display: "flex", 
          justifyContent: "center", 
          alignItems: "center", 
          minHeight: 150,
          width: "100%"
        }}>
          <LoadingSpinner text="Loading market size data..." />
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
            {marketValue !== null && (
              <span style={{ color: "#666", marginLeft: 8 }}>
                ({formatMoney(marketValue)})
              </span>
            )}
          </div>
        </>
      ) : null}

      {/* Comment Block (on hover) */}
      {showComment && sizeState && marketValue !== null && (
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
            lineHeight: "1.5"
          }}
          onMouseEnter={() => setShowComment(true)}
          onMouseLeave={() => setShowComment(false)}
        >
          <div style={{ fontWeight: "bold", marginBottom: "8px", color: "#333", fontSize: "15px" }}>
            📈 Market Size Analysis
          </div>
          <div style={{ color: "#555" }}>
            The total patent investment of <strong>{formatMoney(marketValue)}</strong> places this technology in the{" "}
            <strong style={{ color: STAGES.find(s => s.key === sizeState)?.color }}>{sizeState.toUpperCase()}</strong> market category.
            <br /><br />
            {getInterpretation(sizeState, marketValue)}
          </div>
          <div style={{ 
            marginTop: "10px", 
            fontSize: "12px", 
            color: "#888",
            fontStyle: "italic",
            borderTop: "1px solid #eee",
            paddingTop: "8px"
          }}>
            💡 <strong>Classification Thresholds:</strong><br />
            • Small: &lt; $10M<br />
            • Medium: $10M - $100M<br />
            • Big: &gt; $100M
          </div>
          <div style={{ 
            marginTop: "8px", 
            fontSize: "11px", 
            color: "#999",
            fontStyle: "italic"
          }}>
            <strong>Note:</strong> Based on the sum of patent family costs 
          </div>
        </div>
      )}
    </div>
  );
};

export default MarketSizeCard;