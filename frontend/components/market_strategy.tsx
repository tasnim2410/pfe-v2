// MarketStrategyCard.tsx
import React, { useEffect, useRef, useState } from "react";
import LoadingSpinner from "./LoadingSpinner";

/* ── STAGES ────────────────────────────────────────────────────────────── */
type Level = "local" | "main markets" | "global";

const STAGES: { label: string; color: string; key: Level }[] = [
  { key: "local",    label: "Local",    color: "#F14A37" }, // red
  { key: "main markets", label: "Main Markets", color: "#F2D15F" }, // yellow
  { key: "global",   label: "Global",   color: "#BDD248" }, // green
];

const arrowBoxHeight = 24;
const arrowHeight    = 15;

/* ── COMPONENT ─────────────────────────────────────────────────────────── */
interface Props {
  port?: number;
}

export const MarketStrategyCard: React.FC<Props> = ({ port }) => {
  const [level, setLevel] = useState<Level>("main markets");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [msiValue, setMsiValue] = useState<number | null>(null);
  const [showComment, setShowComment] = useState(false);
  const [commentPosition, setCommentPosition] = useState({ x: 0, y: 0 });
  
  /* arrow centre pos */
  const [arrowLeft, setArrowLeft] = useState(0);
  const [arrowContainerWidth, setArrowContainerWidth] = useState(0);

  /* refs to measure exact cell width */
  const rowRef   = useRef<HTMLDivElement>(null);
  const cellRefs = useRef<HTMLDivElement[]>([]);
  const arrowContainerRef = useRef<HTMLDivElement>(null);
  const hasRunRef = useRef(false);
  const lastPortRef = useRef<number | undefined>(undefined);

  /* Fetch data from APIs */
  useEffect(() => {
    if (hasRunRef.current && lastPortRef.current === port) return;
    hasRunRef.current = true;
    lastPortRef.current = port;
    
    const fetchData = async () => {
      try {
        setLoading(true);
        setError(null);

        let portStr: string | undefined = port ? String(port) : undefined;
        if (!portStr) {
          try {
            const portResponse = await fetch("/backend_port.txt");
            portStr = (await portResponse.text()).trim();
          } catch (e) {
            throw new Error("Failed to read backend port");
          }
        }

        // First: fetch/update legal XML
        const opsResponse = await fetch(`http://localhost:${portStr}/api/legal_status/fetch_xml`, {
          method: "POST",
        });
        
        if (!opsResponse.ok) {
          throw new Error(`Legal status request failed (${opsResponse.status})`);
        }
        
        const familyResponse = await fetch(`http://localhost:${portStr}/api/family_members/ops`, {
          method: "POST",
        });
        
        if (!familyResponse.ok) {
          throw new Error(`Family members request failed (${familyResponse.status})`);
        }
        
        const _opsData = await opsResponse.json();

        // Second: load patent statuses into market_strategy table
        const loadResponse = await fetch(
          `http://localhost:${portStr}/api/market_strategy/load_from_legal_xml`,
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({}),
          }
        );
        
        if (!loadResponse.ok) {
          throw new Error(`Market strategy load request failed (${loadResponse.status})`);
        }
        const loadData = await loadResponse.json();
        
        if (!loadData || loadData.success !== true) {
          throw new Error('Market strategy load did not complete successfully');
        }

        // Third: compute MSI and get summary
        const summaryResponse = await fetch(
          `http://localhost:${portStr}/api/market_strategy/compute_market_strategy`
        );
        
        if (!summaryResponse.ok) {
          throw new Error(`Market strategy summary request failed (${summaryResponse.status})`);
        }
        const summaryData = await summaryResponse.json();

        const avgMsi = Number(
          summaryData?.technology_msi ?? summaryData?.avg_msi ?? 0
        );
        setMsiValue(avgMsi);
        
        if (avgMsi < 0.6) {
          setLevel("local");
        } else if (avgMsi >= 0.9) {
          setLevel("global");
        } else {
          setLevel("main markets");
        }
        
      } catch (err) {
        setError(err instanceof Error ? err.message : 'An unknown error occurred');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [port]);

  /* calculate arrow whenever level or layout changes */
  useEffect(() => {
    const calculateArrowPosition = () => {
      if (!arrowContainerRef.current || !rowRef.current) return;
      
      const containerWidth = arrowContainerRef.current.offsetWidth;
      setArrowContainerWidth(containerWidth);
      
      const idx = STAGES.findIndex((s) => s.key === level);
      const el = cellRefs.current[idx];
      
      if (el && rowRef.current) {
        // Calculate position relative to the container
        const stageRect = el.getBoundingClientRect();
        const containerRect = arrowContainerRef.current.getBoundingClientRect();
        
        // Calculate center of the stage relative to container
        const stageCenter = stageRect.left - containerRect.left + (stageRect.width / 2);
        setArrowLeft(stageCenter);
      }
    };

    // Calculate initially
    calculateArrowPosition();
    
    // Recalculate on window resize
    window.addEventListener('resize', calculateArrowPosition);
    
    return () => {
      window.removeEventListener('resize', calculateArrowPosition);
    };
  }, [level]);

  const getInterpretation = (msi: number, level: Level): string => {
    if (level === "local") {
      return `The MSI of ${msi.toFixed(2)} indicates a LOCAL market strategy. This suggests the technology is protected primarily in specific countries or regions rather than globally. This could indicate:
• Early-stage technology where market testing is ongoing
• Niche applications with limited geographic relevance
• Regulatory constraints limiting global protection
• Cost-conscious IP strategy focusing on key markets
• Technology with regional market preferences or standards`;
    } else if (level === "main markets") {
      return `The MSI of ${msi.toFixed(2)} indicates a MAIN MARKETS strategy. This suggests the technology is protected in key economic regions (typically US, EU, JP, CN). This often indicates:
• Established technology with proven commercial value
• Strategic focus on high-GDP markets with strong IP enforcement
• Balanced approach between protection breadth and cost
• Technology relevant to major industrial economies
• Companies targeting leading markets while managing IP costs`;
    } else {
      return `The MSI of ${msi.toFixed(2)} indicates a GLOBAL market strategy. This suggests comprehensive worldwide patent protection. This typically indicates:
• Breakthrough or foundational technology
• Pharmaceutical or medical device inventions requiring global protection
• Technologies with universal applications across all markets
• Companies establishing dominant market positions
• High-value inventions justifying global IP investment`;
    }
  };

  const getStrategicImplications = (msi: number, level: Level): string => {
    if (level === "local") {
      return "Implications: Lower competitive barriers globally, potential for market entry by competitors in unprotected regions, focus on specific regulatory environments.";
    } else if (level === "main markets") {
      return "Implications: Strong protection in key economic zones, moderate barriers to entry in primary markets, opportunity for regional licensing strategies.";
    } else {
      return "Implications: High barriers to entry globally, strong market exclusivity, potential for broad licensing revenue, significant competitive advantage.";
    }
  };

  if (loading) {
    return (
      <div style={{
        background: "#fff",
        borderRadius: 18,
        boxShadow: "0 2px 18px #B2DBA422",
        padding: "20px",
        textAlign: "center"
      }}>
        <LoadingSpinner text="Loading market data..." height={120} />
      </div>
    );
  }

  if (error) {
    return (
      <div style={{
        background: "#fff",
        borderRadius: 18,
        boxShadow: "0 2px 18px #B2DBA422",
        padding: "20px",
        textAlign: "center",
        color: "#F14A37"
      }}>
        Error: {error}
      </div>
    );
  }

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
          setCommentPosition({ 
            x: Math.min(rect.right, window.innerWidth - 300), 
            y: rect.bottom + 5 
          });
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
        Market Strategy
      </div>

      {/* Arrow container - FIXED POSITIONING */}
      <div
        ref={arrowContainerRef}
        style={{
          height: arrowBoxHeight,
          width: "100%",
          position: "relative",
          display: "flex",
          alignItems: "flex-end",
          marginBottom: "-1px",
          overflow: "visible"
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
            overflow: "visible"
          }}
        >
          {/* Arrow with precise positioning */}
          <polygon
            points={`${arrowLeft - 10},${arrowBoxHeight - arrowHeight}
                     ${arrowLeft + 10},${arrowBoxHeight - arrowHeight}
                     ${arrowLeft},${arrowBoxHeight - 2}`}
            fill="#232526"
            style={{ 
              filter: "drop-shadow(0 2px 2px #B2DBA4AA)",
              transition: "all 0.3s ease"
            }}
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
          position: "relative"
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
              position: "relative"
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

      {/* Display MSI Value */}
      <div
        style={{
          marginTop: 8,
          padding: "8px 16px",
          background: "#F5F7FA",
          borderRadius: 8,
          fontSize: 14,
          fontWeight: 600,
          color: "#232526",
          textAlign: "center",
          width: "100%",
          boxSizing: "border-box",
        }}
      >
        Market Strategy Index:&nbsp;
        <span style={{ color: "#BDD248", fontWeight: 800 }}>
          {msiValue !== null ? msiValue.toFixed(2) : "N/A"}
        </span>
      </div>

      {/* Comment Block (on hover) */}
      {showComment && msiValue !== null && level && (
        <div
          style={{
            position: "fixed",
            left: commentPosition.x,
            top: commentPosition.y,
            width: 320,
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
          <div style={{ 
            fontWeight: "bold", 
            marginBottom: "8px", 
            color: "#333", 
            fontSize: "15px",
            display: "flex",
            alignItems: "center",
            gap: "8px"
          }}>
            <div style={{
              width: "12px",
              height: "12px",
              borderRadius: "50%",
              backgroundColor: STAGES.find(s => s.key === level)?.color
            }}></div>
            Market Strategy Analysis
          </div>
          
          <div style={{ 
            color: "#555",
            marginBottom: "12px",
            paddingBottom: "12px",
            borderBottom: "1px solid #eee"
          }}>
            <strong>Market Strategy Index:</strong> {msiValue.toFixed(2)}<br/>
            <strong>Classification:</strong> {level.toUpperCase()}<br/><br/>
            
            {getInterpretation(msiValue, level)}
          </div>
          
          <div style={{ 
            color: "#555",
            marginBottom: "12px",
            paddingBottom: "12px",
            borderBottom: "1px solid #eee"
          }}>
            <strong>📈 Strategic Implications:</strong><br/>
            {getStrategicImplications(msiValue, level)}
          </div>
          
          <div style={{ 
            marginBottom: "8px", 
            fontSize: "12px", 
            color: "#666",
            fontStyle: "italic"
          }}>
            💡 <strong>MSI Calculation Methodology:</strong><br />
            • Sum of GDP of countries protected by patent family<br />
            • 40% reduction for pending patent applications<br />
            • Normalized to 1.0 for US-only granted patent<br />
            • Considers both alive and dead granted patents
          </div>
          
          <div style={{ 
            fontSize: "11px", 
            color: "#777",
            backgroundColor: "#f9f9f9",
            padding: "8px",
            borderRadius: "4px"
          }}>
            <strong>Classification Thresholds:</strong><br />
            • <span style={{ color: "#F14A37" }}>LOCAL:</span> MSI &lt; 0.6<br />
            • <span style={{ color: "#F2D15F" }}>MAIN MARKETS:</span> 0.6 ≤ MSI &lt; 0.9<br />
            • <span style={{ color: "#BDD248" }}>GLOBAL:</span> MSI ≥ 0.9
          </div>
          
          <div style={{ 
            marginTop: "8px", 
            fontSize: "11px", 
            color: "#888",
            textAlign: "center"
          }}>
            Based on Innosabi Insight methodology
          </div>
        </div>
      )}
    </div>
  );
};

export default MarketStrategyCard; 