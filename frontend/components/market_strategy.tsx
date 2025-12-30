// MarketStrategyCard.tsx
import React, { useEffect, useRef, useState } from "react";

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
  const [msiValue, setMsiValue] = useState<number | null>(null); // Added state for MSI value
  
  /* arrow centre pos */
  const [arrowLeft, setLeft] = useState(0);

  /* refs to measure exact cell width */
  const rowRef   = useRef<HTMLDivElement>(null);
  const cellRefs = useRef<HTMLDivElement[]>([]);
  
  /* Refs to prevent double fetching and abort ongoing requests */
  const hasFetchedRef = useRef(false);
  const abortControllerRef = useRef<AbortController | null>(null);

  /* Fetch data from APIs */
  useEffect(() => {
    // Prevent double calls in Strict Mode
    if (hasFetchedRef.current) {
      return;
    }
    
    // Cancel any existing request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    
    // Create new AbortController
    abortControllerRef.current = new AbortController();
    const signal = abortControllerRef.current.signal;
    
    const fetchData = async () => {
      try {
        hasFetchedRef.current = true;
        setLoading(true);
        setError(null);

        // Resolve backend port
        let portStr: string | undefined = port ? String(port) : undefined;
        if (!portStr) {
          try {
            const portResponse = await fetch("/backend_port.txt", { signal });
            if (signal.aborted) return;
            portStr = (await portResponse.text()).trim();
          } catch (e) {
            if (signal.aborted) return;
            throw new Error("Failed to read backend port");
          }
        }

        // First: fetch/update legal XML
        const opsResponse = await fetch(
          `http://localhost:${portStr}/api/legal_status/fetch_xml`,
          { signal }
        );
        if (signal.aborted) return;
        
        if (!opsResponse.ok) {
          throw new Error(`Legal status request failed (${opsResponse.status})`);
        }
        const _opsData = await opsResponse.json();

        // Second: load patent statuses into market_strategy table
        const loadResponse = await fetch(
          `http://localhost:${portStr}/api/market_strategy/load_from_legal_xml`,
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({}),
            signal
          }
        );
        if (signal.aborted) return;
        
        if (!loadResponse.ok) {
          throw new Error(`Market strategy load request failed (${loadResponse.status})`);
        }
        const loadData = await loadResponse.json();
        
        // Expected format: { success, total, inserted, updated }
        if (!loadData || loadData.success !== true) {
          throw new Error('Market strategy load did not complete successfully');
        }

        // Third: compute MSI and get summary
        const summaryResponse = await fetch(
          `http://localhost:${portStr}/api/market_strategy/compute_market_strategy`,
          { signal }
        );
        if (signal.aborted) return;
        
        if (!summaryResponse.ok) {
          throw new Error(`Market strategy summary request failed (${summaryResponse.status})`);
        }
        const summaryData = await summaryResponse.json();

        // Determine level based on MSI and store value
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
        // Don't set error if request was aborted
        if (signal.aborted) return;
        
        // Reset the fetch flag on error so it can retry
        hasFetchedRef.current = false;
        
        setError(err instanceof Error ? err.message : 'An unknown error occurred');
      } finally {
        // Only update loading state if not aborted
        if (!signal.aborted) {
          setLoading(false);
        }
      }
    };

    fetchData();

    // Cleanup function
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, [port]);

  /* calculate arrow whenever level or layout changes */
  useEffect(() => {
    const idx = STAGES.findIndex((s) => s.key === level);
    const el  = cellRefs.current[idx];
    if (el && rowRef.current) {
      const { offsetLeft, offsetWidth } = el;
      setLeft(offsetLeft + offsetWidth / 2);
    }
  }, [level]);

  if (loading) {
    return (
      <div style={{
        background: "#fff",
        borderRadius: 18,
        boxShadow: "0 2px 18px #B2DBA422",
        padding: "20px",
        textAlign: "center"
      }}>
        Loading market data...
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
    </div>
  );
};

export default MarketStrategyCard;