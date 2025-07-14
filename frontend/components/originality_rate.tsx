

import React, { useEffect, useRef, useState } from "react";
import LoadingSpinner from "./LoadingSpinner";

// ────────────────────────────────────────────────────────────────────────────
//  STAGE DEFINITION
//  ────────────────────────────────────────────────────────────────────────────
//  Thresholds are based on common originality‑index heuristics in the IP‑analytics
//  literature (<0.4 incremental | 0.4‑0.7 emerging | ≥0.7 disruptive).
//  Colours mirror the palette used in InvestmentDynamic for consistency.
// ────────────────────────────────────────────────────────────────────────────
const STAGES = [
  {
    label: "Incremental",
    color: "#F14A37",
    check: (or: number) => or < 0.4,
  },
  {
    label: "Emerging",
    color: "#F2D15F",
    check: (or: number) => or >= 0.4 && or < 0.7,
  },
  {
    label: "Disruptive",
    color: "#BDD248",
    check: (or: number) => or >= 0.7,
  },
] as const;
const totalStages = STAGES.length;

// Arrow geometry (same dimensions as InvestmentDynamic)
const arrowBoxHeight = 24;
const arrowHeight = 17;

// Utility: returns the stage object + its index
const getStage = (rate: number) => {
  for (let i = 0; i < STAGES.length; i++) if (STAGES[i].check(rate)) return { ...STAGES[i], index: i };
  return { ...STAGES[STAGES.length - 1], index: STAGES.length - 1 };
};

export const OriginalityRate: React.FC = () => {
  // Guard so the expensive fetch logic runs only **once**, even in React 18 Strict‑mode
  const hasFetchedRef = useRef(false);

  // ────────────────────────────────────────────────────────────────────────
  //  STATE
  // ────────────────────────────────────────────────────────────────────────
  const [origRate, setOrigRate] = useState<number | null>(null);
  const [totals, setTotals] = useState<{ total: number; valid: number } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [rowWidth, setRowWidth] = useState<number | null>(null);

  // Ref + arrow position info
  const rowRef = useRef<HTMLDivElement>(null);
  const [arrowLeft, setArrowLeft] = useState<number | null>(null);

  // ────────────────────────────────────────────────────────────────────────
  //  HELPERS
  // ────────────────────────────────────────────────────────────────────────
  async function fetchPatentsLimited(port: string, required: number = 30): Promise<number> {
    // Try first with limit 70, then with limit 100 if needed
    let totalAdded = 0;
    const limits = [70, 100];

    for (const limit of limits) {
      const response = await fetch(`http://localhost:${port}/api/originality_rate/fetch?limit=${limit}`, {
        method: "POST",
      });
      const json = await response.json();

      let added = 0;
      if (Array.isArray(json.patents_added)) {
        if (typeof json.patents_added[0] === "number") {
          added = json.patents_added[0];
        } else {
          added = json.patents_added.flat().length;
        }
      }
      totalAdded += added;
      if (totalAdded >= required) break;
    }
    return totalAdded;
  }

  // ────────────────────────────────────────────────────────────────────────
  //  FETCH LOGIC — trigger /fetch then retrieve the data
  //  React‑18 StrictMode renders effects **twice** in development; the ref guard
  //  below makes sure the network requests fire only on the first pass.
  // ────────────────────────────────────────────────────────────────────────
  useEffect(() => {
    if (hasFetchedRef.current) return; // already executed once
    hasFetchedRef.current = true;

    let isMounted = true;

    setLoading(true);
    setError(null);
    setOrigRate(null);
    setTotals(null);

    fetch("/backend_port.txt")
      .then((res) => res.text())
      .then((port) => port.trim())
      .then(async (port) => {
        // 1. Fetch patents with two attempts: limit 70, then 100
        const validPatents = await fetchPatentsLimited(port, 25);
        if (!isMounted) return;
        setTotals({ valid: validPatents, total: validPatents });

        if (validPatents < 25) {
          setError("Not enough valid patents found after 2 tries (limit 70, then 100). Please try again later or check data availability.");
          setLoading(false);
          return;
        }
        // 2. Fetch the originality rate
        try {
          const res  = await fetch(`http://localhost:${port}/api/originality_rate`);
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          const json = await res.json();

          if (!isMounted) return;
          const rate = json?.originality_rate ?? (typeof json === "number" ? json : null);
          if (rate !== null) {
            setOrigRate(rate);
          } else {
            setError("Originality rate not available.");
          }
        } catch (err) {
          if (isMounted) setError("Error loading originality rate.");
        } finally {
          if (isMounted) setLoading(false); // ← always reached
        }
      });

    return () => {
      isMounted = false;
    };
  }, []);

  // Determine stage whenever the rate is known
  const stage = getStage(origRate ?? 0);

  // Compute arrow position on mount + when stage changes or row resizes
  useEffect(() => {
    if (!rowRef.current) return;
    const row = rowRef.current;

    function updateArrow() {
      if (!row) return;
      const width = row.offsetWidth;
      setRowWidth(width);
      const cellWidth = width / totalStages;
      setArrowLeft(cellWidth * stage.index + cellWidth / 2);
    }

    updateArrow(); // initial

    // Modern: ResizeObserver
    const resizeObs = new (window as any).ResizeObserver(updateArrow);
    resizeObs.observe(row);

    window.addEventListener("resize", updateArrow);
    return () => {
      resizeObs.disconnect();
      window.removeEventListener("resize", updateArrow);
    };
  }, [stage.index, origRate]);

  // ────────────────────────────────────────────────────────────────────────
  //  RENDER
  // ────────────────────────────────────────────────────────────────────────
  if (error) return <div style={{ color: "#EA3C53" }}>{error}</div>;

  if (loading) {
    return <LoadingSpinner text="Loading originality rate..." />;
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
      {/* Title bar */}
      <div
        style={{
          marginTop: 12,
          marginBottom: 0,
          padding: "8px 26px",
          fontWeight: 700,
          fontSize: 20,
          letterSpacing: "0.6px",
          background: "#6E6E72",
          color: "#fff",
          borderRadius: 10,
          boxShadow: "0 1px 8px #bdd24816",
        }}
      >
        Originality Rate
      </div>

      {/* Arrow */}
      <div
        style={{
          height: arrowBoxHeight,
          width: "100%",
          display: "flex",
          alignItems: "flex-end",
          justifyContent: "center",
          position: "relative",
        }}
      >
        {arrowLeft != null && rowWidth != null && (
          <svg
            width={rowWidth}
            height={arrowBoxHeight}
            style={{
              position: "absolute",
              left: "50%",
              top: 0,
              transform: "translateX(-50%)",
              pointerEvents: "none",
              zIndex: 10,
            }}
          >
            <polygon
              points={`
                ${arrowLeft - 8},${arrowBoxHeight - arrowHeight}
                ${arrowLeft + 8},${arrowBoxHeight - arrowHeight}
                ${arrowLeft},${arrowBoxHeight - 2}
              `}
              fill="#6E6E72"
              style={{ filter: "drop-shadow(0 2px 2px #B2DBA4AA)" }}
            />
          </svg>
        )}
      </div>

      {/* Stage boxes */}
      <div
        ref={rowRef}
        style={{
          display: "flex",
          flexDirection: "row",
          marginTop: 2,
          marginBottom: 6,
          borderRadius: 12,
          overflow: "hidden",
          boxShadow: "0 1px 6px #bbb5",
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
              padding: "11px 6px 10px 6px",
              textAlign: "center",
              fontSize: 15.5,
              borderRight: i < STAGES.length - 1 ? "2px solid #fff" : undefined,
              opacity: i === stage.index ? 1 : 0.5,
              transition: "all 0.3s",
              minWidth: 0,
            }}
          >
            {stg.label}
          </div>
        ))}
      </div>

      {/* Metrics */}
      <div style={{ color: "#232526", fontSize: 15, fontWeight: 500, textAlign: "center" }}>
        Originality rate:{" "}
        {origRate !== null ? (
          <span style={{ color: "#BDD248", fontWeight: 700 }}>{(origRate * 100).toFixed(2)}%</span>
        ) : (
          <span style={{ color: "#BDD248", fontWeight: 700 }}>N/A</span>
        )}
      </div>
      {totals && (
        <div style={{ color: "#666", fontSize: 13, marginTop: 2 }}>
          {totals.valid}/{totals.total} patents considered
        </div>
      )}
    </div>
  );
};

export default OriginalityRate;

