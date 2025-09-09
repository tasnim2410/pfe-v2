
import React, { useEffect, useRef, useState } from "react";
import LoadingSpinner from "./LoadingSpinner";

// Stages: Label, color, logic for originality rate (0-1 scale converted to percentage)
const STAGES = [
  {
    label: "Incremental",
    color: "#F14A37", // Red
    check: (rate: number) => rate < 0.4,
  },
  {
    label: "Emerging",
    color: "#F2D15F", // Yellow
    check: (rate: number) => rate >= 0.4 && rate < 0.7,
  },
  {
    label: "Disruptive",
    color: "#BDD248", // Green
    check: (rate: number) => rate >= 0.7,
  },
];

const arrowBoxHeight = 24;
const arrowHeight = 15;

function getStage(originalityRate: number) {
  for (let i = 0; i < STAGES.length; i++) {
    if (STAGES[i].check(originalityRate)) return { ...STAGES[i], index: i };
  }
  return { ...STAGES[STAGES.length - 1], index: STAGES.length - 1 };
}

type ChartProps = { width?: number; height?: number };
export const OriginalityDynamic: React.FC<ChartProps> = ({ width, height }) => {
  const [originalityRate, setOriginalityRate] = useState<number | null>(null);
  const [totalPatents, setTotalPatents] = useState<number | null>(null);
  const [validPatents, setValidPatents] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [loadingMessage, setLoadingMessage] = useState<string>("Fetching backend port...");

  const rowRef = useRef<HTMLDivElement>(null);
  const [arrowLeft, setArrowLeft] = useState<number>(0);

  const cellRefs = useRef<HTMLDivElement[]>([]);

  // Fetch patents and then originality rate
  useEffect(() => {
    let isMounted = true;
    
    const fetchData = async () => {
      try {
        setLoadingMessage("Fetching backend port...");
        
        // Get backend port
        const portResponse = await fetch("/backend_port.txt");
        const port = await portResponse.text();
        const baseUrl = `http://localhost:${port.trim()}`;
        
        if (!isMounted) return;
        
        // First, try to fetch patents with limit 30
        setLoadingMessage("Fetching patents (limit: 30)...");
        let fetchResponse = await fetch(`${baseUrl}/api/originality_rate/fetch?limit=30`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
        });
        let fetchData = await fetchResponse.json();
        
        if (!isMounted) return;
        
        let processedPatents = fetchData.processed_patents || 0;
        let addedPatents = Array.isArray(fetchData.patents_added) && Array.isArray(fetchData.patents_added[1]) 
          ? fetchData.patents_added[1].length 
          : 0;
        let totalFetched = processedPatents + addedPatents;
        
        // If we don't have enough patents, try with limit 50
        if (totalFetched < 25) {
          setLoadingMessage("Need more patents, fetching with limit: 50...");
          fetchResponse = await fetch(`${baseUrl}/api/originality_rate/fetch?limit=50`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
          });
          fetchData = await fetchResponse.json();
          
          if (!isMounted) return;
          
          processedPatents = fetchData.processed_patents || 0;
          addedPatents = Array.isArray(fetchData.patents_added) && Array.isArray(fetchData.patents_added[1]) 
            ? fetchData.patents_added[1].length 
            : 0;
          totalFetched = processedPatents + addedPatents;
        }
        
        if (totalFetched < 25) {
          setError(`Not enough patents available (${totalFetched}/25 required)`);
          setIsLoading(false);
          return;
        }
        
        // Now fetch the originality rate
        setLoadingMessage("Calculating originality rate...");
        const rateResponse = await fetch(`${baseUrl}/api/originality_rate`);
        const rateData = await rateResponse.json();
        
        if (!isMounted) return;
        
        if (typeof rateData.originality_rate === 'number') {
          setOriginalityRate(rateData.originality_rate);
          setTotalPatents(rateData.total_patents);
          setValidPatents(rateData.valid_patents);
        } else {
          setError("Invalid originality rate response");
        }
        
        setIsLoading(false);
        
      } catch (err) {
        if (isMounted) {
          setError("Failed to fetch data");
          setIsLoading(false);
        }
      }
    };
    
    fetchData();
    
    return () => { isMounted = false; };
  }, []);

  // Arrow position calculation
  useEffect(() => {
    if (originalityRate === null) return;
    const activeCell = cellRefs.current[getStage(originalityRate).index];
    if (!activeCell) return;
  
    // left-edge of cell + half its width  (= its mid-point)
    const { offsetLeft, offsetWidth } = activeCell;
    setArrowLeft(offsetLeft + offsetWidth / 2);
  }, [originalityRate]);

  if (error) return <div style={{ color: "#EA3C53" }}>{error}</div>;
  if (isLoading) return <LoadingSpinner text={loadingMessage} />;
  if (originalityRate === null) return <LoadingSpinner text="Loading originality dynamic..." />;

  const stage = getStage(originalityRate);

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
      alignItems: "center"
    }}>
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
        Originality Dynamic
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
          ref={el => { if (el) cellRefs.current[i] = el; }}  
            key={stg.label}
            style={{
              background: stg.color,
              color: i === stage.index ? "#232526" : "#3B3C3D",
              fontWeight: i === stage.index ? 800 : 500,
              flex: 1,
              padding: "11px 3px 10px 3px",
              textAlign: "center",
              fontSize: 15.7,
              borderRight: i < STAGES.length - 1 ? "2px solid #fff" : "none",
              opacity: i === stage.index ? 1 : 0.25,
              transition: "all 0.3s",
              position: "relative",
              zIndex: 1,
              minWidth: 0,
              maxWidth: "100%",
              filter: i === stage.index ? "brightness(1.05) saturate(1.25)" : "none",
              wordBreak: "break-word"
            }}
          >
            {stg.label}
          </div>
        ))}
      </div>
      
      {/* Originality Rate and Patent Info */}
      <div style={{
        color: "#232526",
        fontSize: 15,
        fontWeight: 500,
        marginTop: 6,
        marginBottom: 0,
        textAlign: "center",
        letterSpacing: "0.02em"
      }}>
        {`Originality rate: `}
        <span style={{ color: "#BDD248", fontWeight: 700 }}>
          {(originalityRate * 100).toFixed(2)}%
        </span>
        {totalPatents && validPatents && (
          <span style={{ color: "#666", marginLeft: 8 }}>
            ({validPatents}/{totalPatents} patents)
          </span>
        )}
      </div>
    </div>
  );
};

export default OriginalityDynamic;