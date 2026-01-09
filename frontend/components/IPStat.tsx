import React, { useEffect, useRef, useState } from "react";
import LoadingSpinner from "./LoadingSpinner";

// Styles for the value boxes and labels
const valueBoxStyle: React.CSSProperties = {
  background: "linear-gradient(180deg, #d8e79b 0%, #bdd248 100%)",
  borderRadius: "12px",
  boxShadow: "0 2px 6px #bcc84d3a",
  color: "#232526",
  fontWeight: 700,
  fontSize: 32,
  textAlign: "center",
  marginTop: 4,
  marginBottom: 16,
  padding: "14px 0 12px 0",
  border: "2.5px solid #bdd248",
  letterSpacing: "1px",
  width: "100%",
  maxWidth: 180,
};

const labelStyle: React.CSSProperties = {
  color: "#A0A4A8",
  fontWeight: 600,
  fontSize: 16,
  textAlign: "center",
  marginBottom: 0,
  letterSpacing: "0.7px",
  marginTop: 10,
};

// Define the shape of the data returned by the API
interface MarketMetrics {
  market_rate: number;
  market_value: number;
  mean_value: number;
}

type ChartProps = { width?: number; height?: number; onHoverComment?: (text: string) => void };
export const IpStatsBox: React.FC<ChartProps> = ({ width, height, onHoverComment }) => {
  const [metrics, setMetrics] = useState<MarketMetrics | null>(null);
  const [alivePatents, setAlivePatents] = useState<number | null>(null);
  const [totalFamilyMembers, setTotalFamilyMembers] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showComment, setShowComment] = useState(false);
  const [commentPosition, setCommentPosition] = useState({ x: 0, y: 0 });
  const hasRunRef = useRef(false);

  useEffect(() => {
    if (hasRunRef.current) return;
    hasRunRef.current = true;

    const run = async () => {
      try {
        const portRes = await fetch("/backend_port.txt");
        if (!portRes.ok) throw new Error(`Port file HTTP ${portRes.status}`);
        const trimmedPort = (await portRes.text()).trim();
        const baseUrl = `http://localhost:${trimmedPort}`;

        const postOk = async (path: string) => {
          const res = await fetch(`${baseUrl}${path}`, { method: "POST" });
          if (!res.ok) {
            const details = await res.text().catch(() => "");
            throw new Error(`${path} HTTP ${res.status}${details ? `: ${details}` : ""}`);
          }
        };

        await postOk("/api/market_cost");
        await postOk("/api/family_members/ops");
        await postOk("/api/legal_status/fetch_xml");
        await postOk("/api/market_strategy/load_from_legal_xml");

        const metricsRes = await fetch(`${baseUrl}/api/market_metrics?t=${Date.now()}`, {
          cache: "no-store",
        });
        if (!metricsRes.ok) throw new Error(`Metrics HTTP ${metricsRes.status}`);
        const raw: any = await metricsRes.json();

        // Extract metrics based on backend calculation
        const data: MarketMetrics = {
          market_rate: Number(raw?.market_rate ?? 0),
          market_value: Number(raw?.market_value ?? 0),
          mean_value: Number(raw?.mean_value ?? 0),
        };

        // Extract additional debug info if available
        if (raw?.alive_count !== undefined) setAlivePatents(Number(raw.alive_count));
        if (raw?.total_family_members !== undefined) setTotalFamilyMembers(Number(raw.total_family_members));

        console.log("[IPStat] backend port:", trimmedPort);
        console.log("[IPStat] /api/market_metrics response:", raw);

        setMetrics(data);
      } catch (err) {
        console.error(err);
        setError("Failed to load market cost or metrics");
      }
    };

    run();
  }, []);

  // Helper to format numbers as K/M with $ sign
  function formatMoney(value: number): string {
    if (value >= 1_000_000) {
      return `${(value / 1_000_000).toLocaleString(undefined, { maximumFractionDigits: 2, minimumFractionDigits: 2 })}M$`;
    } else if (value >= 1_000) {
      return `${(value / 1_000).toLocaleString(undefined, { maximumFractionDigits: 1, minimumFractionDigits: 1 })}K$`;
    }
    return `${value.toLocaleString()}$`;
  }

  const getInterpretation = (metrics: MarketMetrics) => {
    return `These metrics are calculated based on patent family costs estimated for 2018-2020:
    
• **IP Market Rate (${metrics.market_rate.toFixed(2)})**: Average number of family members per alive patent. 
  ${totalFamilyMembers && alivePatents ? `Calculated as ${totalFamilyMembers} family members ÷ ${alivePatents} alive patents.` : ''}

• **IP Mean Value (${formatMoney(metrics.mean_value)})**: Average cost per alive patent family.
  ${alivePatents ? `Calculated as total market value ÷ ${alivePatents} alive patents.` : ''}

• **IP Total Value (${formatMoney(metrics.market_value)})**: Sum of costs for all family members across alive patents.
  Based on patent age and jurisdiction costs.`;
  };

  const analysisText = metrics ? getInterpretation(metrics) : "";

  if (error) {
    return <div style={{ color: "red" }}>{error}</div>;
  }

  if (!metrics) {
    return <LoadingSpinner text="Loading market metrics..." />;
  }

  const { market_rate, mean_value, market_value } = metrics;

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        width: "100%",
        maxWidth: 180,
        background: "transparent",
        position: "relative"
      }}
    >
      {/* Info icon at top-right corner */}
      <div
        style={{
          position: "absolute",
          top: -5,
          right: 5,
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

          if (onHoverComment && metrics) {
            const pointText = `IP Market Metrics: Rate ${metrics.market_rate.toFixed(2)}, Mean ${formatMoney(metrics.mean_value)}, Total ${formatMoney(metrics.market_value)}`;
            const fullText = analysisText ? `${pointText}\n\n${analysisText}` : pointText;
            onHoverComment(fullText);
          }
        }}
        onMouseLeave={() => setShowComment(false)}
      >
        i
      </div>

      {/* IP Market Rate */}
      <div style={labelStyle}>IP Market Rate</div>
      <div style={valueBoxStyle}>{market_rate.toFixed(2)}</div>

      {/* IP Mean Value */}
      <div style={labelStyle}>IP Mean Value</div>
      <div style={valueBoxStyle}>{formatMoney(mean_value)}</div>

      {/* IP Total Value */}
      <div style={labelStyle}>IP Total Value</div>
      <div style={valueBoxStyle}>{formatMoney(market_value)}</div>

      {/* Comment Block (on hover) */}
      {showComment && metrics && (
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
            📊 IP Market Metrics Analysis
          </div>
          <div style={{ color: "#555" }}>
            {getInterpretation(metrics)}
          </div>
          <div style={{ 
            marginTop: "10px", 
            fontSize: "12px", 
            color: "#888",
            fontStyle: "italic",
            borderTop: "1px solid #eee",
            paddingTop: "8px"
          }}>
            💡 <strong>Backend Details:</strong> Costs are calculated using patent age and jurisdiction mapping to reference countries
          </div>
          {alivePatents !== null && (
            <div style={{ 
              marginTop: "8px", 
              fontSize: "11px", 
              color: "#999",
              fontStyle: "italic",
              backgroundColor: "#f8f9fa",
              padding: "6px",
              borderRadius: "4px"
            }}>
              <strong>Calculation Basis:</strong> Based on {alivePatents} alive patents
              {totalFamilyMembers !== null && ` with ${totalFamilyMembers} family members`}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default IpStatsBox;