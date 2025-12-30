



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

type ChartProps = { width?: number; height?: number };
export const IpStatsBox: React.FC<ChartProps> = ({ width, height }) => {
  const [metrics, setMetrics] = useState<MarketMetrics | null>(null);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    if (abortRef.current) abortRef.current.abort();
    abortRef.current = new AbortController();
    const signal = abortRef.current.signal;

    const run = async () => {
      try {
        const portRes = await fetch("/backend_port.txt", { signal });
        if (!portRes.ok) throw new Error(`Port file HTTP ${portRes.status}`);
        const trimmedPort = (await portRes.text()).trim();
        const baseUrl = `http://localhost:${trimmedPort}`;

        const postOk = async (path: string) => {
          const res = await fetch(`${baseUrl}${path}`, { method: "POST", signal });
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
          signal,
        });
        if (!metricsRes.ok) throw new Error(`Metrics HTTP ${metricsRes.status}`);
        const raw: MarketMetrics = await metricsRes.json();
        const data: MarketMetrics = {
          market_rate: Number((raw as any)?.market_rate ?? 0),
          market_value: Number((raw as any)?.market_value ?? 0),
          mean_value: Number((raw as any)?.mean_value ?? 0),
        };

        console.log("[IPStat] backend port:", trimmedPort);
        console.log("[IPStat] /api/market_metrics response:", raw, "normalized:", data);

        setMetrics(data);
      } catch (err) {
        if (signal.aborted) return;
        console.error(err);
        setError("Failed to load market cost or metrics");
      }
    };

    run();
    return () => {
      if (abortRef.current) abortRef.current.abort();
    };
  }, []);

  if (error) {
    return <div style={{ color: "red" }}>{error}</div>;
  }

  if (!metrics) {
    return <LoadingSpinner text="Loading market metrics..." />;
  }

  const { market_rate, mean_value, market_value } = metrics;

  // Helper to format numbers as K/M with $ sign
  function formatMoney(value: number): string {
    if (value >= 1_000_000) {
      return `${(value / 1_000_000).toLocaleString(undefined, { maximumFractionDigits: 2, minimumFractionDigits: 2 })}M$`;
    } else if (value >= 1_000) {
      return `${(value / 1_000).toLocaleString(undefined, { maximumFractionDigits: 1, minimumFractionDigits: 1 })}K$`;
    }
    return `${value.toLocaleString()}$`;
  }

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        width: "100%",
        maxWidth: 180,
        background: "transparent",
      }}
    >
      {/* IP Market Rate */}
      <div style={labelStyle}>IP Market Rate</div>
      <div style={valueBoxStyle}>{market_rate.toFixed(2)}</div>

      {/* IP Mean Value */}
      <div style={labelStyle}>IP Mean Value</div>
      <div style={valueBoxStyle}>{formatMoney(mean_value)}</div>

      {/* IP Total Value */}
      <div style={labelStyle}>IP Total Value</div>
      <div style={valueBoxStyle}>{formatMoney(market_value)}</div>
    </div>
  );
};

export default IpStatsBox;

