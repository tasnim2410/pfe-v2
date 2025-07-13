// import React from "react";

// const valueBoxStyle: React.CSSProperties = {
//   background: "linear-gradient(180deg, #d8e79b 0%, #bdd248 100%)",
//   borderRadius: "12px",
//   boxShadow: "0 2px 6px #bcc84d3a",
//   color: "#232526",
//   fontWeight: 700,
//   fontSize: 32,
//   textAlign: "center",
//   marginTop: 4,
//   marginBottom: 16,
//   padding: "14px 0 12px 0",
//   border: "2.5px solid #bdd248",
//   letterSpacing: "1px",
//   width: "100%",
//   maxWidth: 180,
// };

// const labelStyle: React.CSSProperties = {
//   color: "#A0A4A8",
//   fontWeight: 600,
//   fontSize: 16,
//   textAlign: "center",
//   marginBottom: 0,
//   letterSpacing: "0.7px",
//   marginTop: 10,
// };

// export const IpStatsBox: React.FC = () => (
//   <div style={{
//     display: "flex",
//     flexDirection: "column",
//     alignItems: "center",
//     width: "100%",
//     maxWidth: 180,
//     background: "transparent"
//   }}>
//     {/* IP Market Rate */}
//     <div style={labelStyle}>IP Market Rate</div>
//     <div style={valueBoxStyle}>2,19</div>
//     {/* IP Mean Value */}
//     <div style={labelStyle}>IP Mean Value</div>
//     <div style={valueBoxStyle}>24,5K$</div>
//     {/* IP Total Value */}
//     <div style={labelStyle}>IP Total Value</div>
//     <div style={valueBoxStyle}>12,22M$</div>
//   </div>
// );

// export default IpStatsBox;





import React, { useEffect, useState } from "react";
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

export const IpStatsBox: React.FC = () => {
  const [metrics, setMetrics] = useState<MarketMetrics | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;
    // First, fetch the backend port dynamically
    fetch("/backend_port.txt")
      .then((res) => {
        if (!res.ok) throw new Error(`Port file HTTP ${res.status}`);
        return res.text();
      })
      .then((port) => {
        if (!isMounted) return;
        const trimmedPort = port.trim();
        // First, call /api/market_cost to update DB
        return fetch(`http://localhost:${trimmedPort}/api/market_cost`, { method: 'POST' })
          .then((costRes) => {
            if (!costRes.ok) throw new Error(`Market cost HTTP ${costRes.status}`);
            // Then call /api/family_members/ops to update DB
            return fetch(`http://localhost:${trimmedPort}/api/family_members/ops`, { method: 'POST' });
          })
          .then((familyRes) => {
            if (!familyRes || !familyRes.ok) throw new Error(`Family members HTTP ${familyRes?.status}`);
            // After both DB updates, fetch metrics
            return fetch(`http://localhost:${trimmedPort}/api/market_metrics`)
              .then((metricsRes) => {
                if (!metricsRes.ok) throw new Error(`Metrics HTTP ${metricsRes.status}`);
                return metricsRes.json();
              })
              .then((data: MarketMetrics) => {
                if (isMounted) setMetrics(data);
              });
          });
      })
      .catch((err) => {
        console.error(err);
        if (isMounted) setError("Failed to load market cost or metrics");
      });
    return () => {
      isMounted = false;
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

