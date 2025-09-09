// SearchHistoryChart.tsx
import React, { useState, useEffect, useMemo } from "react";

interface Keyword {
  field: string;
  keyword: string;
}

interface SearchHistoryItem {
  keywords: Keyword[];
  search_id: string;
  total_results: number;
}

interface SearchHistoryResponse {
  history: SearchHistoryItem[];
  limit: number;
}

interface Props {
  port?: number; // optional override; otherwise read from /backend_port.txt
}

// Small helper: ensure positive integer >= 1
const toPositiveInt = (v: unknown, fallback: number): number => {
  const n = typeof v === "number" ? v : parseInt(String(v), 10);
  return Number.isFinite(n) && n >= 1 ? n : fallback;
};

export const SearchHistoryChart: React.FC<Props> = ({ port }) => {
  const [history, setHistory] = useState<SearchHistoryItem[]>([]);
  const [limit, setLimit] = useState<number>(5);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [backendPort, setBackendPort] = useState<string | null>(null);

  // Preset choices shown in the dropdown
  const presetLimits = useMemo(() => [3, 5, 10, 15], []);
  // Track whether user explicitly chose the custom option in the dropdown
  const [isCustom, setIsCustom] = useState<boolean>(false);

  // Load backend port from public file unless an explicit port prop is provided
  useEffect(() => {
    if (port != null) return; // explicit override
    let mounted = true;
    fetch("/backend_port.txt")
      .then(res => res.text())
      .then(p => {
        if (!mounted) return;
        setBackendPort(p.trim());
      })
      .catch(() => {
        if (!mounted) return;
        setBackendPort(null);
        setError("Backend port not loaded");
        setLoading(false);
      });
    return () => { mounted = false; };
  }, [port]);

  // Fetch search history data
  const fetchSearchHistory = async () => {
    try {
      setLoading(true);
      setError(null);
      const targetPort = port ?? backendPort;
      if (!targetPort) {
        setError("Backend port not loaded");
        setLoading(false);
        return;
      }
      const response = await fetch(`http://localhost:${targetPort}/api/search_ops/history`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ limit }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data: SearchHistoryResponse = await response.json();
      setHistory(data.history);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An unknown error occurred");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const ready = port != null || backendPort != null;
    if (!ready) return;
    fetchSearchHistory();
  }, [limit, port, backendPort]);

  // Format number with commas
  const formatNumber = (num: number): string => {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  };

  // Get keywords as a string
  const getKeywordsString = (keywords: Keyword[]): string => {
    return keywords.map(k => k.keyword).join(", ");
  };

  // Find the maximum total_results for scaling the chart
  const maxResults = history.length > 0 
    ? Math.max(...history.map(item => item.total_results)) 
    : 0;

  if (loading) {
    return (
      <div style={{ 
        padding: "20px", 
        textAlign: "center",
        background: "#f5f5f5",
        borderRadius: "8px"
      }}>
        Loading search history...
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ 
        padding: "20px", 
        textAlign: "center",
        background: "#ffebee",
        borderRadius: "8px",
        color: "#c62828"
      }}>
        Error: {error}
      </div>
    );
  }

  return (
    <div style={{
      background: "#fff",
      borderRadius: "12px",
      boxShadow: "0 4px 12px rgba(0, 0, 0, 0.1)",
      padding: "20px",
      width: "100%",
      maxWidth: "800px",
      margin: "0 auto"
    }}>
      <h2 style={{ 
        marginTop: 0, 
        marginBottom: "20px",
        color: "#232526",
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center"
      }}>
        Search History
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <label htmlFor="limit-select" style={{ fontSize: "14px" }}>
            Show last:
          </label>

          {/* Dropdown with presets + "Custom…" */}
          <select
            id="limit-select"
            value={isCustom ? "custom" : limit}
            onChange={(e) => {
              const v = e.target.value;
              if (v === "custom") {
                setIsCustom(true); // enable custom input
                return;
              }
              setIsCustom(false);
              setLimit(toPositiveInt(v, 5)); // set to selected preset
            }}
            style={{
              padding: "5px 10px",
              borderRadius: "4px",
              border: "1px solid #ddd",
              background: "#fff"
            }}
          >
            {presetLimits.map((n) => (
              <option key={n} value={n}>
                {n} {n === 1 ? "search" : "searches"}
              </option>
            ))}
            <option value="custom">Custom…</option>
          </select>

          {/* Numeric input appears when user chooses "Custom…" */}
          {isCustom && (
            <input
              type="number"
              min={1}
              step={1}
              value={limit}
              onChange={(e) => setLimit(toPositiveInt(e.target.value, limit || 5))}
              style={{
                width: "110px",
                padding: "5px 8px",
                borderRadius: "4px",
                border: "1px solid #ddd"
              }}
              placeholder="Enter limit"
              aria-label="Custom limit"
              title="Enter any positive integer"
            />
          )}
        </div>
      </h2>

      {history.length === 0 ? (
        <div style={{ padding: "20px", textAlign: "center", color: "#666" }}>
          No search history available.
        </div>
      ) : (
        <div style={{ marginTop: "20px" }}>
          {history.map((item, index) => (
            <div key={item.search_id} style={{ marginBottom: "15px" }}>
              <div style={{
                display: "flex",
                justifyContent: "space-between",
                marginBottom: "5px",
                fontSize: "14px"
              }}>
                <span title={getKeywordsString(item.keywords)} style={{
                  whiteSpace: "nowrap",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  maxWidth: "60%"
                }}>
                  {getKeywordsString(item.keywords)}
                </span>
                <span style={{ fontWeight: "bold" }}>
                  {formatNumber(item.total_results)} results
                </span>
              </div>
              <div style={{
                height: "30px",
                background: "#e0e0e0",
                borderRadius: "4px",
                overflow: "hidden"
              }}>
                <div style={{
                  height: "100%",
                  width: `${maxResults > 0 ? (item.total_results / maxResults) * 100 : 0}%`,
                  background: index % 2 === 0 ? "#4fc3f7" : "#7986cb",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "flex-end",
                  paddingRight: "10px",
                  color: "white",
                  fontWeight: "bold",
                  fontSize: "14px",
                  transition: "width 0.5s ease"
                }}>
                  {formatNumber(item.total_results)}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default SearchHistoryChart;
