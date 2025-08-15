import React, { useEffect, useState } from "react";

const CHART_BG = "#fff";
const ACCENT = "#232526";

const cardStyle: React.CSSProperties = {
  background: CHART_BG,
  borderRadius: 18,
  boxShadow: "0 2px 18px #B2DBA422",
  padding: 30,
  width: 860,
  maxWidth: "100%",
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  minHeight: 320,
  margin: "0 auto"
};

const tabStyle = (active: boolean): React.CSSProperties => ({
  padding: "10px 22px",
  borderRadius: 8,
  background: active ? "#BDD248" : "#f2f5ec",
  color: active ? ACCENT : "#63735A",
  fontWeight: 700,
  margin: "0 7px",
  border: "none",
  cursor: "pointer",
  fontSize: 16,
  boxShadow: active ? "0 2px 12px #BDD24866" : undefined,
  transition: "background 0.15s"
});

const weightBar = (weight: number, max: number) => ({
  display: "inline-block",
  height: 10,
  borderRadius: 5,
  minWidth: 20 + (60 * weight / max),
  background: "#BDD248",
  marginLeft: 8,
  verticalAlign: "middle"
});

const KeywordList: React.FC<{ words: { word: string; weight: number }[] }> = ({ words }) => {
  const maxWeight = words.length ? Math.max(...words.map(w => w.weight)) : 1;
  return (
    <div style={{ marginTop: 12 }}>
      {words.map((kw, idx) => (
        <div
          key={kw.word}
          style={{
            margin: "9px 0",
            display: "flex",
            alignItems: "center",
            fontSize: idx === 0 ? 23 : 16,
            fontWeight: idx === 0 ? 800 : 600,
            color: idx === 0 ? "#EA3C53" : ACCENT
          }}
        >
          <span style={{ minWidth: 170, textTransform: "capitalize" }}>
            {kw.word}
          </span>
          <span style={weightBar(kw.weight, maxWeight)} />
          <span style={{ fontSize: 14, color: "#8fa68f", marginLeft: 9 }}>
            {kw.weight.toFixed(2)}
          </span>
        </div>
      ))}
    </div>
  );
};


import LoadingSpinner from "./LoadingSpinner";

const ThemeByTimeWindow: React.FC = () => {
  const [data, setData] = useState<any[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [selected, setSelected] = useState(0);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      setLoading(true);
      setErr(null);
      try {
        const portRes = await fetch("/backend_port.txt");
        const port = (await portRes.text()).trim();
        const endpoints = [
          `/api/research/processed_texts`,
          `/api/research/topic_evolution`,
          
        ];
        // Sequentially fetch each required endpoint
        for (let ep of endpoints) {
          const res = await fetch(`http://localhost:${port}${ep}`);
          if (!res.ok) throw new Error(`Failed to fetch ${ep}`);
        }
        // Now fetch the weighted word clouds
        const apiRes = await fetch(`http://localhost:${port}/api/research/weighted_word_clouds`);
        if (!apiRes.ok) throw new Error("Failed to fetch weighted word clouds");
        const json = await apiRes.json();
        if (!cancelled) setData(json);
        setLoading(false);
      } catch (err: any) {
        setErr(
          err instanceof Error && err.message
            ? err.message
            : "Failed to fetch topic data. Is the backend running?"
        );
        setLoading(false);
      }
    };
    load();
    return () => { cancelled = true; };

  }, []);

  if (loading) return <LoadingSpinner text="Loading Evolving Word Cloud..." />;
  if (err) return (
    <div style={{ color: "#EA3C53", textAlign: "center", margin: 22 }}>
      {err}
    </div>
  );
  if (!data || !data.length) return null;

  const win = data[selected];
  return (
    <div style={cardStyle}>
      {/* Time window tabs */}
      <div style={{ display: "flex", justifyContent: "center", marginBottom: 26 }}>
        {data.map((w, i) => (
          <button
            key={i}
            style={tabStyle(i === selected)}
            onClick={() => setSelected(i)}
          >
            {w.start}–{w.end}
          </button>
        ))}
      </div>
      <div style={{ width: "100%", maxWidth: 640, margin: "0 auto" }}>
        <div style={{ fontSize: 17, color: ACCENT, fontWeight: 600, marginBottom: 16, letterSpacing: 0.5 }}>
          <span style={{ color: "#BDD248", fontWeight: 800, fontSize: 22, marginRight: 12 }}>
            {win.start} – {win.end}
          </span>
          Dominant keywords for this period:
        </div>
        <KeywordList words={win.words.slice(0, 10)} />
      </div>
    </div>
  );
};

export default ThemeByTimeWindow;
