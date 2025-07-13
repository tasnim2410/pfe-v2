import React from "react";

interface LoadingSpinnerProps {
  text?: string;
  height?: number | string;
  size?: number;
  style?: React.CSSProperties;
}

const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  text = "Loading...",
  height = 110,
  size = 36,
  style = {},
}) => (
  <div
    style={{
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      justifyContent: "center",
      height,
      ...style,
    }}
  >
    <div
      style={{
        width: size,
        height: size,
        border: "5px solid #eee",
        borderTop: "5px solid #6E6E72",
        borderRadius: "50%",
        animation: "spin 1s linear infinite",
        marginBottom: 10,
      }}
    />
    <span style={{ color: "#6E6E72", fontWeight: 500 }}>{text}</span>
    <style>{`
      @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
      }
    `}</style>
  </div>
);

export default LoadingSpinner;
