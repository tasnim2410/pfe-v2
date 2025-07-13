"use client"

import React from "react"

export const Spinner: React.FC<{ size?: number; color?: string }> = ({ size = 32, color = "#BDD248" }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 50 50"
    style={{ display: "inline-block", verticalAlign: "middle" }}
    aria-label="Loading"
  >
    <circle
      cx="25"
      cy="25"
      r="20"
      fill="none"
      stroke={color}
      strokeWidth="5"
      strokeDasharray="31.415, 31.415"
      strokeLinecap="round"
      transform="rotate(-90 25 25)"
    >
      <animateTransform
        attributeName="transform"
        type="rotate"
        from="0 25 25"
        to="360 25 25"
        dur="1s"
        repeatCount="indefinite"
      />
    </circle>
  </svg>
)
