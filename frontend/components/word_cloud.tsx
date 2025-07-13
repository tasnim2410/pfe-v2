

"use client";

import React, { useEffect, useRef, useState, useLayoutEffect } from "react";
import D3WordCloud from "react-d3-cloud";

interface ProcessedResponse {
  processed: string[];
}

interface WordDatum {
  text: string;
  value: number;
}

const WordCloud: React.FC = () => {
  const [wordData, setWordData] = useState<WordDatum[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [backendPort, setBackendPort] = React.useState<string | null>(null);
  React.useEffect(() => {
    fetch('/backend_port.txt')
      .then(res => res.text())
      .then(port => setBackendPort(port.trim()))
      .catch(() => setBackendPort(null));
  }, []);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dim, setDim] = useState<{ width: number; height: number }>({ width: 0, height: 0 });

  // Fetch data
  useEffect(() => {
    if (!backendPort) return; // Only fetch if backendPort is loaded
    const fetchProcessed = async () => {
      try {
        const res = await fetch(`http://localhost:${backendPort}/api/processed_texts`);
        if (!res.ok) throw new Error(`Server responded with ${res.status}`);
        const data: ProcessedResponse = await res.json();

        // Flatten sentences → words
        const allWords = data.processed
          .flatMap((sentence) => sentence.split(" "))
          .filter((w) => w.length > 0);

        // Count frequencies
        const freqMap: Record<string, number> = {};
        allWords.forEach((w) => {
          freqMap[w] = (freqMap[w] || 0) + 1;
        });

        // Convert to WordDatum[]
        const cloudArray: WordDatum[] = Object.entries(freqMap).map(
          ([text, value]) => ({ text, value })
        );

        setWordData(cloudArray);
      } catch (err: any) {
        setError(err.message || "Unknown error");
      } finally {
        setIsLoading(false);
      }
    };

    fetchProcessed();
  }, [backendPort]);

  // Use getBoundingClientRect and useLayoutEffect for robust sizing
  const measure = () => {
    if (containerRef.current) {
      const { width, height } = containerRef.current.getBoundingClientRect();
      setDim({ width: Math.floor(width), height: Math.floor(height) });
    }
  };

  useLayoutEffect(() => {
    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, []);

  useLayoutEffect(() => {
    measure();
  }, [wordData]);

  const fontSizeMapper = (word: WordDatum): number => Math.log2(word.value + 1) * 12;
  const rotate = (): number => (Math.random() > 0.5 ? 0 : 90);

  if (isLoading) {
    return (
      <div className="w-full h-96 flex items-center justify-center">
        <p className="text-muted-foreground">Loading word cloud…</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="w-full h-96 flex items-center justify-center">
        <p className="text-red-500">Error: {error}</p>
      </div>
    );
  }

  if (wordData.length === 0) {
    return (
      <div className="w-full h-96 flex items-center justify-center">
        <p className="text-muted-foreground">No data available</p>
      </div>
    );
  }

  return (
    <div ref={containerRef} className="w-full h-96 bg-white rounded-lg shadow-md">
      {dim.width > 0 && dim.height > 0 && (
        <D3WordCloud
          data={wordData}
          fontSize={fontSizeMapper}
          rotate={rotate}
          width={dim.width}
          height={dim.height}
        />
      )}
    </div>
  );
};

export default WordCloud;
