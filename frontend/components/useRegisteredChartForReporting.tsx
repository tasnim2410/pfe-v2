import { useRef, useEffect } from "react";
import { useSetRecoilState } from "recoil";
import { savedChartsState } from "./savedChartsState";

export function useRegisterChartForReporting(cardId: string, cardTitle: string) {
  const chartRef = useRef(null);
  const setSavedCharts = useSetRecoilState(savedChartsState);

  useEffect(() => {
    setSavedCharts(prev => {
      if (prev.find(c => c.id === cardId)) return prev;
      return [...prev, { id: cardId, title: cardTitle, ref: chartRef }];
    });
    // Optional: remove chart when component unmounts
    return () => {
      setSavedCharts(prev => prev.filter(c => c.id !== cardId));
    };
  }, [cardId, cardTitle, setSavedCharts]);

  return chartRef;
}
