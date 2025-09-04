"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from "recharts"

const forecastData = [
  { year: "2020", patents: 150, research: 80, predicted: false },
  { year: "2021", patents: 180, research: 95, predicted: false },
  { year: "2022", patents: 220, research: 110, predicted: false },
  { year: "2023", patents: 280, research: 130, predicted: false },
  { year: "2024", patents: 340, research: 155, predicted: true },
  { year: "2025", patents: 410, research: 180, predicted: true },
  { year: "2026", patents: 490, research: 210, predicted: true },
]

import PatentPublicationChart from "@/components/filing_vs_publication";
import TVPTwoCharts from "@/components/TVPcharts";
import ProphetForecast from "@/components/prophet_forecast";
import ArimaxQuadraticClient from "@/components/arimax_quadratic_client";
import LSTMForecastSeries from "@/components/lstm_forecast";

const forecastingCards = [
  { id: "prophet-forecast", title: "Prophet Model Forecast" },
  { id: "lstm-forecast", title: "LSTM Model Forecast" },
  { id: "patent-publications", title: "Patent Filings vs. Publications" },
  // { id: "forecasted-trends", title: "Forecasted Patent & Publication Trends" },
]
export default function Forecasting() {
  const [visibleCards, setVisibleCards] = useState<string[]>([]);

  const showCard = (cardId: string) => {
    setVisibleCards([...visibleCards, cardId]);
  };

  const hideCard = (cardId: string) => {
    setVisibleCards(visibleCards.filter((id) => id !== cardId));
  };

  const allExpanded = forecastingCards.every(card => visibleCards.includes(card.id));

  const toggleAll = () => {
    if (allExpanded) {
      setVisibleCards([]);
    } else {
      setVisibleCards(forecastingCards.map(card => card.id));
    }
  };

  const renderChart = (cardId: string) => {
    switch (cardId) {
      case "prophet-forecast":
        return <ProphetForecast />;
      case "lstm-forecast":
        return <LSTMForecastSeries />;
      case "patent-publications":
        return <PatentPublicationChart />;
      case "forecasted-trends":
        return <TVPTwoCharts />;
      default:
        return (
          <div className="flex items-center justify-center h-64">
            <p className="text-gray-500">Chart data for {cardId}</p>
          </div>
        );
    }
  };

  return (
    <div className="container mx-auto px-6 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Technology Forecasting</h1>
        <p className="text-gray-600">Predictive analysis of future patent and research trends</p>
      </div>

      <div className="flex justify-end mb-4">
        <Button 
          variant="secondary" 
          onClick={toggleAll}
          className="bg-[#BDD248] text-white hover:bg-[#546472] transition-colors duration-150 focus-visible:ring-2 focus-visible:ring-gray-400"
        >
          {allExpanded ? "Collapse All" : "Expand All"}
        </Button>
      </div>

      <div className="grid grid-cols-1 gap-4">
        {forecastingCards.map((card) => (
          <div key={card.id}>
            {!visibleCards.includes(card.id) ? (
              <Button
                variant="outline"
                className="w-full h-32 flex flex-col items-center justify-center bg-transparent"
                onClick={() => showCard(card.id)}
              >
                <div className="text-lg font-semibold">{card.title}</div>
              </Button>
            ) : (
              <Card className="w-full">
                <CardHeader className="flex flex-row items-center justify-between">
                  <CardTitle>{card.title}</CardTitle>
                  <Button variant="ghost" size="icon" onClick={() => hideCard(card.id)}>
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="h-4 w-4">
                      <line x1="18" y1="6" x2="6" y2="18"></line>
                      <line x1="6" y1="6" x2="18" y2="18"></line>
                    </svg>
                  </Button>
                </CardHeader>
                <CardContent>{renderChart(card.id)}</CardContent>
              </Card>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
