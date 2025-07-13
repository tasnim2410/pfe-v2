"use client"
import { Download, Eye, FileText, Presentation } from "lucide-react"
import html2canvas from "html2canvas"
import { useChartContext } from "../providers/ChartContext";
import { useState } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from "recharts"
import PublicationTrends from "@/components/publication_trends"; // <-- ADDED
import EvolvingWordCloud from "@/components/evolving_word_cloud";
import ApplicantTypePie from "@/components/applicant_type_pie";
import IpStatsBox from "@/components/IPStat";
import Top10Applicants from "@/components/top_10_app";
import { TopIPCCodes } from "@/components/top_ipc_codes";
import TopKeywords from "@/components/top_10_keywords";
import PatentFieldTrends from "@/components/patent_by_field";
import CooccurrenceTrends from "@/components/co-occurunece_trend";
import ApplicantCollaborationNetwork from "@/components/collaboration_network";

const analysisCards = {
  patents: [
    { id: "applicant-analysis", title: "Applicant Type Analysis" },
    { id: "top-ipc", title: "Top IPC Codes" },
    { id: "publication-trend", title: "Publication Trend" },
    { id: "geographic-distribution", title: "Geographic Distribution" },
    { id: "ip-stats", title: "IP Stats" },
    { id: "top-10-applicants", title: "Top 10 Patent Applicants" },
    { id: "top-10-keywords", title: "Top 10 Keywords" },
    { id: "patent-field-trends", title: "Patent Field Trends" },
        { id: "evolving-word-cloud", title: "Evolving Word Cloud" }, // Added
    { id: "cooccurrence-trends", title: "Co-occurrence Trends" },
    { id: "applicant-collaboration-network", title: "Applicant Collaboration Network" },
  ],
  research: [
    { id: "research-trend", title: "Research Publication Trend" },
    { id: "citation-analysis", title: "Citation Analysis" },
    { id: "collaboration-network", title: "Collaboration Network" },
    { id: "funding-analysis", title: "Funding Analysis" },
  ],
}

const mockBarData = [
  { name: "2019", value: 120 },
  { name: "2020", value: 150 },
  { name: "2021", value: 180 },
  { name: "2022", value: 220 },
  { name: "2023", value: 280 },
]

const mockPieData = [
  { name: "Tech Corp", value: 35, color: "#8884d8" },
  { name: "Innovation Ltd", value: 25, color: "#82ca9d" },
  { name: "Future Systems", value: 20, color: "#ffc658" },
  { name: "Others", value: 20, color: "#ff7300" },
]

export default function TrendAnalysis() {
  const [visibleCards, setVisibleCards] = useState<string[]>([]);
  const { selectedCharts, setSelectedCharts, chartComments, setChartComments } = useChartContext();
  

  const showCard = (cardId: string) => {
    setVisibleCards([...visibleCards, cardId]);
    if (!selectedCharts.includes(cardId)) {
      setSelectedCharts([...selectedCharts, cardId]);
    }
  };

  const hideCard = (cardId: string) => {
    setVisibleCards(visibleCards.filter((id) => id !== cardId));
    setSelectedCharts(selectedCharts.filter((id) => id !== cardId));
  };

  const updateComment = (cardId: string, comment: string) => {
    setChartComments({ ...chartComments, [cardId]: comment });
  };

  const renderChart = (cardId: string) => {
    switch (cardId) {
      case "publication-trend":
        return <PublicationTrends />;
      case "applicant-analysis":
        return (
          <div className="w-full flex justify-center items-center">
            <ApplicantTypePie />
          </div>
        );
      case "ip-stats":
        return (
          <div className="flex justify-center items-center w-full h-full">
            <IpStatsBox />
          </div>
        );
      case "top-ipc":
        return <TopIPCCodes />;
      case "top-10-applicants":
        return <Top10Applicants />;
      case "top-10-keywords":
        return <TopKeywords />;
      case "patent-field-trends":
        return <PatentFieldTrends />;
            case "evolving-word-cloud":
        return <EvolvingWordCloud />;
      case "cooccurrence-trends":
        return <CooccurrenceTrends />;
      case "applicant-collaboration-network":
        return <ApplicantCollaborationNetwork />;
      default:
        return (
          <div className="flex items-center justify-center h-64">
            <p className="text-gray-500">Chart data for {cardId}</p>
          </div>
        );
    }
  }

  return (
    <div className="container mx-auto px-6 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Trend Analysis</h1>
        <p className="text-gray-600">Comprehensive analysis of technology trends and patterns</p>
      </div>

      <Tabs defaultValue="patents" className="w-full">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="patents">Patent Analysis</TabsTrigger>
          <TabsTrigger value="research">Scientific Research</TabsTrigger>
        </TabsList>

        <TabsContent value="patents" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-2 gap-4">
            {analysisCards.patents.map((card) => {
  const isTop10 = card.id === "top-10-applicants";
  return (
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
        <Card className={isTop10 || card.id === "top-ipc" ? "min-w-[400px] max-w-[98vw] w-fit" : "w-full"}>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle>{card.title}</CardTitle>
            <Button variant="ghost" size="icon" onClick={() => hideCard(card.id)}>
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="h-4 w-4"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
            </Button>
          </CardHeader>
          <CardContent className={isTop10 || card.id === "top-ipc" ? "p-0 min-w-[400px] max-w-[98vw] w-auto" : undefined}>{renderChart(card.id)}</CardContent>
        </Card>
      )}
    </div>
  );
})}

          </div>
        </TabsContent>

        <TabsContent value="research" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {analysisCards.research.map((card) => (
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
                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="h-4 w-4"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
                      </Button>
                    </CardHeader>
                    <CardContent>{renderChart(card.id)}</CardContent>
                  </Card>
                )}
              </div>
            ))}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}
