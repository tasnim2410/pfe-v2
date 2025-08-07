"use client"
import { Download, Eye, FileText, Presentation } from "lucide-react"
import html2canvas from "html2canvas"
import { useState } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Button } from "@/components/ui/button"
import { ExternalLink, X } from "lucide-react"
import MarketStrategyCard from "./market_strategy";
import MarketSizeCard from "./market_size";

interface SearchResultsProps {
  hasSearched: boolean;
  results?: any[];
  loading?: boolean;
}

const summaryCards = [
  /* left column first row  */ { id: "market-size",      title: "Market Size",        description: "Market valuation analysis" },
  /* centre column first row*/ { id: "ipstat",           title: "IP Stats",           description: "Intellectual-property metrics" },
  /* right column first row */ { id: "investment",       title: "Investment Dynamic", description: "Investment trend analysis" },
  /* left column second row */ { id: "market-strategy",  title: "Market Strategy",    description: "Strategic market insights" },
  /* centre column second row*/{ id: "innovation",       title: "Innovation Cycle",   description: "Technology lifecycle" },
  /* right column second row*/ { id: "originality",      title: "Originality Rate",   description: "Patent originality" },
  /* full-width last row    */{ id: "summary",          title: "Analysis Summary",   description: "Comprehensive overview" },
];

// No more mockPatents; using real API results

import { Spinner } from "@/components/ui/spinner"
import LoadingSpinner from "./LoadingSpinner";
import styles from './search-results.module.css';
import React from 'react';
import { InnovationCycle } from './innovation-cycle';
import InvestmentDynamic from './investment_dynamic';
import OriginalityRate from './originality_rate';
import IpStatsBox from './IPStat';
import AnalysisSummaryCard from './analysis-summary';

export function SearchResults({ hasSearched, results, loading }: SearchResultsProps) {
  const [visibleCards, setVisibleCards] = useState<string[]>([])
  const [rowCount, setRowCount] = useState<number | 'all'>(10);
  const [activeTab, setActiveTab] = useState<'patents' | 'summary'>('patents');

  const showCard = (cardId: string) => {
    setVisibleCards([...visibleCards, cardId])
  }

  const hideCard = (cardId: string) => {
    setVisibleCards(visibleCards.filter((id) => id !== cardId))
  }

  const allSummaryIds = summaryCards.map(c => c.id);           // every card ID in Summary
  const allExpanded   = allSummaryIds.every(id => visibleCards.includes(id));

  const toggleAllSummary = () => {
    /* If all are visible → collapse them,
       else → expand any that are still hidden. */
    setVisibleCards(vc =>
      allExpanded
        ? vc.filter(id => !allSummaryIds.includes(id))               // collapse
        : Array.from(new Set([...vc, ...allSummaryIds]))             // expand
    );
  };



  const renderCard = (card: any) => {
    switch (card.id) {
      case "ipstat":
        return (
          <Card className="w-full relative">
            <Button
              variant="ghost"
              size="sm"
              className="absolute top-2 right-2 h-6 w-6 p-0 hover:bg-gray-100"
              onClick={() => hideCard(card.id)}
            >
              <X className="h-4 w-4" />
            </Button>
            <CardHeader>
              <CardTitle>IP Stats</CardTitle>
            </CardHeader>
            <CardContent className="flex items-center justify-center w-full p-2" id={`chart-${card.id}`}>
              <React.Suspense fallback={<LoadingSpinner text="Loading IP stats..." size={28} height={60} />}> 
                <IpStatsBox />
              </React.Suspense>
            </CardContent>
          </Card>
        );
      case "originality":

        return (
          <Card className="w-full relative">
            <Button
              variant="ghost"
              size="sm"
              className="absolute top-2 right-2 h-6 w-6 p-0 hover:bg-gray-100"
              onClick={() => hideCard(card.id)}
            >
              <X className="h-4 w-4" />
            </Button>
            <CardHeader>
              <CardTitle>Originality Rate</CardTitle>
            </CardHeader>
            <CardContent className="flex items-center justify-center w-full p-2" id={`chart-${card.id}`}>
              <React.Suspense fallback={<LoadingSpinner text="Loading originality rate..." size={28} height={60} />}> 
                <OriginalityRate />
              </React.Suspense>
            </CardContent>
          </Card>
        )
      case "innovation":
        return (
          <Card className="w-full relative">
            <Button
              variant="ghost"
              size="sm"
              className="absolute top-2 right-2 h-6 w-6 p-0 hover:bg-gray-100"
              onClick={() => hideCard(card.id)}
            >
              <X className="h-4 w-4" />
            </Button>
            <CardHeader>
              <CardTitle>Innovation Cycle</CardTitle>
            </CardHeader>
            <CardContent>
  <div className="flex items-center justify-center w-full" id={`chart-${card.id}`}>
    <React.Suspense fallback={<LoadingSpinner text="Loading innovation cycle..." size={28} height={60} />}> 
      <InnovationCycle />
    </React.Suspense>
  </div>
</CardContent>

          </Card>
        )
      case "market-strategy":
        return (
          <Card className="w-full relative">
            <Button
              variant="ghost"
              size="sm"
              className="absolute top-2 right-2 h-6 w-6 p-0 hover:bg-gray-100"
              onClick={() => hideCard(card.id)}
            >
              <X className="h-4 w-4" />
            </Button>
            <CardHeader>
              <CardTitle>Market Strategy</CardTitle>
            </CardHeader>
            <CardContent className="flex items-center justify-center w-full p-2" id={`chart-${card.id}`}>
              <MarketStrategyCard level="global"/>
            </CardContent>
          </Card>
        )

      case "market-size":
        return (
          <Card className="w-full relative">
            <Button
              variant="ghost"
              size="sm"
              className="absolute top-2 right-2 h-6 w-6 p-0 hover:bg-gray-100"
              onClick={() => hideCard(card.id)}
            >
              <X className="h-4 w-4" />
            </Button>
            <CardHeader>
              <CardTitle>Market Size</CardTitle>
            </CardHeader>
            <CardContent className="flex items-center justify-center w-full p-2" id={`chart-${card.id}`}>
              <React.Suspense fallback={<LoadingSpinner text="Loading market size..." size={28} height={60} />}>
                <MarketSizeCard size="big" />
              </React.Suspense>
            </CardContent>
          </Card>
        )

      case "investment":
        return (
          <Card className="w-full relative">
            <Button
              variant="ghost"
              size="sm"
              className="absolute top-2 right-2 h-6 w-6 p-0 hover:bg-gray-100"
              onClick={() => hideCard(card.id)}
            >
              <X className="h-4 w-4" />
            </Button>
            <CardHeader>
              <CardTitle>Investment Dynamic</CardTitle>
            </CardHeader>
            <CardContent className="flex justify-center w-full p-2">
            <div className="w-full" id={`chart-${card.id}`}>
    <React.Suspense fallback={<LoadingSpinner text="Loading investment dynamic..." size={28} height={60} />}> 
      <InvestmentDynamic />
    </React.Suspense>
  </div>
</CardContent>

          </Card>
        )
      case "summary":
        return (
          <Card className="w-full relative">
            <Button
              variant="ghost"
              size="sm"
              className="absolute top-2 right-2 h-6 w-6 p-0 hover:bg-gray-100"
              onClick={() => hideCard(card.id)}
            >
              <X className="h-4 w-4" />
            </Button>
            <CardHeader>
              {/* <CardTitle>Analysis Summary</CardTitle> */}
            </CardHeader>
            <CardContent className="flex items-center justify-center w-full p-2" id={`chart-${card.id}`}>
              <React.Suspense fallback={<LoadingSpinner text="Loading analysis summary..." size={28} height={60} />}> 
                <AnalysisSummaryCard />
              </React.Suspense>
            </CardContent>
          </Card>
        )
      default:
        return (
          <Card className="w-full relative">
            <Button
              variant="ghost"
              size="sm"
              className="absolute top-2 right-2 h-6 w-6 p-0 hover:bg-gray-100"
              onClick={() => hideCard(card.id)}
            >
              <X className="h-4 w-4" />
            </Button>
            <CardHeader>
              <CardTitle>{card.title}</CardTitle>
            </CardHeader>
            <CardContent>
              <p>Analysis data for {card.title.toLowerCase()}</p>
            </CardContent>
          </Card>
        )
    }
  }

  if (!hasSearched) {
    return (
      <Card className="w-full">
        <CardContent className="flex items-center justify-center h-32">
          {loading ? (
            <div className="flex flex-col items-center justify-center">
              <Spinner size={40} />
              <span className="mt-2 text-gray-500">Searching...</span>
            </div>
          ) : (
            <p className="text-gray-500">Make a search to see results</p>
          )}
        </CardContent>
      </Card>
    )
  }

  return (
    <Tabs
  defaultValue="patents"
  className="w-full"
  value={activeTab}
  onValueChange={val => setActiveTab(val as 'patents' | 'summary')}
>
  <TabsList className="grid w-full grid-cols-2">
    <TabsTrigger
      value="patents"
      className={`${styles.tabTrigger} ${activeTab !== 'patents' ? styles.tabTriggerInactive : ''}`}
    >
      Patents
    </TabsTrigger>
    <TabsTrigger
      value="summary"
      className={`${styles.tabTrigger} ${activeTab !== 'summary' ? styles.tabTriggerInactive : ''}`}
    >
      Summary
    </TabsTrigger>
  </TabsList>

      <TabsContent value="patents" className="space-y-4">
        {hasSearched ? (
          <Card>
            <CardHeader>
  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 16 }}>
    <CardTitle style={{ margin: 0 }}>Patent Search Results</CardTitle>
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <label htmlFor="rowCountSelect" style={{ fontWeight: 500, fontSize: 14, marginRight: 4 }}>Show:</label>
      <select
        id="rowCountSelect"
        value={rowCount}
        onChange={e => setRowCount(e.target.value === 'all' ? 'all' : Number(e.target.value))}
        style={{ padding: '4px 8px', borderRadius: 4, border: '1px solid #ccc', fontSize: 14 }}
      >
        <option value={10}>10</option>
        <option value={20}>20</option>
        <option value={50}>50</option>
        <option value="all">All</option>
      </select>
    </div>
  </div>
  <CardDescription>Found {results && results.length} patents matching your criteria</CardDescription>
</CardHeader>
<CardContent>
  <Table>
    <TableHeader>
                  <TableRow>
                    <TableHead>Publication Number</TableHead>
                    <TableHead>Title</TableHead>
                    <TableHead>Applicants</TableHead>
                    <TableHead>Inventors</TableHead>
                    <TableHead>CPC</TableHead>
                    <TableHead>IPC</TableHead>
                    <TableHead>Earliest Priority Year</TableHead>
                    <TableHead>Earliest Publication</TableHead>
                    <TableHead>First Filing Year</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {results && results.length > 0 ?
  (rowCount === 'all'
    ? results
    : results.slice(0, rowCount)
  ).map((patent: any, idx: number) => (
    <TableRow key={patent["Publication number"] || idx}>
      <TableCell>
  {typeof patent["Publication number"] === "string"
    ? patent["Publication number"].split(/\r?\n/).map((pubNum: string, i: number) => (
        pubNum.trim() ? (
          <div key={pubNum + i} style={{ marginBottom: 2 }}>
            <span style={{ position: 'relative', display: 'inline-block' }}>
  <a
    href={`https://worldwide.espacenet.com/patent/search/family/${patent["Family number"]}/publication/${pubNum.trim()}?q=pn%3D${pubNum.trim()}`}
    target="_blank"
    rel="noopener noreferrer"
    style={{ color: "#2563eb", textDecoration: "underline" }}
    onMouseEnter={e => {
      const tooltip = e.currentTarget.nextSibling as HTMLElement;
      if (tooltip) tooltip.style.display = 'block';
    }}
    onMouseLeave={e => {
      const tooltip = e.currentTarget.nextSibling as HTMLElement;
      if (tooltip) tooltip.style.display = 'none';
    }}
  >
    {pubNum.trim()}
  </a>
  <span
    style={{
      display: 'none',
      position: 'absolute',
      left: '100%',
      top: '50%',
      transform: 'translateY(-50%)',
      background: '#f1f1f1',
      color: '#222',
      padding: '6px 12px',
      borderRadius: 6,
      whiteSpace: 'nowrap',
      fontSize: 13,
      zIndex: 10,
      marginLeft: 8,
      boxShadow: '0 2px 8px rgba(0,0,0,0.15)'
    }}
  >
    see more details on EPO
  </span>
</span>
          </div>
        ) : null
      ))
    : null}
</TableCell>
      <TableCell>{patent["Title"]}</TableCell>
      <TableCell>{patent["Applicants"]}</TableCell>
      <TableCell>{patent["Inventors"]}</TableCell>
      <TableCell>{Array.isArray(patent["CPC"]) ? patent["CPC"].join(", ") : patent["CPC"]}</TableCell>
      <TableCell>{Array.isArray(patent["IPC"]) ? patent["IPC"].join(", ") : patent["IPC"]}</TableCell>
      <TableCell>{patent["earliest_priority_year"]}</TableCell>
      <TableCell>{patent["earliest_publication"]}</TableCell>
      <TableCell>{patent["first_filing_year"]}</TableCell>
    </TableRow>
  )) : (
                    <TableRow>
                      <TableCell colSpan={9} className="text-center text-gray-500">No results found</TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        ) : loading ? (
          <Card className="w-full">
            <CardContent className="flex items-center justify-center h-32">
              <LoadingSpinner size={36} />
            </CardContent>
          </Card>
        ) : (
          <Card className="w-full">
            <CardContent className="flex items-center justify-center h-32">
              <p className="text-gray-500">Make a search to see results</p>
            </CardContent>
          </Card>
        )}
      </TabsContent>

      <TabsContent value="summary" className="space-y-4">
      <div className="flex justify-end">
          <Button
            onClick={toggleAllSummary}
            className="bg-[#BDD248] text-white hover:bg-[#546472] transition-colors duration-150
                       focus-visible:ring-2 focus-visible:ring-gray-400"
          >
            {allExpanded ? "Collapse All" : "Expand All"}
          </Button>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 auto-rows-auto">
          {summaryCards.map((card) => (
            <div
            key={card.id}
            className={`${
              card.id === "summary" ? "md:col-span-3" : ""   // ⬅ full width on desktops
            }`}
          >
              {!visibleCards.includes(card.id) ? (
                <Button
                  variant="outline"
                  className="w-full h-32 flex flex-col items-center justify-center bg-transparent"
                  onClick={() => showCard(card.id)}
                >
                  <div className="text-lg font-semibold">{card.title}</div>
                  <div className="text-sm text-gray-600 mt-1">{card.description}</div>
                </Button>
              ) : (
                renderCard(card)
              )}
            </div>
          ))}
        </div>
      </TabsContent>
    </Tabs>
  )
}
