'use client';
import React, { useEffect, useState } from 'react';

type MarketMetrics = { market_value: number; market_rate: number; mean_value: number };
type Coapplicant = { coapplicant_rate: number; total_applications: number; coapplicant_count: number };
type Innovation = string; // typically a float string
type Growth = { growth_rate: number };
type Originality = { originality_rate: number; total_patents: number; valid_patents: number };

export default function AnalysisSummaryCard() {
  const [market, setMarket] = useState<MarketMetrics | null>(null);
  const [coapp, setCoapp] = useState<Coapplicant | null>(null);
  const [innovation, setInnovation] = useState<string | null>(null);
  const [growth, setGrowth] = useState<Growth | null>(null);
  const [originality, setOriginality] = useState<Originality | null>(null);
  // Returns a market size description and reach based on value
function getMarketSummary(marketValue: number) {
  if (!isFinite(marketValue)) return "";
  if (marketValue >= 1e9) return "The IP market is large, with a global reach.";
  if (marketValue >= 1e7) return "The IP market is medium, with an international reach.";
  if (marketValue >= 1e5) return "The IP market is moderate, with niche opportunities.";
  return "The IP market is small and emerging.";
}
// Returns innovation summary based on innovation cycle (percent) and growth
function getInnovationSummary(innovationPct: number, growthRate: number) {
  if (!isFinite(innovationPct)) return "";
  if (innovationPct >= 50) return "Innovation cycle is ending, dominated by top applicants.";
  if (innovationPct >= 30) return "Innovation cycle is slowing, but active development continues.";
  if (innovationPct >= 20) return "Innovation cycle is ongoing, showing active development.";
  if (innovationPct >= 10) return "Innovation cycle is beginning, with new players entering.";
  return "Innovation is emerging, with high potential for new entrants.";
}
// Returns disruption comment based on originality rate
function getDisruptiveSummary(originality: number) {
  if (!isFinite(originality)) return "";
  if (originality >= 0.7) return "This is a disruptive innovation—new value is being created.";
  if (originality >= 0.4) return "Emerging innovation—sector is evolving.";
  return "Incremental innovation—changes are evolutionary rather than revolutionary.";
}




  function formatCompactNumber(num: number) {
    if (typeof num !== "number" || isNaN(num)) return "N/A";
    if (Math.abs(num) >= 1.0e9)
      return (num / 1.0e9).toFixed(2).replace(/\.00$/, '') + "B";
    if (Math.abs(num) >= 1.0e6)
      return (num / 1.0e6).toFixed(2).replace(/\.00$/, '') + "M";
    if (Math.abs(num) >= 1.0e3)
      return (num / 1.0e3).toFixed(2).replace(/\.00$/, '') + "K";
    return num.toString();
  }

  useEffect(() => {
    fetch("/backend_port.txt")
      .then((res) => res.text())
      .then((port) => port.trim())
      .then((port) => {
        Promise.all([
          fetch(`http://localhost:${port}/api/market_metrics`).then(res => res.json()),
          fetch(`http://localhost:${port}/api/coapplicant_rate`).then(res => res.json()),
          fetch(`http://localhost:${port}/api/innovation_cycle`).then(res => res.json()),
          fetch(`http://localhost:${port}/api/growth_rate`).then(res => res.json()),
          fetch(`http://localhost:${port}/api/originality_rate`).then(res => res.json())
        ]).then(([market, coapp, innovation, growth, originality]) => {
          setMarket(market);
          setCoapp(coapp);
          setInnovation(innovation);
          setGrowth(growth);
          setOriginality(originality);
        });
      });
  }, []);

  if (!market || !coapp || !innovation || !growth || !originality) {
    return <div className="py-12 flex justify-center text-gray-400">Loading analysis summary...</div>;
  }

  return (
    <div className="w-full bg-white rounded-xl shadow-md py-10 px-8 my-8">
      <h2 className="text-3xl font-bold text-center mb-8">ANALYSIS SUMMARY</h2>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-8 text-gray-700">
        {/* MARKET PRESENCE & APPLICATION SCOPE */}
        <div>
          <h3 className="font-semibold uppercase mb-2">Market Presence & Application Scope</h3>
          <ul className="list-disc ml-4 space-y-1">
            <li>
              <b>{coapp.coapplicant_rate}%</b> co-applicant rate ({coapp.coapplicant_count} multi-applicant out of {coapp.total_applications}).
            </li>
            <li>
              Mean patent market value: <b>${formatCompactNumber((market.mean_value))}</b>
            </li>
            <li>
              Total estimated market: <b>${formatCompactNumber(market.market_value)}</b>
            </li>
          </ul>
        </div>
        {/* INNOVATION DYNAMICS */}
        <div>
          <h3 className="font-semibold uppercase mb-2">Innovation Dynamics</h3>
          <ul className="list-disc ml-4 space-y-1">
            <li>
              Innovation cycle: <b>{innovation}</b>
            </li>
            <li>
              Growth rate: <b>
                {Array.isArray(growth.growth_rate) && isFinite(Number(growth.growth_rate[0]))
                  ? Number(growth.growth_rate[0]).toFixed(2)
                  : "N/A"}%
              </b>
            </li>
            <li>
              Market expansion rate: <b>
                {Array.isArray(market.market_rate) && isFinite(Number(market.market_rate[0]))
                  ? Number(market.market_rate[0]).toFixed(2)
                  : "N/A"}%
              </b>
            </li>
          </ul>
        </div>
        {/* DISRUPTIVE POTENTIAL */}
        <div>
          <h3 className="font-semibold uppercase mb-2">Disruptive Potential</h3>
          <ul className="list-disc ml-4 space-y-1">
            <li>
              Originality index: <b>{isFinite(Number(originality.originality_rate)) ? Number(originality.originality_rate).toFixed(2) : "N/A"}</b> ({originality.valid_patents} evaluated)
            </li>
            <li>
              High innovation means new value for the sector.
            </li>
            <li>
              {/* Add more logic-based sentences here */}
              Technology trend is <b>{growth.growth_rate > 0 ? 'rising' : 'declining'}</b> based on recent patent filings.
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
}
