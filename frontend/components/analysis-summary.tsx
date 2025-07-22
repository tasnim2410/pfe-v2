// analysis-summary.tsx  (watermark-free)

'use client';
import React, { useEffect, useState } from 'react';

type MarketMetrics = { market_value: number; market_rate: number; mean_value: number };
type Coapplicant   = { coapplicant_rate: number; total_applications: number; coapplicant_count: number };
type Innovation    = string;
type Growth        = { growth_rate: number };
type Originality   = { originality_rate: number; total_patents: number; valid_patents: number };

const nfmt = (n: number) =>
  !isFinite(n)
    ? 'N/A'
    : Math.abs(n) >= 1e9
    ? `${(n / 1e9).toFixed(2).replace(/\.00$/, '')}B`
    : Math.abs(n) >= 1e6
    ? `${(n / 1e6).toFixed(2).replace(/\.00$/, '')}M`
    : Math.abs(n) >= 1e3
    ? `${(n / 1e3).toFixed(2).replace(/\.00$/, '')}K`
    : n.toFixed(2).replace(/\.00$/, '');

const marketSentence = (value: number) => {
  if (!isFinite(value)) return '';
  if (value >= 1e9) return 'The IP market is large, with a global reach.';
  if (value >= 1e7) return 'The IP market is medium, with a global reach.';
  if (value >= 1e5) return 'The IP market is moderate, with niche opportunities.';
  return 'The IP market is small and emerging.';
};
const innovationSentence = (pct: number) => {
  if (!isFinite(pct)) return '';
  if (pct >= 50) return 'Innovation cycle is ending, technology is mature.';
  if (pct >= 30) return 'Innovation cycle is slowing, but active development continues.';
  if (pct >= 20) return 'Innovation cycle is ongoing, showing active development.';
  if (pct >= 10) return 'Innovation cycle is beginning, early players are entering.';
  return 'Innovation is emerging with high growth potential.';
};
const disruptiveSentence = (ori: number) => {
  if (!isFinite(ori)) return '';
  if (ori >= 0.7) return 'Sector is disruptive—new value is being created.';
  if (ori >= 0.4) return 'Sector is emerging—innovation is accelerating.';
  return 'Sector is incremental—evolutionary rather than revolutionary.';
};

export default function AnalysisSummaryCard() {
  const [market,      setMarket]      = useState<MarketMetrics | null>(null);
  const [coapp,       setCoapp]       = useState<Coapplicant | null>(null);
  const [innovation,  setInnovation]  = useState<number | null>(null);
  const [growth,      setGrowth]      = useState<Growth | null>(null);
  const [originality, setOriginality] = useState<Originality | null>(null);

  useEffect(() => {
    (async () => {
      const port = (await fetch('/backend_port.txt').then(r => r.text()).catch(() => '49473')).trim();
      const url = (p: string) => `http://localhost:${port}${p}`;

      const [m, c, i, g, o] = await Promise.all([
        fetch(url('/api/market_metrics')).then(r => r.json()),
        fetch(url('/api/coapplicant_rate')).then(r => r.json()),
        fetch(url('/api/innovation_cycle')).then(r => r.json()),
        fetch(url('/api/growth_rate')).then(r => r.json()),
        fetch(url('/api/originality_rate')).then(r => r.json()),
      ]);

      setMarket(m);
      setCoapp(c);
      setInnovation(parseFloat(i));
      setGrowth(g);
      setOriginality(o);
    })();
  }, []);

  if (!market || !coapp || innovation === null || !growth || !originality)
    return <div className="py-12 flex justify-center text-gray-400">Loading analysis summary…</div>;

  const bulletsPresence = [
    `${coapp.coapplicant_rate}% co-applicant rate (${coapp.coapplicant_count} multi-applicant out of ${coapp.total_applications}).`,
    `Mean patent market value: $${nfmt(market.mean_value)}.`,
    `Total estimated market: $${nfmt(market.market_value)}.`,
    marketSentence(market.market_value),
  ];
  const bulletsInnovation = [
    innovationSentence(innovation),
    growth.growth_rate > 20
      ? 'Technology is rapidly evolving with new capabilities emerging.'
      : growth.growth_rate > 0
      ? 'Technology shows steady improvements.'
      : 'Technology trend is declining based on recent filings.',

  ];
  const bulletsDisruptive = [
    `Originality index: ${originality.originality_rate.toFixed(2)} (${originality.valid_patents} evaluated).`,
    disruptiveSentence(originality.originality_rate),
    growth.growth_rate > 0
      ? 'High innovation is creating new value for the sector.'
      : 'Low innovation may limit value creation.',
  ];

  return (
    <div className="w-full bg-white rounded-xl shadow-md py-10 px-8 my-8">
      <h2 className="text-4xl font-bold text-center mb-10">ANALYSIS SUMMARY</h2>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-8 text-gray-800">
        <section>
          <h3 className="font-semibold uppercase mb-2">Market Presence & Application Scope</h3>
          <ul className="list-disc ml-4 space-y-1">
            {bulletsPresence.map(t => <li key={t}>{t}</li>)}
          </ul>
        </section>

        <section>
          <h3 className="font-semibold uppercase mb-2">Innovation Dynamics</h3>
          <ul className="list-disc ml-4 space-y-1">
            {bulletsInnovation.map(t => <li key={t}>{t}</li>)}
          </ul>
        </section>

        <section>
          <h3 className="font-semibold uppercase mb-2">Disruptive Potential</h3>
          <ul className="list-disc ml-4 space-y-1">
            {bulletsDisruptive.map(t => <li key={t}>{t}</li>)}
          </ul>
        </section>
      </div>
    </div>
  );
}
