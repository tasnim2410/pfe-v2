import React, { useEffect, useState } from 'react';
import { Card } from './ui/card';
import { ChartContainer } from './ui/chart';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface PublicationYearData {
  count: number;
  year: number;
}

const PublicationsByYear: React.FC = () => {
  const [data, setData] = useState<PublicationYearData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch('http://localhost:53076/api/research_publications_by_year')
      .then((res) => {
        if (!res.ok) throw new Error('Failed to fetch data');
        return res.json();
      })
      .then((json) => {
        setData(json);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  if (loading) return <Card><div>Loading...</div></Card>;
  if (error) return <Card><div>Error: {error}</div></Card>;

  // Prepare data for chart
  const chartData = data.map((d) => ({ year: d.year, count: d.count }));

  return (
    <Card style={{ minHeight: 500, display: 'flex', flexDirection: 'column', justifyContent: 'center', height: '100%' }}>
      <div style={{ flex: 1, display: 'flex' }}>
        <ChartContainer config={{}}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 32, right: 32, left: 32, bottom: 32 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
              <XAxis dataKey="year" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="count" fill="#3692eb" />
            </BarChart>
          </ResponsiveContainer>
        </ChartContainer>
      </div>
    </Card>
  );
};

export default PublicationsByYear;
