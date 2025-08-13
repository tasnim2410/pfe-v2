import React, { useEffect, useState } from 'react';
import { Card } from './ui/card';
import { ChartContainer } from './ui/chart';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LabelList } from 'recharts';

interface PublicationYearData {
  count: number;
  year: number;
}

const PublicationsByYear: React.FC = () => {
  const [data, setData] = useState<PublicationYearData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const port = (await (await fetch("/backend_port.txt")).text()).trim();
        
        const res = await fetch(`http://localhost:${port}/api/research_publications_by_year`);
        if (!res.ok) throw new Error('Failed to fetch data');
        const json = await res.json();
        setData(json);
      } catch (err: any) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  if (loading) return <Card><div>Loading...</div></Card>;
  if (error) return <Card><div>Error: {error}</div></Card>;

  // Prepare data for chart
  const chartData = data.map((d) => ({ year: d.year, count: d.count }));

  return (
    <Card className="w-full h-full flex flex-col">
      <div className="flex-1 min-h-[400px] w-full">
        <ChartContainer config={{}}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 32, right: 32, left: 32, bottom: 32 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" vertical={false} />
              <XAxis 
                dataKey="year" 
                tick={{ fontSize: 12 }}
                label={{ 
                  value: 'Year', 
                  position: 'bottom', 
                  offset: 0,
                  style: { fontSize: 14, fontWeight: 500 }
                }}
              />
              <YAxis 
                tick={{ fontSize: 12 }}
                label={{ 
                  value: 'Number of Publications', 
                  angle: -90, 
                  position: 'insideLeft',
                  offset: 10,
                  style: { fontSize: 14, fontWeight: 500 }
                }}
              />
              <Tooltip 
                formatter={(value) => [`${value} publications`, 'Count']}
                labelFormatter={(year) => `Year: ${year}`}
              />
                            <Bar 
                dataKey="count" 
                fill="#3692eb" 
                radius={[4, 4, 0, 0]}
              >
                <LabelList 
                  dataKey="count" 
                  position="top" 
                  formatter={(value: unknown) => 
                    typeof value === 'number' && value > 0 ? value : ''
                  }
                  fill="#333"
                  fontSize={12}
                />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartContainer>
      </div>
    </Card>
  );
};

export default PublicationsByYear;
