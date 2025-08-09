"use client";
import React, { useState } from 'react';
import { Card } from './ui/card';
import { Button } from './ui/button';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
  ResponsiveContainer
} from 'recharts';

// Updated to include combined fields
const AVAILABLE_FIELDS = [
  'Medicine',
  'Computer Science',
  'Biology',
  'Materials Science',
  'Environmental Science',
  'Chemistry',
  'Physics',
  'Mathematics',
  'Engineering',
  'Agriculture',
  'Geology',
  'Psychology',
  'Economics',
  'Statistics',
];

interface TrendData {
  count: number;
  field: string;
  year: number;
}

// Color palette for lines
const COLORS = [
  '#3692eb', '#f39c12', '#e74c3c', '#2ecc71', '#8e44ad', 
  '#16a085', '#d35400', '#7f8c8d', '#e67e22', '#34495e',
];

export default function ResearchFieldTrends() {
  const [selectedFields, setSelectedFields] = useState<string[]>([]);
  const [trendData, setTrendData] = useState<TrendData[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchTrends = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('http://localhost:57495/api/research_field_trends', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fields: selectedFields }),
      });
      if (!response.ok) throw new Error('Failed to fetch trends');
      const data = await response.json();
      setTrendData(data);
    } catch (err: any) {
      setError(err.message || 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  // Transform data for recharts
  const years = Array.from(new Set(trendData.map(d => d.year))).sort((a, b) => a - b);
  const chartData = years.map(year => {
    const entry: any = { year };
    selectedFields.forEach(field => {
      const found = trendData.find(d => d.year === year && d.field === field);
      entry[field] = found ? found.count : 0;
    });
    return entry;
  });

  return (
    <div className="p-4">
      <h2 className="text-xl font-bold mb-4">Trends for Selected Research Fields of Study</h2>
      
      <div className="mb-6">
        <label className="block font-medium mb-2">Select Fields:</label>
        <div className="flex flex-wrap gap-2 mb-4">
          {AVAILABLE_FIELDS.map(field => (
            <Card 
              key={field}
              className={`p-3 cursor-pointer border rounded-lg transition-all ${
                selectedFields.includes(field)
                  ? 'bg-blue-100 border-blue-500'
                  : 'hover:bg-gray-50'
              }`}
              onClick={() => {
                setSelectedFields(prev => 
                  prev.includes(field)
                    ? prev.filter(f => f !== field)
                    : [...prev, field]
                );
              }}
            >
              {field}
            </Card>
          ))}
        </div>
        
        <Button 
          className="mt-2"
          onClick={fetchTrends}
          disabled={selectedFields.length === 0 || loading}
        >
          {loading ? 'Loading...' : 'Show Trends'}
        </Button>
      </div>
      
      {error && <div className="text-red-500 mb-4 p-3 bg-red-50 rounded-md">{error}</div>}
      
      {trendData.length > 0 && selectedFields.length > 0 && (
        <div className="mt-8">
          <h3 className="font-semibold text-center mb-4">Number of Papers (Smoothed, 3-year avg)</h3>
          <div className="w-full h-[400px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={chartData}
                margin={{ top: 20, right: 30, left: 20, bottom: 30 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis 
                  dataKey="year" 
                  type="number"
                  domain={['dataMin', 'dataMax']}
                  tickCount={10}
                  tick={{ fontSize: 13 }}
                  dy={10}
                />
                <YAxis 
                  tick={{ fontSize: 13 }} 
                  allowDecimals={false}
                />
                <Tooltip 
                  formatter={(value) => [`${value} papers`, 'Count']}
                  labelFormatter={(year) => `Year: ${year}`}
                />
                <Legend 
                  verticalAlign="top" 
                  height={40}
                  wrapperStyle={{ paddingBottom: 20 }}
                />
                {selectedFields.map((field, idx) => (
                  <Line
                    key={field}
                    type="monotone"
                    dataKey={field}
                    stroke={COLORS[idx % COLORS.length]}
                    strokeWidth={2}
                    dot={false}
                    activeDot={{ r: 6 }}
                    name={field}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
      
      {trendData.length === 0 && !loading && !error && (
        <div className="text-gray-500 text-center py-8">
          Select fields and click "Show Trends" to display data
        </div>
      )}
    </div>
  );
}