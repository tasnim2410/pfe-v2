// import { useEffect, useRef } from 'react';
// import { Chart } from 'chart.js/auto';

// const PatentPublicationChart = () => {
//   const chartRef = useRef<HTMLCanvasElement>(null);
//   const chartInstance = useRef<Chart | null>(null);

//   useEffect(() => {
//     const fetchData = async () => {
//       try {
//         // Get backend port
//         const portResponse = await fetch("/backend_port.txt");
//         const port = (await portResponse.text()).trim();
        
//         // Fetch patent data
//         const patentRes = await fetch(`http://localhost:${port}/api/patents/first_filing_years`);
//         const patentData = await patentRes.json();
        
//         // Fetch publication data
//         const pubRes = await fetch(`http://localhost:${port}/api/research_publications_by_year`);
//         const pubData = await pubRes.json();
        
//         // Process patent data
//         const patentYears = patentData.labels;
//         const patentCounts = patentData.datasets[0].data;
        
//         // Process publication data
//         const pubCountsMap = new Map<number, number>();
//         pubData.forEach((item: { year: number; count: number }) => {
//           pubCountsMap.set(item.year, item.count);
//         });
        
//         // Combine years from both datasets
//         const allYears = Array.from(
//           new Set([
//             ...patentYears,
//             ...pubData.map((item: { year: number }) => item.year)
//           ])
//         ).sort((a, b) => a - b);
        
//         // Align data to common years
//         const alignedPatentData = allYears.map(year => 
//           patentYears.includes(year) 
//             ? patentCounts[patentYears.indexOf(year)] 
//             : 0
//         );
        
//         const alignedPubData = allYears.map(year => 
//           pubCountsMap.get(year) || 0
//         );
        
//         // Create chart
//         if (chartRef.current) {
//           const ctx = chartRef.current.getContext('2d');
//           if (ctx) {
//             // Destroy previous chart if exists
//             if (chartInstance.current) {
//               chartInstance.current.destroy();
//             }
            
//             chartInstance.current = new Chart(ctx, {
//               type: 'line',
//               data: {
//                 labels: allYears,
//                 datasets: [
//                   {
//                     label: 'Patent Filings',
//                     data: alignedPatentData,
//                     borderColor: 'rgb(75, 192, 192)',
//                     backgroundColor: 'rgba(75, 192, 192, 0.1)',
//                     tension: 0.1,
//                     yAxisID: 'y'
//                   },
//                   {
//                     label: 'Publications',
//                     data: alignedPubData,
//                     borderColor: 'rgb(255, 99, 132)',
//                     backgroundColor: 'rgba(255, 99, 132, 0.1)',
//                     tension: 0.1,
//                     yAxisID: 'y1'
//                   }
//                 ]
//               },
//               options: {
//                 responsive: true,
//                 interaction: {
//                   mode: 'index',
//                   intersect: false,
//                 },
//                 scales: {
//                   x: {
//                     title: {
//                       display: true,
//                       text: 'Year'
//                     }
//                   },
//                   y: {
//                     type: 'linear',
//                     display: true,
//                     position: 'left',
//                     title: {
//                       display: true,
//                       text: 'Patents'
//                     },
//                     min: 0
//                   },
//                   y1: {
//                     type: 'linear',
//                     display: true,
//                     position: 'right',
//                     title: {
//                       display: true,
//                       text: 'Publications'
//                     },
//                     min: 0,
//                     grid: {
//                       drawOnChartArea: false,
//                     },
//                   }
//                 },
//                 plugins: {
//                   title: {
//                     display: true,
//                     text: 'Patent vs Publication Trends'
//                   },
//                   tooltip: {
//                     callbacks: {
//                       label: (context) => {
//                         const label = context.dataset.label || '';
//                         const value = context.parsed.y || 0;
//                         return `${label}: ${value}`;
//                       }
//                     }
//                   }
//                 }
//               }
//             });
//           }
//         }
//       } catch (error) {
//         console.error('Error fetching data:', error);
//       }
//     };

//     fetchData();

//     // Cleanup function
//     return () => {
//       if (chartInstance.current) {
//         chartInstance.current.destroy();
//       }
//     };
//   }, []);

//   return (
//     <div className="chart-container">
//       <canvas ref={chartRef} />
//     </div>
//   );
// };

// export default PatentPublicationChart;







import { useEffect, useRef } from 'react';
import { Chart } from 'chart.js/auto';

const PatentPublicationChart = () => {
  const chartRef = useRef<HTMLCanvasElement>(null);
  const chartInstance = useRef<Chart | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        // Get backend port
        const portResponse = await fetch("/backend_port.txt");
        const port = (await portResponse.text()).trim();
        
        // Fetch patent data
        const patentRes = await fetch(`http://localhost:${port}/api/patents/first_filing_years`);
        const patentData = await patentRes.json();
        
        // Fetch publication data
        const pubRes = await fetch(`http://localhost:${port}/api/research_publications_by_year`);
        const pubData = await pubRes.json();
        
        // Process patent data
        const patentYears = patentData.labels;
        const patentCounts = patentData.datasets[0].data;
        
        // Process publication data
        const pubCountsMap = new Map<number, number>();
        pubData.forEach((item: { year: number; count: number }) => {
          pubCountsMap.set(item.year, item.count);
        });
        
        // Combine years from both datasets
        const allYears = Array.from(
          new Set([
            ...patentYears,
            ...pubData.map((item: { year: number }) => item.year)
          ])
        ).sort((a, b) => a - b);
        
        // Align data to common years
        const alignedPatentData = allYears.map(year => 
          patentYears.includes(year) 
            ? patentCounts[patentYears.indexOf(year)] 
            : 0
        );
        
        const alignedPubData = allYears.map(year => 
          pubCountsMap.get(year) || 0
        );
        
        // Create chart
        if (chartRef.current) {
          const ctx = chartRef.current.getContext('2d');
          if (ctx) {
            // Destroy previous chart if exists
            if (chartInstance.current) {
              chartInstance.current.destroy();
            }
            
            chartInstance.current = new Chart(ctx, {
              type: 'line',
              data: {
                labels: allYears,
                datasets: [
                  {
                    label: 'Patent Filings',
                    data: alignedPatentData,
                    borderColor: 'rgb(75, 192, 192)',
                    backgroundColor: 'rgba(75, 192, 192, 0.1)',
                    tension: 0.1,
                    borderWidth: 2,
                  },
                  {
                    label: 'Publications',
                    data: alignedPubData,
                    borderColor: 'rgb(255, 99, 132)',
                    backgroundColor: 'rgba(255, 99, 132, 0.1)',
                    tension: 0.1,
                    borderWidth: 2,
                  }
                ]
              },
              options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                  mode: 'index',
                  intersect: false,
                },
                scales: {
                  x: {
                    title: {
                      display: true,
                      text: 'Year',
                      font: {
                        weight: 'bold',
                        size: 14
                      }
                    },
                    grid: {
                      display: false
                    }
                  },
                  y: {
                    type: 'linear',
                    display: true,
                    title: {
                      display: true,
                      text: 'Count',
                      font: {
                        weight: 'bold',
                        size: 14
                      }
                    },
                    min: 0,
                    ticks: {
                      stepSize: 5
                    }
                  }
                },
                plugins: {
                  // title: {
                  //   display: true,
                  //   text: 'Patent Filings vs Publications',
                  //   font: {
                  //     size: 18,
                  //     weight: 'bold'
                  //   },
                  //   padding: {
                  //     top: 10,
                  //     bottom: 20
                  //   }
                  // },
                  legend: {
                    position: 'top',
                    labels: {
                      boxWidth: 15,
                      padding: 15,
                      font: {
                        size: 13
                      }
                    }
                  },
                  tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.7)',
                    padding: 10,
                    titleFont: {
                      size: 13
                    },
                    bodyFont: {
                      size: 13
                    }
                  }
                }
              }
            });
          }
        }
      } catch (error) {
        console.error('Error fetching data:', error);
      }
    };

    fetchData();

    // Cleanup function
    return () => {
      if (chartInstance.current) {
        chartInstance.current.destroy();
      }
    };
  }, []);

  return (
    <div className="chart-container" style={{ position: 'relative', height: '60vh', width: '100%' }}>
      <canvas ref={chartRef} />
    </div>
  );
};

export default PatentPublicationChart;