// "use client"
// import html2canvas from "html2canvas"
// import { useState } from "react"
// import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
// import { Button } from "@/components/ui/button"
// import { Checkbox } from "@/components/ui/checkbox"
// import { Textarea } from "@/components/ui/textarea"
// import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
// import { Download, Eye, FileText, Presentation } from "lucide-react"

// const availableCharts = [
//   { id: "publication-trend", title: "Publication Trend", type: "summary" },
//   { id: "applicant-analysis", title: "Applicant Analysis", type: "trend" },
//   { id: "innovation-cycle", title: "Innovation Cycle", type: "summary" },
//   { id: "market-size", title: "Market Size", type: "summary" },
//   { id: "forecast-chart", title: "Technology Forecast", type: "forecast" },
// ]

// const savedReports = [
//   { id: "1", title: "Q4 2023 Technology Analysis", date: "2023-12-15", type: "PDF" },
//   { id: "2", title: "Patent Landscape Report", date: "2023-11-20", type: "PowerPoint" },
//   { id: "3", title: "Innovation Trends Summary", date: "2023-10-30", type: "PDF" },
// ]

// export default function Reporting() {
//   const { selectedCharts, setSelectedCharts, chartComments, setChartComments } = useChartContext();

//   const generatePptx = async () => {
//     const ppt = new PptxGenJS()
//     const chartDivs = Array.from(
//       document.querySelectorAll<HTMLDivElement>('[id^="chart-"]')
//     )
//     for (const div of chartDivs) {
//       const canvas = await html2canvas(div)
//       const imgData = canvas.toDataURL("image/png")
//       const slide = ppt.addSlide()
//       slide.addImage({ data: imgData, x: 0.5, y: 0.5, w: 9, h: 5 })
//       slide.addText(div.id, { x: 0.5, y: 5.6, fontSize: 10 })
//     }
//     await ppt.writeFile({ fileName: "charts-report.pptx" })
//   }

//   const toggleChart = (chartId: string) => {
//     setSelectedCharts((prev: string[]) =>
//       prev.includes(chartId) ? prev.filter((id) => id !== chartId) : [...prev, chartId]
//     );
//   };

//   const updateComment = (chartId: string, comment: string) => {
//     setChartComments((prev: Record<string, string>) => ({ ...prev, [chartId]: comment }));
//   };

//   const generateReport = (format: "pdf" | "pptx") => {
//     console.log("Generating report:", { format, selectedCharts, chartComments });
//     // Here you would call your backend API to generate the report
//   };

//   return (
//     <div className="container mx-auto px-6 py-8">
//       <div className="mb-8">
//         <h1 className="text-3xl font-bold text-gray-900 mb-2">Report Generation</h1>
//         <p className="text-gray-600">Create comprehensive reports from your analysis</p>
//       </div>

//       <Tabs defaultValue="create" className="w-full">
//         <TabsList className="grid w-full grid-cols-2">
//           <TabsTrigger value="create">Create Report</TabsTrigger>
//           <TabsTrigger value="saved">Saved Reports</TabsTrigger>
//         </TabsList>

//         <TabsContent value="create" className="space-y-6">
//           <Card>
//             <CardHeader>
//               <CardTitle>Select Charts and Analysis</CardTitle>
//             </CardHeader>
//             <CardContent className="space-y-4">
//               {availableCharts.map((chart) => (
//                 <div key={chart.id} className="space-y-3">
//                   <div className="flex items-center space-x-2">
//                     <Checkbox
//                       id={chart.id}
//                       checked={selectedCharts.includes(chart.id)}
//                       onCheckedChange={() => toggleChart(chart.id)}
//                     />
//                     <label htmlFor={chart.id} className="text-sm font-medium">
//                       {chart.title} ({chart.type})
//                     </label>
//                   </div>

//                   {selectedCharts.includes(chart.id) && (
//                     <div className="ml-6">
//                       <Textarea
//                         placeholder="Add comments for this chart..."
//                         value={chartComments[chart.id] || ""}
//                         onChange={(e) => updateComment(chart.id, e.target.value)}
//                         className="w-full"
//                       />
//                     </div>
//                   )}
//                 </div>
//               ))}
//             </CardContent>
//           </Card>

//           <Card>
//             <CardHeader>
//               <CardTitle>Report Actions</CardTitle>
//             </CardHeader>
//             <CardContent className="flex space-x-4">
//               <Button variant="outline" className="flex items-center space-x-2 bg-transparent">
//                 <Eye className="h-4 w-4" />
//                 <span>Preview Report</span>
//               </Button>

//               <Button
//                 onClick={() => generateReport("pdf")}
//                 className="flex items-center space-x-2"
//                 disabled={selectedCharts.length === 0}
//               >
//                 <Download className="h-4 w-4" />
//                 <span>Download PDF</span>
//               </Button>

//               <Button
//                 onClick={() => generatePptx()}
//                 className="flex items-center space-x-2"
//                 disabled={selectedCharts.length === 0}
//               >
//                 <Download className="h-4 w-4" />
//                 <span>Download PowerPoint</span>
//               </Button>
//             </CardContent>
//           </Card>
//         </TabsContent>

//         <TabsContent value="saved" className="space-y-4">
//           <Card>
//             <CardHeader>
//               <CardTitle>Previously Generated Reports</CardTitle>
//             </CardHeader>
//             <CardContent>
//               <div className="space-y-4">
//                 {savedReports.map((report) => (
//                   <div key={report.id} className="flex items-center justify-between p-4 border rounded-lg">
//                     <div className="flex items-center space-x-3">
//                       {report.type === "PDF" ? (
//                         <FileText className="h-5 w-5 text-red-500" />
//                       ) : (
//                         <Presentation className="h-5 w-5 text-orange-500" />
//                       )}
//                       <div>
//                         <h3 className="font-medium">{report.title}</h3>
//                         <p className="text-sm text-gray-500">
//                           {report.date} • {report.type}
//                         </p>
//                       </div>
//                     </div>
//                     <div className="flex space-x-2">
//                       <Button variant="outline" size="sm">
//                         <Eye className="h-4 w-4 mr-1" />
//                         View
//                       </Button>
//                       <Button variant="outline" size="sm">
//                         <Download className="h-4 w-4 mr-1" />
//                         Download
//                       </Button>
//                     </div>
//                   </div>
//                 ))}
//               </div>
//             </CardContent>
//           </Card>
//         </TabsContent>
//       </Tabs>
//     </div>
//   )
// }







// working button , empty ppt 

// File: /mnt/data/page.tsx
// "use client"

// import html2canvas from "html2canvas"
// import { useState } from "react"
// import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
// import { Button } from "@/components/ui/button"
// import { Checkbox } from "@/components/ui/checkbox"
// import { Textarea } from "@/components/ui/textarea"
// import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
// import { Download, Eye, FileText, Presentation } from "lucide-react"

// // Available charts
// const availableCharts = [
//   { id: "publication-trend", title: "Publication Trend", type: "summary" },
//   { id: "applicant-analysis", title: "Applicant Analysis", type: "trend" },
//   { id: "innovation-cycle", title: "Innovation Cycle", type: "summary" },
//   { id: "market-size", title: "Market Size", type: "summary" },
//   { id: "forecast-chart", title: "Technology Forecast", type: "forecast" },
// ]

// const savedReports = [
//   { id: "1", title: "Q4 2023 Technology Analysis", date: "2023-12-15", type: "PDF" },
//   { id: "2", title: "Patent Landscape Report", date: "2023-11-20", type: "PowerPoint" },
//   { id: "3", title: "Innovation Trends Summary", date: "2023-10-30", type: "PDF" },
// ]

// export default function Reporting() {
//   const [selectedCharts, setSelectedCharts] = useState<string[]>([])
//   const [chartComments, setChartComments] = useState<Record<string, string>>({})

//   // Chart toggle
//   const toggleChart = (id: string) => {
//     setSelectedCharts((prev) =>
//       prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
//     )
//   }

//   // Comment updater
//   const updateComment = (id: string, text: string) => {
//     setChartComments((prev) => ({ ...prev, [id]: text }))
//   }

//   // PDF stub
//   const generateReport = (format: "pdf" | "pptx") => {
//     console.log("Generating report:", { format, selectedCharts, chartComments })
//     alert(`Report (${format}) is not implemented yet.`)
//   }

//   // PPTX via backend
//   const generatePptx = async () => {
//     console.log("generatePptx clicked", selectedCharts)
//     const divs = Array.from(document.querySelectorAll<HTMLDivElement>(`[id^="chart-"]`))
//     if (!divs.length) return alert("No charts to export.")
//     const images = await Promise.all(
//       divs.map(async (d) => ({ id: d.id, data: (await html2canvas(d)).toDataURL() }))
//     )
//     try {
//       // Dynamically get the backend port
//       const portRes = await fetch("/backend_port.txt");
//       const port = (await portRes.text()).trim();
//       const res = await fetch(`http://localhost:${port}/api/report/generate-pptx`, {
//         method: 'POST',
//         headers: { 'Content-Type': 'application/json' },
//         body: JSON.stringify({ images, comments: chartComments }),
//       })
//       if (!res.ok) throw new Error(await res.text())
//       const blob = await res.blob()
//       const url = URL.createObjectURL(blob)
//       const a = document.createElement('a')
//       a.href = url
//       a.download = 'charts-report.pptx'
//       a.click()
//     } catch (e) {
//       console.error(e)
//       alert('Export failed: ' + e)
//     }
//   }

//   return (
//     <div className="container mx-auto p-6">
//       <Tabs defaultValue="create" className="w-full">
//         <TabsList className="grid grid-cols-2">
//           <TabsTrigger value="create">Create Report</TabsTrigger>
//           <TabsTrigger value="saved">Saved Reports</TabsTrigger>
//         </TabsList>

//         <TabsContent value="create">
//           <Card className="mb-6">
//             <CardHeader>
//               <CardTitle>Select Charts</CardTitle>
//             </CardHeader>
//             <CardContent>
//               {availableCharts.map(({ id, title, type }) => (
//                 <div key={id} className="mb-4">
//                   <div className="flex items-center">
//                     <Checkbox
//                       id={id}
//                       checked={selectedCharts.includes(id)}
//                       onCheckedChange={() => toggleChart(id)}
//                     />
//                     <label htmlFor={id} className="ml-2">
//                       {title} ({type})
//                     </label>
//                   </div>
//                   {selectedCharts.includes(id) && (
//                     <div className="mt-2 ml-6" id={`chart-${id}`}> 
//                       <Textarea
//                         placeholder="Comment"
//                         value={chartComments[id] || ''}
//                         onChange={(e) => updateComment(id, e.target.value)}
//                       />
//                     </div>
//                   )}
//                 </div>
//               ))}
//             </CardContent>
//           </Card>

//           <Card>
//             <CardHeader>
//               <CardTitle>Actions</CardTitle>
//             </CardHeader>
//             <CardContent className="flex space-x-4">
//               <Button className="flex items-center space-x-2">
//                 <Eye /><span>Preview</span>
//               </Button>
//               <Button
//                 className="flex items-center space-x-2"
//                 onClick={() => generateReport('pdf')}
//                 disabled={!selectedCharts.length}
//               >
//                 <Download /><span>PDF</span>
//               </Button>
//               <Button
//                 className="flex items-center space-x-2"
//                 onClick={generatePptx}
//                 disabled={!selectedCharts.length}
//               >
//                 <Download /><span>PPTX</span>
//               </Button>
//             </CardContent>
//           </Card>
//         </TabsContent>

//         <TabsContent value="saved">
//           <Card>
//             <CardHeader>
//               <CardTitle>Saved Reports</CardTitle>
//             </CardHeader>
//             <CardContent>
//               {savedReports.map(({ id, title, date, type }) => (
//                 <div key={id} className="flex justify-between items-center mb-4">
//                   <div className="flex items-center">
//                     {type === 'PDF' ? <FileText /> : <Presentation />} 
//                     <div className="ml-2">
//                       <p>{title}</p>
//                       <p className="text-xs text-gray-500">{date}</p>
//                     </div>
//                   </div>
//                   <div className="flex space-x-2">
//                     <Button className="flex items-center"><Eye /></Button>
//                     <Button className="flex items-center"><Download /></Button>
//                   </div>
//                 </div>
//               ))}
//             </CardContent>
//           </Card>
//         </TabsContent>
//       </Tabs>
//     </div>
//   )
// }




// File: /mnt/data/page.tsx
// "use client"

// import html2canvas from "html2canvas"
// import dynamic from "next/dynamic"
// import { useState } from "react"
// import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
// import { Button } from "@/components/ui/button"
// import { Checkbox } from "@/components/ui/checkbox"
// import { Textarea } from "@/components/ui/textarea"
// import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
// import { Download, Eye, FileText, Presentation } from "lucide-react"
// import { useChartContext } from "../providers/ChartContext";

// // Map chart IDs to dynamically imported components
// const ChartRegistry: Record<string, React.ComponentType<{width: number; height: number}>> = {
//   "publication-trend": dynamic<{ width: number; height: number }>(() => import("@/components/publication_trends")),
//   "applicant-analysis": dynamic<{ width: number; height: number }>(() => import("@/components/applicant_type_pie")),
//   // "innovation-cycle": dynamic<{ width: number; height: number }>(() => import("@/components/innovation-cycle")),
//   "market-size": dynamic<{ width: number; height: number }>(() => import("@/components/IPStat")),
// }

// // Available charts metadata
// const availableCharts = [
//   { id: "publication-trend", title: "Publication Trend", type: "summary" },
//   { id: "applicant-analysis", title: "Applicant Analysis", type: "trend" },
//   // { id: "innovation-cycle", title: "Innovation Cycle", type: "summary" },
//   { id: "market-size", title: "Market Size", type: "summary" },
//   { id: "forecast-chart", title: "Technology Forecast", type: "forecast" },
// ]

// const savedReports = [
//   { id: "1", title: "Q4 2023 Technology Analysis", date: "2023-12-15", type: "PDF" },
//   { id: "2", title: "Patent Landscape Report", date: "2023-11-20", type: "PowerPoint" },
//   { id: "3", title: "Innovation Trends Summary", date: "2023-10-30", type: "PDF" },
// ]

// export default function Reporting() {
//   const [selectedCharts, setSelectedCharts] = useState<string[]>([])
//   const [chartComments, setChartComments] = useState<Record<string, string>>({})
  

//   // Toggle chart selection
//   const toggleChart = (id: string) => {
//     setSelectedCharts(prev =>
//       prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]
//     )
//   }

//   // Update comments
//   const updateComment = (id: string, text: string) => {
//     setChartComments(prev => ({ ...prev, [id]: text }))
//   }

//   // Placeholder PDF report
//   const generateReport = (format: "pdf" | "pptx") => {
//     alert(`Report (${format}) is not implemented yet.`)
//   }

//   // Export PPTX via backend
//   const generatePptx = async () => {
//     const divs = Array.from(document.querySelectorAll<HTMLDivElement>(`[id^="chart-"]`))
//     if (!divs.length) return alert("No charts to export.")
//     const images = await Promise.all(
//       divs.map(async d => ({ id: d.id, data: (await html2canvas(d)).toDataURL() }))
//     )
//     try {
//             // Dynamically get the backend port
//             const portRes = await fetch("/backend_port.txt");
//             const port = (await portRes.text()).trim();
//             const res = await fetch(`http://localhost:${port}/api/report/generate-pptx`, {
//               method: 'POST',
//               headers: { 'Content-Type': 'application/json' },
//               body: JSON.stringify({ images, comments: chartComments }),
//             })
//       if (!res.ok) throw new Error(await res.text())
//       const blob = await res.blob()
//       const url = URL.createObjectURL(blob)
//       const a = document.createElement('a')
//       a.href = url; a.download = 'charts-report.pptx'; a.click()
//     } catch (e) {
//       console.error(e);
//       alert('Export failed: ' + e)
//     }
//   }

//   return (
//     <div className="container mx-auto p-6">
//       <Tabs defaultValue="create" className="w-full">
//         <TabsList className="grid grid-cols-2">
//           <TabsTrigger value="create">Create Report</TabsTrigger>
//           <TabsTrigger value="saved">Saved Reports</TabsTrigger>
//         </TabsList>

//         {/* Create Report Tab */}
//         <TabsContent value="create" className="space-y-6">
//           <Card>
//             <CardHeader><CardTitle>Select Charts</CardTitle></CardHeader>
//             <CardContent className="space-y-4">
//               {availableCharts.map(({ id, title, type }) => (
//                 <div key={id} className="space-y-2">
//                   <div className="flex items-center">
//                     <Checkbox
//                       id={id}
//                       checked={selectedCharts.includes(id)}
//                       onCheckedChange={() => toggleChart(id)}
//                     />
//                     <label htmlFor={id} className="ml-2 font-medium">
//                       {title} ({type})
//                     </label>
//                   </div>
//                   {/* Render chart preview when selected */}
//                   {selectedCharts.includes(id) && (
//                     <div className="p-4 border rounded" id={`chart-${id}`}>
//                       {/* Dynamic chart component */}
//                       {ChartRegistry[id] ? (
//                         (() => {
//                           const ChartComponent = ChartRegistry[id];
//                           return <ChartComponent width={600} height={400} />;
//                         })()
//                       ) : (
//                         <p>Chart not available</p>
//                       )}
//                       {/* Comment box */}
//                       <Textarea
//                         placeholder="Add comment"
//                         value={chartComments[id] || ''}
//                         onChange={e => updateComment(id, e.target.value)}
//                         className="mt-2 w-full"
//                       />
//                     </div>
//                   )}
//                 </div>
//               ))}
//             </CardContent>
//           </Card>

//           <Card>
//             <CardHeader><CardTitle>Actions</CardTitle></CardHeader>
//             <CardContent className="flex space-x-4">
//               <Button onClick={() => generateReport('pdf')} className="flex items-center space-x-2" disabled={!selectedCharts.length}>
//                 <Download /><span>PDF</span>
//               </Button>
//               <Button onClick={generatePptx} className="flex items-center space-x-2" disabled={!selectedCharts.length}>
//                 <Download /><span>Export PPTX</span>
//               </Button>
//             </CardContent>
//           </Card>
//         </TabsContent>

//         {/* Saved Reports Tab */}
//         <TabsContent value="saved">
//           <Card>
//             <CardHeader><CardTitle>Saved Reports</CardTitle></CardHeader>
//             <CardContent>
//               {savedReports.map(({ id, title, date, type }) => (
//                 <div key={id} className="flex justify-between items-center mb-4">
//                   <div className="flex items-center">
//                     {type === 'PDF' ? <FileText /> : <Presentation />}  
//                     <div className="ml-2">
//                       <p>{title}</p>
//                       <p className="text-xs text-gray-500">{date}</p>
//                     </div>
//                   </div>
//                   <div className="flex space-x-2">
//                     <Button className="flex items-center space-x-1"><Eye /></Button>
//                     <Button className="flex items-center space-x-1"><Download /></Button>
//                   </div>
//                 </div>
//               ))}
//             </CardContent>
//           </Card>
//         </TabsContent>
//       </Tabs>
//     </div>
//   )
// }









// File: /app/reporting/page.tsx
"use client"

import html2canvas from "html2canvas"
import dynamic from "next/dynamic"
import { useChartContext } from "../providers/ChartContext"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import { Textarea } from "@/components/ui/textarea"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Download, Eye, FileText, Presentation } from "lucide-react"

// Map chart IDs to dynamically imported components
const ChartComponents = {
  "publication-trend": dynamic<{ width: number; height: number }>(
    () => import("@/components/publication_trends").then(mod => mod.default || mod),
    { ssr: false }
  ),
  "applicant-analysis": dynamic<{ width: number; height: number }>(
    () => import("@/components/applicant_type_pie").then(mod => mod.default || mod),
    { ssr: false }
  ),
  "market-size": dynamic<{ width: number; height: number }>(
    () => import("@/components/IPStat").then(mod => mod.default || mod),
    { ssr: false }
  ),
  // add more chart mappings as needed...
} as const;

type ChartId = keyof typeof ChartComponents;

const availableCharts = [
  { id: "publication-trend", title: "Publication Trend", type: "summary" },
  { id: "applicant-analysis", title: "Applicant Analysis", type: "trend" },
  { id: "market-size", title: "Market Size", type: "summary" },
  { id: "forecast-chart", title: "Technology Forecast", type: "forecast" },
]

const savedReports = [
  { id: "1", title: "Q4 2023 Technology Analysis", date: "2023-12-15", type: "PDF" },
  { id: "2", title: "Patent Landscape Report", date: "2023-11-20", type: "PowerPoint" },
  { id: "3", title: "Innovation Trends Summary", date: "2023-10-30", type: "PDF" },
]

export default function Reporting() {
  // pull shared state from context
  const {
    selectedCharts,
    setSelectedCharts,
    chartComments,
    setChartComments,
  } = useChartContext()

  // Toggle chart selection
  const toggleChart = (id: string) => {
    setSelectedCharts(prev =>
      prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]
    )
  }

  // Update comments
  const updateComment = (id: string, text: string) => {
    setChartComments(prev => ({ ...prev, [id]: text }))
  }

  // Placeholder PDF export
  const generateReport = (format: "pdf" | "pptx") => {
    alert(`Report (${format}) is not implemented yet.`)
  }

  // Export PPTX via backend
  const generatePptx = async () => {
    const divs = Array.from(
      document.querySelectorAll<HTMLDivElement>(`[id^="chart-"]`)
    )
    if (!divs.length) return alert("No charts to export.")
    const images = await Promise.all(
      divs.map(async d => ({
        id: d.id,
        data: (await html2canvas(d)).toDataURL(),
      }))
    )
    try {
      // dynamically read backend port
      const portRes = await fetch("/backend_port.txt")
      const port = (await portRes.text()).trim()
      const res = await fetch(`http://localhost:${port}/api/report/generate-pptx`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ images, comments: chartComments }),
      })
      if (!res.ok) throw new Error(await res.text())
      const blob = await res.blob()
      const url = URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = "charts-report.pptx"
      a.click()
    } catch (e) {
      console.error(e)
      alert("Export failed: " + e)
    }
  }

  return (
    <div className="container mx-auto p-6">
      <Tabs defaultValue="create" className="w-full">
        <TabsList className="grid grid-cols-2">
          <TabsTrigger value="create">Create Report</TabsTrigger>
          <TabsTrigger value="saved">Saved Reports</TabsTrigger>
        </TabsList>

        {/* Create Report Tab */}
        <TabsContent value="create" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Select Charts</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {availableCharts.map(({ id, title, type }) => (
                <div key={id} className="space-y-2">
                  <div className="flex items-center">
                    <Checkbox
                      id={id}
                      checked={selectedCharts.includes(id)}
                      onCheckedChange={() => toggleChart(id)}
                    />
                    <label htmlFor={id} className="ml-2 font-medium">
                      {title} ({type})
                    </label>
                  </div>
                  {selectedCharts.includes(id) && (
                    <div className="p-4 border rounded" id={`chart-${id}`}>
                      {(() => {
                        if (id in ChartComponents) {
                          const ChartComponent = ChartComponents[id as ChartId];
                          return (
                            <>
                              <ChartComponent width={600} height={400} />
                              <Textarea
                                placeholder="Add comment"
                                value={chartComments[id] || ""}
                                onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => updateComment(id, e.target.value)}
                                className="mt-2 w-full"
                              />
                            </>
                          );
                        }
                        return <p>Chart not available</p>;
                      })()}
                    </div>
                  )}
                </div>
              ))}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Actions</CardTitle>
            </CardHeader>
            <CardContent className="flex space-x-4">
              <Button
                disabled={!selectedCharts.length}
                onClick={() => generateReport("pdf")}
                className="flex items-center space-x-2"
              >
                <Download />
                <span>PDF</span>
              </Button>
              <Button
                disabled={!selectedCharts.length}
                onClick={generatePptx}
                className="flex items-center space-x-2"
              >
                <Download />
                <span>Export PPTX</span>
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Saved Reports Tab */}
        <TabsContent value="saved">
          <Card>
            <CardHeader>
              <CardTitle>Saved Reports</CardTitle>
            </CardHeader>
            <CardContent>
              {savedReports.map(({ id, title, date, type }) => (
                <div
                  key={id}
                  className="flex justify-between items-center mb-4"
                >
                  <div className="flex items-center">
                    {type === "PDF" ? <FileText /> : <Presentation />}
                    <div className="ml-2">
                      <p>{title}</p>
                      <p className="text-xs text-gray-500">{date}</p>
                    </div>
                  </div>
                  <div className="flex space-x-2">
                    <Button className="flex items-center space-x-1">
                      <Eye />
                    </Button>
                    <Button className="flex items-center space-x-1">
                      <Download />
                    </Button>
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
