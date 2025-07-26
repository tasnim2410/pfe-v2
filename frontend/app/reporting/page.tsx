


// /* app/reporting/page.tsx */
// "use client"

// import { useState, useEffect } from "react"
// import { DndContext, DragEndEvent, closestCenter } from "@dnd-kit/core"
// import { arrayMove, SortableContext, useSortable } from "@dnd-kit/sortable"
// import { CSS } from "@dnd-kit/utilities"
// import html2canvas from "html2canvas"
// import { 
//   Download, Eye, FileText, Presentation, 
//   LayoutGrid, Lock, Unlock, Plus, Trash 
// } from "lucide-react"
// import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
// import { Checkbox } from "@/components/ui/checkbox";
// import { Textarea } from "@/components/ui/textarea";
// import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

// import { Button } from '@/components/ui/button'
// import { useChartContext } from "../providers/ChartContext"

// // Function to render the appropriate chart based on chartId
// const renderChart = (chartId: string) => {
//   switch (chartId) {
//     case 'publication-trends':
//       return <PublicationTrends />;
//     case 'evolving-word-cloud':
//       return <EvolvingWordCloud />;
//     case 'applicant-type-pie':
//       return <ApplicantTypePie />;
//     case 'ip-stats':
//       return <IpStatsBox />;
//     case 'top-10-applicants':
//       return <Top10Applicants />;
//     case 'top-ipc-codes':
//       return <TopIPCCodes />;
//     case 'top-keywords':
//       return <TopKeywords />;
//     case 'patent-field-trends':
//       return <PatentFieldTrends />;
//     case 'cooccurrence-trends':
//       return <CooccurrenceTrends />;
//     default:
//       return <div className="text-red-500">Chart not found</div>;
//   }
// };
// import PublicationTrends from "@/components/publication_trends"; // <-- ADDED
// import EvolvingWordCloud from "@/components/evolving_word_cloud";
// import ApplicantTypePie from "@/components/applicant_type_pie";
// import IpStatsBox from "@/components/IPStat";
// import Top10Applicants from "@/components/top_10_app";
// import { TopIPCCodes } from "@/components/top_ipc_codes";
// import TopKeywords from "@/components/top_10_keywords";
// import PatentFieldTrends from "@/components/patent_by_field";
// import CooccurrenceTrends from "@/components/co-occurunece_trend";
// import ApplicantCollaborationNetwork from "@/components/collaboration_network";
// import OriginalityRate from "@/components/originality_rate";
// import FamilyMemberCountChart from "@/components/family-member-count";
// import FamilySizeDistributionChart from "@/components/family-size-distribution";
// import InternationalProtectionMatrixChart from "@/components/international-protection-matrix";
// import InternationalPatentFlowChart from "@/components/international-patent-flow";
// import GeographicalDistribution from "@/components/geographical_distribution";  
// import jsPDF from "jspdf";

// import PptxGenJS from 'pptxgenjs';

// // Chart palette definitions for reporting
// const availableCharts = [
//   { id: "publication-trend", title: "Publication Trend", type: "line" },
//   { id: "applicant-analysis", title: "Applicant Type Analysis", type: "pie" },
//   { id: "top-ipc", title: "Top IPC Codes", type: "bar" },
//   { id: "top-10-applicants", title: "Top 10 Applicants", type: "bar" },
//   { id: "top-10-keywords", title: "Top 10 Keywords", type: "bar" },
//   { id: "patent-field-trends", title: "Patent Field Trends", type: "line" },
//   { id: "evolving-word-cloud", title: "Evolving Word Cloud", type: "wordcloud" },
//   { id: "family-member-count", title: "Family Member Count", type: "bar" },
//   { id: "family-size-distribution", title: "Family Size Distribution", type: "histogram" },
//   { id: "international-protection-matrix", title: "International Protection Matrix", type: "matrix" },
//   { id: "international-patent-flow", title: "International Patent Flow", type: "flow" },
//   { id: "geographic-distribution", title: "Geographic Distribution", type: "map" },
//   // Add more as needed
// ];

// // Define template types
// type Template = {
//   id: string
//   name: string
//   dimensions: { width: number; height: number }
//   slots: Slot[]
// }

// type Slot = {
//   id: string
//   chartId: string | null
//   comment: string
//   width?: number
//   height?: number
//   x?: number
//   y?: number
// }

// // ChartId type alias
// type ChartId = string;

// // Predefined templates
// const TEMPLATES: Template[] = [
//   {
//     id: "executive-pdf",
//     name: "A4 Executive PDF",
//     dimensions: { width: 595, height: 842 }, // A4 in points (pt)
//     slots: [
//       { id: "s1", chartId: null, comment: "", x: 40, y: 40, width: 515, height: 200 },
//       { id: "s2", chartId: null, comment: "", x: 40, y: 260, width: 250, height: 200 },
//       { id: "s3", chartId: null, comment: "", x: 305, y: 260, width: 250, height: 200 },
//       { id: "s4", chartId: null, comment: "", x: 40, y: 480, width: 515, height: 300 },
//     ]
//   },
//   {
//     id: "landscape-pdf",
//     name: "A4 Landscape PDF",
//     dimensions: { width: 842, height: 595 }, 
//     slots: [
//       { id: "s1", chartId: null, comment: "", x: 40, y: 40, width: 762, height: 200 },
//       { id: "s2", chartId: null, comment: "", x: 40, y: 260, width: 380, height: 295 },
//       { id: "s3", chartId: null, comment: "", x: 440, y: 260, width: 380, height: 295 },
//     ]
//   },
//   {
//     id: "ppt-deck",
//     name: "16×9 PPT Deck",
//     dimensions: { width: 1280, height: 720 }, 
//     slots: [
//       { id: "s1", chartId: null, comment: "", x: 40, y: 40, width: 1200, height: 640 },
//     ]
//   }
// ]



// export default function Reporting() {
//   const { selectedCharts, setSelectedCharts, chartComments, setChartComments } = useChartContext()
//   const [selectedTemplate, setSelectedTemplate] = useState<Template | null>(null)
//   const [layout, setLayout] = useState<Slot[]>([])
//   const [isLayoutUnlocked, setIsLayoutUnlocked] = useState(false)
//   const [activeId, setActiveId] = useState<string | null>(null)

//   // Initialize layout when template is selected
//   useEffect(() => {
//     if (selectedTemplate) {
//       setLayout([...selectedTemplate.slots])
//     }
//   }, [selectedTemplate])

//   // Handle drag end events
//   const handleDragEnd = (event: DragEndEvent) => {
//     const { active, over } = event
    
//     if (over?.id && active.id !== over.id) {
//       const activeChartId = active.id.toString()
//       const slotId = over.id.toString()
      
//       setLayout(prev => prev.map(slot => 
//         slot.id === slotId ? { ...slot, chartId: activeChartId } : slot
//       ))
//     }
    
//     setActiveId(null)
//   }

//   // Add new slot in custom layout mode
//   const addNewSlot = () => {
//     const newSlot: Slot = {
//       id: `slot-${Date.now()}`,
//       chartId: null,
//       comment: "",
//       x: 50,
//       y: 50,
//       width: 200,
//       height: 150
//     }
//     setLayout([...layout, newSlot])
//   }

//   // Remove a slot
//   const removeSlot = (slotId: string) => {
//     setLayout(layout.filter(slot => slot.id !== slotId))
//   }

//   // Update slot comment
//   const updateSlotComment = (slotId: string, comment: string) => {
//     setLayout(prev => 
//       prev.map(slot => 
//         slot.id === slotId ? { ...slot, comment } : slot
//       )
//     )
//   }

//   // Toggle chart selection
//   const toggleChart = (id: ChartId) => {
//     setSelectedCharts(prev =>
//       prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]
//     )
//   }

//   // Export report as PDF
// const generatePdf = async () => {
//   if (!selectedTemplate) return;
//   const previewNode = document.querySelector('.relative.mx-auto.bg-white.border');
//   if (!previewNode) return;

//   const canvas = await html2canvas(previewNode as HTMLElement, { scale: 2 });
//   const imgData = canvas.toDataURL('image/png');
//   const pdf = new jsPDF({
//     orientation:
//       selectedTemplate.dimensions.width > selectedTemplate.dimensions.height
//         ? 'landscape'
//         : 'portrait',
//     unit: 'pt',
//     format: [selectedTemplate.dimensions.width, selectedTemplate.dimensions.height],
//   });

//   pdf.addImage(
//     imgData,
//     'PNG',
//     0,
//     0,
//     selectedTemplate.dimensions.width,
//     selectedTemplate.dimensions.height
//   );
//   pdf.save(`${selectedTemplate.name}.pdf`);
// };

// // Export report as PowerPoint
// // ─── Replace your old generatePptx with this ─────────────────────────────────

// const generatePptx = async () => {
//   // 1. Grab all rendered charts
//   const divs = Array.from(
//     document.querySelectorAll<HTMLDivElement>('[id^="chart-"]')
//   );
//   if (!divs.length) {
//     return alert("No charts to export.");
//   }

//   // 2. Screenshot each one
//   const images = await Promise.all(
//     divs.map(async (d) => ({
//       // strip off the "chart-" prefix so your backend can key off the raw id
//       id: d.id.replace(/^chart-/, ""),
//       data: (await html2canvas(d)).toDataURL("image/png"),
//     }))
//   );

//   try {
//     // 3. Find the port your local service is running on
//     const port = (await (await fetch("/backend_port.txt")).text()).trim();

//     // 4. POST images + comments to your API
//     const res = await fetch(`http://localhost:${port}/api/report/generate-pptx`, {
//       method: "POST",
//       headers: { "Content-Type": "application/json" },
//       body: JSON.stringify({ images, comments: chartComments }),
//     });

//     if (!res.ok) {
//       throw new Error(await res.text());
//     }

//     // 5. Download the blob as a .pptx
//     const blob = await res.blob();
//     const url = URL.createObjectURL(blob);
//     const a = document.createElement("a");
//     a.href = url;
//     a.download = "charts-report.pptx";
//     a.click();
//   } catch (e) {
//     console.error(e);
//     alert("Export failed: " + e);
//   }
// };


//   // Sortable slot component
//   const SortableSlot = ({ slot }: { slot: Slot }) => {
//     const { 
//       attributes, 
//       listeners, 
//       setNodeRef, 
//       transform, 
//       transition 
//     } = useSortable({ id: slot.id })
    
//     const style = {
//       transform: CSS.Transform.toString(transform),
//       transition,
//     }

//     const renderChart = (chartId: string | null, width?: number, height?: number) => {
//   if (!chartId) return null;
//   switch (chartId) {
//     case "publication-trend":
//       return <PublicationTrends />;
//     case "applicant-analysis":
//       return (
//         <div className="w-full flex justify-center items-center">
//           <ApplicantTypePie />
//         </div>
//       );
//     case "top-ipc":
//       return <TopIPCCodes />;
//     case "top-10-applicants":
//       return <Top10Applicants />;
//     case "top-10-keywords":
//       return <TopKeywords />;
//     case "patent-field-trends":
//       return <PatentFieldTrends />;
//     case "evolving-word-cloud":
//       return <EvolvingWordCloud />;
//     case "cooccurrence-trends":
//       return <CooccurrenceTrends />;
//     case "applicant-collaboration-network":
//       return <ApplicantCollaborationNetwork />;
//     case "family-member-count":
//       return <FamilyMemberCountChart />;
//     case "family-size-distribution":  
//       return <FamilySizeDistributionChart />;
//     case "international-protection-matrix":
//       return <InternationalProtectionMatrixChart />;
//     case "international-patent-flow":
//       return <InternationalPatentFlowChart />;
//     case "geographic-distribution":
//       return <GeographicalDistribution />;
//     // Add additional chart mappings as needed
//     default:
//       return (
//         <div className="flex items-center justify-center h-64">
//           <p className="text-gray-500">Chart data for {chartId}</p>
//         </div>
//       );
//   }
// };


//     return (
//       <div
//         ref={setNodeRef}
//         style={style}
//         className={`relative border rounded p-2 bg-white ${
//           slot.chartId ? "" : "bg-gray-100 border-dashed"
//         }`}
//       >
//         {isLayoutUnlocked && (
//           <div className="absolute top-1 right-1 flex gap-1 z-10">
//             <button 
//               onClick={() => removeSlot(slot.id)}
//               className="p-1 bg-red-500 text-white rounded-full"
//             >
//               <Trash size={14} />
//             </button>
//             <button 
//               {...attributes}
//               {...listeners}
//               className="p-1 bg-gray-800 text-white rounded-full cursor-move"
//             >
//               <LayoutGrid size={14} />
//             </button>
//           </div>
//         )}
        
//         {slot.chartId ? (
//           <div className="h-full">
//             {slot.chartId ? (
//   <>
//     {renderChart(slot.chartId, slot.width, slot.height)}
//     <Textarea
//       placeholder="Add comment"
//       value={slot.comment}
//       onChange={(e) => updateSlotComment(slot.id, e.target.value)}
//       className="mt-2 w-full text-xs"
//     />
//   </>
// ) : (
//   <div className="bg-red-100 border border-red-300 p-4 text-red-700">
//     Data unavailable
//   </div>
// )}
//           </div>
//         ) : (
//           <div className="flex items-center justify-center h-full text-gray-400">
//             Drop chart here
//           </div>
//         )}
//       </div>
//     )
//   }

//   // Template selector modal
//   if (!selectedTemplate) {
//     return (
//       <div className="container mx-auto p-6">
//         <Card>
//           <CardHeader>
//             <CardTitle>Choose Report Template</CardTitle>
//           </CardHeader>
//           <CardContent className="grid grid-cols-1 md:grid-cols-3 gap-4">
//             {TEMPLATES.map(template => (
//               <div 
//                 key={template.id}
//                 className="border rounded-lg p-4 cursor-pointer hover:bg-gray-50"
//                 onClick={() => setSelectedTemplate(template)}
//               >
//                 <div className="text-lg font-medium">{template.name}</div>
//                 <div className="text-sm text-gray-500 mt-2">
//                   {template.dimensions.width} × {template.dimensions.height} pt
//                 </div>
//                 <div className="mt-4 flex justify-center">
//                   <div 
//                     className="relative bg-gray-100 border"
//                     style={{ 
//                       width: template.dimensions.width / 4, 
//                       height: template.dimensions.height / 4 
//                     }}
//                   >
//                     {template.slots.map(slot => (
//                       <div
//                         key={slot.id}
//                         className="absolute border border-dashed border-gray-400 bg-gray-50 bg-opacity-50"
//                         style={{
//                           left: slot.x! / 4,
//                           top: slot.y! / 4,
//                           width: slot.width! / 4,
//                           height: slot.height! / 4
//                         }}
//                       />
//                     ))}
//                   </div>
//                 </div>
//               </div>
//             ))}
//           </CardContent>
//         </Card>
//       </div>
//     )
//   }

//   // Calculate preview scale (70% of original)
//   const previewScale = 0.7
//   const previewWidth = selectedTemplate.dimensions.width * previewScale
//   const previewHeight = selectedTemplate.dimensions.height * previewScale

//   return (
//     <div className="container mx-auto p-6">
//       <div className="flex justify-between items-center mb-6">
//         <h1 className="text-2xl font-bold">
//           {selectedTemplate.name} Report
//         </h1>
//         <div className="flex gap-2">
//           <Button
//             variant={isLayoutUnlocked ? "default" : "outline"}
//             onClick={() => setIsLayoutUnlocked(!isLayoutUnlocked)}
//           >
//             {isLayoutUnlocked ? <><Unlock size={16} className="mr-2" /> Lock Layout</> : 
//                                 <><Lock size={16} className="mr-2" /> Unlock Layout</>}
//           </Button>
//           {isLayoutUnlocked && (
//             <Button onClick={addNewSlot}>
//               <Plus size={16} className="mr-2" /> Add Slot
//             </Button>
//           )}
//         </div>
//       </div>

//       <div className="flex gap-6">
//         {/* Chart Palette */}
//         <div className="w-1/4">
//           <Card>
//             <CardHeader>
//               <CardTitle>Charts</CardTitle>
//             </CardHeader>
//             <CardContent className="space-y-3">
//               {availableCharts.map(({ id, title, type }) => (
//                 <div 
//                   key={id}
//                   className="flex items-center p-3 border rounded hover:bg-gray-50 cursor-move"
//                   draggable
//                   onDragStart={(e) => {
//                     e.dataTransfer.setData("chartId", id)
//                     setActiveId(id)
//                   }}
//                 >
//                   <Checkbox
//                     id={id}
//                     checked={selectedCharts.includes(id)}
//                     onCheckedChange={() => toggleChart(id)}
//                     className="mr-3"
//                   />
//                   <label htmlFor={id} className="flex-1 font-medium">
//                     {title} <span className="text-gray-500 text-sm">({type})</span>
//                   </label>
//                 </div>
//               ))}
//             </CardContent>
//           </Card>
//         </div>

//         {/* Template Preview */}
//         <div className="flex-1">
//   <Card>
//     <CardHeader>
//       <CardTitle>Report Preview</CardTitle>
//     </CardHeader>
//     <CardContent>
//       <div 
//         className="relative mx-auto bg-white border"
//         style={{
//           width: previewWidth,
//           height: previewHeight,
//           maxWidth: "100%",
//           position: "relative"
//         }}
//       >
//         {layout.map(slot => (
//           <div
//             key={slot.id}
//             id={"chart-" + slot.chartId}
//             className="absolute border border-gray-300 bg-white shadow rounded overflow-hidden"
//             style={{
//               left: slot.x! * previewScale,
//               top: slot.y! * previewScale,
//               width: slot.width! * previewScale,
//               height: slot.height! * previewScale,
//               display: "flex",
//               flexDirection: "column"
//             }}
//           >
//             {isLayoutUnlocked && (
//               <div className="absolute top-1 right-1 z-10 flex gap-1">
//                 <button
//                   onClick={() => removeSlot(slot.id)}
//                   className="bg-red-500 text-white p-1 rounded-full"
//                 >
//                   <Trash size={14} />
//                 </button>
//               </div>
//             )}
//             {slot.chartId ? (
//               <div className="flex-1 overflow-auto">
//                 {renderChart(slot.chartId)}
//               </div>
//             ) : (
//               <div className="flex-1 flex items-center justify-center text-gray-400">
//                 Drop chart here
//               </div>
//             )}
//             <Textarea
//               placeholder="Add comment"
//               value={slot.comment}
//               onChange={(e) => updateSlotComment(slot.id, e.target.value)}
//               className="text-xs border-t mt-auto"
//             />
//           </div>
//         ))}
//       </div>
//     </CardContent>
//   </Card>
// </div>
//       </div>
//     </div>
//   )
// }







































"use client"

import { useState, useEffect, useRef, memo } from "react"
import { Trash, Plus, Lock, Unlock, Download, MessageCircle } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Checkbox } from "@/components/ui/checkbox"
import { Textarea } from "@/components/ui/textarea"
import { Button } from "@/components/ui/button"
import { useChartContext } from "../providers/ChartContext"
import PublicationTrends from "@/components/publication_trends"
import EvolvingWordCloud from "@/components/evolving_word_cloud"
import ApplicantTypePie from "@/components/applicant_type_pie"
import Top10Applicants from "@/components/top_10_app"
import { TopIPCCodes } from "@/components/top_ipc_codes"
import TopKeywords from "@/components/top_10_keywords"
import PatentFieldTrends from "@/components/patent_by_field"
import CooccurrenceTrends from "@/components/co-occurunece_trend"
import ApplicantCollaborationNetwork from "@/components/collaboration_network"
import FamilyMemberCountChart from "@/components/family-member-count"
import FamilySizeDistributionChart from "@/components/family-size-distribution"
import InternationalProtectionMatrixChart from "@/components/international-protection-matrix"
import InternationalPatentFlowChart from "@/components/international-patent-flow"
import GeographicalDistribution from "@/components/geographical_distribution"
import jsPDF from "jspdf"
import html2canvas from "html2canvas"

// Chart palette definitions
const availableCharts = [
  { id: "publication-trend", title: "Publication Trend", type: "line" },
  { id: "applicant-analysis", title: "Applicant Type Analysis", type: "pie" },
  { id: "top-ipc", title: "Top IPC Codes", type: "bar" },
  { id: "top-10-applicants", title: "Top 10 Applicants", type: "bar" },
  { id: "top-10-keywords", title: "Top 10 Keywords", type: "bar" },
  { id: "patent-field-trends", title: "Patent Field Trends", type: "line" },
  { id: "evolving-word-cloud", title: "Evolving Word Cloud", type: "wordcloud" },
  { id: "family-member-count", title: "Family Member Count", type: "bar" },
  { id: "family-size-distribution", title: "Family Size Distribution", type: "histogram" },
  { id: "international-protection-matrix", title: "International Protection Matrix", type: "matrix" },
  { id: "international-patent-flow", title: "International Patent Flow", type: "flow" },
  { id: "geographic-distribution", title: "Geographic Distribution", type: "map" },
]

type Template = {
  id: string
  name: string
  dimensions: { width: number; height: number }
  slots: Slot[]
}
type Slot = {
  id: string
  chartId: string | null
  x: number
  y: number
  width: number
  height: number
}
type StickyComment = {
  id: string
  text: string
  x: number
  y: number
  width: number
  height: number
}
const TEMPLATES: Template[] = [
  {
    id: "executive-pdf",
    name: "A4 Executive PDF",
    dimensions: { width: 595, height: 842 },
    slots: [
      { id: "s1", chartId: null, x: 40, y: 40, width: 515, height: 200 },
      { id: "s2", chartId: null, x: 40, y: 260, width: 250, height: 200 },
      { id: "s3", chartId: null, x: 305, y: 260, width: 250, height: 200 },
      { id: "s4", chartId: null, x: 40, y: 480, width: 515, height: 300 },
    ]
  },
  {
    id: "landscape-pdf",
    name: "A4 Landscape PDF",
    dimensions: { width: 842, height: 595 }, 
    slots: [
      { id: "s1", chartId: null, x: 40, y: 40, width: 762, height: 200 },
      { id: "s2", chartId: null, x: 40, y: 260, width: 380, height: 295 },
      { id: "s3", chartId: null, x: 440, y: 260, width: 380, height: 295 },
    ]
  },
  {
    id: "ppt-deck",
    name: "16×9 PPT Deck",
    dimensions: { width: 1280, height: 720 }, 
    slots: [
      { id: "s1", chartId: null, x: 40, y: 40, width: 1200, height: 640 },
    ]
  }
]

// Render chart by id (memoized)
const renderChart = (chartId: string | null) => {
  if (!chartId) return null
  switch (chartId) {
    case "publication-trend":
      return <PublicationTrends />
    case "applicant-analysis":
      return <ApplicantTypePie />
    case "top-ipc":
      return <TopIPCCodes />
    case "top-10-applicants":
      return <Top10Applicants />
    case "top-10-keywords":
      return <TopKeywords />
    case "patent-field-trends":
      return <PatentFieldTrends />
    case "evolving-word-cloud":
      return <EvolvingWordCloud />
    case "cooccurrence-trends":
      return <CooccurrenceTrends />
    case "applicant-collaboration-network":
      return <ApplicantCollaborationNetwork />
    case "family-member-count":
      return <FamilyMemberCountChart />
    case "family-size-distribution":
      return <FamilySizeDistributionChart />
    case "international-protection-matrix":
      return <InternationalProtectionMatrixChart />
    case "international-patent-flow":
      return <InternationalPatentFlowChart />
    case "geographic-distribution":
      return <GeographicalDistribution />
    default:
      return (
        <div className="flex items-center justify-center h-64">
          <p className="text-gray-500">Chart data for {chartId}</p>
        </div>
      )
  }
}
const ChartSlot = memo(
  function ChartSlot({ chartId }: { chartId: string | null }) {
    return renderChart(chartId)
  },
  (prev, next) => prev.chartId === next.chartId
)



function moveOverlappingSlots(movedSlot: Slot, prevLayout: Slot[], minGap = 16) {
  let layout = prevLayout.map(s => ({ ...s }))
  const idx = layout.findIndex(s => s.id === movedSlot.id)
  layout[idx] = { ...movedSlot }
  let hasChanges = true
  while (hasChanges) {
    hasChanges = false
    for (let i = 0; i < layout.length; ++i) {
      for (let j = 0; j < layout.length; ++j) {
        if (i === j) continue
        const a = layout[i]
        const b = layout[j]
        if (
          a.x < b.x + b.width &&
          a.x + a.width > b.x &&
          a.y < b.y + b.height &&
          a.y + a.height > b.y
        ) {
          if (b.y < a.y + a.height) {
            layout[j] = {
              ...b,
              y: a.y + a.height + minGap
            }
            hasChanges = true
          }
        }
      }
    }
  }
  return layout
}

const MIN_SLOT_WIDTH = 120
const MIN_SLOT_HEIGHT = 90
const MIN_COMMENT_WIDTH = 140
const MIN_COMMENT_HEIGHT = 60
const PADDING = 32

export default function Reporting() {
  const { selectedCharts, setSelectedCharts } = useChartContext()
  const [selectedTemplate, setSelectedTemplate] = useState<Template | null>(null)
  const [layout, setLayout] = useState<Slot[]>([])
  const [comments, setComments] = useState<StickyComment[]>([])
  const [isLayoutUnlocked, setIsLayoutUnlocked] = useState(false)
  const [activeId, setActiveId] = useState<string | null>(null)

  // Initialize layout when template is selected
  useEffect(() => {
    if (selectedTemplate) setLayout([...selectedTemplate.slots])
  }, [selectedTemplate])

  // Add new slot in custom layout mode
  const addNewSlot = () => {
    const newSlot: Slot = {
      id: `slot-${Date.now()}`,
      chartId: null,
      x: 60,
      y: 60,
      width: 200,
      height: 150
    }
    setLayout([...layout, newSlot])
  }
  // Add new comment sticky note
  const addNewComment = () => {
    const newComment: StickyComment = {
      id: `comment-${Date.now()}`,
      text: "",
      x: 120,
      y: 120,
      width: 180,
      height: 80,
    }
    setComments([...comments, newComment])
  }
  // Remove a slot
  const removeSlot = (slotId: string) => {
    setLayout(layout.filter(slot => slot.id !== slotId))
  }
  // Remove a sticky note
  const removeComment = (commentId: string) => {
    setComments(comments.filter(c => c.id !== commentId))
  }
  // Update sticky comment text
  const updateCommentText = (id: string, text: string) => {
    setComments(comments => comments.map(c => c.id === id ? { ...c, text } : c))
  }

  // Toggle chart selection
  const toggleChart = (id: string) => {
    setSelectedCharts(prev =>
      prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]
    )
  }

  // Export report as PDF
  const pageWidth = Math.max(
    ...[...layout.map(slot => slot.x + slot.width), ...comments.map(c => c.x + c.width)],
    selectedTemplate?.dimensions.width || 600
  ) + PADDING
  const pageHeight = Math.max(
    ...[...layout.map(slot => slot.y + slot.height), ...comments.map(c => c.y + c.height)],
    selectedTemplate?.dimensions.height || 600
  ) + PADDING

  const generatePdf = async () => {
    if (!selectedTemplate) return
    const previewNode = document.querySelector('.preview-canvas')
    if (!previewNode) return
    const canvas = await html2canvas(previewNode as HTMLElement, { scale: 2 })
    const imgData = canvas.toDataURL('image/png')
    const pdf = new jsPDF({
      orientation:
        pageWidth > pageHeight
          ? 'landscape'
          : 'portrait',
      unit: 'pt',
      format: [pageWidth, pageHeight],
    })
    pdf.addImage(
      imgData,
      'PNG',
      0,
      0,
      pageWidth,
      pageHeight
    )
    pdf.save(`${selectedTemplate.name}.pdf`)
  }

  // --- SLOT DRAG LOGIC ---
  function handleSlotDrag(id: string, startX: number, startY: number) {
    const slotIdx = layout.findIndex(s => s.id === id)
    if (slotIdx === -1) return
    const slot = layout[slotIdx]
    const origX = slot.x
    const origY = slot.y
  
    function onMouseMove(e: MouseEvent) {
      const deltaX = e.clientX - startX
      const deltaY = e.clientY - startY
      setLayout(prev => {
        const updatedSlot = { ...slot, x: origX + deltaX, y: origY + deltaY }
        return moveOverlappingSlots(updatedSlot, prev)
      })
    }
    function onMouseUp() {
      window.removeEventListener("mousemove", onMouseMove)
      window.removeEventListener("mouseup", onMouseUp)
    }
    window.addEventListener("mousemove", onMouseMove)
    window.addEventListener("mouseup", onMouseUp)
  }
  
  
  // --- COMMENT DRAG LOGIC ---
  function handleCommentDrag(id: string, startX: number, startY: number) {
    const commentIdx = comments.findIndex(c => c.id === id)
    if (commentIdx === -1) return
    const sticky = comments[commentIdx]
    const origX = sticky.x
    const origY = sticky.y
    function onMouseMove(e: MouseEvent) {
      const deltaX = e.clientX - startX
      const deltaY = e.clientY - startY
      setComments(comments =>
        comments.map(c =>
          c.id === id
            ? { ...c, x: origX + deltaX, y: origY + deltaY }
            : c
        )
      )
    }
    function onMouseUp() {
      window.removeEventListener("mousemove", onMouseMove)
      window.removeEventListener("mouseup", onMouseUp)
    }
    window.addEventListener("mousemove", onMouseMove)
    window.addEventListener("mouseup", onMouseUp)
  }
  // --- SLOT RESIZE ---
  function handleSlotResizeMouseDown(slot: Slot, e: React.MouseEvent) {
    e.preventDefault()
    e.stopPropagation()
    const startX = e.clientX
    const startY = e.clientY
    const startW = slot.width
    const startH = slot.height
    function onMouseMove(moveEvent: MouseEvent) {
      const deltaX = moveEvent.clientX - startX
      const deltaY = moveEvent.clientY - startY
      const newWidth = Math.max(MIN_SLOT_WIDTH, startW + deltaX)
      const newHeight = Math.max(MIN_SLOT_HEIGHT, startH + deltaY)
      setLayout(prev => {
        const updatedSlot = { ...slot, width: newWidth, height: newHeight }
        return moveOverlappingSlots(updatedSlot, prev)
      })
    }
    function onMouseUp() {
      window.removeEventListener("mousemove", onMouseMove)
      window.removeEventListener("mouseup", onMouseUp)
    }
    window.addEventListener("mousemove", onMouseMove)
    window.addEventListener("mouseup", onMouseUp)
  }
  
  // --- COMMENT RESIZE ---
  function handleCommentResizeMouseDown(sticky: StickyComment, e: React.MouseEvent) {
    e.preventDefault()
    e.stopPropagation()
    const startX = e.clientX
    const startY = e.clientY
    const startW = sticky.width
    const startH = sticky.height
    function onMouseMove(moveEvent: MouseEvent) {
      const deltaX = moveEvent.clientX - startX
      const deltaY = moveEvent.clientY - startY
      const newWidth = Math.max(MIN_COMMENT_WIDTH, startW + deltaX)
      const newHeight = Math.max(MIN_COMMENT_HEIGHT, startH + deltaY)
      setComments(comments =>
        comments.map(c =>
          c.id === sticky.id
            ? { ...c, width: newWidth, height: newHeight }
            : c
        )
      )
    }
    function onMouseUp() {
      window.removeEventListener("mousemove", onMouseMove)
      window.removeEventListener("mouseup", onMouseUp)
    }
    window.addEventListener("mousemove", onMouseMove)
    window.addEventListener("mouseup", onMouseUp)
  }

  // --- Chart drop support ---
  function onChartDrop(e: React.DragEvent, slotId: string) {
    e.preventDefault()
    const chartId = e.dataTransfer.getData("chartId")
    if (!chartId) return
    setLayout(prev =>
      prev.map(s =>
        s.id === slotId
          ? { ...s, chartId }
          : s
      )
    )
  }

  // --------- Template selector modal ---------
  if (!selectedTemplate) {
    return (
      <div className="container mx-auto p-6">
        <Card>
          <CardHeader>
            <CardTitle>Choose Report Template</CardTitle>
          </CardHeader>
          <CardContent className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {TEMPLATES.map(template => (
              <div
                key={template.id}
                className="border rounded-lg p-4 cursor-pointer hover:bg-gray-50"
                onClick={() => setSelectedTemplate(template)}
              >
                <div className="text-lg font-medium">{template.name}</div>
                <div className="text-sm text-gray-500 mt-2">
                  {template.dimensions.width} × {template.dimensions.height} pt
                </div>
                <div className="mt-4 flex justify-center">
                  <div
                    className="relative bg-gray-100 border"
                    style={{
                      width: template.dimensions.width / 4,
                      height: template.dimensions.height / 4
                    }}
                  >
                    {template.slots.map(slot => (
                      <div
                        key={slot.id}
                        className="absolute border border-dashed border-gray-400 bg-gray-50 bg-opacity-50"
                        style={{
                          left: slot.x / 4,
                          top: slot.y / 4,
                          width: slot.width / 4,
                          height: slot.height / 4
                        }}
                      />
                    ))}
                  </div>
                </div>
              </div>
            ))}
          </CardContent>
        </Card>
      </div>
    )
  }

  // ----------- Main Render -----------
  return (
    <div className="container mx-auto p-6">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold">
          {selectedTemplate.name} Report
        </h1>
        <div className="flex gap-2">
          <Button
            variant={isLayoutUnlocked ? "default" : "outline"}
            onClick={() => setIsLayoutUnlocked(!isLayoutUnlocked)}
          >
            {isLayoutUnlocked ? <><Unlock size={16} className="mr-2" /> Lock Layout</> : 
                                <><Lock size={16} className="mr-2" /> Unlock Layout</>}
          </Button>
          {isLayoutUnlocked && (
            <>
              <Button onClick={addNewSlot}>
                <Plus size={16} className="mr-2" /> Add Slot
              </Button>
              <Button onClick={addNewComment} variant="outline">
                <MessageCircle size={16} className="mr-2" /> Add Comment
              </Button>
            </>
          )}
        </div>
      </div>
      <div className="flex gap-6">
        {/* Chart Palette */}
        <div className="w-full md:w-1/4 mb-6 md:mb-0">
          <Card>
            <CardHeader>
              <CardTitle>Charts</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {availableCharts.map(({ id, title, type }) => (
                <div
                  key={id}
                  className="flex items-center p-3 border rounded hover:bg-gray-50 cursor-move"
                  draggable
                  onDragStart={e => {
                    e.dataTransfer.setData("chartId", id)
                    setActiveId(id)
                  }}
                >
                  <Checkbox
                    id={id}
                    checked={selectedCharts.includes(id)}
                    onCheckedChange={() => toggleChart(id)}
                    className="mr-3"
                  />
                  <label htmlFor={id} className="flex-1 font-medium text-sm">
                    {title} <span className="text-gray-500 text-xs">({type})</span>
                  </label>
                </div>
              ))}
            </CardContent>
          </Card>
        </div>
        {/* Template Preview */}
        <div className="flex-1">
          <Card>
            <CardHeader>
              <CardTitle>Report Preview</CardTitle>
            </CardHeader>
            <CardContent className="overflow-auto">
              <div className="flex justify-center">
                <div
                  className="preview-canvas relative bg-white border"
                  style={{
                    width: pageWidth,
                    height: pageHeight,
                    minWidth: 300,
                    minHeight: 300,
                    transition: "width 0.15s, height 0.15s"
                  }}
                >
                  {/* Slots */}
                  {layout.map(slot => (
                    <div
                      key={slot.id}
                      style={{
                        position: "absolute",
                        left: slot.x,
                        top: slot.y,
                        width: slot.width,
                        height: slot.height,
                        border: "1px solid #ccc",
                        borderRadius: 8,
                        background: slot.chartId ? "#fff" : "#f9fafb",
                        boxShadow: "0 1px 3px #0001",
                        zIndex: 5,
                        display: "flex",
                        flexDirection: "column",
                        overflow: "hidden"
                      }}
                      onDrop={e => onChartDrop(e, slot.id)}
                      onDragOver={e => e.preventDefault()}
                    >
                      {/* Move handle */}
                      {isLayoutUnlocked && (
                        <div
                          onMouseDown={e => handleSlotDrag(slot.id, e.clientX, e.clientY)}
                          style={{
                            position: "absolute",
                            left: 0,
                            top: 0,
                            width: "100%",
                            height: 18,
                            background: "rgba(0,0,0,0.04)",
                            cursor: "grab",
                            zIndex: 10,
                            borderTopLeftRadius: 8,
                            borderTopRightRadius: 8,
                            userSelect: "none"
                          }}
                        ></div>
                      )}
                      {/* Delete */}
                      {isLayoutUnlocked && slot.chartId && (
                        <button
                          onClick={() => removeSlot(slot.id)}
                          className="absolute top-1 right-1 p-1 bg-red-500 text-white rounded-full hover:bg-red-600 z-20"
                        >
                          <Trash size={14} />
                        </button>
                      )}
                      <div style={{ flex: 1, minHeight: 0, paddingTop: isLayoutUnlocked ? 18 : 0 }}>
                        {slot.chartId ? (
                          <ChartSlot chartId={slot.chartId} />
                        ) : (
                          <div className="w-full h-full flex items-center justify-center text-gray-400">
                            Drop chart here
                          </div>
                        )}
                      </div>
                      {/* Resize handle */}
                      {isLayoutUnlocked && (
                        <div
                          onMouseDown={e => handleSlotResizeMouseDown(slot, e)}
                          style={{
                            position: "absolute",
                            right: 0,
                            bottom: 0,
                            width: 18,
                            height: 18,
                            background: "rgba(0,0,0,0.1)",
                            cursor: "nwse-resize",
                            zIndex: 10,
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            borderRadius: "0 0 8px 0"
                          }}
                        >
                          <svg width="14" height="14">
                            <polyline
                              points="0,14 14,14 14,0"
                              stroke="#888"
                              strokeWidth="2"
                              fill="none"
                            />
                          </svg>
                        </div>
                      )}
                    </div>
                  ))}
                  {/* Movable, resizable sticky comments */}
                  {comments.map(sticky => (
                    <div
                      key={sticky.id}
                      style={{
                        position: "absolute",
                        left: sticky.x,
                        top: sticky.y,
                        width: sticky.width,
                        height: sticky.height,
                        zIndex: 10,
                        background: "#f8fcff",
                        border: "1.5px solid #bbdefb",
                        borderRadius: 6,
                        boxShadow: "0 1px 6px #0001",
                        display: "flex",
                        flexDirection: "column",
                        pointerEvents: isLayoutUnlocked ? "auto" : "none"
                      }}
                    >
                      {isLayoutUnlocked && (
                        <div
                          onMouseDown={e => handleCommentDrag(sticky.id, e.clientX, e.clientY)}
                          style={{
                            width: "100%",
                            height: 18,
                            cursor: "grab",
                            borderTopLeftRadius: 6,
                            borderTopRightRadius: 6,
                            background: "rgba(33,150,243,0.06)"
                          }}
                        />
                      )}
                      {isLayoutUnlocked && (
                        <button
                          onClick={() => removeComment(sticky.id)}
                          className="absolute top-1 right-1 p-1 bg-red-500 text-white rounded-full hover:bg-red-600 z-20"
                        >
                          <Trash size={14} />
                        </button>
                      )}
                      <Textarea
                        className="w-full text-xs px-2 py-1 flex-1 bg-transparent border-0 resize-none outline-none"
                        value={sticky.text}
                        onChange={e => updateCommentText(sticky.id, e.target.value)}
                        placeholder="Comment"
                        style={{
                          background: "transparent",
                          border: "none",
                          boxShadow: "none",
                          minHeight: 32,
                          resize: "none",
                          pointerEvents: isLayoutUnlocked ? "auto" : "none"
                        }}
                        disabled={!isLayoutUnlocked}
                      />
                      {/* Resize handle */}
                      {isLayoutUnlocked && (
                        <div
                          onMouseDown={e => handleCommentResizeMouseDown(sticky, e)}
                          style={{
                            position: "absolute",
                            right: 0,
                            bottom: 0,
                            width: 18,
                            height: 18,
                            background: "rgba(33,150,243,0.11)",
                            cursor: "nwse-resize",
                            zIndex: 12,
                            borderRadius: "0 0 6px 0",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center"
                          }}
                        >
                          <svg width="14" height="14">
                            <polyline
                              points="0,14 14,14 14,0"
                              stroke="#2196f3"
                              strokeWidth="2"
                              fill="none"
                            />
                          </svg>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
          {/* Export Actions */}
          <div className="mt-6 flex justify-center gap-4">
            <Button
              onClick={generatePdf}
              className="flex items-center space-x-2"
              size="lg"
            >
              <Download size={18} />
              <span>Export PDF</span>
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}
