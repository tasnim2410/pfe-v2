


"use client"

import { useState, useEffect, useRef, memo } from "react"
import { Trash, Plus, Lock, Unlock, Download, MessageCircle, FilePlus2, File } from "lucide-react"
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
import PptxGenJS from "pptxgenjs"



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
type Page = {
  id: string
  name: string
  layout: Slot[]
  comments: StickyComment[]
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


const COMMENT_STYLE = {
  background: "#f8f9fa",
  border: "1.5px solid #e9ecef",
  handleBackground: "rgba(108, 117, 125, 0.06)",
  resizeBackground: "rgba(108, 117, 125, 0.11)",
  textColor: "#495057"
};

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
    const [isLoading, setIsLoading] = useState(true);
    
    useEffect(() => {
      if (chartId) {
        setIsLoading(true);
        // Simulate chart loading
        const timer = setTimeout(() => {
          setIsLoading(false);
        }, 800);
        
        return () => clearTimeout(timer);
      }
    }, [chartId]);
    
    return (
      <div className="w-full h-full relative">
        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-gray-50 bg-opacity-80 z-10">
            <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-500"></div>
          </div>
        )}
        {renderChart(chartId)}
      </div>
    );
  },
  (prev, next) => prev.chartId === next.chartId
);

const MIN_SLOT_WIDTH = 120
const MIN_SLOT_HEIGHT = 90
const MIN_COMMENT_WIDTH = 140
const MIN_COMMENT_HEIGHT = 60
const PADDING = 32

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

export default function Reporting() {
  const { selectedCharts, setSelectedCharts } = useChartContext()
  const [selectedTemplate, setSelectedTemplate] = useState<Template | null>(null)
  const [pages, setPages] = useState<Page[]>([])
  const [activePageIdx, setActivePageIdx] = useState(0)
  const [isLayoutUnlocked, setIsLayoutUnlocked] = useState(false)
  const [activeId, setActiveId] = useState<string | null>(null)
  const [showWarning, setShowWarning] = useState(false)
  const [isExporting, setIsExporting] = useState(false);
  const [chartLoading, setChartLoading] = useState<Record<string, boolean>>({});
  // When template picked, create first page
  useEffect(() => {
    if (selectedTemplate) {
      setPages([{
        id: `page-1`,
        name: "Page 1",
        layout: [...selectedTemplate.slots],
        comments: [],
      }])
      setActivePageIdx(0)
    }
  }, [selectedTemplate])
  const activePage = pages[activePageIdx]
  const getCurrentPageChartIds = () => {
    return activePage?.layout
      .filter(slot => slot.chartId)
      .map(slot => slot.id) || [];
  };

  const waitForCharts = (selector: string) => {
    return new Promise<void>(resolve => {
      const check = () => {
        const charts = document.querySelectorAll(selector);
        if (charts.length > 0) {
          resolve();
        } else {
          setTimeout(check, 100);
        }
      };
      check();
    });
  };
  

  const { allLoaded: chartsLoaded, markAsLoaded } = useChartLoading(getCurrentPageChartIds());
  const getPageDimensions = (page: Page) => {
    const width = Math.max(
      ...(page.layout?.map(slot => slot.x + slot.width) || []),
      ...(page.comments?.map(c => c.x + c.width) || []),
      selectedTemplate?.dimensions.width || 600
    ) + PADDING;
    
    const height = Math.max(
      ...(page.layout?.map(slot => slot.y + slot.height) || []),
      ...(page.comments?.map(c => c.y + c.height) || []),
      selectedTemplate?.dimensions.height || 600
    ) + PADDING;
    
    return { width, height };
  };

  // Add new page (duplicate template structure, empty)
  const addNewPage = () => {
    if (!selectedTemplate) return
    setPages(pages => [
      ...pages,
      {
        id: `page-${pages.length + 1}`,
        name: `Page ${pages.length + 1}`,
        layout: [...selectedTemplate.slots.map(s => ({ ...s, chartId: null }))],
        comments: [],
      }
    ])
    setActivePageIdx(pages.length)
  }
  // Remove page (if more than 1)
  const removePage = (idx: number) => {
    if (pages.length === 1) return
    setPages(p => p.filter((_, i) => i !== idx))
    setActivePageIdx(idx === 0 ? 0 : idx - 1)
  }
  

  // ---- Slot & comment operations on active page ----
  function updateLayout(updater: (layout: Slot[]) => Slot[]) {
    setPages(pages =>
      pages.map((pg, i) =>
        i === activePageIdx ? { ...pg, layout: updater(pg.layout) } : pg
      )
    )
  }
  function updateComments(updater: (comments: StickyComment[]) => StickyComment[]) {
    setPages(pages =>
      pages.map((pg, i) =>
        i === activePageIdx ? { ...pg, comments: updater(pg.comments) } : pg
      )
    )
  }
  const addNewSlot = () => {
    updateLayout(layout => [
      ...layout,
      { id: `slot-${Date.now()}`, chartId: null, x: 60, y: 60, width: 200, height: 150 }
    ])
  }
  const addNewComment = () => {
    updateComments(comments => [
      ...comments,
      { id: `comment-${Date.now()}`, text: "", x: 120, y: 120, width: 180, height: 80 }
    ])
  }
  const removeSlot = (slotId: string) => updateLayout(layout => layout.filter(slot => slot.id !== slotId))
  const removeComment = (commentId: string) => updateComments(comments => comments.filter(c => c.id !== commentId))
  const updateCommentText = (id: string, text: string) =>
    updateComments(comments => comments.map(c => c.id === id ? { ...c, text } : c))

  const toggleChart = (id: string) => {
    setSelectedCharts(prev =>
      prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]
    )
  }

  // --- PDF Download (enforce lock) ---
  const pageWidth = Math.max(
    ...(activePage?.layout?.map(slot => slot.x + slot.width) || []),
    ...(activePage?.comments?.map(c => c.x + c.width) || []),
    selectedTemplate?.dimensions.width || 600
  ) + PADDING
  const pageHeight = Math.max(
    ...(activePage?.layout?.map(slot => slot.y + slot.height) || []),
    ...(activePage?.comments?.map(c => c.y + c.height) || []),
    selectedTemplate?.dimensions.height || 600
  ) + PADDING

  // … inside your Reporting() component, just above the return:
  // Add this utility function
const waitForAll = (predicate: () => boolean, timeout = 10000) => {
  return new Promise<void>((resolve, reject) => {
    const startTime = Date.now();
    const check = () => {
      if (predicate()) {
        resolve();
      } else if (Date.now() - startTime > timeout) {
        reject(new Error('Timeout waiting for condition'));
      } else {
        setTimeout(check, 100);
      }
    };
    check();
  });
};

// In the generatePdf function:
const generatePdf = async () => {
  if (isLayoutUnlocked) {
    setShowWarning(true);
    setTimeout(() => setShowWarning(false), 2100);
    return;
  }
  
  if (!selectedTemplate) return;
  setIsExporting(true);
  
  try {
    const pdf = new jsPDF();
    const originalPageIdx = activePageIdx;
    
    for (let i = 0; i < pages.length; i++) {
      setActivePageIdx(i);
      // Wait for UI to update
      await new Promise(resolve => setTimeout(resolve, 500));
      
      const previewNode = document.querySelector('.preview-canvas');
      if (!previewNode) continue;
      
      // Wait for all charts to be present
      await waitForAll(() => {
        return previewNode.querySelectorAll('[id^="chart-"]').length > 0;
      });
      
      // Wait for charts to finish rendering
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      const { width, height } = getPageDimensions(pages[i]);
      const canvas = await html2canvas(previewNode as HTMLElement, { 
        scale: 1,
        backgroundColor: "#FFFFFF",
        useCORS: true,
        logging: true,
        ignoreElements: (element) => {
          // Ignore the loading overlay if present
          return element.classList.contains('export-loading-overlay');
        }
      });
      
      const imgData = canvas.toDataURL('image/png');
      const imgWidth = canvas.width;
      const imgHeight = canvas.height;
      
      // Calculate aspect ratio
      const aspectRatio = imgWidth / imgHeight;
      const pdfPageWidth = pdf.internal.pageSize.getWidth();
      const pdfPageHeight = pdf.internal.pageSize.getHeight();
      
      // Calculate dimensions to fit the page
      let renderWidth = pdfPageWidth;
      let renderHeight = pdfPageHeight;
      
      if (aspectRatio > pdfPageWidth / pdfPageHeight) {
        renderHeight = pdfPageWidth / aspectRatio;
      } else {
        renderWidth = pdfPageHeight * aspectRatio;
      }
      
      // Center the image
      const x = (pdfPageWidth - renderWidth) / 2;
      const y = (pdfPageHeight - renderHeight) / 2;
      
      if (i > 0) pdf.addPage();
      
      pdf.addImage(
        imgData,
        'PNG',
        x,
        y,
        renderWidth,
        renderHeight
      );
    }
    
    setActivePageIdx(originalPageIdx);
    pdf.save(`${selectedTemplate.name}_report.pdf`);
  } catch (e) {
    console.error("PDF export failed:", e);
    alert("PDF export failed: " + e);
  } finally {
    setIsExporting(false);
  }
};

// In the generatePpt function:
const generatePpt = async () => {
  if (isLayoutUnlocked) {
    setShowWarning(true);
    setTimeout(() => setShowWarning(false), 2100);
    return;
  }
  
  if (!selectedTemplate) return;
  setIsExporting(true);
  
  try {
    const images = [];
    const originalPageIdx = activePageIdx;
    
    
    for (let i = 0; i < pages.length; i++) {
      setActivePageIdx(i);
      // Wait for UI to update
      await new Promise(resolve => setTimeout(resolve, 500));
      
      const previewNode = document.querySelector('.preview-canvas');
      if (!previewNode) continue;
      
      // Wait for all charts to be present
      await waitForAll(() => {
        return previewNode.querySelectorAll('[id^="chart-"]').length > 0;
      });
      
      // Wait for charts to finish rendering
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      const canvas = await html2canvas(previewNode as HTMLElement, { 
        scale: 1,
        backgroundColor: "#FFFFFF",
        useCORS: true,
        logging: true,
        ignoreElements: (element) => {
          return element.classList.contains('export-loading-overlay');
        }
      });
      
      // Capture actual pixel dimensions
      const imgWidthPx = canvas.width;
      const imgHeightPx = canvas.height;
      
      images.push({
        id: `page-${i+1}`,
        data: canvas.toDataURL("image/png"),
        width: imgWidthPx,
        height: imgHeightPx
      });
    }
    
    setActivePageIdx(originalPageIdx);
    
    // Get backend port
    const portResponse = await fetch("/backend_port.txt");
    const port = await portResponse.text();
    
    // Send to backend with page dimensions
    const res = await fetch(`http://localhost:${port}/api/report/generate-pptx`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ 
        images,
        dimensions: {
          width: selectedTemplate.dimensions.width,
          height: selectedTemplate.dimensions.height
        }
      }),
    });

    if (!res.ok) throw new Error(await res.text());

    // Download the PPT
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "report.pptx";
    a.click();
  } catch (e) {
    console.error("PPT export failed:", e);
    alert("PPT export failed: " + e);
  } finally {
    setIsExporting(false);
  }
};


  // --- Slot drag/resize handlers: always auto-push other slots out of the way
  function handleSlotDrag(id: string, startX: number, startY: number) {
    const slotIdx = activePage.layout.findIndex(s => s.id === id)
    if (slotIdx === -1) return
    const slot = activePage.layout[slotIdx]
    const origX = slot.x
    const origY = slot.y

    function onMouseMove(e: MouseEvent) {
      const deltaX = e.clientX - startX
      const deltaY = e.clientY - startY
      updateLayout(prev => {
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
  function useChartLoading(chartIds: string[]) {
    const [loadedCharts, setLoadedCharts] = useState<Record<string, boolean>>({});
    const [allLoaded, setAllLoaded] = useState(false);
  
    useEffect(() => {
      const allLoaded = chartIds.every(id => loadedCharts[id]);
      setAllLoaded(allLoaded);
    }, [loadedCharts, chartIds]);
  
    const markAsLoaded = (id: string) => {
      setLoadedCharts(prev => ({ ...prev, [id]: true }));
    };
  
    return { allLoaded, markAsLoaded };
  }
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
      updateLayout(prev => {
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
  function handleCommentDrag(id: string, startX: number, startY: number) {
    const commentIdx = activePage.comments.findIndex(c => c.id === id)
    if (commentIdx === -1) return
    const sticky = activePage.comments[commentIdx]
    const origX = sticky.x
    const origY = sticky.y
    function onMouseMove(e: MouseEvent) {
      const deltaX = e.clientX - startX
      const deltaY = e.clientY - startY
      updateComments(comments =>
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
      updateComments(comments =>
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
  function onChartDrop(e: React.DragEvent, slotId: string) {
    e.preventDefault()
    const chartId = e.dataTransfer.getData("chartId")
    if (!chartId) return
    updateLayout(prev =>
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
      <div className="container mx-auto p-6" style={{ paddingTop: 'var(--header-height)' }}>
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
      {/* PAGE TABS */}
      <div className="flex gap-2 mb-4">
        {pages.map((pg, i) => (
          
          <button
            key={pg.id}
            onClick={() => setActivePageIdx(i)}
            className={`flex items-center gap-2 px-3 py-1 border-b-2 ${
              i === activePageIdx 
                ? "border-gray-700 bg-gray-200 text-gray-900 font-bold" 
                : "border-transparent bg-gray-100 text-gray-700"
            } rounded-t transition-colors`}
            style={{ minWidth: 68 }}
          >
            <File size={15} className="mr-1" />
            {pg.name}
            {pages.length > 1 && isLayoutUnlocked && (
              <span
                className="ml-2 text-xs text-red-400 cursor-pointer hover:underline"
                onClick={e => { e.stopPropagation(); removePage(i); }}
                title="Remove page"
              >
                &times;
              </span>
            )}
          </button>
        ))}
        {isLayoutUnlocked && (
          <button
            onClick={addNewPage}
            className="flex items-center gap-2 px-3 py-1 text-gray-800 bg-gray-100 border-b-2 border-transparent rounded-t hover:bg-gray-200 transition-colors"
    >
            <FilePlus2 size={16} />
            Add Page
          </button>
        )}
      </div>
      <div className="flex gap-6">
        {/* Chart Palette Sidebar (sticky!) */}
        <div
          className="w-full md:w-1/4 mb-6 md:mb-0"
          style={{
            position: "sticky",
            top: 'var(--header-height)',
            alignSelf: "flex-start",
            zIndex: 999,
            minWidth: 225,
            maxHeight: "90vh",
            height: "fit-content"
          }}
        >

<Card style={{ maxHeight: '70vh', overflow: 'hidden' }}>
  <CardHeader>
    <CardTitle>Charts</CardTitle>
  </CardHeader>
  <CardContent 
    className="space-y-3" 
    style={{ 
      maxHeight: 'calc(70vh - 57px)', // Adjust based on header height
      overflowY: 'auto'
    }}
  >
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
        {/* <Checkbox
          id={id}
          checked={selectedCharts.includes(id)}
          onCheckedChange={() => toggleChart(id)}
          className="mr-3"
        /> */}
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
          {/* Top controls */}
          <div className="flex justify-between mb-2">
  <div>
    <Button
      variant={isLayoutUnlocked ? "default" : "outline"}
      onClick={() => setIsLayoutUnlocked(!isLayoutUnlocked)}
    >
      {isLayoutUnlocked
        ? <><Unlock size={16} className="mr-2" /> Lock Layout</>
        : <><Lock   size={16} className="mr-2" /> Unlock Layout</>}
    </Button>
    {isLayoutUnlocked && (
      <>
        <Button onClick={addNewSlot} className="ml-2">
          <Plus size={16} className="mr-2" /> Add Slot
        </Button>
        <Button onClick={addNewComment} className="ml-2" variant="outline">
          <MessageCircle size={16} className="mr-2" /> Add Comment
        </Button>
      </>
    )}
  </div>

  <div className="flex items-center space-x-2">
    {/* PDF */}
    <Button
      onClick={generatePdf}
      className="flex items-center"
      size="lg"
      disabled={isLayoutUnlocked}
      style={{ opacity: isLayoutUnlocked ? 0.7 : 1 }}
    >
      <Download size={18} className="mr-2" />
      Export PDF
    </Button>

    {/* PPT */}
    <Button
      onClick={generatePpt}
      className="flex items-center"
      size="lg"
      disabled={isLayoutUnlocked}
      style={{ opacity: isLayoutUnlocked ? 0.7 : 1 }}
    >
      <Download size={18} className="mr-2" />
      Export PPT
    </Button>

    {showWarning && (
      <span className="ml-3 text-red-600 font-medium bg-yellow-50 border border-yellow-300 px-3 py-1 rounded shadow">
        Please lock layout before downloading!
      </span>
    )}
  </div>
</div>

          <Card>
            <CardHeader>
              <CardTitle>{activePage?.name} Preview</CardTitle>
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
                  {activePage?.layout.map(slot => (
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
                            <div 
                            id={`chart-${slot.id}`}
                            style={{ 
                              width: '100%', 
                              height: '100%',
                              display: 'flex',
                              flexDirection: 'column',
                              position: 'relative',
                              overflow: 'hidden',
                            }}
                          >
                            <ChartSlot chartId={slot.chartId} onLoad={() => markAsLoaded(slot.id)} />
                          </div>
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
                  {activePage?.comments.map(sticky => (
                    <div
                      key={sticky.id}
                      style={{
                        position: "absolute",
                        left: sticky.x,
                        top: sticky.y,
                        width: sticky.width,
                        height: sticky.height,
                        zIndex: 10,
                        background: COMMENT_STYLE.background,
                        border: COMMENT_STYLE.border,
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
                            background: COMMENT_STYLE.background
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
                          pointerEvents: isLayoutUnlocked ? "auto" : "none",
                          color: COMMENT_STYLE.textColor,
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
                            background: COMMENT_STYLE.resizeBackground,
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
        </div>
      </div>
      {isExporting && (
  <div className="fixed inset-0 bg-black bg-opacity-50 z-[1000] flex items-center justify-center">
    <div className="bg-white p-6 rounded-lg shadow-xl flex flex-col items-center">
      <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-gray-900 mb-4"></div>
      <p className="text-lg font-medium text-gray-800">
        Preparing export...
      </p>
      <p className="text-gray-600 mt-1">
        Rendering page {activePageIdx + 1} of {pages.length}
      </p>
      {!chartsLoaded && (
        <p className="text-gray-500 text-sm mt-2">
          Waiting for charts to load...
        </p>
      )}
    </div>
  </div>
)}
    </div>
  )
}
