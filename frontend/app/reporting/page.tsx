


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

  // Utility for active page data
  const activePage = pages[activePageIdx]

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

  const generatePpt = async () => {
    // 1) Gather all rendered chart divs
    const divs = Array.from(document.querySelectorAll<HTMLDivElement>('[id^="chart-"]'));
    if (!divs.length) return alert("No charts to export.");
  
    // 2) Screenshot each with html2canvas
    const images = await Promise.all(
      divs.map(async d => ({
        id: d.id.replace("chart-", ""),
        data: (await html2canvas(d)).toDataURL("image/png"),
      }))
    );
    const commentMap = (activePage?.comments || []).reduce((acc, comment) => {
      acc[comment.id] = comment.text;
      return acc;
    }, {} as Record<string, string>);
  
    try {
      // 3) Ask your local backend port
      const portResponse = await fetch("/backend_port.txt");
      const port = await portResponse.text();
      // 4) Send to your PPTX-generation endpoint
      const res = await fetch(`http://localhost:${port}/api/report/generate-pptx`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ images, comments: commentMap }),
      });
  
      if (!res.ok) throw new Error(await res.text());
  
      // 5) Download the returned PPTX blob
      const blob = await res.blob();
      const url  = URL.createObjectURL(blob);
      const a    = document.createElement("a");
      a.href     = url;
      a.download = "charts-report.pptx";
      a.click();
    } catch (e) {
      console.error(e);
      alert("Export failed: " + e);
    }
  };
  const chartStyle: React.CSSProperties = isLayoutUnlocked 
  ? {} 
  : { position: 'relative', zIndex: 1 };
     
  


  const generatePdf = async () => {
    if (isLayoutUnlocked) {
      setShowWarning(true)
      setTimeout(() => setShowWarning(false), 2100)
      return
    }
    
    if (!selectedTemplate || !activePage) return
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
    pdf.save(`${selectedTemplate.name}_${activePage.name}.pdf`)
  }

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
            className={`flex items-center gap-2 px-3 py-1 border-b-2 ${i === activePageIdx ? "border-blue-500 font-bold" : "border-transparent"} bg-white rounded-t`}
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
            className="flex items-center gap-2 px-3 py-1 text-blue-700 bg-blue-50 border-b-2 border-transparent rounded-t hover:bg-blue-100"
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

  <div className="space-x-2">
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
                            id={`chart-${slot.id}`} // Add this ID
                            style={{ ...chartStyle, width: '100%', height: '100%' }}
                          >
                            <ChartSlot chartId={slot.chartId} />
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
        </div>
      </div>
    </div>
  )
}
