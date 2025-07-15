


/* app/reporting/page.tsx */
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

/*───────────────────────────────────────────────────────────────────────────
  1.  Dynamic imports for **all** charts we might want in a report
  ───────────────────────────────────────────────────────────────────────────*/

// width/height props are optional for most components; keep them generic
type ChartProps = { width?: number; height?: number }

// prettier-ignore
const ChartComponents = {
  "publication-trend"              : dynamic<ChartProps>(() => import("@/components/publication_trends").then(m => m.default || m), { ssr:false }),
  "applicant-analysis"             : dynamic<ChartProps>(() => import("@/components/applicant_type_pie").then(m => m.default || m),   { ssr:false }),
  "market-size"                    : dynamic<ChartProps>(() => import("@/components/IPStat").then(m => m.default || m),                { ssr:false }),
  // >>> NEW
  "top-ipc"                        : dynamic<ChartProps>(() => import("@/components/top_ipc_codes").then(m => m.default || m),         { ssr:false }),
  // "geographic-distribution"        : dynamic<ChartProps>(() => import("@/components/geographic_distribution").then(m => m.default || m),{ ssr:false }),
  "top-10-applicants"              : dynamic<ChartProps>(() => import("@/components/top_10_app").then(m => m.default || m),            { ssr:false }),
  "top-10-keywords"                : dynamic<ChartProps>(() => import("@/components/top_10_keywords").then(m => m.default || m),       { ssr:false }),
  "patent-field-trends"            : dynamic<ChartProps>(() => import("@/components/patent_by_field").then(m => m.default || m),       { ssr:false }),
  "evolving-word-cloud"            : dynamic<ChartProps>(() => import("@/components/evolving_word_cloud").then(m => m.default || m),   { ssr:false }),
  "cooccurrence-trends"            : dynamic<ChartProps>(() => import("@/components/co-occurunece_trend").then(m => m.default || m),   { ssr:false }),
  "applicant-collaboration-network": dynamic<ChartProps>(() => import("@/components/collaboration_network").then(m => m.default || m), { ssr:false }),
  "originality-rate"               : dynamic<ChartProps>(() => import("@/components/originality_rate").then(m => m.default || m),      { ssr:false }),
  "family-member-count"            : dynamic(() => import("@/components/family-member-count").then(m => m.default || m),   { ssr:false }),
"family-size-distribution"       : dynamic(() => import("@/components/family-size-distribution").then(m => m.default || m),{ ssr:false }),
"international-protection-matrix": dynamic(() => import("@/components/international-protection-matrix").then(m => m.default || m),{ ssr:false }),
"international-patent-flow"      : dynamic(() => import("@/components/international-patent-flow").then(m => m.default || m),{ ssr:false }),
} as const

type ChartId = keyof typeof ChartComponents

/*───────────────────────────────────────────────────────────────────────────
  2.  Master list drives both the checkbox UI and import map
  ───────────────────────────────────────────────────────────────────────────*/
const availableCharts: { id: ChartId; title: string; type: string }[] = [
  { id: "publication-trend",               title: "Publication Trend",                type: "summary" },
  { id: "applicant-analysis",              title: "Applicant Analysis",               type: "trend"   },
  { id: "market-size",                     title: "Market Size",                      type: "summary" },
  // >>> NEW
  { id: "top-ipc",                         title: "Top IPC Codes",                    type: "summary" },
  // { id: "geographic-distribution",         title: "Geographic Distribution",          type: "summary" },
  { id: "top-10-applicants",               title: "Top 10 Applicants",                type: "summary" },
  { id: "top-10-keywords",                 title: "Top 10 Keywords",                  type: "summary" },
  { id: "patent-field-trends",             title: "Patent Field Trends",              type: "trend"   },
  { id: "evolving-word-cloud",             title: "Evolving Word Cloud",              type: "trend"   },
  { id: "cooccurrence-trends",             title: "Co-occurrence Trends",             type: "trend"   },
  { id: "applicant-collaboration-network", title: "Collaboration Network",            type: "network" },
  { id: "originality-rate",                title: "Originality Rate",                 type: "summary" },
  { id: "family-member-count",             title: "Family Member Count",              type: "summary" },
  { id: "family-size-distribution",        title: "Family Size Distribution",         type: "summary" },
  { id: "international-protection-matrix", title: "International Protection Matrix",  type: "summary" },
  { id: "international-patent-flow",       title: "International Patent Flow",        type: "summary" },
]

/* savedReports list unchanged */
const savedReports = [
  { id: "1", title: "Q4 2023 Technology Analysis", date: "2023-12-15", type: "PDF"        },
  { id: "2", title: "Patent Landscape Report",     date: "2023-11-20", type: "PowerPoint" },
  { id: "3", title: "Innovation Trends Summary",   date: "2023-10-30", type: "PDF"        },
]

export default function Reporting() {
  const {
    selectedCharts,
    setSelectedCharts,
    chartComments,
    setChartComments,
  } = useChartContext()

  /*──────── helpers ────────*/
  const toggleChart = (id: ChartId) =>
    setSelectedCharts(prev =>
      prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]
    )

  const updateComment = (id: ChartId, text: string) =>
    setChartComments(prev => ({ ...prev, [id]: text }))

  /*──────── stub PDF export ────────*/
  const generateReport = (format: "pdf" | "pptx") =>
    alert(`Report (${format}) is not implemented yet.`)

  /*──────── PPTX export (html2canvas → backend) ────────*/
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

  /*───────────────────────────────────────────────────────────────*/
  return (
    <div className="container mx-auto p-6">
      <Tabs defaultValue="create" className="w-full">
        <TabsList className="grid grid-cols-2">
          <TabsTrigger value="create">Create Report</TabsTrigger>
          <TabsTrigger value="saved">Saved Reports</TabsTrigger>
        </TabsList>

        {/*────────────── Create Report ──────────────*/}
        <TabsContent value="create" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Select Charts</CardTitle>
            </CardHeader>

            <CardContent className="space-y-4">
              {availableCharts.map(({ id, title, type }) => {
                const Selected = ChartComponents[id]
                const checked = selectedCharts.includes(id)

                return (
                  <div key={id} className="space-y-2">
                    {/* checkbox */}
                    <div className="flex items-center">
                      <Checkbox
                        id={id}
                        checked={checked}
                        onCheckedChange={() => toggleChart(id)}
                      />
                      <label htmlFor={id} className="ml-2 font-medium">
                        {title} ({type})
                      </label>
                    </div>

                    {/* chart preview + comment box */}
                    {checked && (
                      <div id={`chart-${id}`} className="p-4 border rounded">
                        {Selected ? (
                          <>
                            <Selected width={640} height={400} />
                            <Textarea
                              placeholder="Add comment"
                              value={chartComments[id] || ""}
                              onChange={e => updateComment(id, e.target.value)}
                              className="mt-2 w-full"
                            />
                          </>
                        ) : (
                          <p>Component not available</p>
                        )}
                      </div>
                    )}
                  </div>
                )
              })}
            </CardContent>
          </Card>

          {/* actions */}
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

        {/*────────────── Saved Reports ──────────────*/}
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
