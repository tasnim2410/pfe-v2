"use client"
import { Download, Eye, FileText, Presentation } from "lucide-react"
import html2canvas from "html2canvas"
import AdvancedSearchBar from "@/components/advanced-search";


export default function Page() {
  return (
    <div className="container mx-auto px-6 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Technology Trend Dashboard</h1>
        <p className="text-gray-600">Advanced patent and research analysis platform</p>
      </div>
      <div className="space-y-6">
        <AdvancedSearchBar />
      </div>
    </div>
  );
}
