// import type React from "react"
// import type { Metadata } from "next"
// import { Inter } from "next/font/google"
// import "./globals.css"
// import { ThemeProvider } from "@/components/theme-provider"
// import { Header } from "@/components/header"
// import { RecoilRoot } from "recoil";
// const inter = Inter({ subsets: ["latin"] })

// export const metadata: Metadata = {
//   title: "Actia - Technology Trend Analysis",
//   description: "Advanced technology trend analysis and forecasting platform",
//     generator: 'v0.dev'
// }

// export default function RootLayout({
//   children,
// }: {
//   children: React.ReactNode
// }) {
//   return (
//     <html lang="en" suppressHydrationWarning>
// <body className={inter.className}>
//         <RecoilRoot> {/* <-- Add this! */}
//           <ThemeProvider attribute="class" defaultTheme="light" enableSystem disableTransitionOnChange>
//             <div className="min-h-screen bg-gray-50">
//               <Header />
//               <main className="pt-16">{children}</main>
//             </div>
//           </ThemeProvider>
//         </RecoilRoot>
//       </body>
//     </html>
//   )
// }

import type React from "react";
import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { ThemeProvider } from "@/components/theme-provider";
import { Header } from "@/components/header";


const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Actia - Technology Trend Analysis",
  description: "Advanced technology trend analysis and forecasting platform",
  generator: "v0.dev",
};

import { ChartProvider } from "./providers/ChartContext";

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={inter.className}>
        <ChartProvider>
          <ThemeProvider
            attribute="class"
            defaultTheme="light"
            enableSystem
            disableTransitionOnChange
          >
            <div className="min-h-screen bg-gray-50">
              <Header />
              <main className="pt-16">{children}</main>
            </div>
          </ThemeProvider>
        </ChartProvider>
      </body>
    </html>
  );
}
