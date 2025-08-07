// "use client"

// import Link from "next/link"
// import { usePathname } from "next/navigation"
// import { Button } from "@/components/ui/button"
// import { cn } from "@/lib/utils"
// import { useState } from "react"
// import { LoginDialog } from "@/components/login-dialog"

// const navigation = [
//   { name: "Dashboard", href: "/" },
//   { name: "Trend Analysis", href: "/trend-analysis" },
//   { name: "Forecasting", href: "/forecasting" },
//   { name: "Reporting", href: "/reporting" },
// ]

// export function Header() {
//   const pathname = usePathname()
//   const [showLogin, setShowLogin] = useState(false)

//   return (
//     <>
//       <header className="fixed top-0 left-0 right-0 z-50 bg-black text-white shadow-lg">
//         <div className="flex items-center justify-between px-6 py-3">
//           <div className="flex items-center space-x-8">
//             <div className="flex items-center space-x-3">
//   <img
//     src="https://lab-engineering.actia.tn/wp-content/uploads/2021/02/logo-actia.png"
//     alt="ACTIA Logo"
//     width={180}
//     height={60}
//     className="object-contain h-10"
//     style={{ maxWidth: 180 }}
//   />
// </div>

//             <nav className="flex space-x-6">
//               {navigation.map((item) => (
//                 <Link
//                   key={item.name}
//                   href={item.href}
//                   className={cn(
//                     "px-3 py-2 text-sm font-medium transition-colors hover:text-gray-300",
//                     pathname === item.href ? "text-white border-b-2 border-white" : "text-gray-300",
//                   )}
//                 >
//                   {item.name}
//                 </Link>
//               ))}
//             </nav>
//           </div>

//           <div className="flex items-center space-x-4">
//             <Link href="/settings">
//               <Button variant="ghost" size="sm" className="text-white hover:text-gray-300">
//                 Settings
//               </Button>
//             </Link>
//             <Button
//               variant="outline"
//               size="sm"
//               className="border-white text-white hover:bg-white hover:text-black bg-transparent"
//               onClick={() => setShowLogin(true)}
//             >
//               Sign In
//             </Button>
//           </div>
//         </div>
//       </header>

//       <LoginDialog open={showLogin} onOpenChange={setShowLogin} />
//     </>
//   )
// }


"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { useState } from "react";
import { LoginDialog } from "@/components/login-dialog";
import { useScrollProgress } from "@/hooks/useScrollProgress";

/* ─────────────────────────────────────────
   STYLE CONFIG  (edit these when you need)
   ───────────────────────────────────────── */
const BRAND_GREEN   = "#BDD248";
const BAR_HEIGHT_PX = 6;
const TAGLINE_TEXT  =
  "Technology Trend-Analysis Platform";

const C = {
  header:  "fixed inset-x-0 top-0 z-50 bg-black text-white shadow-lg",
  bar:     "block",
  navbar:  "flex items-center justify-between px-6 py-3",
  logoCol: "flex flex-col space-y-1",               // ← new
  tagline: "text-xs italic text-white",
  navLink: "px-3 py-2 text-sm font-medium transition-colors hover:text-gray-300",
  navLinkActive: "text-white border-b-2 border-white",
};
/* ───────────────────────────────────────── */

const NAV = [
  { name: "Home", href: "/home" },
  { name: "Dashboard", href: "/" },

  { name: "Trend Analysis", href: "/trend-analysis" },
  { name: "Forecasting", href: "/forecasting" },
  { name: "Reporting", href: "/reporting" },
];

export function Header() {
  const pathname = usePathname();
  const [showLogin, setShowLogin] = useState(false);
  const progress = useScrollProgress();           // 0–100 %

  const barDynamic = {
    height: BAR_HEIGHT_PX,
    backgroundColor: BRAND_GREEN,
    width: `${progress}%`,
    opacity: progress === 0 ? 0 : 1,
    transition: "width .2s linear, opacity .2s linear",
  } as const;

  const barStatic = {
    height: BAR_HEIGHT_PX,
    backgroundColor: BRAND_GREEN,
  } as const;

  return (
    <>
      <header className={C.header}>
        {/* TOP progress bar */}
        <span className={C.bar} style={barDynamic} />

        {/* NAVBAR */}
        <div className={C.navbar}>
          {/* left: logo + tagline + nav links */}
          <div className="flex items-center space-x-8">
            {/* logo + tagline stacked */}
            <div className={C.logoCol}>
              <img
                src="https://lab-engineering.actia.tn/wp-content/uploads/2021/02/logo-actia.png"
                alt="ACTIA Logo"
                width={180}
                height={60}
                className="h-10 object-contain"
              />
              <span className={C.tagline}>{TAGLINE_TEXT}</span>
            </div>

            {/* nav links */}
            <nav className="flex space-x-6">
              {NAV.map((item) => (
                <Link
                  key={item.name}
                  href={item.href}
                  className={cn(
                    C.navLink,
                    pathname === item.href && C.navLinkActive
                  )}
                >
                  {item.name}
                </Link>
              ))}
            </nav>
          </div>

          {/* right: actions */}
          <div className="flex items-center space-x-4">
            <Link href="/settings">
              <Button variant="ghost" size="sm" className="text-white hover:text-gray-300">
                Settings
              </Button>
            </Link>
            <Button
              variant="outline"
              size="sm"
              className="border-white text-black hover:bg-white hover:text-black"
              onClick={() => setShowLogin(true)}
            >
              Sign In
            </Button>
          </div>
        </div>

        {/* BOTTOM permanent bar */}
        <span className={C.bar} style={barStatic} />
      </header>

      {/* login dialog + spacer */}
      <LoginDialog open={showLogin} onOpenChange={setShowLogin} />
      <div style={{ height: 60 + BAR_HEIGHT_PX * 2 + 20 }} /> {/* spacer */}
    </>
  );
}

