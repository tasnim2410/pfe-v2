import { useEffect, useState } from "react";

export function useScrollProgress() {
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const calc = () => {
      const scrollTop =
        window.pageYOffset ||
        document.documentElement.scrollTop ||
        document.body.scrollTop ||
        0;

      const docHeight =
        Math.max(
          document.body.scrollHeight,
          document.documentElement.scrollHeight
        ) - window.innerHeight;

      setProgress(docHeight ? (scrollTop / docHeight) * 100 : 0);
    };

    calc(); // initialise
    window.addEventListener("scroll", calc, { passive: true });
    window.addEventListener("resize", calc); // handle viewport resizes
    return () => {
      window.removeEventListener("scroll", calc);
      window.removeEventListener("resize", calc);
    };
  }, []);

  return progress; // 0-100
}
