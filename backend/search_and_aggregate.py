# technology-trend-analysis/backend/search_and_aggregate.py
"""
Pipeline:
  For each SEARCH_TERMS entry:
    1) POST /api/search (patents) using the AND + nested-OR request shape (title field)
    2) POST /api/scientific_search_merge (publications) with {"query": term}
    3) GET yearly counts from:
         - /api/patents/yearly_counts
         - /api/publications/yearly_counts
       (tries ?tech_label=<term>; falls back to plain endpoint)
  Then write wide, continuous-year CSVs:
    - outputs/patents.csv      -> columns: year, pat1, pat2, ...
    - outputs/publications.csv -> columns: year, pub1, pub2, ...
    - outputs/techs_order.txt  -> mapping to columns

Assumption:
  - After the search POSTs, the yearly_counts endpoints reflect that term
    (either via a "current series" cache or by honoring ?tech_label).
"""

import os
import re
import sys
import time
import json
import argparse
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote_plus

import requests
import pandas as pd

# ------------------------ Config ------------------------

SEARCH_TERMS = [
    "Edge Computing",
    "Supply Chain",
    "Online Learning",
    "Optical Character Recognition",
]

DEFAULT_OUTDIR = os.path.join(os.getcwd(), "outputs")
DEFAULT_TIMEOUT = 300.0  # 5 minutes (long ops)
DEFAULT_RETRIES = 2
DEFAULT_BACKOFF = 1.5
POST_TO_GET_SETTLE_SEC = 0.25  # small delay so backend can update any "current" cache

# ------------------------ shared_path.txt helpers ------------------------

def default_shared_path() -> str:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    candidate = os.path.join(repo_root, "shared_path.txt")
    if os.path.isfile(candidate):
        return candidate
    return os.path.join(os.getcwd(), "shared_path.txt")

def read_port_from_shared_path(shared_path: str) -> Optional[str]:
    if not os.path.isfile(shared_path):
        return None
    try:
        with open(shared_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read().strip()
            if content and content.isdigit():
                return content
    except Exception:
        return None
    return None

# ------------------------ HTTP helpers ------------------------

def post_with_retries(url: str, payload: Dict, timeout: float, retries: int, backoff: float) -> Optional[requests.Response]:
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            return requests.post(url, json=payload, timeout=timeout)
        except (requests.Timeout, requests.ConnectionError) as e:
            last_err = e
            if attempt == retries:
                print(f"[ERROR] POST {url} failed after {attempt}/{retries}: {e}")
                return None
            sleep_s = backoff ** attempt
            print(f"[WARN] POST {url} attempt {attempt}/{retries} failed: {e}. Retrying in {sleep_s:.1f}s...")
            time.sleep(sleep_s)
        except Exception as e:
            print(f"[ERROR] Unexpected error POST {url}: {e}")
            return None
    if last_err:
        print(f"[ERROR] Giving up on {url}: {last_err}")
    return None

def get_json(url: str, timeout: float) -> Optional[Dict]:
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            print(f"[WARN] GET {url} -> {r.status_code}")
            return None
        ct = r.headers.get("Content-Type", "")
        if "application/json" not in ct:
            print(f"[WARN] GET {url} returned non-JSON content-type: {ct}")
            return None
        return r.json()
    except Exception as e:
        print(f"[WARN] GET {url} failed: {e}")
        return None

# ------------------------ Payload builders ------------------------

def _tokenize(term: str) -> List[str]:
    return [w for w in re.findall(r"[A-Za-z0-9]+", term) if w]

def build_patents_payload(term: str) -> Dict:
    """
    Build the /api/search body using the pattern the user provided:
    {
      "query": {
        "group1": {
          "type": "group",
          "operator": "AND",
          "keywords": [
            {"type":"keyword","word": MAIN,"rule_op":"all","field":"title"},
            {"type":"group","operator":"OR","keywords":[
                {"type":"keyword","word": TOK2,"rule_op":"any","field":"title"},
                {"type":"keyword","word": TOK3,"rule_op":"any","field":"title"},
                ...
            ]}
          ]
        }
      }
    }
    If the term has only one token, we omit the OR group.
    """
    toks = _tokenize(term)
    if not toks:
        # fall back to a single keyword with the raw term
        toks = [term]

    main_kw = {
        "type": "keyword",
        "word": toks[0],
        "rule_op": "all",
        "field": "title",
    }

    keywords: List[Dict] = [main_kw]

    if len(toks) > 1:
        or_group = {
            "type": "group",
            "operator": "OR",
            "keywords": [
                {"type": "keyword", "word": w, "rule_op": "any", "field": "title"}
                for w in toks[1:]
            ],
        }
        keywords.append(or_group)

    return {
        "query": {
            "group1": {
                "type": "group",
                "operator": "AND",
                "keywords": keywords,
            }
        }
    }

# ------------------------ Yearly counts parsing ------------------------

def parse_yearly_counts(payload: Dict) -> Tuple[List[int], List[int]]:
    """
    Parse the shape:
      { "datasets": [ {"data": [..], "label": "..."} ], "labels": [years...] }
    Sum across datasets if >1.
    """
    if not isinstance(payload, dict):
        raise ValueError("yearly_counts payload is not a dict")
    labels = payload.get("labels", [])
    datasets = payload.get("datasets", [])
    if not isinstance(labels, list) or not datasets:
        raise ValueError("yearly_counts missing labels or datasets")
    years = [int(x) for x in labels]
    agg = [0] * len(years)
    for ds in datasets:
        data = ds.get("data", [])
        if len(data) != len(years):
            raise ValueError("datasets.data length != labels length")
        agg = [int(a) + int(b) for a, b in zip(agg, data)]
    return years, agg

def fetch_yearly_counts(base: str, endpoint: str, term: str, timeout: float) -> Tuple[List[int], List[int]]:
    """
    Try with ?tech_label=<term>, then fallback to plain endpoint if the first returns empty/invalid.
    """
    # 1) try with tech_label
    from urllib.parse import quote_plus
    url_with = f"{base}{endpoint}?tech_label={quote_plus(term)}"
    payload = get_json(url_with, timeout=timeout)
    if payload:
        try:
            return parse_yearly_counts(payload)
        except Exception:
            pass  # fall back to plain

    # 2) plain endpoint
    url_plain = f"{base}{endpoint}"
    payload = get_json(url_plain, timeout=timeout)
    if not payload:
        raise RuntimeError(f"Failed to GET yearly counts from {url_with} or {url_plain}")
    return parse_yearly_counts(payload)

# ------------------------ Wide DF builder (continuous years) ------------------------

def to_wide_continuous(per_term_counts: List[Tuple[List[int], List[int]]], prefix: str) -> pd.DataFrame:
    """
    per_term_counts: list of (years, counts) tuples
    Returns a continuous-year wide DF with columns: year, <prefix>1, <prefix>2, ...
    """
    all_years = set()
    for years, _ in per_term_counts:
        all_years.update(years)
    if not all_years:
        return pd.DataFrame(columns=["year"])
    years_full = list(range(min(all_years), max(all_years) + 1))
    df = pd.DataFrame({"year": years_full})
    for i, (years, counts) in enumerate(per_term_counts, start=1):
        m = {int(y): int(c) for y, c in zip(years, counts)}
        df[f"{prefix}{i}"] = [m.get(y, 0) for y in years_full]
    return df

# ------------------------ Main ------------------------

def main():
    parser = argparse.ArgumentParser(description="Search-by-term then fetch yearly counts from API; write aligned CSVs.")
    parser.add_argument("--shared-path", default=default_shared_path(), help="Path to shared_path.txt that contains the port number")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Output directory for CSVs and mapping")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="HTTP timeout seconds")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="HTTP retries on failure (POST-only)")
    parser.add_argument("--backoff", type=float, default=DEFAULT_BACKOFF, help="Exponential backoff base (POST-only)")
    parser.add_argument("--settle-sec", type=float, default=POST_TO_GET_SETTLE_SEC, help="Delay after POSTs before GET yearly_counts")
    args = parser.parse_args()

    # --- 1) Read the port number from shared_path.txt
    port_str = read_port_from_shared_path(args.shared_path)
    if not port_str or not port_str.strip().isdigit():
        print(f"[FATAL] Could not read a valid port from: {args.shared_path}")
        sys.exit(1)
    port = int(port_str.strip())

    # --- 2) Build URLs (NOTE: patents search moved to /api/search)
    base = f"http://localhost:{port}"
    patents_search_url = f"{base}/api/search"  # changed from /api/search_ops
    science_search_url = f"{base}/api/scientific_search_merge"
    patents_counts_url = f"{base}/api/patents/yearly_counts"
    publications_counts_url = f"{base}/api/publications/yearly_counts"

    print(f"[OK] Base: {base}")
    print(f"[OK] Search endpoints: {patents_search_url} | {science_search_url}")
    print(f"[OK] Yearly counts:    {patents_counts_url} | {publications_counts_url}")

    pat_series: List[Tuple[List[int], List[int]]] = []
    pub_series: List[Tuple[List[int], List[int]]] = []

    for idx, term in enumerate(SEARCH_TERMS, start=1):
        print(f"\n=== [{idx}/{len(SEARCH_TERMS)}] Term: {term} ===")

        # 3.1) Trigger search on patents via /api/search (AND + nested OR over title)
        pat_payload = build_patents_payload(term)
        r_pat = post_with_retries(patents_search_url, pat_payload, args.timeout, args.retries, args.backoff)
        if r_pat is not None:
            print(f"    Patents search (/api/search): {r_pat.status_code}")
        else:
            print("    Patents search: FAILED (continuing)")

        # 3.2) Trigger search on publications (unchanged)
        r_pub = post_with_retries(science_search_url, {"query": term}, args.timeout, args.retries, args.backoff)
        if r_pub is not None:
            print(f"    Publications search: {r_pub.status_code}")
        else:
            print("    Publications search: FAILED (continuing)")

        # small settle delay so backend can update any "current" cache (if applicable)
        if args.settle_sec > 0:
            time.sleep(args.settle_sec)

        # 3.3) Fetch yearly counts for THIS term (tries ?tech_label=term then plain)
        try:
            pat_years, pat_counts = fetch_yearly_counts(base, "/api/patents/yearly_counts", term, args.timeout)
            print(f"    Patents yearly counts: {len(pat_counts)} points, span {min(pat_years)}–{max(pat_years)}")
        except Exception as e:
            print(f"[WARN] Failed to fetch patents yearly counts for '{term}': {e}")
            pat_years, pat_counts = [], []

        try:
            pub_years, pub_counts = fetch_yearly_counts(base, "/api/publications/yearly_counts", term, args.timeout)
            print(f"    Publications yearly counts: {len(pub_counts)} points, span {min(pub_years)}–{max(pub_years)}")
        except Exception as e:
            print(f"[WARN] Failed to fetch publications yearly counts for '{term}': {e}")
            pub_years, pub_counts = [], []

        pat_series.append((pat_years, pat_counts))
        pub_series.append((pub_years, pub_counts))

    # --- 4) Build wide continuous DataFrames
    df_pat = to_wide_continuous(pat_series, "pat")
    df_pub = to_wide_continuous(pub_series, "pub")

    print(f"\nDataFrame shapes -> patents: {df_pat.shape} | pubs: {df_pub.shape}")

    # --- 5) Save outputs
    os.makedirs(args.outdir, exist_ok=True)
    patents_csv = os.path.join(args.outdir, "patents.csv")
    publications_csv = os.path.join(args.outdir, "publications.csv")
    mapping_txt = os.path.join(args.outdir, "techs_order.txt")

    try:
        df_pat.to_csv(patents_csv, index=False)
        print(f"✓ Wrote patents CSV: {patents_csv}")
    except Exception as e:
        print(f"✗ Error writing patents CSV: {e}")

    try:
        df_pub.to_csv(publications_csv, index=False)
        print(f"✓ Wrote publications CSV: {publications_csv}")
    except Exception as e:
        print(f"✗ Error writing publications CSV: {e}")

    try:
        with open(mapping_txt, "w", encoding="utf-8") as f:
            for i, term in enumerate(SEARCH_TERMS, start=1):
                f.write(f"pat{i}; {term}\n")
            for i, term in enumerate(SEARCH_TERMS, start=1):
                f.write(f"pub{i}; {term}\n")
        print(f"✓ Wrote mapping file: {mapping_txt}")
    except Exception as e:
        print(f"✗ Error writing mapping file: {e}")

    print("\nDone.")

if __name__ == "__main__":
    main()
