"""
market_strategy.py
Utilities for:
- Loading EPO OPS credentials and handling token caching/rotation
- Fetching and parsing EPO OPS Legal status for a DOCDB publication number (JSON and XML fallback)
- Classifying per-country patent status (GRANTED / PENDING / DEAD)
- Loading a country GDP file and computing a Market Strategy Index (MSI)

Drop this file next to your backend app (e.g., backend/market_strategy.py) and import the functions in app.py.

Design notes:
- We keep functions side-effect free (no Flask imports). Callers pass paths/objs as needed.
- We implement robust parsing for both OPS JSON and OPS XML payloads.
- Status rules: kind B* => GRANTED; explicit lapse/withdraw/expired => DEAD; else PENDING.
- MSI = sum( GDP[country] * weight(status) ) / GDP[USA], weights: {GRANTED:1.0, PENDING:0.6, DEAD:0.0}.
"""

from __future__ import annotations

import os
import re
import time
from typing import Dict, List, Tuple, Optional, Any
import requests
import xml.etree.ElementTree as ET
from pathlib import Path

# Pandas is optional until you load GDP. We import softly.
try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - allow usage without pandas until MSI is needed
    pd = None  # type: ignore

# -------------------------
# Constants & Regex helpers
# -------------------------

# EPO OPS endpoints (defaults; can be overridden via function args)
TOKEN_URL_DEFAULT = "https://ops.epo.org/3.2/auth/accesstoken"
LEGAL_URL_DEFAULT = "https://ops.epo.org/3.2/rest-services/legal/publication/docdb/"

# Strong signals that an application is no longer active in that country.
DEAD_PATTERNS = re.compile(
    r"(LAPS|LAPSE|LAPSED|EXPIRE|EXPIRED|WITHDRAWN|REFUS|REVOK|ABANDON|CEASED)",
    re.IGNORECASE
)

# Minimal practical WIPO ST.3 (DOCDB) -> ISO3 mapping for GDP lookup.
ST3_TO_ISO3: Dict[str, Optional[str]] = {
    # Americas
    "US": "USA", "CA": "CAN", "MX": "MEX", "BR": "BRA", "AR": "ARG", "CL": "CHL",
    "CO": "COL", "PE": "PER",
    # Europe
    "GB": "GBR", "UK": "GBR", "DE": "DEU", "FR": "FRA", "IT": "ITA", "ES": "ESP",
    "NL": "NLD", "SE": "SWE", "CH": "CHE", "AT": "AUT", "BE": "BEL", "DK": "DNK",
    "FI": "FIN", "NO": "NOR", "IE": "IRL", "PL": "POL", "CZ": "CZE",
    # Asia
    "CN": "CHN", "JP": "JPN", "KR": "KOR", "IN": "IND", "SG": "SGP", "MY": "MYS",
    "TH": "THA", "ID": "IDN", "TW": "TWN", "HK": "HKG",
    # Oceania
    "AU": "AUS", "NZ": "NZL",
    # Middle East
    "IL": "ISR", "SA": "SAU", "AE": "ARE", "TR": "TUR",
    # Africa
    "ZA": "ZAF", "EG": "EGY", "MA": "MAR",
    # Special authorities (not countries)
    "WO": None,  # PCT publication
    "EP": None,  # EPO regional publication (handled specially by caller if needed)
}

# -------------------------
# Credentials & Token cache
# -------------------------

def load_api_credentials() -> List[Dict[str, str]]:
    """
    Load EPO OPS API credentials from environment variables.

    Supports:
      CONSUMER_KEY / CONSUMER_SECRET
      CONSUMER_KEY_0..9 / CONSUMER_SECRET_0..9
    """
    # Ensure we load the backend/.env that sits next to this file if present.
    # Fallback to default search (current dir/parents) otherwise.
    try:
        from dotenv import load_dotenv
        here = Path(__file__).resolve()
        candidates = [
            here.with_name('.env'),                    # backend/.env
            here.parent.with_name('.env'),             # technology-trend-analysis/.env
            here.parent.parent.with_name('.env'),      # repo-root .env
        ]
        used_any = []
        for p in candidates:
            try:
                if p.exists():
                    load_dotenv(dotenv_path=p, override=True)
                    used_any.append(str(p))
            except Exception:
                continue
        # Fallback to default search if none found
        if not used_any:
            load_dotenv(override=True)
        # Optional diagnostics without exposing secrets
        if os.getenv("OPS_DEBUG", "0") == "1":
            keys = sorted([k for k in os.environ.keys() if k.startswith("CONSUMER_KEY")])
            secs = sorted([k for k in os.environ.keys() if k.startswith("CONSUMER_SECRET")])
            print({
                "ops_debug": True,
                "env_paths_tried": [str(x) for x in candidates],
                "env_paths_used": used_any,
                "keys_found": keys,
                "secrets_found": secs,
                "key_count": len(keys),
                "secret_count": len(secs),
            })
    except Exception:
        pass

    creds: List[Dict[str, str]] = []

    # Base pair (optional)
    base_key = os.getenv("CONSUMER_KEY")
    base_sec = os.getenv("CONSUMER_SECRET")
    if base_key and base_sec:
        creds.append({"key": base_key.strip(), "secret": base_sec.strip()})

    # Indexed pairs (optional)
    for i in range(10):
        k = os.getenv(f"CONSUMER_KEY_{i}")
        s = os.getenv(f"CONSUMER_SECRET_{i}")
        if k and s:
            creds.append({"key": k.strip(), "secret": s.strip()})

    if not creds:
        raise RuntimeError("No EPO OPS credentials found (.env: CONSUMER_KEY[_i]/CONSUMER_SECRET[_i]).")
    return creds



def build_token_cache(creds: List[Dict[str, str]]) -> List[Dict[str, float]]:
    """
    Build a simple per-credential token cache.
    Each element holds {"token": str|None, "expiry": float_unix_seconds}.
    """
    return [{"token": None, "expiry": 0.0} for _ in creds]


def get_access_token(
    cred_idx: int,
    creds: List[Dict[str, str]],
    token_cache: List[Dict[str, float]],
    token_url: str = TOKEN_URL_DEFAULT,
    timeout: int = 15
) -> str:
    """
    Returns a Bearer token for OPS, using a per-credential cache.

    - cred_idx selects which credential to use (for rotation).
    - Token cached approx 58 minutes (3500 s).

    Raises:
        requests.HTTPError if the token request fails.
    """
    now = time.time()
    info = token_cache[cred_idx]
    if info["token"] and now < info["expiry"]:
        return str(info["token"])
    cred = creds[cred_idx]
    data = {
        "grant_type": "client_credentials",
        "client_id": cred["key"],
        "client_secret": cred["secret"]
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    resp = requests.post(token_url, data=data, headers=headers, timeout=timeout)
    resp.raise_for_status()
    token = resp.json()["access_token"]
    info["token"] = token
    info["expiry"] = now + 3500  # ~58 minutes
    return token


# -------------------------
# OPS Legal fetch & parsing
# -------------------------

def fetch_legal_raw(
    publication_number_docdb: str,
    cred_idx: int,
    creds: List[Dict[str, str]],
    token_cache: List[Dict[str, float]],
    legal_base_url: str = LEGAL_URL_DEFAULT,
    token_url: str = TOKEN_URL_DEFAULT,
    timeout: int = 20
) -> Tuple[Optional[Any], Optional[str]]:
    """
    Fetch OPS 'legal' data for a DOCDB publication number.
    Tries JSON first (Accept: application/json), falls back to XML if needed.

    Args:
        publication_number_docdb: e.g., "US.10339814.B2" (DOCDB format). Caller ensures correct formatting.
        cred_idx: credential index to use (rotate to spread the load).
        creds, token_cache: objects returned by load_api_credentials() and build_token_cache().

    Returns:
        (payload_obj, error_msg)
        - If JSON available, payload_obj is dict.
        - If only XML available, payload_obj is {"__xml__": "<raw xml text>"}.
        - On error, payload_obj=None and error_msg contains reason (HTTP code or parse error).
    """
    token = get_access_token(cred_idx, creds, token_cache, token_url=token_url)
    url = f"{legal_base_url}{requests.utils.quote(str(publication_number_docdb))}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json"
    }
    r = requests.get(url, headers=headers, timeout=timeout)
    # Forbidden or Not Found: return error
    if r.status_code in (403, 404):
        return None, f"{r.status_code} {r.reason}"
    # JSON happy path
    if r.headers.get("Content-Type", "").lower().startswith("application/json"):
        try:
            return r.json(), None
        except Exception as e:
            return None, f"JSON parse error: {e}"
    # Fallback to XML
    if r.ok:
        try:
            return {"__xml__": r.text}, None
        except Exception as e:
            return None, f"XML read error: {e}"
    return None, f"HTTP {r.status_code} {r.reason}"


def parse_legal_json_or_xml(payload_obj: Any) -> List[Dict[str, Any]]:
    """
    Normalize OPS legal payload into a list of per-country members:

    Returns:
        [
          {
            'country': 'US',
            'kind': 'B2',
            'events': [{'code': 'MAFP', 'text': '...'}, ...]
          },
          ...
        ]
    """
    members: List[Dict[str, Any]] = []

    # JSON path (ops:world-patent-data -> ops:patent-family -> ops:family-member[*])
    if payload_obj and "__xml__" not in (payload_obj if isinstance(payload_obj, dict) else {}):
        wpd = payload_obj.get("ops:world-patent-data", {})
        pf  = wpd.get("ops:patent-family", {})
        ms  = pf.get("ops:family-member", [])
        if isinstance(ms, dict):  # single member case
            ms = [ms]

        def _get_text(x):
            # Some OPS JSON nodes wrap value as {"$": "..."}; accept both plain and wrapped.
            if isinstance(x, dict) and "$" in x:
                return x.get("$")
            return x

        for m in ms or []:
            pubref = m.get("publication-reference", {})
            docs   = pubref.get("document-id", [])
            if isinstance(docs, dict):
                docs = [docs]
            country, kind = None, None
            for d in docs:
                if d.get("@document-id-type") == "docdb":
                    country = _get_text(d.get("country"))
                    kind    = _get_text(d.get("kind"))
            events: List[Dict[str, str]] = []
            raw_legals = m.get("ops:legal", [])
            if isinstance(raw_legals, dict):
                raw_legals = [raw_legals]
            if isinstance(raw_legals, list):
                for le in raw_legals:
                    if not le:
                        continue
                    code = le.get("@code")
                    # L500EP/L510EP contains the human-readable free-text
                    txt = ""
                    l500 = le.get("ops:L500EP", {})
                    if isinstance(l500, dict):
                        l510 = l500.get("ops:L510EP")
                        if isinstance(l510, dict):
                            txt = _get_text(l510) or ""
                        elif isinstance(l510, list):
                            txt = " | ".join([_get_text(x) or "" for x in l510])
                    events.append({"code": code, "text": txt})
            if country:
                members.append({"country": country, "kind": kind or "", "events": events})
        return members

    # XML path
    xml_txt = payload_obj.get("__xml__") if isinstance(payload_obj, dict) else None
    if not xml_txt:
        return members

    root = ET.fromstring(xml_txt)
    ns = {"ops": "http://ops.epo.org", "ex": "http://www.epo.org/exchange"}
    for m in root.findall(".//ops:family-member", ns):
        # publication-reference/document-id[@document-id-type='docdb']
        country = ""
        kind = ""
        for d in m.findall(".//ex:publication-reference/ex:document-id", ns):
            if d.get("document-id-type") == "docdb":
                c = d.findtext("ex:country", default="", namespaces=ns) or ""
                k = d.findtext("ex:kind", default="", namespaces=ns) or ""
                country = c or country
                kind    = k or kind
        events: List[Dict[str, str]] = []
        for le in m.findall(".//ops:legal", ns):
            code = le.get("code")
            # try to get free text line L510EP
            l510 = le.find(".//ops:L510EP", ns)
            txt  = l510.text if l510 is not None else ""
            events.append({"code": code, "text": txt or ""})
        if country:
            members.append({"country": country, "kind": kind or "", "events": events})
    return members


def classify_member_status(member: Dict[str, Any]) -> str:
    """
    Heuristic classification for a single family member in a given country.

    Returns:
        'GRANTED' | 'PENDING' | 'DEAD'

    Rules:
      - If DOCDB kind starts with 'B' -> GRANTED (counted even if later lapsed).
      - Else if any legal text/code matches DEAD_PATTERNS -> DEAD.
      - Else -> PENDING.
    """
    kind = (member.get("kind") or "").upper()
    if kind.startswith("B"):
        return "GRANTED"
    for ev in member.get("events", []) or []:
        t = f"{ev.get('code', '')}-{ev.get('text', '')}"
        if DEAD_PATTERNS.search(t or ""):
            return "DEAD"
    return "PENDING"


# -------------------------
# GDP & MSI
# -------------------------

def load_gdp_map(csv_path: str) -> Tuple[Dict[str, float], float]:
    """
    Load a GDP CSV (World Bank-like format) and return:
      - iso3_to_gdp: dict ISO3 -> latest available GDP (float)
      - us_gdp: shortcut GDP for USA (float)

    The CSV is expected to have:
      - a 'Country Code' column (ISO3 codes),
      - multiple year columns 'YYYY'.

    We take the last year column and forward/back fill if needed.
    """
    if pd is None:
        raise RuntimeError("pandas is required to load GDP; please install pandas.")

    import pandas as _pd  # local alias to avoid shadowing

    gdf = _pd.read_csv(csv_path, sep=None, engine="python")
    # Identify year columns
    year_cols = [c for c in gdf.columns if re.fullmatch(r"\d{4}", str(c))]
    if not year_cols:
        raise ValueError("GDP CSV should contain year columns like 2018, 2019, ...")

    # Use the last chronological year as 'latest'
    year_cols = sorted(year_cols, key=lambda x: int(x))
    # forward/back fill across years to reduce NaNs; then take last column
    gdf[year_cols] = gdf[year_cols].ffill(axis=1).bfill(axis=1)
    latest_col = year_cols[-1]

    iso3_to_gdp: Dict[str, float] = {}
    for iso3, val in zip(gdf["Country Code"], gdf[latest_col]):
        try:
            # Convert to float; skip NaN or None
            f = float(val)
            if f == f:  # not NaN
                iso3_to_gdp[str(iso3)] = f
        except Exception:
            continue

    us_gdp = iso3_to_gdp.get("USA", 0.0)
    return iso3_to_gdp, us_gdp


def compute_msi(
    status_by_country: Dict[str, str],
    iso3_gdp_map: Dict[str, float],
    us_gdp: float,
    st3_to_iso3_map: Dict[str, Optional[str]] = ST3_TO_ISO3,
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Compute Market Strategy Index (MSI):

        MSI = sum( GDP[country] * weight(status) ) / GDP[USA]

    Default weights:
        GRANTED: 1.0
        PENDING: 0.6
        DEAD:    0.0

    EP/WO are skipped (not countries). Caller may handle EP expansion if desired.
    """
    if weights is None:
        weights = {"GRANTED": 1.0, "PENDING": 0.6, "DEAD": 0.0}

    denom = float(us_gdp or 1.0)
    total = 0.0

    for st3, status in (status_by_country or {}).items():
        st3 = (st3 or "").upper()
        if st3 in ("WO", "EP"):  # not countries -> skip to avoid double counting
            continue
        iso3 = st3_to_iso3_map.get(st3)
        if not iso3:
            continue
        gdp = iso3_gdp_map.get(iso3)
        if gdp is None:
            continue
        total += float(gdp) * float(weights.get(status, 0.0))

    return total / denom if denom else 0.0


__all__ = [
    "TOKEN_URL_DEFAULT",
    "LEGAL_URL_DEFAULT",
    "DEAD_PATTERNS",
    "ST3_TO_ISO3",
    "load_api_credentials",
    "build_token_cache",
    "get_access_token",
    "fetch_legal_raw",
    "parse_legal_json_or_xml",
    "classify_member_status",
    "load_gdp_map",
    "compute_msi",
]
