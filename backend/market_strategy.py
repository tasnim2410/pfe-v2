"""
market_strategy.py
Utilities for:
- Loading EPO OPS credentials and handling token caching/rotation
- Fetching and parsing EPO OPS Legal status for a DOCDB publication number (JSON and XML fallback)
- Classifying per-country patent status (GRANTED / PENDING / DEAD)
- Loading a country GDP file and computing a Market Strategy Index (MSI)

Spec alignment (Innosabi):
- Protected = alive applications + any granted patent (either dead or alive)
- Pending countries get a 40% reduction
- WO / (pending) EP market size is calculated statistically across authorities
- Normalize so: US-only grant scores 1.0
"""

from __future__ import annotations

import os
import re
import time
from typing import Dict, List, Tuple, Optional, Any
import requests
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None  # type: ignore

TOKEN_URL_DEFAULT = "https://ops.epo.org/3.2/auth/accesstoken"
LEGAL_URL_DEFAULT = "https://ops.epo.org/3.2/rest-services/legal/publication/docdb/"

DEAD_PATTERNS = re.compile(
    r"(LAPS|LAPSE|LAPSED|EXPIRE|EXPIRED|WITHDRAWN|REFUS|REVOK|ABANDON|CEASED)",
    re.IGNORECASE
)

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
    # Special authorities (not direct countries)
    "WO": None,  # PCT
    "EP": None,  # EPO (regional)
}

# Default statistical distributions (coarse priors) for EP/WO contributions.
# Values approximately sum to 1.0.
EP_EXPECTED: Dict[str, float] = {
    "DEU": 0.22, "FRA": 0.15, "GBR": 0.15, "ITA": 0.10, "ESP": 0.08,
    "NLD": 0.07, "SWE": 0.05, "CHE": 0.06, "AUT": 0.04, "DNK": 0.03,
    "FIN": 0.03, "IRL": 0.02
}
WO_EXPECTED: Dict[str, float] = {
    "USA": 0.35, "CHN": 0.20, "JPN": 0.15, "KOR": 0.10, "DEU": 0.08,
    "FRA": 0.06, "GBR": 0.06
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
    try:
        from dotenv import load_dotenv
        here = Path(__file__).resolve()
        candidates = [
            here.with_name('.env'),
            here.parent.with_name('.env'),
            here.parent.parent.with_name('.env'),
        ]
        used_any = []
        for p in candidates:
            try:
                if p.exists():
                    load_dotenv(dotenv_path=p, override=True)
                    used_any.append(str(p))
            except Exception:
                continue
        if not used_any:
            load_dotenv(override=True)
    except Exception:
        pass

    creds: List[Dict[str, str]] = []
    base_key = os.getenv("CONSUMER_KEY")
    base_sec = os.getenv("CONSUMER_SECRET")
    if base_key and base_sec:
        creds.append({"key": base_key.strip(), "secret": base_sec.strip()})
    for i in range(10):
        k = os.getenv(f"CONSUMER_KEY_{i}")
        s = os.getenv(f"CONSUMER_SECRET_{i}")
        if k and s:
            creds.append({"key": k.strip(), "secret": s.strip()})
    if not creds:
        raise RuntimeError("No EPO OPS credentials found (.env: CONSUMER_KEY[_i]/CONSUMER_SECRET[_i]).")
    return creds


def build_token_cache(creds: List[Dict[str, str]]) -> List[Dict[str, float]]:
    return [{"token": None, "expiry": 0.0} for _ in creds]


def get_access_token(
    cred_idx: int,
    creds: List[Dict[str, str]],
    token_cache: List[Dict[str, float]],
    token_url: str = TOKEN_URL_DEFAULT,
    timeout: int = 15
) -> str:
    now = time.time()
    info = token_cache[cred_idx]
    if info["token"] and now < info["expiry"]:
        return str(info["token"])
    cred = creds[cred_idx]
    data = {"grant_type": "client_credentials", "client_id": cred["key"], "client_secret": cred["secret"]}
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    resp = requests.post(token_url, data=data, headers=headers, timeout=timeout)
    resp.raise_for_status()
    token = resp.json()["access_token"]
    info["token"] = token
    info["expiry"] = now + 3500
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
    token = get_access_token(cred_idx, creds, token_cache, token_url=token_url)
    url = f"{legal_base_url}{requests.utils.quote(str(publication_number_docdb))}"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    r = requests.get(url, headers=headers, timeout=timeout)
    if r.status_code in (403, 404):
        return None, f"{r.status_code} {r.reason}"
    if r.headers.get("Content-Type", "").lower().startswith("application/json"):
        try:
            return r.json(), None
        except Exception as e:
            return None, f"JSON parse error: {e}"
    if r.ok:
        try:
            return {"__xml__": r.text}, None
        except Exception as e:
            return None, f"XML read error: {e}"
    return None, f"HTTP {r.status_code} {r.reason}"


def parse_legal_json_or_xml(payload_obj: Any) -> List[Dict[str, Any]]:
    members: List[Dict[str, Any]] = []
    if payload_obj and "__xml__" not in (payload_obj if isinstance(payload_obj, dict) else {}):
        wpd = payload_obj.get("ops:world-patent-data", {})
        pf  = wpd.get("ops:patent-family", {})
        ms  = pf.get("ops:family-member", [])
        if isinstance(ms, dict):
            ms = [ms]

        def _get_text(x):
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

    xml_txt = payload_obj.get("__xml__") if isinstance(payload_obj, dict) else None
    if not xml_txt:
        return members
    root = ET.fromstring(xml_txt)
    ns = {"ops": "http://ops.epo.org", "ex": "http://www.epo.org/exchange"}
    for m in root.findall(".//ops:family-member", ns):
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
            l510 = le.find(".//ops:L510EP", ns)
            txt  = l510.text if l510 is not None else ""
            events.append({"code": code, "text": txt or ""})
        if country:
            members.append({"country": country, "kind": kind or "", "events": events})
    return members


def classify_member_status(member: Dict[str, Any]) -> str:
    """
    Returns: 'GRANTED' | 'PENDING' | 'DEAD'
    Rules:
      - If DOCDB kind starts with 'B' or 'C' -> GRANTED (count fully, even if later lapsed).
      - Else if any legal text/code matches DEAD_PATTERNS -> DEAD.
      - Else -> PENDING.
    """
    kind = (member.get("kind") or "").upper()
    if kind.startswith(("B", "C")):
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
    if pd is None:
        raise RuntimeError("pandas is required to load GDP; please install pandas.")
    import pandas as _pd
    gdf = _pd.read_csv(csv_path, sep=None, engine="python")
    year_cols = [c for c in gdf.columns if re.fullmatch(r"\d{4}", str(c))]
    if not year_cols:
        raise ValueError("GDP CSV should contain year columns like 2018, 2019, ...")
    year_cols = sorted(year_cols, key=lambda x: int(x))
    gdf[year_cols] = gdf[year_cols].ffill(axis=1).bfill(axis=1)
    latest_col = year_cols[-1]
    iso3_to_gdp: Dict[str, float] = {}
    for iso3, val in zip(gdf["Country Code"], gdf[latest_col]):
        try:
            f = float(val)
            if f == f:
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
    weights: Optional[Dict[str, float]] = None,
    *,
    ep_expected: Optional[Dict[str, float]] = None,
    wo_expected: Optional[Dict[str, float]] = None,
    apply_ep_for_granted: bool = True
) -> float:
    """
    MSI = Σ GDP(country) * weight(status) / GDP(USA)

    Defaults:
        weight(GRANTED)=1.0, weight(PENDING)=0.6, weight(DEAD)=0.0

    EP/WO handling:
      - If ep_expected / wo_expected are provided, add a distributed contribution:
            Σ ( GDP[iso3] * p(iso3) * weight(status) )
        for the corresponding authority (EP or WO).
      - If not provided, EP/WO are skipped.
      - Set apply_ep_for_granted=False to only distribute EP when PENDING (strict reading).
    """
    if weights is None:
        weights = {"GRANTED": 1.0, "PENDING": 0.6, "DEAD": 0.0}

    denom = float(us_gdp or 1.0)
    total = 0.0

    # First, add direct national contributions
    for st3, status in (status_by_country or {}).items():
        st3 = (st3 or "").upper()
        if st3 in ("WO", "EP"):
            continue  # handled below if distributions provided
        iso3 = st3_to_iso3_map.get(st3)
        if not iso3:
            continue
        gdp = iso3_gdp_map.get(iso3)
        if gdp is None:
            continue
        total += float(gdp) * float(weights.get(status, 0.0))

    # EP distributed contribution
    if "EP" in (status_by_country or {}) and ep_expected:
        st = status_by_country["EP"]
        if st != "DEAD" and (apply_ep_for_granted or st == "PENDING"):
            w = float(weights.get(st, 0.0))
            ep_sum = 0.0
            for iso3, p in ep_expected.items():
                g = iso3_gdp_map.get(iso3)
                if g is not None:
                    ep_sum += float(g) * float(p)
            total += ep_sum * w

    # WO distributed contribution
    if "WO" in (status_by_country or {}) and wo_expected:
        st = status_by_country["WO"]
        if st != "DEAD":  # doc emphasizes pending, but granting at nationals later; we still allow contribution
            w = float(weights.get(st, 0.0))
            wo_sum = 0.0
            for iso3, p in wo_expected.items():
                g = iso3_gdp_map.get(iso3)
                if g is not None:
                    wo_sum += float(g) * float(p)
            total += wo_sum * w

    return total / denom if denom else 0.0


__all__ = [
    "TOKEN_URL_DEFAULT",
    "LEGAL_URL_DEFAULT",
    "DEAD_PATTERNS",
    "ST3_TO_ISO3",
    "EP_EXPECTED",
    "WO_EXPECTED",
    "load_api_credentials",
    "build_token_cache",
    "get_access_token",
    "fetch_legal_raw",
    "parse_legal_json_or_xml",
    "classify_member_status",
    "load_gdp_map",
    "compute_msi",
]
