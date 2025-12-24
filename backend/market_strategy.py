"""
market_strategy.py
Enhanced with improved legal event classification and XML fetch utilities
"""

from __future__ import annotations
from db import db,EpoEventCode
import os
import re
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
import requests
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    import pandas as pd
except Exception:
    pd = None

logger = logging.getLogger(__name__)

TOKEN_URL_DEFAULT = "https://ops.epo.org/3.2/auth/accesstoken"
LEGAL_URL_DEFAULT = "https://ops.epo.org/3.2/rest-services/legal/publication/docdb/"

# Enhanced keyword sets from notebook
DEAD_KEYWORDS = [
    "annulment", "annulled", "revoked", "revocation", "cancelled",
    "cancellation", "void", "declared void", "null and void",
    "ceased", "cessation", "extinction", "forfeiture",
    "abandoned", "deemed abandoned", "surrender", "surrendered",
    "renunciation", "renounced", "waiver",
    "dedication filed", "disclaimer and dedication",
    "non payment of the annual fee", "renewal fees not paid",
    "loss of rights", "completion of term", "expired",
    "patent refused", "request refused", "refusal decision now final",
    "supplementary protection certificate rejected",
    "spc annulled", "paediatric extension rejected",
    "extension of term refused",
    "no longer valid", "deemed void", "not in force",
    "always having been void", "no legal effect",
    "termination", "lapse", "expiry", "withdrawal",
    "discontinuation", "non-payment", "expiration",
    "nullification", "invalid", "ceased to have effect"
]

ALIVE_KEYWORDS = [
    "grant", "maintenance", "fee payment",
    "right in force", "in force", "validated", "validation",
    "renewal", "annuity",
    "reinstated", "reinstatement", "restored", "restoration",
    "re-establishment", "reestablishment",
    "revived", "revival", "resumption",
    "authorization for restitution", "authorisation for restitution",
    "restoration of rights", "patent restored", "restoration after lapse"
]

PENDING_KEYWORDS = [
    "application", "examination", "publication",
    "pre-grant", "granting procedure",
    "suspension of granting procedure",
    "opposition pending", "appeal pending", "procedural status"
]

NEUTRAL_KEYWORDS = [
    "correction", "corresponds to",
    "translation of claims", "translation filed",
    "protection beyond ip right term"
]

GRANTED_KEYWORDS = [
    "patent granted", "grant of patent", "decision to grant", "granted patent",
    "b1 document published", "b2 document published",
    "patent specification published",
    "takes effect as a national patent",
    "ep patent valid", "european patent takes effect",
    "validated european patent",
    "patent sealed", "patent issued",
    "valid patent", "right in force",
]

PREGRANT_BLOCKERS = [
    "application", "pre-grant", "granting procedure",
    "suspension of granting", "intention to grant"
]

ST3_TO_ISO3: Dict[str, Optional[str]] = {
    "US": "USA", "CA": "CAN", "MX": "MEX", "BR": "BRA", "AR": "ARG", "CL": "CHL",
    "CO": "COL", "PE": "PER",
    "GB": "GBR", "UK": "GBR", "DE": "DEU", "FR": "FRA", "IT": "ITA", "ES": "ESP",
    "NL": "NLD", "SE": "SWE", "CH": "CHE", "AT": "AUT", "BE": "BEL", "DK": "DNK",
    "FI": "FIN", "NO": "NOR", "IE": "IRL", "PL": "POL", "CZ": "CZE",
    "CN": "CHN", "JP": "JPN", "KR": "KOR", "IN": "IND", "SG": "SGP", "MY": "MYS",
    "TH": "THA", "ID": "IDN", "TW": "TWN", "HK": "HKG",
    "AU": "AUS", "NZ": "NZL",
    "IL": "ISR", "SA": "SAU", "AE": "ARE", "TR": "TUR",
    "ZA": "ZAF", "EG": "EGY", "MA": "MAR",
    "WO": None, "EP": None,
}

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
    """Load EPO OPS API credentials from environment variables."""
    try:
        from dotenv import load_dotenv
        here = Path(__file__).resolve()
        candidates = [
            here.with_name('.env'),
            here.parent.with_name('.env'),
            here.parent.parent.with_name('.env'),
        ]
        for p in candidates:
            try:
                if p.exists():
                    load_dotenv(dotenv_path=p, override=True)
                    break
            except Exception:
                continue
        else:
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
        raise RuntimeError("No EPO OPS credentials found.")
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


def fetch_legal_xml(
    publication_number_docdb: str,
    cred_idx: int,
    creds: List[Dict[str, str]],
    token_cache: List[Dict[str, float]],
    legal_base_url: str = LEGAL_URL_DEFAULT,
    token_url: str = TOKEN_URL_DEFAULT,
    timeout: int = 20
) -> Dict[str, Any]:
    """
    Fetch XML for a single publication number.
    
    Returns:
        {
            'publication_number': str,
            'xml': str or None,
            'error': str or None
        }
    """
    try:
        token = get_access_token(cred_idx, creds, token_cache, token_url=token_url)
        url = f"{legal_base_url}{requests.utils.quote(str(publication_number_docdb))}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/xml"
        }
        
        r = requests.get(url, headers=headers, timeout=timeout)
        
        if r.status_code == 403:
            return {
                "publication_number": publication_number_docdb,
                "xml": None,
                "error": "403 Forbidden"
            }
        
        if r.status_code == 404:
            return {
                "publication_number": publication_number_docdb,
                "xml": None,
                "error": "404 Not Found"
            }
        
        r.raise_for_status()
        
        return {
            "publication_number": publication_number_docdb,
            "xml": r.text,
            "error": None
        }
        
    except requests.HTTPError as he:
        code = getattr(he.response, "status_code", "Unknown")
        return {
            "publication_number": publication_number_docdb,
            "xml": None,
            "error": f"HTTPError: {code}"
        }
    except Exception as e:
        return {
            "publication_number": publication_number_docdb,
            "xml": None,
            "error": str(e)
        }


def parse_legal_json_or_xml(payload_obj: Any) -> List[Dict[str, Any]]:
    """Parse legal data and extract events with enhanced classification."""
    members: List[Dict[str, Any]] = []
    
    # JSON parsing
    if payload_obj and "__xml__" not in (payload_obj if isinstance(payload_obj, dict) else {}):
        wpd = payload_obj.get("ops:world-patent-data", {})
        pf = wpd.get("ops:patent-family", {})
        ms = pf.get("ops:family-member", [])
        if isinstance(ms, dict):
            ms = [ms]

        def _get_text(x):
            if isinstance(x, dict) and "$" in x:
                return x.get("$")
            return x

        for m in ms or []:
            pubref = m.get("publication-reference", {})
            docs = pubref.get("document-id", [])
            if isinstance(docs, dict):
                docs = [docs]
            country, kind = None, None
            for d in docs:
                if d.get("@document-id-type") == "docdb":
                    country = _get_text(d.get("country"))
                    kind = _get_text(d.get("kind"))
            
            events: List[Dict[str, str]] = []
            raw_legals = m.get("ops:legal", [])
            if isinstance(raw_legals, dict):
                raw_legals = [raw_legals]
            if isinstance(raw_legals, list):
                for le in raw_legals:
                    if not le:
                        continue
                    code = le.get("@code", "")
                    desc = le.get("@desc", "")
                    influence = le.get("@infl", "")
                    txt = ""
                    l500 = le.get("ops:L500EP", {})
                    if isinstance(l500, dict):
                        l510 = l500.get("ops:L510EP")
                        if isinstance(l510, dict):
                            txt = _get_text(l510) or ""
                        elif isinstance(l510, list):
                            txt = " | ".join([_get_text(x) or "" for x in l510])
                    events.append({
                        "code": code,
                        "desc": desc,
                        "influence": influence,
                        "text": txt
                    })
            
            if country:
                members.append({
                    "country": country,
                    "kind": kind or "",
                    "events": events
                })
        return members

    # XML parsing
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
                kind = k or kind
        
        events: List[Dict[str, str]] = []
        for le in m.findall(".//ops:legal", ns):
            code = le.get("code", "")
            desc = le.get("desc", "")
            influence = le.get("infl", "")
            l510 = le.find(".//ops:L510EP", ns)
            txt = l510.text if l510 is not None else ""
            events.append({
                "code": code,
                "desc": desc,
                "influence": influence,
                "text": txt or ""
            })
        
        if country:
            members.append({
                "country": country,
                "kind": kind or "",
                "events": events
            })
    
    return members


def classify_member_status(member: Dict[str, Any]) -> str:
    """
    Enhanced classification using the notebook's logic.
    
    Returns: 'GRANTED' | 'ALIVE' | 'PENDING' | 'DEAD'
    """
    events = member.get("events", [])
    kind = (member.get("kind") or "").upper()
    
    has_grant = False
    has_dead = False
    has_alive = False
    has_pending = False
    
    # Check kind code (B/C = granted)
    if kind.startswith(("B", "C")):
        has_grant = True
    
    # Analyze events
    for ev in events:
        code = (ev.get("code") or "").strip()
        desc = (ev.get("desc") or "").strip()
        influence = (ev.get("influence") or "").strip()
        text = (ev.get("text") or "").strip()
        
        combined = f"{code} {desc} {text}".lower()
        
        # Check for grant (not blocked by pre-grant context)
        if not any(blocker in combined for blocker in PREGRANT_BLOCKERS):
            if any(kw in combined for kw in GRANTED_KEYWORDS):
                has_grant = True
        
        # Check death
        if influence == "-" and any(kw in combined for kw in DEAD_KEYWORDS):
            has_dead = True
        
        # Check alive
        if influence == "+" and any(kw in combined for kw in ALIVE_KEYWORDS):
            has_alive = True
        
        # Check pending
        if any(kw in combined for kw in PENDING_KEYWORDS):
            has_pending = True
    
    # Decision logic
    if has_grant:
        if has_dead:
            return "DEAD"
        return "GRANTED"
    
    if has_dead:
        return "DEAD"
    
    if has_alive:
        return "ALIVE"
    
    if has_pending:
        return "PENDING"
    
    return "PENDING"


def to_docdb(pub: str) -> str:
    """
    Best-effort normalization to DOCDB shape 'CC.NUM[.KIND]' for OPS legal endpoint.
    """
    s = str(pub or "").strip()
    if not s:
        return s
    if "." in s:
        return s
    alnum = re.sub(r"[^A-Za-z0-9]", "", s)
    m = re.match(r"^([A-Za-z]{2})(\d+)([A-Za-z]\d{0,2})?$", alnum)
    if m:
        cc, num, kind = m.group(1).upper(), m.group(2), (m.group(3) or "").upper()
        return f"{cc}.{num}.{kind}" if kind else f"{cc}.{num}"
    return s


def infer_status_offline(kind_code: str, family_jurs: List[str]) -> str:
    """Infer status from kind code for offline mode."""
    if kind_code:
        clean = re.sub(r"[^A-Za-z0-9]", "", str(kind_code).upper())
        m = re.search(r"([A-Z]\d{0,2})$", clean)
        if m:
            k0 = m.group(1)[0]
            if k0 in ("B", "C"):
                return "GRANTED"
            elif k0 == "A":
                return "PENDING"
    return "PENDING"

# -------------------------
# GDP & MSI
# -------------------------

def load_gdp_map(csv_path: str) -> Tuple[Dict[str, float], float]:
    if pd is None:
        raise RuntimeError("pandas required")
    gdf = pd.read_csv(csv_path, sep=None, engine="python")
    year_cols = [c for c in gdf.columns if re.fullmatch(r"\d{4}", str(c))]
    if not year_cols:
        raise ValueError("GDP CSV should contain year columns")
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
    """
    if weights is None:
        weights = {
            "GRANTED": 1.0,
            "ALIVE": 1.0,
            "PENDING": 0.6,
            "DEAD": 0.0
        }

    denom = float(us_gdp or 1.0)
    total = 0.0

    # Direct national contributions
    for st3, status in (status_by_country or {}).items():
        st3 = (st3 or "").upper()
        if st3 in ("WO", "EP"):
            continue
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
            ep_sum = sum(
                float(iso3_gdp_map.get(iso3, 0)) * float(p)
                for iso3, p in ep_expected.items()
            )
            total += ep_sum * w

    # WO distributed contribution
    if "WO" in (status_by_country or {}) and wo_expected:
        st = status_by_country["WO"]
        if st != "DEAD":
            w = float(weights.get(st, 0.0))
            wo_sum = sum(
                float(iso3_gdp_map.get(iso3, 0)) * float(p)
                for iso3, p in wo_expected.items()
            )
            total += wo_sum * w

    return total / denom if denom else 0.0


__all__ = [
    "load_api_credentials",
    "build_token_cache",
    "get_access_token",
    "fetch_legal_raw",
    "fetch_legal_xml",
    "parse_legal_json_or_xml",
    "classify_member_status",
    "to_docdb",
    "infer_status_offline",
    "load_gdp_map",
    "compute_msi",
    "DEAD_KEYWORDS",
    "ALIVE_KEYWORDS",
    "PENDING_KEYWORDS",
    "GRANTED_KEYWORDS",
    "ST3_TO_ISO3",
    "EP_EXPECTED",
    "WO_EXPECTED",
]

def load_event_codes() -> bool:
    event_codes_path = r"C:\Users\tasni\OneDrive\Documents\PFE\code\technology-trend-analysis\backend related folders\epo_event_codes.csv"
    try:
        if pd is None:
            raise RuntimeError("pandas is not available")

        event_codes = pd.read_csv(event_codes_path)

        # Normalize known column variants
        event_codes = event_codes.rename(columns={
            "Event code": "Event-code",
            "Event Code": "Event-code",
            "Event-class": "Event-class",
            "Event class": "Event-class",
            "Event Class": "Event-class",
        })

        required = [
            "Authority",
            "Event-code",
            "Influence",
            "Description ENG",
            "Description ORI",
            "Event-class",
            "Event-class Description",
            "mapped_status",
            "is_granted",
        ]
        missing = [c for c in required if c not in event_codes.columns]
        if missing:
            raise KeyError(
                f"Missing columns in epo_event_codes.csv: {missing}. Got: {list(event_codes.columns)}"
            )

        for index, row in event_codes.iterrows():
            influence_raw = row["Influence"]
            influence = "" if influence_raw is None else str(influence_raw).strip()
            if len(influence) > 1:
                influence = influence[0]
            event_code_db = EpoEventCode(
                id=index,
                authority=row["Authority"],
                event_code=str(row["Event-code"]).upper().strip(),
                influence=influence,
                description_eng=row["Description ENG"],
                description_ori=row["Description ORI"],
                event_class=row["Event-class"],
                event_class_description=row["Event-class Description"],
                mapped_status=row["mapped_status"],
                is_granted=bool(row["is_granted"]),
            )
            db.session.add(event_code_db)

        db.session.commit()
        return True

    except Exception:
        db.session.rollback()
        logger.exception("Failed to load epo_event_codes.csv into DB")
        return False


def build_event_code_lookups() -> tuple[dict, dict]:
    """Build (code_lookup, is_granted_lookup) from EpoEventCode table.

    Must be called inside a Flask app context (or request context), because it uses db.session.
    """
    rows = db.session.query(EpoEventCode.event_code, EpoEventCode.influence, EpoEventCode.mapped_status, EpoEventCode.is_granted).all()
    code_lookup: dict = {}
    is_granted_lookup: dict = {}
    for event_code, influence, mapped_status, is_granted in rows:
        key = ((event_code or "").upper().strip(), (influence or "").strip())
        code_lookup[key] = mapped_status
        is_granted_lookup[key] = bool(is_granted)
    return code_lookup, is_granted_lookup


from pyexpat import XML_PARAM_ENTITY_PARSING_ALWAYS
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, List, Tuple

class PatentStatusExtractor:
    """Extract and determine patent legal status from EPO patent XML data."""
    
    # Keywords indicating patent is dead/expired
    # normalize for safe matching
    
    
    def __init__(self, xml_content: str, code_lookup: dict | None = None, is_granted_lookup: dict | None = None):
        """Initialize with XML content string."""
        self.root = ET.fromstring(xml_content)
        self.namespaces = {
            "ops": "http://ops.epo.org",
            "ex": "http://www.epo.org/exchange"
            }
        self.code_lookup = code_lookup or {}
        self.is_granted_lookup = is_granted_lookup or {}

    
    def extract_legal_events(self) -> List[Dict]:
        """Extract legal events and enrich them using df_codes mapping."""
        legal_events = []

        for legal in self.root.findall(".//ops:legal", self.namespaces):

            code = legal.get("code", "").upper().strip()
            influence = legal.get("infl", "").strip()
            desc = legal.get("desc", "").strip()

            event = {
                "code": code,
                "desc": desc,
                "influence": influence,
                "date": "",
                "free_text": "",
                "mapped_status": None,
                "is_granted": False,
            }

        # -----------------------------
        # Date
        # -----------------------------
            date_elem = legal.find(".//ops:L007EP", self.namespaces)
            if date_elem is not None and date_elem.text:
                event["date"] = date_elem.text.strip()

        # -----------------------------
        # Free text
        # -----------------------------
            text_elem = legal.find(".//ops:L510EP", self.namespaces)
            if (
                text_elem is not None
                and text_elem.get("desc", "").lower() == "free format text"
                and text_elem.text
            ):
                event["free_text"] = text_elem.text.strip()

        # -----------------------------
        # Mapping via df_codes
        # -----------------------------
            key = (code, influence)

            if key in self.code_lookup:
                event["mapped_status"] = self.code_lookup[key]

        # is_granted lookup (safe)
            grant_flag = self.is_granted_lookup.get(key)
            if grant_flag is True:
                event["is_granted"] = True

            legal_events.append(event)

        return legal_events

    
    def sort_events_by_date(self, events: List[Dict]) -> List[Dict]:
        """Sort legal events by date (most recent first)."""
        def parse_date(date_str):
            try:
                return datetime.strptime(date_str, '%Y-%m-%d')
            except:
                return datetime.min
        
        return sorted(events, key=lambda x: parse_date(x['date']), reverse=True)
    
    from typing import List, Dict, Tuple

    def determine_status(self, events: List[Dict]) -> Tuple[str, str, Dict]:
        """
        Status rules:
        - Use mapped_status of MOST RECENT event
        - If ANY event is_granted == True → status = GRANTED
        (regardless of alive/dead)
        - If no grant → status = mapped_status
        """

        if not events:
            return "UNKNOWN", "No legal events found", {}

        sorted_events = self.sort_events_by_date(events)
        latest_event = sorted_events[0]

        latest_status = latest_event.get("mapped_status")

        has_grant = any(e.get("is_granted") is True for e in sorted_events)

    # -----------------------------
    # GRANTED overrides
    # -----------------------------
        if has_grant and latest_status in {"ALIVE", "DEAD"}:
            return (
                "GRANTED",
                f"Grant found; latest status = {latest_status}",
                latest_event,
            )

    # -----------------------------
    # Normal life status
    # -----------------------------
        if latest_status in {"ALIVE", "DEAD", "PENDING", "NEUTRAL"}:
            return (
                latest_status,
                f"Latest event mapped to {latest_status}",
                latest_event,
            )

        return "UNKNOWN", "No mapped decisive event", latest_event


    
    
    
    def get_patent_status_report(self) -> Dict:
        """Generate a complete patent status report."""
        events = self.extract_legal_events()
        sorted_events = self.sort_events_by_date(events)
        status, reason, latest_event = self.determine_status(events)
        
        # Get application and publication info
        app_number = ''
        pub_number = ''
        
        app_ref = self.root.find('.//application-reference', self.namespaces)
        if app_ref is not None:
            app_doc = app_ref.find('.//doc-number', self.namespaces)
            if app_doc is not None and app_doc.text:
                app_number = app_doc.text.strip()
        
        pub_ref = self.root.find('.//publication-reference', self.namespaces)
        if pub_ref is not None:
            pub_doc = pub_ref.find('.//doc-number', self.namespaces)
            if pub_doc is not None and pub_doc.text:
                pub_number = pub_doc.text.strip()
        
        return {
            'status': status,
            'reason': reason,
            'application_number': app_number,
            'publication_number': pub_number,
            'latest_event': latest_event,
            'all_events': sorted_events,
            'total_events': len(events)
        }
    
    @staticmethod
    def build_patent_status_df(
        legal_xml_df: pd.DataFrame,
        code_lookup: dict | None = None,
        is_granted_lookup: dict | None = None,
    ) -> pd.DataFrame:
        """
        Build a patent status DataFrame from a DataFrame
        with columns: id, publication_number, xml
        """

        required_cols = {"publication_number", "xml"}
        if not required_cols.issubset(legal_xml_df.columns):
            raise ValueError(
                f"Input DataFrame must contain columns: {required_cols}"
            )

        rows = []

        for row in legal_xml_df.itertuples(index=False):
            publication_number = row.publication_number
            xml = row.xml

            if not xml or not isinstance(xml, str):
                rows.append({
                    "publication_number": publication_number,
                    "status": None,
                    "is_granted": False,
                    "xml": xml,
                    "error": "Empty or invalid XML"
                })
                continue

            try:
                extractor = PatentStatusExtractor(
                    xml,
                    code_lookup=code_lookup,
                    is_granted_lookup=is_granted_lookup,
                )
                report = extractor.get_patent_status_report()
                status = report.get("status")
                is_granted = any(
                    e.get("is_granted") is True
                    for e in (report.get("all_events") or [])
                )
                error = None
            except Exception as e:
                status = None
                is_granted = False
                error = str(e)

            rows.append({
                "publication_number": publication_number,
                "status": status,
                "is_granted": is_granted,
                "xml": xml,
                "error": error
            })

        df = pd.DataFrame(rows)
        return df.rename(columns={
            "publication_number": "Publication number",
            "status": "Status",
            "is_granted": "Is granted",
            "xml": "XML",
            "error": "Error",
        })



if __name__ == "__main__":
    try:
        df_status = PatentStatusExtractor.build_patent_status_df(legal_xml_results)
        print(df_status.head())
    except Exception as e:
        print(f"Error processing patent data: {e}")
    


