# ---------- backend/ops_search.py ----------
import os, time, requests, xml.etree.ElementTree as ET, pandas as pd
from dotenv import load_dotenv
load_dotenv()

# ── 1.  OAuth helper ─────────────────────────────────────────────
_TOKEN = None
_EXPIRY = 0
def _get_access_token() -> str:
    global _TOKEN, _EXPIRY
    if _TOKEN and time.time() < _EXPIRY:           # still valid
        return _TOKEN
    data = {
        "grant_type": "client_credentials",
        "client_id":  os.getenv("CONSUMER_KEY").strip(),
        "client_secret": os.getenv("CONSUMER_SECRET").strip()
    }
    r = requests.post("https://ops.epo.org/3.2/auth/accesstoken",
                      data=data,
                      headers={"Content-Type": "application/x-www-form-urlencoded"},
                      timeout=20)
    r.raise_for_status()
    _TOKEN = r.json()["access_token"]
    _EXPIRY = time.time() + 3500         # renew ~5 min before true expiry
    return _TOKEN

# ── 2.  JSON→CQL builder (same rules as your scraper) ────────────
_FIELD_MAP = {
    'title': 'ti', 'abstract': 'ab', 'claims': 'cl',
    'title,abstract or claims': 'ctxt', 'all text fields': 'ftxt',
    'title or abstract': 'ta', 'description': 'desc',
    'all text fields or names': 'nftxt', 'title , abstract or names': 'ntxt'
}
def _build(node: dict) -> str:
    if node["type"] == "keyword":
        code = _FIELD_MAP[node["field"]]
        return f'{code} {node["rule_op"].lower()} "{node["word"]}"'
    if node["type"] == "group":
        parts = [_build(child) for child in node["keywords"]]
        return "(" + f' {node["operator"].upper()} '.join(parts) + ")"
    raise ValueError("Unknown node type")

def json_to_cql(query_input: dict) -> str:
    return _build(query_input["query"]["group1"])

# ── 3.  One-page fetch & parse helpers ───────────────────────────
_NS = {'ops': 'http://ops.epo.org', 'ex': 'http://www.epo.org/exchange'}
def _get_text(el) -> str:
    """
    Return stripped text or '' if element is None / empty.
    This avoids AttributeError when optional tags are missing.
    """
    return el.text.strip() if el is not None and el.text else ''

def _doc_to_dict(doc) -> dict:
    """
    Convert one <exchange-document/> element to a flat dict that our
    DataFrame can ingest. Uses _get_text() for every optional node so
    missing tags never crash the parser.
    """
    biblio = doc.find('ex:bibliographic-data', _NS)

    # --- publication-reference fields ---
    pub      = biblio.find('ex:publication-reference/ex:document-id[@document-id-type="docdb"]', _NS)
    country  = _get_text(pub.find('ex:country',     _NS))
    doc_no   = _get_text(pub.find('ex:doc-number',  _NS))
    kind     = _get_text(pub.find('ex:kind',        _NS))
    pub_date = _get_text(pub.find('ex:date',        _NS))

    # --- text fields ---
    title_el   = biblio.find('ex:invention-title[@lang="en"]', _NS)
    title      = _get_text(title_el)

    abstract_el = doc.find('ex:abstract[@lang="en"]/ex:p', _NS)   # may be None
    abstract    = _get_text(abstract_el)

    appl_el = biblio.find('ex:parties/ex:applicants/ex:applicant[@data-format="epodoc"]/ex:applicant-name/ex:name', _NS)
    appl    = _get_text(appl_el)

    invs_el = biblio.findall('ex:parties/ex:inventors/ex:inventor[@data-format="epodoc"]/ex:inventor-name/ex:name', _NS)
    inventors = "; ".join(_get_text(i) for i in invs_el) if invs_el else ''

    # --- classifications ---
    ipc_list = [
        _get_text(c) for c in
        biblio.findall('ex:classifications-ipcr/ex:classification-ipcr/ex:text', _NS)
    ]
    cpc_list = [
        f"{_get_text(p.find('ex:section',_NS))}"
        f"{_get_text(p.find('ex:class',_NS))}"
        f"{_get_text(p.find('ex:subclass',_NS))}"
        for p in biblio.findall('ex:patent-classifications/ex:patent-classification', _NS)
    ]

    # --- family + status ---
    status_kind = kind                          # keep the logic but safe now
    is_active   = status_kind.startswith('B')   # crude heuristic

    earliest_prio = biblio.find('ex:priority-claims/ex:priority-claim/ex:document-id/ex:date', _NS)
    if earliest_prio is not None:
        earliest_prio = _get_text(earliest_prio)[:4]

    return {
        "Title":                title,
        "Applicants":           appl,
        "Inventors":            inventors,
        "Publication number":   f"{country}{doc_no} {kind}".strip(),
        "Publication date":     pub_date,
        "IPC":                  "; ".join(ipc_list),
        "CPC":                  "; ".join(cpc_list) or None,
        "Earliest priority":    earliest_prio,
        "First filing year":    _get_text(biblio.find('ex:application-reference/ex:document-id/ex:date', _NS))[:4],
        # extras
        "family_id":            doc.attrib.get('family-id'),
        "app_country":          country,
        "is_active":            is_active
    }
def extract_keyword_pairs(group):
    """
    Recursively extract (field, word) pairs from nested query groups.
    Accepts group dict and returns list of (field, word) tuples.
    """
    pairs = []
    if not group or not isinstance(group, dict):
        return pairs
    for item in group.get("keywords", []):
        if item.get("type") == "keyword":
            field = item.get("field")
            word = item.get("word")
            if field and word:
                pairs.append((field, word))
        elif item.get("type") == "group":
            pairs.extend(extract_keyword_pairs(item))
    return pairs


def fetch_to_dataframe(cql: str, max_records: int = 500) -> tuple[pd.DataFrame, int]:
    out, total = [], None
    start, step = 1, 100
    token = _get_access_token()
    headers = {"Accept": "application/xml", "Authorization": f"Bearer {token}"}

    while len(out) < max_records:
        end = min(start + step - 1, max_records)
        url = f"https://ops.epo.org/3.2/rest-services/published-data/search/biblio?q={requests.utils.quote(cql)}&Range={start}-{end}"
        try:
            r = requests.get(url, headers=headers, timeout=30)
            if r is None or r.status_code != 200:
                raise RuntimeError(f"EPO OPS API did not return data (status={getattr(r, 'status_code', 'no response')})")
            # Defensive: Make sure content is not empty
            if not r.content:
                raise RuntimeError("EPO OPS API returned empty content")
            root = ET.fromstring(r.content)
        except Exception as e:
            print(f"Error in fetch_to_dataframe: {e}")
            raise

        # grab total-result-count only once (first page)
        if total is None:
            total_el = root.find(".//ops:biblio-search", _NS)
            if total_el is None or "total-result-count" not in total_el.attrib:
                raise RuntimeError("Could not find total-result-count in OPS API response")
            total = int(total_el.attrib["total-result-count"])

        docs = root.findall(".//ex:exchange-document", _NS)
        if not docs:
            break
        out.extend(_doc_to_dict(d) for d in docs)
        if len(docs) < step:
            break
        start += step

    return pd.DataFrame(out), total or 0

