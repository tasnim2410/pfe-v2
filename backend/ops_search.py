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
def _doc_to_dict(doc) -> dict:
    biblio = doc.find('ex:bibliographic-data', _NS)
    pub = biblio.find('ex:publication-reference/ex:document-id[@document-id-type="docdb"]', _NS)
    title = biblio.find('ex:invention-title[@lang="en"]', _NS)
    abstract = doc.find('ex:abstract[@lang="en"]/ex:p', _NS)  # might be None
    appl = biblio.find('ex:parties/ex:applicants/ex:applicant[@data-format="epodoc"]/ex:applicant-name/ex:name', _NS)
    invs = biblio.findall('ex:parties/ex:inventors/ex:inventor[@data-format="epodoc"]/ex:inventor-name/ex:name', _NS)

    ipc_list  = [c.text.strip()                 for c in biblio.findall('ex:classifications-ipcr/ex:classification-ipcr/ex:text', _NS)]
    cpc_list  = [f"{p.find('ex:section',_NS).text}{p.find('ex:class',_NS).text}{p.find('ex:subclass',_NS).text}"
                 for p in biblio.findall('ex:patent-classifications/ex:patent-classification', _NS)]

    status_kind = pub.find('ex:kind', _NS).text
    is_active = status_kind.startswith('B')    # crude: B* often = granted; adapt to your logic

    return {
        "Title":      title.text.strip() if title is not None else '',
        "Applicants": appl.text.strip() if appl is not None else '',
        "Inventors":  "; ".join(i.text.strip() for i in invs) if invs else '',
        "Publication number": f'{pub.find("ex:country",_NS).text}{pub.find("ex:doc-number",_NS).text} {status_kind}',
        "Publication date":   pub.find('ex:date', _NS).text,
        "IPC":        "; ".join(ipc_list),
        "CPC":        "; ".join(cpc_list) or None,
        "Earliest priority":  biblio.find('ex:priority-claims/ex:priority-claim/ex:document-id/ex:date', _NS).text if biblio.find('ex:priority-claims', _NS) is not None else None,
        "Family number":      doc.attrib.get('family-id'),
        # extras
        "family_id":  doc.attrib.get('family-id'),
        "app_country": pub.find('ex:country', _NS).text,
        "is_active":  is_active
    }

# def fetch_to_dataframe(cql: str, max_records: int = 500) -> pd.DataFrame:
#     out = []
#     start, step = 1, 100
#     token = _get_access_token()
#     headers = {"Accept": "application/xml", "Authorization": f"Bearer {token}"}
#     while len(out) < max_records:
#         end = min(start + step - 1, max_records)
#         url = f"https://ops.epo.org/3.2/rest-services/published-data/search/biblio?q={requests.utils.quote(cql)}&Range={start}-{end}"
#         r = requests.get(url, headers=headers, timeout=30)
#         r.raise_for_status()
#         root = ET.fromstring(r.content)
#         docs = root.findall(".//ex:exchange-document", _NS)
#         if not docs:
#             break
#         out.extend(_doc_to_dict(d) for d in docs)
#         if len(docs) < step:     # last page
#             break
#         start += step
#     df = pd.DataFrame(out)
#     return df


def extract_keyword_pairs(node):
    if node["type"] == "keyword":
        return [(node["field"], node["word"])]
    elif node["type"] == "group":
        pairs = []
        for child in node["keywords"]:
            pairs.extend(extract_keyword_pairs(child))
        return pairs
    return []



def fetch_to_dataframe(cql: str, max_records: int = 500) -> tuple[pd.DataFrame, int]:
    out, total = [], None
    start, step = 1, 100
    token = _get_access_token()
    headers = {"Accept": "application/xml", "Authorization": f"Bearer {token}"}

    while len(out) < max_records:
        end = min(start + step - 1, max_records)
        url = f"https://ops.epo.org/3.2/rest-services/published-data/search/biblio?q={requests.utils.quote(cql)}&Range={start}-{end}"
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        root = ET.fromstring(r.content)

        # grab total-result-count only once (first page)
        if total is None:
            total = int(root.find(".//ops:biblio-search", _NS).attrib["total-result-count"])

        docs = root.findall(".//ex:exchange-document", _NS)
        if not docs:
            break
        out.extend(_doc_to_dict(d) for d in docs)
        if len(docs) < step:
            break
        start += step

    return pd.DataFrame(out), total or 0
