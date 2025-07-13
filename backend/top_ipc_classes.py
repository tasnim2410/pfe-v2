import requests
import xml.etree.ElementTree as ET
import re
from typing import List, Dict, Optional
import os
import time
from dotenv import load_dotenv

# --- Espacenet API Auth ---
TOKEN = None
TOKEN_EXPIRY = 0
TOKEN_URL = "https://ops.epo.org/3.2/auth/accesstoken"
BASE_URL = "https://ops.epo.org/3.2/rest-services"
load_dotenv()
CONSUMER_KEY = os.getenv("CONSUMER_KEY").strip()
CONSUMER_SECRET = os.getenv("CONSUMER_SECRET").strip()

def get_access_token() -> str:
    """Get or refresh the OAuth access token."""
    global TOKEN, TOKEN_EXPIRY
    if TOKEN and time.time() < TOKEN_EXPIRY:
        return TOKEN
    data = {
        "grant_type": "client_credentials",
        "client_id": CONSUMER_KEY,
        "client_secret": CONSUMER_SECRET
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    response = requests.post(TOKEN_URL, data=data, headers=headers, timeout=15)
    response.raise_for_status()
    TOKEN = response.json()["access_token"]
    # Token expires in ~58 minutes; refresh slightly before expiry.
    TOKEN_EXPIRY = time.time() + 3500  
    return TOKEN

def normalize_ipc_code(ipc_code: str) -> Optional[str]:
    """
    Normalize IPC/CPC code for Espacenet API: keep only letters/numbers, add /00 if missing.
    """
    code = re.sub(r'[^A-Za-z0-9]', '', ipc_code)
    # If code already contains a slash, keep as is
    if '/' not in code:
        code += '/00'
    return code

def get_ipc_meaning(ipc_code: str) -> Optional[Dict[str, str]]:
    """
    Query Espacenet API for a given CPC code and return its title and explanation.
    """
    base_url = "https://ops.epo.org/3.2/rest-services/classification/cpc/"
    code = normalize_ipc_code(ipc_code)
    if not code:
        print(f"Could not normalize code: {ipc_code}")
        return None
    url = f"{base_url}{code}"
    token = get_access_token()
    headers = {
        "Accept": "application/xml",
        "Authorization": f"Bearer {token}"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"API error for {url}: {e}")
        return None
    try:
        root = ET.fromstring(response.content)
        ns = {'cpc': 'http://www.epo.org/cpcexport'}
        item = root.find('.//cpc:classification-item', ns)
        if item is None:
            print(f"No classification-item for {url}")
            return None
        symbol = item.find('cpc:classification-symbol', ns)
        class_title = item.find('cpc:class-title/cpc:title-part/cpc:text', ns)
        explanation = item.find('cpc:class-title/cpc:title-part/cpc:explanation/cpc:text', ns)
        return {
            'ipc_code': code,
            'symbol': symbol.text if symbol is not None else '',
            'title': class_title.text if class_title is not None else '',
            'explanation': explanation.text if explanation is not None else ''
        }
    except Exception as e:
        print(f"XML parse error for {url}: {e}")
        return None

def get_ipc_codes_meanings(ipc_codes: List[str]) -> Dict[str, Dict[str, str]]:
    """
    For a list of IPC codes, return a dict mapping each code to its title and explanation.
    """
    results = {}
    for code in ipc_codes:
        meaning = get_ipc_meaning(code)
        if meaning:
            results[code] = meaning
        else:
            results[code] = {'ipc_code': code, 'symbol': '', 'title': '', 'explanation': ''}
    return results

# #Example usage:
# codes = ["A61B34", "B25J9", "G05D1", "A61B90", "{A61B34", "{B25J9", "G05B19", "A61B17", "{A01D34", "{B25J13"]
# meanings = get_ipc_codes_meanings(codes)
# print(meanings)
