import os
import time
import threading
import requests
from urllib.parse import quote
from dotenv import load_dotenv
from flask import Flask, jsonify
from tqdm import tqdm   # for progress bar
from sqlalchemy.orm import scoped_session
from db import RawPatent, db  # Adjust this import!

# --- Helper: Load credentials (rotating list) ---
def load_api_credentials():
    load_dotenv()
    creds = []
    for i in range(10):  # expand as needed
        key = os.getenv(f"CONSUMER_KEY_{i}")
        secret = os.getenv(f"CONSUMER_SECRET_{i}")
        if key and secret:
            creds.append({'key': key.strip(), 'secret': secret.strip()})
    # Add base creds if present
    base_key = os.getenv("CONSUMER_KEY")
    base_secret = os.getenv("CONSUMER_SECRET")
    if base_key and base_secret:
        creds.insert(0, {'key': base_key.strip(), 'secret': base_secret.strip()})
    if not creds:
        raise ValueError("No valid API credentials found in environment.")
    return creds

API_CREDENTIALS = load_api_credentials()
TOKEN_URL = "https://ops.epo.org/3.2/auth/accesstoken"
BASE_URL = "https://ops.epo.org/3.2/rest-services"
# One token per credential
TOKEN_CACHE = [{'token': None, 'expiry': 0} for _ in API_CREDENTIALS]
TOKEN_LOCKS = [threading.Lock() for _ in API_CREDENTIALS]

def get_access_token(cred_idx):
    # Thread-safe token cache
    with TOKEN_LOCKS[cred_idx]:
        token_info = TOKEN_CACHE[cred_idx]
        now = time.time()
        if token_info['token'] and now < token_info['expiry']:
            return token_info['token']
        cred = API_CREDENTIALS[cred_idx]
        data = {
            "grant_type": "client_credentials",
            "client_id": cred['key'],
            "client_secret": cred['secret']
        }
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        resp = requests.post(TOKEN_URL, data=data, headers=headers, timeout=15)
        resp.raise_for_status()
        access_token = resp.json()["access_token"]
        token_info['token'] = access_token
        token_info['expiry'] = now + 3500  # ~58 min
        return access_token

def validate_patent_number(patent):
    return bool(patent and len(str(patent).strip()) >= 4)

def extract_jurisdictions_and_members(data):
    try:
        jurisdictions = set()
        family_members = []
        world_data = data.get('ops:world-patent-data', {})
        patent_family = world_data.get('ops:patent-family', {})
        members = patent_family.get('ops:family-member', [])
        if isinstance(members, dict):
            members = [members]
        for member in members:
            pub_ref = member.get('publication-reference', {})
            docs = pub_ref.get('document-id', [])
            if isinstance(docs, dict):
                docs = [docs]
            for doc in docs:
                if doc.get('@document-id-type') == 'docdb':
                    country = doc.get('country')
                    if isinstance(country, dict):
                        country = country.get('$')
                    doc_number = doc.get('doc-number')
                    if isinstance(doc_number, dict):
                        doc_number = doc_number.get('$')
                    kind = doc.get('kind')
                    if isinstance(kind, dict):
                        kind = kind.get('$')
                    if country and doc_number and kind:
                        jurisdictions.add(country)
                        family_members.append(f"{country}{doc_number}{kind}")
        return {
            'jurisdictions': sorted(jurisdictions),
            'family_members': sorted(set(family_members))
        }
    except Exception as e:
        print(f"Error parsing response: {e}")
        return {'jurisdictions': None, 'family_members': None}

def fetch_family_data_api(pub_number, cred_idx):
    try:
        token = get_access_token(cred_idx)
        url = f"{BASE_URL}/family/publication/docdb/{quote(str(pub_number))}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json"
        }
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 403:
            return None, "403 forbidden"
        if response.status_code == 404:
            return None, "404 not found"
        response.raise_for_status()
        data = response.json()
        fam = extract_jurisdictions_and_members(data)
        # Empty family = failed (for fallback to scraping)
        if not fam['jurisdictions'] and not fam['family_members']:
            return None, "API returned empty"
        return fam, None
    except Exception as e:
        return None, f"API error: {str(e)}"

# Optional: Espacenet Scraping fallback
def fetch_family_data_scrape(pub_number, family_number):
    # -- Placeholder for actual scraping logic (reuse from family_members2.py) --
    # For brevity: returns None (implement this using undetected-chromedriver as in your file)
    return None, "Scraping fallback not implemented"

# --- Flask Route ---

app = Flask(__name__)



