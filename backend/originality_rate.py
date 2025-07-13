# import os
# import time
# import requests
# import pandas as pd
# import xml.etree.ElementTree as ET
# from dotenv import load_dotenv
# from db import db
# # Global token cache
# TOKEN = None
# TOKEN_EXPIRY = 0

# # Constants for API endpoints
# TOKEN_URL = "https://ops.epo.org/3.2/auth/accesstoken"
# BASE_URL = "https://ops.epo.org/3.2/rest-services"

# # Load credentials from .env file
# load_dotenv()
# CREDENTIALS = []
# for i in range(10):
#     key = os.getenv(f"CONSUMER_KEY_{i}")
#     secret = os.getenv(f"CONSUMER_SECRET_{i}")
#     if key and secret:
#         CREDENTIALS.append((key.strip(), secret.strip()))
# # Always add the default if present
# if os.getenv("CONSUMER_KEY") and os.getenv("CONSUMER_SECRET"):
#     CREDENTIALS.insert(0, (os.getenv("CONSUMER_KEY").strip(), os.getenv("CONSUMER_SECRET").strip()))
# if not CREDENTIALS:
#     raise RuntimeError("No EPO OPS credentials found in .env")
# CURRENT_CREDENTIAL_IDX = 0

# def get_access_token() -> str:
#     """Get or refresh the OAuth access token, rotating credentials if needed."""
#     global TOKEN, TOKEN_EXPIRY, CURRENT_CREDENTIAL_IDX
#     if TOKEN and time.time() < TOKEN_EXPIRY:
#         return TOKEN
#     tries = 0
#     while tries < len(CREDENTIALS):
#         key, secret = CREDENTIALS[CURRENT_CREDENTIAL_IDX]
#         data = {
#             "grant_type": "client_credentials",
#             "client_id": key,
#             "client_secret": secret
#         }
#         headers = {"Content-Type": "application/x-www-form-urlencoded"}
#         try:
#             response = requests.post(TOKEN_URL, data=data, headers=headers, timeout=15)
#             response.raise_for_status()
#             TOKEN = response.json()["access_token"]
#             TOKEN_EXPIRY = time.time() + 3500
#             return TOKEN
#         except requests.HTTPError as e:
#             print(f"Token fetch failed for credential {CURRENT_CREDENTIAL_IDX}: {e}")
#             CURRENT_CREDENTIAL_IDX = (CURRENT_CREDENTIAL_IDX + 1) % len(CREDENTIALS)
#             tries += 1
#             time.sleep(1)
#     raise RuntimeError("All EPO OPS credentials failed to obtain access token.")



# class OriginalityData(db.Model):
#     __tablename__ = 'originality_data'
#     id = db.Column(db.Integer, primary_key=True)
#     first_publication_number = db.Column(db.String(50), unique=True, nullable=False)
#     ipc = db.Column(db.Text, nullable=True)
#     citation_numbers = db.Column(db.ARRAY(db.String), nullable=True)
#     citations_ipc = db.Column(db.ARRAY(db.String), nullable=True)
#     ipc_count = db.Column(db.Integer, nullable=True)
#     citations_ipc_count = db.Column(db.Integer, nullable=True)
#     ratio = db.Column(db.Float, nullable=True)
#     ratio_squared = db.Column(db.Float, nullable=True)


# import re

# def is_valid_pub_number(pub_num):
#     # EPO expects something like US1234567A1, EP1234567B1, etc.
#     return bool(re.match(r'^[A-Z]{2}[0-9]+[A-Z0-9]+$', str(pub_num)))

# def fetch_and_store_originality_data(limit=50):
#     """
#     Fetches patent data from raw_patents, retrieves citation numbers and IPCs, calculates ratios, and stores in originality_data table.
#     """
#     from db import RawPatent
#     import pandas as pd
#     db.session.query(OriginalityData).delete()
#     patents = RawPatent.query.with_entities(RawPatent.first_publication_number, RawPatent.ipc).limit(limit).all()
#     skipped = []
#     failed = []
#     added = 0
#     for pub_num, ipc in patents:
#         if not is_valid_pub_number(pub_num):
#             print(f"Skipping invalid pub_num: {pub_num}")
#             skipped.append(pub_num)
#             continue
#         try:
#             citation_numbers = retrieve_citation_publication_numbers(get_patent_biblio(pub_num))
#             citations_ipc = get_all_citations_ipc(citation_numbers)
#             ipc_count = len(ipc.split(',')) if ipc else 0
#             citations_ipc_count = len(citations_ipc)
#             if citations_ipc_count == 0 or ipc_count > citations_ipc_count:
#                 continue
#             ratio = ipc_count / citations_ipc_count if citations_ipc_count else 0
#             ratio_squared = ratio ** 2
#             entry = OriginalityData(
#                 first_publication_number=pub_num,
#                 ipc=ipc,
#                 citation_numbers=citation_numbers,
#                 citations_ipc=citations_ipc,
#                 ipc_count=ipc_count,
#                 citations_ipc_count=citations_ipc_count,
#                 ratio=ratio,
#                 ratio_squared=ratio_squared
#             )
#             db.session.add(entry)
#             added += 1
#         except Exception as e:
#             print(f"Error processing {pub_num}: {type(e).__name__}: {e}")
#             failed.append(pub_num)
#     db.session.commit()
#     print(f"OriginalityData: {added} added, {len(skipped)} skipped, {len(failed)} failed")
#     if skipped:
#         print(f"Skipped publication numbers: {skipped}")
#     if failed:
#         print(f"Failed publication numbers: {failed}")
#     return added, skipped, failed



# def calculate_originality_rate_from_db():
#     """
#     Calculates the originality rate from data stored in originality_data table.
#     Returns (originality_rate, total_patents, valid_patents)
#     """
#     valid_entries = OriginalityData.query.all()
#     n_valid = len(valid_entries)
#     if n_valid < 25:
#         return None, n_valid, n_valid
#     mean_squared_ratio = sum([e.ratio_squared for e in valid_entries]) / n_valid
#     originality_rate = 1 - mean_squared_ratio
#     total_patents = db.session.query(db.func.count()).select_from(OriginalityData).scalar()
#     return originality_rate, total_patents, n_valid


# def get_patent_biblio(publication_number: str) -> str:
#     """
#     Fetch bibliographic data for a given patent number from the EPO OPS API, rotating credentials on error.
#     """
#     tries = 0
#     global CURRENT_CREDENTIAL_IDX, TOKEN, TOKEN_EXPIRY
#     while tries < len(CREDENTIALS):
#         token = get_access_token()
#         url = f"{BASE_URL}/published-data/publication/docdb/{publication_number}/biblio"
#         headers = {
#             "Authorization": f"Bearer {token}",
#             "Accept": "application/xml"
#         }
#         try:
#             session = requests.Session()
#             response = session.get(url, headers=headers, timeout=30) 
#             response.raise_for_status()
#             return response.text
#         except requests.HTTPError as e:
#             print(f"EPO API error for pub {publication_number} with credential {CURRENT_CREDENTIAL_IDX}: {e}")
#             # Check for rate limit or 400/401/403
#             if response.status_code in [400, 401, 403, 429] or 'rate limit' in response.text.lower():
#                 # Rotate credential and force token refresh
#                 CURRENT_CREDENTIAL_IDX = (CURRENT_CREDENTIAL_IDX + 1) % len(CREDENTIALS)
#                 TOKEN = None
#                 TOKEN_EXPIRY = 0
#                 tries += 1
#                 time.sleep(1)
#                 continue
#             raise
#     raise RuntimeError(f"All EPO OPS credentials failed for publication {publication_number}.")


# def retrieve_citation_publication_numbers(xml_string: str) -> list:
#     """
#     Parses an EPO patent XML string and retrieves citation publication numbers 
#     from each citation's <document-id> element with document-id-type="docdb".
#     The publication number is constructed as: country + doc-number + kind.
    
#     Args:
#         xml_string (str): The XML string containing patent data.
    
#     Returns:
#         list of str: A list of citation publication numbers.
#     """
#     ns = {
#         'ex': "http://www.epo.org/exchange",
#         'ops': "http://ops.epo.org"
#     }
    
#     publication_numbers = []
#     root = ET.fromstring(xml_string)
    
#     citations = root.findall(".//ex:bibliographic-data/ex:references-cited/ex:citation", ns)
    
#     for citation in citations:
#         docdb = citation.find(".//ex:document-id[@document-id-type='docdb']", ns)
#         if docdb is not None:
#             country = docdb.findtext("ex:country", default="", namespaces=ns)
#             doc_number = docdb.findtext("ex:doc-number", default="", namespaces=ns)
#             kind = docdb.findtext("ex:kind", default="", namespaces=ns)
#             pub_number = f"{country}{doc_number}{kind}"
#             if pub_number:
#                 publication_numbers.append(pub_number)
    
#     return publication_numbers

# def retrieve_ipc_classifications(xml_string: str) -> list:
#     """
#     Parses the given patent XML string and extracts the IPC classification texts
#     from the <classifications-ipcr> element. For each classification text:
#       - Everything after (and including) the '/' character is removed.
#       - All spaces are removed from the remaining text.
      
#     Args:
#         xml_string (str): The XML string from the OPS API.
        
#     Returns:
#         list of str: A list of cleaned IPC classification texts.
#     """
#     ns = {
#         'ex': "http://www.epo.org/exchange",
#         'ops': "http://ops.epo.org"
#     }
    
#     ipcs = []
#     root = ET.fromstring(xml_string)
    
#     for cl in root.findall(".//ex:classifications-ipcr/ex:classification-ipcr", ns):
#         text = cl.findtext("ex:text", default="", namespaces=ns)
#         if text:
#             # Remove everything after the first '/'
#             cleaned_text = text.strip().split('/')[0].strip()
#             # Remove all spaces from the cleaned text
#             cleaned_text = cleaned_text.replace(" ", "")
#             ipcs.append(cleaned_text)
    
#     return ipcs

# def get_citations_ipc_for_patent(publication_number: str) -> list:
#     """
#     For a given citation publication number, fetch bibliographic data and
#     return its IPC classifications.
    
#     Args:
#         publication_number (str): A citation publication number.
        
#     Returns:
#         list: A list of cleaned IPC classification texts.
#     """
#     try:
#         xml_data = get_patent_biblio(publication_number)
#         ipc_classifications = retrieve_ipc_classifications(xml_data)
#         return ipc_classifications
#     except Exception as e:
#         print(f"Error fetching IPC for {publication_number}: {e}")
#         return []

# def get_all_citations_ipc(citation_nums: list) -> list:
#     """
#     Given a list of citation publication numbers, retrieve the IPC classifications
#     for each citation and aggregate them into one list.
    
#     Args:
#         citation_nums (list): List of citation publication numbers.
        
#     Returns:
#         list: Aggregated list of cleaned IPC classification texts from the citations.
#     """
#     ipc_results = []
#     for num in citation_nums:
#         ipc = get_citations_ipc_for_patent(num)
#         ipc_results.extend(ipc)
#     return ipc_results





























import os
import time
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
from db import db
# Global token cache
TOKEN = None
TOKEN_EXPIRY = 0

# Constants for API endpoints
TOKEN_URL = "https://ops.epo.org/3.2/auth/accesstoken"
BASE_URL = "https://ops.epo.org/3.2/rest-services"

# Load credentials from .env file
load_dotenv()
CREDENTIALS = []
for i in range(10):
    key = os.getenv(f"CONSUMER_KEY_{i}")
    secret = os.getenv(f"CONSUMER_SECRET_{i}")
    if key and secret:
        CREDENTIALS.append((key.strip(), secret.strip()))
# Always add the default if present
if os.getenv("CONSUMER_KEY") and os.getenv("CONSUMER_SECRET"):
    CREDENTIALS.insert(0, (os.getenv("CONSUMER_KEY").strip(), os.getenv("CONSUMER_SECRET").strip()))
if not CREDENTIALS:
    raise RuntimeError("No EPO OPS credentials found in .env")
CURRENT_CREDENTIAL_IDX = 0

def get_access_token() -> str:
    """Get or refresh the OAuth access token, rotating credentials only on failure."""
    global TOKEN, TOKEN_EXPIRY, CURRENT_CREDENTIAL_IDX

    if TOKEN and time.time() < TOKEN_EXPIRY:
        return TOKEN

    key, secret = CREDENTIALS[CURRENT_CREDENTIAL_IDX]
    data = {
        "grant_type": "client_credentials",
        "client_id": key,
        "client_secret": secret
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    try:
        response = requests.post(TOKEN_URL, data=data, headers=headers, timeout=15)
        response.raise_for_status()
        TOKEN = response.json()["access_token"]
        TOKEN_EXPIRY = time.time() + 3500
        return TOKEN
    except requests.HTTPError as e:
        print(f"Token fetch failed for credential {CURRENT_CREDENTIAL_IDX}: {e}")
        # Rotate credential and reset token
        CURRENT_CREDENTIAL_IDX = (CURRENT_CREDENTIAL_IDX + 1) % len(CREDENTIALS)
        TOKEN = None
        TOKEN_EXPIRY = 0
        raise



class OriginalityData(db.Model):
    __tablename__ = 'originality_data'
    id = db.Column(db.Integer, primary_key=True)
    first_publication_number = db.Column(db.String(50), unique=True, nullable=False)
    ipc = db.Column(db.Text, nullable=True)
    citation_numbers = db.Column(db.ARRAY(db.String), nullable=True)
    citations_ipc = db.Column(db.ARRAY(db.String), nullable=True)
    ipc_count = db.Column(db.Integer, nullable=True)
    citations_ipc_count = db.Column(db.Integer, nullable=True)
    ratio = db.Column(db.Float, nullable=True)
    ratio_squared = db.Column(db.Float, nullable=True)


import re

def is_valid_pub_number(pub_num):
    # EPO expects something like US1234567A1, EP1234567B1, etc.
    return bool(re.match(r'^[A-Z]{2}[0-9]+[A-Z0-9]+$', str(pub_num)))

def fetch_and_store_originality_data(limit=50):
    """
    Fetches patent data from raw_patents, retrieves citation numbers and IPCs, calculates ratios, and stores in originality_data table.
    """
    from db import RawPatent
    import pandas as pd
    db.session.query(OriginalityData).delete()
    patents = RawPatent.query.with_entities(RawPatent.first_publication_number, RawPatent.ipc).limit(limit).all()
    skipped = []
    failed = []
    added = 0
    for pub_num, ipc in patents:
        if not is_valid_pub_number(pub_num):
            print(f"Skipping invalid pub_num: {pub_num}")
            skipped.append(pub_num)
            continue
        try:
            citation_numbers = retrieve_citation_publication_numbers(get_patent_biblio(pub_num))
            citations_ipc = get_all_citations_ipc(citation_numbers)
            ipc_count = len(ipc.split(',')) if ipc else 0
            citations_ipc_count = len(citations_ipc)
            if citations_ipc_count == 0 or ipc_count > citations_ipc_count:
                continue
            ratio = ipc_count / citations_ipc_count if citations_ipc_count else 0
            ratio_squared = ratio ** 2
            entry = OriginalityData(
                first_publication_number=pub_num,
                ipc=ipc,
                citation_numbers=citation_numbers,
                citations_ipc=citations_ipc,
                ipc_count=ipc_count,
                citations_ipc_count=citations_ipc_count,
                ratio=ratio,
                ratio_squared=ratio_squared
            )
            db.session.add(entry)
            added += 1
        except Exception as e:
            print(f"Error processing {pub_num}: {type(e).__name__}: {e}")
            failed.append(pub_num)
    db.session.commit()
    print(f"OriginalityData: {added} added, {len(skipped)} skipped, {len(failed)} failed")
    if skipped:
        print(f"Skipped publication numbers: {skipped}")
    if failed:
        print(f"Failed publication numbers: {failed}")
    return added, skipped, failed



def calculate_originality_rate_from_db():
    """
    Calculates the originality rate from data stored in originality_data table.
    Returns (originality_rate, total_patents, valid_patents)
    """
    valid_entries = OriginalityData.query.all()
    n_valid = len(valid_entries)
    if n_valid < 30:
        return None, n_valid, n_valid
    mean_squared_ratio = sum([e.ratio_squared for e in valid_entries]) / n_valid
    originality_rate = 1 - mean_squared_ratio
    total_patents = db.session.query(db.func.count()).select_from(OriginalityData).scalar()
    return originality_rate, total_patents, n_valid


def get_patent_biblio(publication_number: str) -> str:
    """
    Fetch bibliographic data for a given patent number from the EPO OPS API.
    Rotates credentials only on 401/403/429.
    """
    global CURRENT_CREDENTIAL_IDX, TOKEN, TOKEN_EXPIRY

    token = get_access_token()
    url = f"{BASE_URL}/published-data/publication/docdb/{publication_number}/biblio"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/xml"
    }

    try:
        session = requests.Session()
        response = session.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.text
    except requests.HTTPError as e:
        print(f"EPO API error for pub {publication_number} with credential {CURRENT_CREDENTIAL_IDX}: {e}")
        if response.status_code in [401, 403, 429]:
            # Rotate credential and reset token
            CURRENT_CREDENTIAL_IDX = (CURRENT_CREDENTIAL_IDX + 1) % len(CREDENTIALS)
            TOKEN = None
            TOKEN_EXPIRY = 0
            # Retry once with new credential
            return get_patent_biblio(publication_number)
        raise


def retrieve_citation_publication_numbers(xml_string: str) -> list:
    """
    Parses an EPO patent XML string and retrieves citation publication numbers 
    from each citation's <document-id> element with document-id-type="docdb".
    The publication number is constructed as: country + doc-number + kind.
    
    Args:
        xml_string (str): The XML string containing patent data.
    
    Returns:
        list of str: A list of citation publication numbers.
    """
    ns = {
        'ex': "http://www.epo.org/exchange",
        'ops': "http://ops.epo.org"
    }
    
    publication_numbers = []
    root = ET.fromstring(xml_string)
    
    citations = root.findall(".//ex:bibliographic-data/ex:references-cited/ex:citation", ns)
    
    for citation in citations:
        docdb = citation.find(".//ex:document-id[@document-id-type='docdb']", ns)
        if docdb is not None:
            country = docdb.findtext("ex:country", default="", namespaces=ns)
            doc_number = docdb.findtext("ex:doc-number", default="", namespaces=ns)
            kind = docdb.findtext("ex:kind", default="", namespaces=ns)
            pub_number = f"{country}{doc_number}{kind}"
            if pub_number:
                publication_numbers.append(pub_number)
    
    return publication_numbers

def retrieve_ipc_classifications(xml_string: str) -> list:
    """
    Parses the given patent XML string and extracts the IPC classification texts
    from the <classifications-ipcr> element. For each classification text:
      - Everything after (and including) the '/' character is removed.
      - All spaces are removed from the remaining text.
      
    Args:
        xml_string (str): The XML string from the OPS API.
        
    Returns:
        list of str: A list of cleaned IPC classification texts.
    """
    ns = {
        'ex': "http://www.epo.org/exchange",
        'ops': "http://ops.epo.org"
    }
    
    ipcs = []
    root = ET.fromstring(xml_string)
    
    for cl in root.findall(".//ex:classifications-ipcr/ex:classification-ipcr", ns):
        text = cl.findtext("ex:text", default="", namespaces=ns)
        if text:
            # Remove everything after the first '/'
            cleaned_text = text.strip().split('/')[0].strip()
            # Remove all spaces from the cleaned text
            cleaned_text = cleaned_text.replace(" ", "")
            ipcs.append(cleaned_text)
    
    return ipcs

def get_citations_ipc_for_patent(publication_number: str) -> list:
    """
    For a given citation publication number, fetch bibliographic data and
    return its IPC classifications.
    
    Args:
        publication_number (str): A citation publication number.
        
    Returns:
        list: A list of cleaned IPC classification texts.
    """
    try:
        xml_data = get_patent_biblio(publication_number)
        ipc_classifications = retrieve_ipc_classifications(xml_data)
        return ipc_classifications
    except Exception as e:
        print(f"Error fetching IPC for {publication_number}: {e}")
        return []

def get_all_citations_ipc(citation_nums: list) -> list:
    """
    Given a list of citation publication numbers, retrieve the IPC classifications
    for each citation and aggregate them into one list.
    
    Args:
        citation_nums (list): List of citation publication numbers.
        
    Returns:
        list: Aggregated list of cleaned IPC classification texts from the citations.
    """
    ipc_results = []
    for num in citation_nums:
        ipc = get_citations_ipc_for_patent(num)
        ipc_results.extend(ipc)
    return ipc_results





