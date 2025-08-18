

""""claude code """



import requests
import pandas as pd
import time
import traceback
from datetime import datetime
from sqlalchemy import or_
from db import db, ResearchData2, ImpactFactor ,ResearchData3
from impact_factor_processor import clean_and_process_data , store_processed_data
from flask import current_app
import re, pandas as pd
from datetime import date
import pandas as pd
from thefuzz import fuzz
import numpy as np

# Field-specific JIF thresholds (match your impact factor processor)
FIELD_THRESHOLDS = {
    'Medicine': 1.0,
    'Biology': 1.0,
    'Materials Science': 1.5,
    'Computer Science': 0.8,
    'Chemistry': 1.5,
    'Mathematics': 0.7,
    'Geology': 0.8,
    'Physics': 1.0,
    'Engineering': 1.5,
    'Environmental Science': 0.5
}

def fetch_research_data3(query):
    url = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
    params = {
        'query': query,
        'fields': 'paperId,title,authors,venue,publicationVenue,year,publicationDate,citationCount,abstract,influentialCitationCount,fieldsOfStudy,publicationTypes',
        'limit': 500
    }
    for attempt in range(5):
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json().get('data', [])
        elif response.status_code == 429:
            time.sleep(60)
    return []

def safe_float_convert(value):
    """Safely convert value to float, handling NaN, None, and invalid strings"""
    if value is None or value == '' or pd.isna(value):
        return None
    
    if isinstance(value, str):
        # Check if it's a valid float string
        try:
            float_val = float(value)
            # Check for NaN or infinity
            if np.isnan(float_val) or np.isinf(float_val):
                return None
            return float_val
        except (ValueError, TypeError):
            return None
    
    if isinstance(value, (int, float)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    
    return None

def safe_int_convert(value):
    """Safely convert value to int, handling None and invalid values"""
    if value is None or value == '' or pd.isna(value):
        return None
    
    try:
        if isinstance(value, str):
            # Remove any non-numeric characters except minus sign
            cleaned = re.sub(r'[^\d-]', '', value)
            if not cleaned or cleaned == '-':
                return None
            return int(cleaned)
        return int(value)
    except (ValueError, TypeError):
        return None

def process_research_data3(papers, impact_factors=None):
    rows = []
    for paper in papers:
        pub_venue = paper.get('publicationVenue') or {}
        pub_venue_type = str(pub_venue.get('type', 'unknown'))[:50] if isinstance(pub_venue, dict) else 'unknown'
        pub_venue_name = pub_venue.get('name') if isinstance(pub_venue, dict) else paper.get('venue', '')
        issn = pub_venue.get('issn') if isinstance(pub_venue, dict) else None

        authors = paper.get('authors', [])
        author_names = [author.get('name', '') for author in authors]
        
        # Extract authorship countries from author affiliations
        authorship_countries = []
        for author in authors:
            affiliations = author.get('affiliations', [])
            for affiliation in affiliations:
                if isinstance(affiliation, dict) and affiliation.get('country'):
                    authorship_countries.append(affiliation['country'])
        # Remove duplicates while preserving order
        authorship_countries = list(dict.fromkeys(authorship_countries))
        
        pub_date = parse_date(paper.get('publicationDate'))
        
        doi = None
        disclaimer = paper.get('openAccessPdf', {}).get('disclaimer', '')
        doi_match = re.search(r'https?://doi\.org/([^\s]+)', disclaimer)
        if doi_match:
            doi = doi_match.group(1).rstrip('.')

        row = {
            'paper_id': paper.get('paperId', '')[:255],
            'title': paper.get('title', ''),
            'abstract': paper.get('abstract', ''),
            'publication_venue_name': pub_venue_name,
            'publication_venue_type': pub_venue_type,
            'year': safe_int_convert(paper.get('year')),
            'citation_count': safe_int_convert(paper.get('citationCount')),
            'influential_citation_count': safe_int_convert(paper.get('influentialCitationCount')),
            'fields_of_study': paper.get('fieldsOfStudy', []),
            'publication_types': paper.get('publicationTypes', []),
            'publication_date': pub_date,
            'authors': ', '.join(author_names)[:255] if author_names else None,
            'authorship_countries': authorship_countries,
            'DOI': doi,
            'ISSN': issn
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    
    # Clean and filter data
    df = clean_base_data(df)
    
    # Apply impact factor filtering
    df = filter_by_impact_factor(df, impact_factors)
    
    df = df.rename(columns={'Field': 'field', 'JIF5Years': 'jif_5years'})
    
    return df

def parse_date(date_str):
    try:
        return datetime.strptime(date_str, '%Y-%m-%d').date() if date_str else None
    except:
        return None

def clean_base_data(df):
    # Convert and filter year - using safe conversion
    df['year'] = df['year'].apply(safe_int_convert)
    df = df[~((df['publication_venue_name'] == '') | (df['year'].isna()))]
    
    return df.reset_index(drop=True)
  
def normalize_name(col):
    return col.str.lower().str.strip().str.replace(r'\s*&\s*', ' and ', regex=True)

def filter_by_impact_factor(df, impact_factors=None):
    # Normalize publication venue name
    df = df.copy()
    df['venue_norm'] = normalize_name(df['publication_venue_name'])

    # Get the ImpactFactor data from the database
    final_data = pd.DataFrame([{
        'id': j.id,
        'Name': j.name,
        'Abbr Name': j.abbrev_name,
        'issn': j.issn,
        'JIF5Years': j.jif_5yr,
        'subcategory': j.subcategory,
        'Field': j.field,
    } for j in impact_factors])

    # Normalize Name and Abbr Name in final_data
    final_data['name_norm'] = normalize_name(final_data['Name'])
    final_data['abbr_norm'] = normalize_name(final_data['Abbr Name'])

    # Merge on the normalized columns (first on full name, then on abbr)
    merged = df.merge(
        final_data[['name_norm', 'subcategory', 'Field', 'JIF5Years']],
        how='left',
        left_on='venue_norm',
        right_on='name_norm'
    ).merge(
        final_data[['abbr_norm', 'subcategory', 'Field', 'JIF5Years']],
        how='left',
        left_on='venue_norm',
        right_on='abbr_norm',
        suffixes=('', '_abbr')
    )

    # Coalesce the two matches
    for col in ('subcategory', 'Field', 'JIF5Years'):
        if f"{col}_abbr" in merged.columns:
            merged[col] = merged[col].combine_first(merged[f"{col}_abbr"])

    lookup = final_data.set_index(['name_norm','abbr_norm'])[['subcategory','Field','JIF5Years']]
    mask = merged['subcategory'].isna()
    for idx in merged.index[mask]:
        venue = merged.at[idx, 'venue_norm']
        match = None
        venue = str(venue) if venue is not None else ""

        # Look for any normalized name or abbr inside venue
        for (nm, ab), row in lookup.iterrows():
            if nm in venue or ab in venue:
                match = row
                break

        if match is not None:
            merged.loc[idx, ['subcategory','Field','JIF5Years']] = match.values
    
    # Drop helper columns
    merged = merged.drop(columns=[
        'name_norm', 'abbr_norm', 'venue_norm',
        'subcategory_abbr', 'Field_abbr', 'JIF5Years_abbr'
    ], errors='ignore')
    
    keep_types = {'JournalArticle', 'Review'}
    mask = merged['publication_types'].apply(lambda types: isinstance(types, list) and bool(set(types) & keep_types))
    clean_df = merged[mask].reset_index(drop=True)
    mask_conf_only = merged['publication_types'].apply(lambda x: x == ['conference'])
    clean_df = merged.loc[~mask_conf_only].reset_index(drop=True)
    df = clean_df
    df = df[df['subcategory'].notna()].reset_index(drop=True)

    return df

def fetch_openalex_works(query, max_docs=300, per_page=200):
    # Validate per_page (OpenAlex allows 1-200 per request)
    per_page = min(max(per_page, 1), 200)
    
    filters = {
    "title_and_abstract.search": query
}

    url = "https://api.openalex.org/works"
    params = {
        "filter": ",".join([f"{k}:{v}" for k, v in filters.items()]),
        "per_page": per_page,
        "cursor": "*"
    }

    all_works = []
    total_docs_retrieved = 0
    
    while total_docs_retrieved < max_docs:
        response = requests.get(url, params=params)
        results = response.json()
        
        current_works = results.get("results", [])
        all_works.extend(current_works)
        
        total_docs_retrieved += len(current_works)
        
        if not current_works or total_docs_retrieved >= max_docs:
            break
            
        params["cursor"] = results.get("meta", {}).get("next_cursor")

    return all_works[:max_docs]
  
def clean_issn(issn: str) -> str | None:
    """
    Clean ISSN string but preserve hyphens.
    Only removes whitespace and converts to uppercase.
    Returns None for NaN or empty values.
    """
    if pd.isna(issn):
        return None
    s = str(issn).strip().upper()
    return s if s else None

def process_documents(documents, journals_df):

    
    # Clean journals_df ISSN and create mapping
    journal_data = [{
        'issn': j.issn,
        'field': j.field
    } for j in journals_df]
    
    # Create ISSN to field mapping
    issn_to_field = {}
    for journal in journal_data:
        if journal['issn']:
            clean_issn_val = clean_issn(journal['issn'])
            if clean_issn_val:
                issn_to_field[clean_issn_val] = journal['field']
    
    rows = []
    for doc in documents:
        # Handle nested structures safely with fallback defaults
        primary_location = doc.get('primary_location') or {}
        source = primary_location.get('source') or {}
        primary_topic = doc.get('primary_topic') or {}
        field = primary_topic.get('field') or {}
        domain = primary_topic.get('domain') or {}
        
        # Clean ISSN from source
        journal_issn = clean_issn(source.get('issn_l'))
        
        # Process keywords
        keywords = doc.get('keywords', []) or []
        keyword_names = [k.get('display_name', '') for k in keywords]
        keyword_scores = [k.get('score', '') for k in keywords]
        
        # Process locations
        locations = doc.get('locations', []) or []
        location_sources = [
            (loc.get('source') or {}).get('display_name', '')  
            for loc in locations if loc
        ]
        
        # Extract DOI
        raw_doi = doc.get('doi') or (doc.get('ids') or {}).get('doi') or ''
        doi = raw_doi.split('doi.org/')[-1].strip() if raw_doi else ''
        
        # Process authorships countries
        authorships = doc.get('authorships', []) or []
        countries = [auth.get('countries') for auth in authorships if auth.get('countries')]
        flat_countries = []
        for c in countries:
            if isinstance(c, list):
                flat_countries.extend(c)
            else:
                flat_countries.append(c)
        unique_countries = list(dict.fromkeys(flat_countries))
        
        row = {
            'doi': doi,
            'title': doc.get('title', ''),
            'official_title': doc.get('display_name', ''),
            'abstract': doc.get('abstract', ''),
            'relevance': safe_float_convert(doc.get('relevance_score')),
            'year': safe_int_convert(doc.get('publication_year')),
            'publication_date': None,  # OpenAlex doesn't provide this
            'publication_types': [doc.get('type', '')] if doc.get('type') else [],
            'source_type': source.get('type'),
            'publication_venue_name': source.get('display_name'),
            'journal_name': source.get('display_name'),
            'journal_issn_l': source.get('issn_l'),
            'journal_issn_l_clean': journal_issn,
            'issn': source.get('issn_l'),
            'publisher': source.get('host_organization_name'),
            'publisher_hierarchy': source.get('host_organization_lineage_names', []),
            'main_topic': primary_topic.get('display_name'),
            'fields_of_study': [field.get('display_name')] if field.get('display_name') else [],
            'academic_domain': domain.get('display_name'),
            'subcategory': None,  # Will be filled from journals mapping
            'keyword_relevance_score': safe_float_convert(keyword_scores[0]) if keyword_scores else None,
            'keywords': keyword_names,
            'all_journal_sources': [
                loc.get('source', {}).get('display_name')
                for loc in doc.get('locations', []) if isinstance(loc, dict) and isinstance(loc.get('source', {}), dict)
            ],
            'authorships_countries': unique_countries,
            'reference_count': safe_int_convert(doc.get('referenced_works_count')),
            'citation_count': safe_int_convert(doc.get('cited_by_count')),
            'influential_citation_count': None,  # OpenAlex doesn't provide this
            'field_weighted_citation_impact': safe_float_convert(doc.get('works_count')),
            'jif_5years': None,  # Will be filled from journals mapping
            'Field': issn_to_field.get(journal_issn)  # Map the field from journals_df
        }
        
        rows.append(row)
    
    return pd.DataFrame(rows)



def match_papers_by_issn(
    journal_df: pd.DataFrame,
    paper_df: pd.DataFrame,
    issn_col_journal: str = 'ISSN',
    issn_col_paper: str = 'journal_issn_l',
    field_col: str = 'Field'
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:

    # Clean ISSN columns
    jdf = journal_df.copy()
    jdf['ISSN_clean'] = jdf[issn_col_journal].apply(clean_issn)

    pdf = paper_df.copy()
    pdf['ISSN_clean'] = pdf[issn_col_paper].apply(clean_issn)

    # Build ISSN → field lookup
    issn_to_field: dict[str, str] = {}
    for _, row in jdf.iterrows():
        issn = row['ISSN_clean']
        if issn:
            issn_to_field[issn] = row[field_col]

    # Map field over papers
    pdf['Field'] = pdf['ISSN_clean'].map(issn_to_field)

    # Partition matched / unmatched
    matched_mask = pdf['Field'].notna()
    matched_df   = pdf[matched_mask].reset_index(drop=True)
    unmatched_df = pdf[~matched_mask].reset_index(drop=True)

    # Build summary
    summary = {
        'total':    len(pdf),
        'matched':  matched_mask.sum(),
        'unmatched': len(pdf) - matched_mask.sum()
    }

    return matched_df, unmatched_df, summary

def renaming_columns(
    sem_df: pd.DataFrame,
    AIex_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:

    sem_df.rename(
        columns={
            'PublicationVenueName': 'journal_name',
            'Title': 'title',
        },
        inplace=True
    )
    
    AIex_df.rename(
        columns={
            'document_type':   'PublicationTypes',
            'Journal_issn_l':  'ISSN',
            'research_field':  'FieldsOfStudy',
            'publication_year':'Year',
            
        },
        inplace=True
    )
    
    return sem_df, AIex_df
def clean_doi(raw):
    if pd.isna(raw): return None
    s = str(raw).strip().lower()
    s = re.sub(r'^https?://(dx\.)?doi\.org/', '', s)
    s = re.sub(r'^doi:\s*', '', s)
    s = re.split(r'[\?#]', s)[0]
    return s.rstrip('/') or None
  #sem_df['doi_clean']   = sem_df['DOI'].apply(clean_doi)
  #AIex_df['doi_clean']  = AIex_df['DOI'].apply(clean_doi)


def merge_unique_by_doi(
    sem_df: pd.DataFrame,
    AIex_df: pd.DataFrame,
    doi_col: str = 'doi_clean',
    title_col: str = 'title',
    fuzz_threshold: int = 85
) -> pd.DataFrame:

    # Exact matches on DOI
    sem_dois = sem_df[[doi_col]].dropna().drop_duplicates()
    aiex_dois = AIex_df[[doi_col]].dropna().drop_duplicates()
    exact_matches = pd.merge(sem_dois, aiex_dois, on=doi_col, how='inner')

    # Filter sem_df to those not in exact matches
    sem_non = sem_df[~sem_df[doi_col].isin(exact_matches[doi_col])].copy()

    # Prepare AIex pool for fuzzy matching
    aiex_pool = AIex_df[AIex_df[doi_col].isin(exact_matches[doi_col])].copy()
    aiex_pool = aiex_pool[aiex_pool[doi_col].notna()]

    # Fuzzy-match by title
    fuzzy_dois = set()
    for _, s_row in sem_non.iterrows():
        s_doi = s_row[doi_col]
        if pd.isna(s_doi):
            continue
        s_title = str(s_row[title_col]).lower().strip()
        best_score = 0
        best_doi = None
        for _, a_row in aiex_pool.iterrows():
            score = fuzz.token_set_ratio(s_title, str(a_row[title_col]).lower().strip())
            if score > best_score:
                best_score = score
                best_doi = a_row[doi_col]
        if best_score >= fuzz_threshold:
            fuzzy_dois.add(s_doi)

    # Drop any sem_df rows with DOIs in exact or fuzzy matches
    to_drop = set(exact_matches[doi_col]).union(fuzzy_dois)
    sem_filtered = sem_df[~sem_df[doi_col].isin(to_drop)].copy()

    # Concatenate AIex_df + filtered sem_df
    final_df = pd.concat([AIex_df, sem_filtered], ignore_index=True)
    final_df['doi'] = final_df['doi'].combine_first(final_df['doi'])
    final_df['year'] = final_df['year'].combine_first(final_df['year'])
    final_df.drop(columns=['DOI', 'Year'], inplace=True, errors='ignore')
    return final_df

def store_research_data3(df):
    """Store research data with proper type conversion and validation"""
    
    # Clear existing data
    db.session.query(ResearchData3).delete()
    
    records = df.to_dict('records')
    array_fields = [
        'publication_types', 'fields_of_study', 'publisher_hierarchy',
        'keywords', 'all_journal_sources', 'authorships_countries'
    ]
    float_fields = [
        'relevance', 'keyword_relevance_score', 'field_weighted_citation_impact', 'jif_5years'
    ]
    int_fields = ['year', 'citation_count', 'influential_citation_count', 'reference_count']

    for rec in records:
        # Ensure array fields are lists
        for field in array_fields:
            value = rec.get(field)
            if not isinstance(value, list):
                if isinstance(value, str) and value:
                    # Split comma-separated strings into lists
                    rec[field] = [item.strip() for item in value.split(',') if item.strip()]
                else:
                    rec[field] = []
                
        # Safe conversion of float fields
        for field in float_fields:
            rec[field] = safe_float_convert(rec.get(field))
                
        # Safe conversion of int fields
        for field in int_fields:
            rec[field] = safe_int_convert(rec.get(field))

        # Handle publication_date
        pub_date = rec.get('publication_date')
        if isinstance(pub_date, str):
            try:
                rec['publication_date'] = date.fromisoformat(pub_date)
            except (ValueError, TypeError):
                rec['publication_date'] = None
        elif not isinstance(pub_date, date):
            rec['publication_date'] = None

        # Create model instance with safe values
        research_entry = ResearchData3(
            paper_id=str(rec.get('paper_id', ''))[:255],  # Ensure string and limit length
            doi=str(rec.get('doi') or rec.get('DOI', ''))[:255] if rec.get('doi') or rec.get('DOI') else None,            title=str(rec.get('title', ''))[:5000],  # Limit title length
            official_title=str(rec.get('official_title', ''))[:5000] if rec.get('official_title') else None,
            abstract=str(rec.get('abstract', ''))[:10000] if rec.get('abstract') else None,
            relevance=rec.get('relevance'),
            year=rec.get('year') if 'year' in rec else rec.get('Year'),            publication_date=rec.get('publication_date'),
            publication_types=rec.get('publication_types', []),
            source_type=str(rec.get('source_type', ''))[:50] if rec.get('source_type') else None,
            publication_venue_name=str(rec.get('publication_venue_name', ''))[:500] if rec.get('publication_venue_name') else None,
            publication_venue_type=str(rec.get('publication_venue_type', ''))[:100] if rec.get('publication_venue_type') else None,
            journal_name=str(rec.get('journal_name', ''))[:255] if rec.get('journal_name') else None,
            journal_issn_l=str(rec.get('journal_issn_l', ''))[:50] if rec.get('journal_issn_l') else None,
            journal_issn_l_clean=str(rec.get('journal_issn_l_clean', ''))[:50] if rec.get('journal_issn_l_clean') else None,
            issn=str(rec.get('issn', ''))[:50] if rec.get('issn') else None,
            publisher=str(rec.get('publisher', ''))[:255] if rec.get('publisher') else None,
            publisher_hierarchy=rec.get('publisher_hierarchy', []),
            main_topic=str(rec.get('main_topic', ''))[:100] if rec.get('main_topic') else None,
            fields_of_study=rec.get('fields_of_study', []),
            academic_domain=str(rec.get('academic_domain', ''))[:100] if rec.get('academic_domain') else None,
            subcategory=str(rec.get('subcategory', ''))[:100] if rec.get('subcategory') else None,
            keyword_relevance_score=rec.get('keyword_relevance_score'),
            keywords=rec.get('keywords', []),
            all_journal_sources=rec.get('all_journal_sources', []),
            authorships_countries=rec.get('authorships_countries', []),
            reference_count=rec.get('reference_count'),
            citation_count=rec.get('citation_count'),
            influential_citation_count=rec.get('influential_citation_count'),
            field_weighted_citation_impact=rec.get('field_weighted_citation_impact'),
            jif_5years=rec.get('jif_5years')
        )
        db.session.add(research_entry)
    
    db.session.commit()
    
    
    
    
    
    
    def tokenize_query(q: str):
        """
        Tokenize a flat boolean string with AND/OR, parentheses and quoted phrases.
        Returns a list like: ["(", "radar system", "AND", "mimo", ")", "OR", "fmcw"]
        """
        tokens = []
        i, n = 0, len(q or "")
        while i < n:
            c = q[i]
            if c.isspace():
                i += 1
                continue
            if c in "()":
                tokens.append(c)
                i += 1
                continue
            if c in '"“”':  # quoted phrase
                quote = c
                i += 1
                start = i
                while i < n and q[i] not in '"“”':
                    i += 1
                phrase = q[start:i].strip()
                tokens.append(phrase)
                i += 1 if i < n else 0
            continue
            # word (consume until space or paren)
            start = i
            while i < n and (not q[i].isspace()) and q[i] not in "()":
                i += 1
            word = q[start:i].strip()
            if word:
                tokens.append(word)
        return tokens


def to_rpn(tokens):
    """
    Shunting-yard to convert infix tokens (AND/OR) to RPN.
    Precedence: AND > OR. Parentheses respected.
    Non-operator tokens are treated as terms/phrases.
    """
    prec = {"AND": 2, "OR": 1}
    out, op = [], []
    for t in tokens:
        u = t.upper()
        if u in ("AND", "OR"):
            while op and op[-1] in prec and prec[op[-1]] >= prec[u]:
                out.append(op.pop())
            op.append(u)
        elif t == "(":
            op.append(t)
        elif t == ")":
            while op and op[-1] != "(":
                out.append(op.pop())
            if op and op[-1] == "(":
                op.pop()
        else:
            out.append(t)
    while op:
        out.append(op.pop())
    return out


def leaf_condition(term: str):
    """
    Build the per-term condition against multiple text fields (OR across fields).
    We search title + abstract + fields_of_study string. Adjust/add fields if needed.
    """
    pat = f"%{term}%"
    conds = []
    # Defensive getattr to avoid AttributeErrors if schema differs
    if hasattr(ResearchData3, "title"):
        conds.append(ResearchData3.title.ilike(pat))
    if hasattr(ResearchData3, "abstract"):
        conds.append(ResearchData3.abstract.ilike(pat))
    if hasattr(ResearchData3, "fields_of_study"):
        # fields_of_study may be JSON/text; string match is fine
        conds.append(ResearchData3.fields_of_study.ilike(pat))
    # If none of the above exist, fall back to a harmless true filter
    return or_(*conds) if conds else True


def build_sqlalchemy_filter(query_str: str):
    """
    Parse the boolean string into a SQLAlchemy expression:
    - tokens -> RPN
    - build stack of conditions using AND/OR
    - leaves are OR across (title, abstract, fields_of_study)
    If parsing fails or query empty, returns True (no filter).
    """
    if not query_str or not query_str.strip():
        return True
    tokens = tokenize_query(query_str)
    # Normalize AND/OR tokens, keep phrases/words as-is; drop stray parentheses-only input
    if not any(t for t in tokens if t not in ("(", ")")):
        return True
    rpn = to_rpn(tokens)
    stack = []
    for t in rpn:
        u = t.upper()
        if u == "AND":
            if len(stack) >= 2:
                b = stack.pop(); a = stack.pop()
                stack.append(and_(a, b))
        elif u == "OR":
            if len(stack) >= 2:
                b = stack.pop(); a = stack.pop()
                stack.append(or_(a, b))
        else:
            stack.append(leaf_condition(t))
    return stack[0] if stack else True


def row_to_paper_dict(r):
    """
    Serialize a research_data3 ORM row into the columns needed by the Papers tab.
    Safely handles JSON/text for fields_of_study.
    """
    # journal_name can exist under different names; prefer journal_name, else publication_venue_name
    jname = getattr(r, "journal_name", None) or getattr(r, "publication_venue_name", None)
    fos_raw = getattr(r, "fields_of_study", None)
    fos = None
    if isinstance(fos_raw, str):
        try:
            tmp = json.loads(fos_raw)
            if isinstance(tmp, list):
                fos = tmp
            elif isinstance(tmp, str):
                fos = [tmp]
        except Exception:
            # treat as comma-separated string
            fos = [s.strip() for s in fos_raw.split(",") if s.strip()]
    elif isinstance(fos_raw, list):
        fos = fos_raw
    # build dict
    return {
        "id": getattr(r, "id", None),
        "title": getattr(r, "title", None),
        "year": getattr(r, "year", None),
        "journal_name": jname,
        "fields_of_study": fos,
        "citation_count": getattr(r, "citation_count", None),
        "reference_count": getattr(r, "reference_count", None),
        "subcategory": getattr(r, "subcategory", None),
    }