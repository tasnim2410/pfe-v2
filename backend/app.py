# backendTry1/app.py
import os
import time
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import text
# import your scraper & post‐download processor
from scraping_raw_data import EspacenetScraper, process_downloaded_data ,DatabaseManager ,extract_field_keyword_pairs
#from family_members import ensure_columns_exist
from db import db, RawPatent ,ImpactFactor
from family_members import process_dataframe_parallel
from flask import Flask, request, jsonify
import concurrent.futures
from family_members import process_patent_api , PatentsSearch , process_rows
from flask import Flask, request, jsonify
import pandas as pd
import logging
import os
import concurrent.futures
import time
from urllib.parse import quote
import requests
import sqlalchemy
from dotenv import load_dotenv
import threading
import json
from sqlalchemy import text
import uuid
from family_members2 import PatentsSearch , build_espacenet_url
import ast
from keyword_analysis import preprocess_text , extract_keywords ,analyze_topic_evolution
from cleaners import clean_family_members , extract_country_codes  # Import from your module
from family_analysis import get_family_counts_by_country
from research_retrieve import fetch_research_data , process_research_data, store_research_data
from research_retrieve2 import fetch_research_data2 , process_research_data2, store_research_data2
from impact_factor_processor import clean_and_process_data , store_processed_data
from sqlalchemy.exc import SQLAlchemyError
from pandas.errors import ParserError
from research_retrieve2 import fetch_research_data2, process_research_data2, store_research_data2
from research_retrieve3 import (
    fetch_research_data3, process_research_data3, fetch_openalex_works, process_documents,
    clean_issn, clean_doi, renaming_columns, merge_unique_by_doi, store_research_data3
)
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from flask import jsonify
import pandas as pd
import math
from collections import Counter
from keywords_coccurrence_trends import track_cooccurrence_trends , clean_text_remove_stopwords
from growth_rate import patent_growth_summary
from collections import defaultdict
from keyword_analysis import preprocess_text, analyze_topic_evolution
import logging
from patent_trend_by_field import get_ipc_meaning , normalize_engineering, get_patent_fields
from db import Window, Topic , PatentKeyword ,Divergence ,IPCClassification , ClassifiedPatent , IPCFieldOfStudy , Applicant , ApplicantType , Cost ,PatentCost
from applicant_analysis import get_applicants_df, get_applicant_type_df , get_top_10_applicants,get_innovation_cycle
from market_cost import update_cost_df , calculate_age ,assign_cost, get_market_metrics
import socket
import re
from dotenv import set_key
from top_ipc_classes import get_ipc_codes_meanings
import pandas as pd
from collections import Counter
from originality_rate import retrieve_citation_publication_numbers, get_patent_biblio, get_all_citations_ipc
import random
load_dotenv()
import signal
import threading
from werkzeug.serving import make_server
from applicant_analysis import extract_applicant_collaboration_network

from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from dotenv import load_dotenv
from urllib.parse import quote
import os, time, requests
from originality_rate import fetch_and_store_originality_data, calculate_originality_rate_from_db, OriginalityData
from datetime import datetime
import uuid

import os
import time
import threading
import requests
from urllib.parse import quote
from dotenv import load_dotenv
from flask import Flask, jsonify
from tqdm import tqdm   # for progress bar
from sqlalchemy.orm import scoped_session
from db import RawPatent, db
from family_ops import get_access_token, validate_patent_number, extract_jurisdictions_and_members , load_api_credentials ,fetch_family_data_api , fetch_family_data_scrape
from flask import Flask, request, send_file
from pptx import Presentation
from pptx.util import Inches
import io
import base64


class ServerThread(threading.Thread):
    def __init__(self, app, port):
        threading.Thread.__init__(self)
        self.srv = make_server('127.0.0.1', port, app)
        self.ctx = app.app_context()
        self.ctx.push()

    def run(self):
        print(f"Running Flask app on http://127.0.0.1:{port}")
        self.srv.serve_forever()

    def shutdown(self):
        self.srv.shutdown()

def create_app():
    
    app = Flask(__name__)
    CORS(app, origins=["http://localhost:3000"])  

    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db.init_app(app)
    # logging.basicConfig(
    # level=logging.INFO,
    # format='%(asctime)s - %(levelname)s - %(message)s'
    # )
    logger = logging.getLogger(__name__)    
    
# Set up logging
    # Database connection
    db_url = os.getenv("DATABASE_URL")
    if db_url is None:
        raise ValueError("DATABASE_URL not found. Please check your .env file.")

# Create the SQLAlchemy engine
    engine = sqlalchemy.create_engine(db_url)

    
    @app.route('/')
    def home():
      return 'app is running!', 200
  
    @app.route('/api/last_search_keywords', methods=['GET'])
    def get_last_search_keywords():
        try:
            with engine.connect() as connection: 
                result = connection.execute(
                    text("SELECT search_id FROM search_keywords WHERE id =(SELECT MAX(id) FROM search_keywords)")
                )
                row = result.fetchone()
                if not row: 
                    return jsonify({"message":"No search keywords found"}), 404
                search_id = row[0]

                result = connection.execute(
                    text("SELECT keyword , field FROM search_keywords WHERE search_id = :search_id"),
                    {"search_id": search_id}
                )
                rows = result.mappings().all()
                search_query = {row["keyword"]:row["field"] for row in rows}
                return jsonify(search_query), 200
        except Exception as e:
            return jsonify({"error": f"Failed to fetch last search keywords: {str(e)}"}), 500

    #search keywords with field mapping
     
    # @app.route('/api/search', methods=['POST'])
    # def search_patents():
    #     # 1) Parse complex query parameters from JSON body
    #     data = request.get_json()
    #     keywords_list = data.get('keywords', []) if data else []
    #     if not keywords_list:
    #         return jsonify({"error": "Use JSON body with 'keywords' list: [{\"field\": \"field_name\", \"keyword\": \"keyword\"}]"}), 400
    
    #     search_id = str(uuid.uuid4())
    
    #     # Build search_map from the list of objects
    #     search_map = {}
    #     for item in keywords_list:
    #         field = item.get('field')
    #         keyword = item.get('keyword')
    #         if not field or not keyword:
    #             return jsonify({"error": "Each keyword item must have 'field' and 'keyword'"}), 400
    #         search_map[keyword.strip()] = field.strip().lower()  # Normalize field name

    #     # 2) Build scraper with the parsed query
    #     db_manager = DatabaseManager()
    #     scraper = EspacenetScraper(
    #         search_map,
    #         headless=True,
    #         options_args=[
    #             "--disable-blink-features=AutomationControlled",
    #             "--no-sandbox",
    #             "--disable-dev-shm-usage",
    #             "--remote-debugging-port=9222",
    #             "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36"
    #         ]
    #     )
    #     try:
    #         # Load the Espacenet page
    #         if not scraper.get_page_html():
    #             return jsonify({"error": "Failed to load Espacenet page"}), 500

    #         # Download the CSV file
    #         if not scraper.download_csv(max_results=500):
    #             return jsonify({"error": "Failed to download CSV"}), 500

    #         # Wait to ensure file download completes
    #         time.sleep(10)

    #         # Process the downloaded CSV
    #         df = process_downloaded_data(os.path.expanduser("~/Downloads"))
    #         if df is None or df.empty:
    #             return jsonify({"error": "Couldn’t parse downloaded CSV or DataFrame is empty"}), 500

    #         print(f"DataFrame contains {len(df)} rows")

    #         # Store data in the database
    #         if not db_manager.store_patents(df):
    #             return jsonify({"error": "Failed to store data in database"}), 500

    #         print("Data stored successfully")

    #     finally:
    #         if scraper:
    #             scraper.close()

    #     # 3) Insert search keywords into the database
    #     try:
    #         with db_manager.engine.connect() as connection:
    #             keyword_params = [
    #                 {"search_id": search_id, "field": field, "keyword": keyword}
    #                 for keyword, field in search_map.items()
    #             ]
    #             connection.execute(
    #                 text("INSERT INTO search_keywords (search_id, field, keyword) VALUES (:search_id, :field, :keyword)"),
    #                 keyword_params
    #             )
    #             connection.commit()
    #             print("Keyword data inserted successfully")
    #     except Exception as e:
    #         print(f"Error inserting keywords into database: {e}")
    #         return jsonify({"error": f"Failed to insert search keywords: {str(e)}"}), 500

    #     df = df.where(pd.notnull(df), None)

    #     for dtcol in df.select_dtypes(include=['datetime64[ns]']).columns:
    #         df[dtcol] = df[dtcol].apply(lambda x: x.isoformat() if x is not None else None)
    
    #     # 4) Return the response
    #     response_data = {
    #         "search_id": search_id,
    #         "results": df.to_dict(orient='records')
    #     }

    #     return jsonify(response_data), 200
        def safe_number(val):
            return val if isinstance(val, (int, float)) and not math.isnan(val) else None

    
    # @app.route('/api/search', methods=['POST'])
    # def search_patents():
    #     data = request.get_json()
    #     keywords_list = data.get('keywords', [])
    #     operators = data.get('operators', [])
    #     if not keywords_list or len(keywords_list) < 2:
    #         return jsonify({"error": "Provide at least two keywords."}), 400
    #     if len(operators) != len(keywords_list) - 1:
    #         return jsonify({"error": "Operators count must be one less than keywords count."}), 400

    #     search_id = str(uuid.uuid4())
    #     search_pairs = []
    #     for item in keywords_list:
    #         field = item.get('field')
    #         keyword = item.get('keyword')
    #         if not field or not keyword:
    #             return jsonify({"error": "Each keyword item must have 'field' and 'keyword'"}), 400
    #         search_pairs.append((field.strip().lower(), keyword.strip()))

    #     db_manager = DatabaseManager()
    #     scraper = EspacenetScraper(
    #         search_pairs,
    #         operators=operators,
    #         headless=True,
    #         options_args=[
    #             "--disable-blink-features=AutomationControlled",
    #             "--no-sandbox",
    #             "--disable-dev-shm-usage",
    #             "--remote-debugging-port=9222",
    #             "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36"
    #         ]
    #     )
    #     # ...rest of your code...
    #     try:
    #         # Load the Espacenet page
    #         if not scraper.get_page_html():
    #             return jsonify({"error": "Failed to load Espacenet page"}), 500

    #         # Download the CSV file
    #         if not scraper.download_csv(max_results=500):
    #             return jsonify({"error": "Failed to download CSV"}), 500

    #         # Wait to ensure file download completes
    #         time.sleep(10)

    #         # Process the downloaded CSV
    #         df = process_downloaded_data(os.path.expanduser("~/Downloads"))
    #         if df is None or df.empty:
    #             return jsonify({"error": "Couldn’t parse downloaded CSV or DataFrame is empty"}), 500

    #         print(f"DataFrame contains {len(df)} rows")

    #         # Store data in the database
    #         if not db_manager.store_patents(df):
    #             return jsonify({"error": "Failed to store data in database"}), 500

    #         print("Data stored successfully")

    #     finally:
    #         if scraper:
    #             scraper.close()

    #     # Create search_map from search_pairs for database storage
    #     search_map = {keyword: field for field, keyword in search_pairs}

    #     # 3) Insert search keywords into the database
    #     try:
    #         with db_manager.engine.connect() as connection:
    #             keyword_params = [
    #                 {"search_id": search_id, "field": field, "keyword": keyword}
    #                 for keyword, field in search_map.items()
    #             ]
    #             connection.execute(
    #                 text("INSERT INTO search_keywords (search_id, field, keyword) VALUES (:search_id, :field, :keyword)"),
    #                 keyword_params
    #             )
    #             connection.commit()
    #             print("Keyword data inserted successfully")
    #     except Exception as e:
    #         print(f"Error inserting keywords into database: {e}")
    #         return jsonify({"error": f"Failed to insert search keywords: {str(e)}"}), 500

    #     df = df.replace({np.nan: None})
        
    #     for dtcol in df.select_dtypes(include=['datetime64[ns]']).columns:
    #         df[dtcol] = df[dtcol].apply(lambda x: x.isoformat() if x is not None else None)
    
    #     # 4) Return the response
    #     response_data = {
    #         "search_id": search_id,
    #         "results": df.to_dict(orient='records')  
    #     }

    #     return jsonify(response_data), 200
    
    
    
    
    @app.route('/api/search', methods=['POST'])
    def search_patents():
            query_input = request.get_json()

            db_manager = DatabaseManager()
            scraper = EspacenetScraper(
                search_keywords=None,
                headless=True
            )
           
            try:
                # Load the Espacenet page
                if not scraper.get_page_html(query_input):
                    return jsonify({"error": "Failed to load Espacenet page"}), 500

                # Download the CSV file
                if not scraper.download_csv(max_results=500):
                    return jsonify({"error": "Failed to download CSV"}), 500

                # Wait to ensure file download completes
                time.sleep(10)

                # Process the downloaded CSV
                df = process_downloaded_data(os.path.expanduser("~/Downloads"))
                if df is None or df.empty:
                    return jsonify({"error": "Couldn’t parse downloaded CSV or DataFrame is empty"}), 500

                print(f"DataFrame contains {len(df)} rows")

                # Store data in the database
                if not db_manager.store_patents(df):
                    return jsonify({"error": "Failed to store data in database"}), 500

                print("Data stored successfully")

            finally:
                if scraper:
                    scraper.close()

            root_group = query_input.get("query", {}).get("group1")
            if not root_group:
                return jsonify({"error": "Query must contain a group labeled 'group1' under 'query'"}), 400
            search_pairs = extract_field_keyword_pairs(root_group)
            search_id = str(uuid.uuid4())

            # 3) Insert search keywords into the database
            try:
                with db_manager.engine.connect() as connection:
                    keyword_params = [
                        {"search_id": search_id, "field": field, "keyword": keyword}
                        for field, keyword in search_pairs
                    ]
                    connection.execute(
                    text("""
                     INSERT INTO search_keywords (search_id, field, keyword)
                 VALUES (:search_id, :field, :keyword)
                """),
                    keyword_params
                )
                    connection.commit()
                    print("Keyword data inserted successfully")
            except Exception as e:
                print(f"Error inserting keywords into database: {e}")
                return jsonify({"error": f"Failed to insert search keywords: {str(e)}"}), 500

            df = df.replace({np.nan: None})
        
            for dtcol in df.select_dtypes(include=['datetime64[ns]']).columns:
                df[dtcol] = df[dtcol].apply(lambda x: x.isoformat() if x is not None else None)
    
            # 4) Return the response
            response_data = {
                "search_id": search_id,
                "results": df.to_dict(orient='records')  
            }

            return jsonify(response_data), 200
    

    
    
    
    

    def fetch_last_search_keywords_from_db():
        """
        Returns a dict mapping keyword→field for the most
        recent search_id, or raises ValueError if none found.
        """
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT search_id FROM search_keywords ORDER BY id DESC LIMIT 1")
            ).fetchone()
            if not row:
                raise ValueError("No search keywords found")

            search_id = row[0]
            mappings = conn.execute(
                text("SELECT keyword, field FROM search_keywords WHERE search_id = :sid"),
                {"sid": search_id}
            ).mappings().all()

        # build and return the dict
            return {m["keyword"]: m["field"] for m in mappings}
        

    @app.route('/api/family_members/scraping', methods=['POST'])
    def get_family_members():
    # Fetch data from database
        field_mapping = {
            "title": "ti",
            "abstract": "ab",
            "claims": "cl",
            "title,abstract or claims": "ctxt",
            "all text fields": "ftxt",
            "title or abstract": "ta",
            "description": "desc",
            "all text fields or names": "nftxt",
            "title , abstract or names": "ntxt"
        }
        keywords = fetch_last_search_keywords_from_db()
    
        # Load existing data
        query = 'SELECT *, "No" as id FROM raw_patents'
        df = pd.read_sql(query, engine)
    
        scraper = PatentsSearch(headless=False)
    
        try:
            # Process each patent record
            for index, row in df.iterrows():
                print(f"Processing patent {index+1}/{len(df)}")
            
            # 1. Scrape family members
                url = build_espacenet_url(row, keywords, field_mapping)
                html = scraper.get_page_html(url)
            
                if not html:
                    print(f"Skipping {row['first_publication_number']} - failed to retrieve page")
                    df.at[index, 'family_members'] = []
                    df.at[index, 'family_jurisdictions'] = []
                    continue
                
            # 2. Get raw family members from HTML
                raw_members = scraper.parse_html(html)
            
            # 3. Clean the scraped data
                cleaned_members = clean_family_members(raw_members)
            
            # 4. Extract country codes from CLEANED members
                jurisdictions = extract_country_codes(cleaned_members)
            
            # 5. Update both columns
                df.at[index, 'family_members'] = cleaned_members
                df.at[index, 'family_jurisdictions'] = jurisdictions
            
                print(f"Updated {row['first_publication_number']} with {len(cleaned_members)} members")

            # Prepare database updates
            updates = []
            for _, row in df.iterrows():
                updates.append({
                    'id': row['id'],
                    'members': row['family_members'],
                    'jurisdictions': row['family_jurisdictions']
                })

            # Update database in a single transaction
            with db.engine.begin() as connection:  # Automatically commits/rolls back
                connection.execute(
                    text("""
                        UPDATE raw_patents 
                        SET family_members = :members,
                            family_jurisdictions = :jurisdictions
                        WHERE "No" = :id
                    """),
                    updates
                )

            # Prepare response statistics
            total_members = sum(len(m) for m in df['family_members'])
            unique_countries = list({cc for codes in df['family_jurisdictions'] for cc in codes})
        
            return jsonify({
                "success": True,
                "updated_records": len(df),
                "total_family_members": total_members,
                "unique_jurisdictions": sorted(unique_countries),
                "sample_entry": {
                    "publication_number": df.iloc[0]['first_publication_number'],
                    "members": df.iloc[0]['family_members'],
                    "jurisdictions": df.iloc[0]['family_jurisdictions']
                }
            })

        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e),
                "failed_at": f"Failed at record {index}: {row['first_publication_number']}" if 'index' in locals() else "Initialization"
            }), 500

        finally:
            if 'scraper' in locals():
                scraper.close()

    @app.route('/api/research', methods=['POST'])
    def store_research():
        query = request.json.get('query')
        if not query:
            return jsonify({"error": "Query parameter is required"}), 400
    
        papers = fetch_research_data(query)
        if not papers:
            return jsonify({"error": "Failed to fetch data from API"}), 500
    
        df = process_research_data(papers)
        try:
            store_research_data(df)
            return jsonify({"message": "Data stored successfully"}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        
        
        
    @app.route('/api/research_retrieve_and_filtering', methods=['POST'])
    def store_research2():
        query = request.json.get('query')
        papers = fetch_research_data2(query)
        impact_factors = ImpactFactor.query.all()
    
        if not papers:
            return jsonify({"error": "No papers found"}), 404
    
        df = process_research_data2(papers , impact_factors)
    
        try:
            store_research_data2(df)
            return jsonify({
                "message": f"Stored {len(df)} papers",
                "stats": {
                    
                    "top_journal": df['publication_venue_name'].mode()[0]
                }
            }), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        
        
    """enpoint to fetch and store research data that's supposed to show the number of papers before and after filtering but not working yet"""
    # @app.route('/api/research_retrieve', methods=['POST'])
    # def store_research2():
    #     query = request.json.get('query')
    #     papers = fetch_research_data(query)

    #     if not papers:
    #         return jsonify({"error": "No papers found"}), 404

    # # Must match the two‐value return from process_research_data:
    #     result = process_research_data(papers)
    #     print(f"Received result type: {type(result)}, length: {len(result) if hasattr(result, '__len__') else 'N/A'}")
    #     if len(result) != 2:
    #         return jsonify({"error": "Unexpected data format from processing"}), 500
    #     df, stats = result
    #     if df.empty:
    #         return jsonify({"error": "No papers survived processing filters"}), 40

    #     try:
    #         store_research_data(df)
    #         return jsonify({
    #             "message": f"Stored {len(df)} papers",
    #             "stats": {
    #                 "before_clean": stats['before_clean'],
    #                 "after_clean": stats['after_clean'],
    #                 "after_filter": stats['after_filter'],
    #                 "top_journal": df['publication_venue_name'].mode()[0] if len(df) else None
    #             }
    #         }), 200

    #     except Exception as e:
    #         return jsonify({"error": str(e)}), 500





        
    @app.route('/process_and_store', methods=['GET'])
    def process_and_store():
        """
        Endpoint to check if store_processed_data works.
        Requires ?confirm=true query parameter to proceed.
        Returns JSON response indicating success or failure.
        """
        confirm = request.args.get('confirm', 'false').lower() == 'true'
        if not confirm:
            return jsonify({"message": "Operation not confirmed. Add ?confirm=true to proceed."}), 400
    
        try:
            logging.info("Starting data processing and storage.")
            message = store_processed_data()
            logging.info("Data processed and stored successfully.")
            return jsonify({"message": message}), 200
        except ValueError as e:
            logging.error(f"Configuration error: {str(e)}")
            return jsonify({"error": f"Configuration error: {str(e)}"}), 400
        except FileNotFoundError as e:
            logging.error(f"File not found: {str(e)}")
            return jsonify({"error": f"File not found: {str(e)}"}), 404
        except ParserError as e:
            logging.error(f"Error parsing file: {str(e)}")
            return jsonify({"error": f"Error parsing file: {str(e)}"}), 400
        except SQLAlchemyError as e:
            logging.error(f"Database error: {str(e)}")
            return jsonify({"error": f"Database error: {str(e)}"}), 500
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
            return jsonify({"error": f"Unexpected error: {str(e)}"}), 500




    @app.route('/fetch_research_data', methods=['POST'])
    def fetch_research_data():
        # Get query from the request
        data = request.get_json()
        query = data.get('query')

        if not query:
            return jsonify({'error': 'Query is required'}), 400

        # Load impact factors from the database for processing
        impact_factors_df = ImpactFactor.query.all()

        # Fetch and process Semantic Scholar data
        papers = fetch_research_data3(query)
        if not papers:
            return jsonify({'error': 'No papers found from Semantic Scholar or API error occurred'}), 500
        sem_df = process_research_data3(papers, impact_factors_df)

        # Add DOI cleaning for Semantic Scholar DataFrame
        sem_df['doi_clean'] = sem_df['DOI'].apply(clean_doi)

        # Fetch and process OpenAlex data
        openalex_works = fetch_openalex_works(max_docs=300, per_page=200)
        if not openalex_works:
            return jsonify({'error': 'No works found from OpenAlex or API error occurred'}), 500
    
        # Use the new process_documents function with impact_factors_df
        openalex_df = process_documents(openalex_works, journals_df=impact_factors_df)

        # Clean DOI for OpenAlex DataFrame (ISSN cleaning is now handled in process_documents)
        openalex_df['doi_clean'] = openalex_df['doi'].apply(clean_doi)

        # Rename columns to ensure consistency before merging
        sem_df, openalex_df = renaming_columns(sem_df, openalex_df)

        # Merge the two DataFrames using merge_unique_by_doi
        final_df = merge_unique_by_doi(sem_df, openalex_df)

        # Store the merged DataFrame
        store_research_data3(final_df)

        # Return a success response
        return jsonify({
            'message': 'Research data fetched, merged, and stored successfully',
            'papers_processed': len(final_df),
            'semantic_scholar_papers': len(sem_df),
            'openalex_papers': len(openalex_df)
        }), 200
        
        
    @app.route('/top_keyword', methods=['GET'])
    def keyword_analysis():
        
        
        try:
            # Fetch patent data from the database
            query = 'SELECT * FROM raw_patents'
            logger.info("Fetching patent data from raw_patents table")
            df = pd.read_sql(query, engine)
        
            if df.empty:
                logger.warning("No patent data found in the database")
                return jsonify({"error": "No patent data found in the database"}), 404
        
            # Preprocess the Titre column
            logger.info("Preprocessing patent titles")
            df['title'] = df['Title'].apply(preprocess_text)
        
            # Extract texts, dropping any nulls
            texts = df['title'].dropna().tolist()
            if not texts:
                logger.warning("No valid texts after preprocessing")
                return jsonify({"error": "No valid texts available after preprocessing"}), 400
        
            # Extract top keywords
            logger.info(f"Extracting top keywords from {len(texts)} texts")
            top_keywords = extract_keywords(texts)
        
            if not top_keywords:
                logger.warning("No keywords extracted (possibly due to insufficient documents)")
                return jsonify({"warning": "No keywords extracted, try adjusting min_df or adding more data"}), 200
        
            # Format keywords for JSON response
            keywords_response = [{"keyword": keyword, "score": float(score)} for keyword, score in top_keywords]
        
            logger.info("Successfully extracted keywords")
            return jsonify({
                "status": "success",
                "keywords": keywords_response,
                "count": len(keywords_response)
            }), 200
    
        except sqlalchemy.exc.SQLAlchemyError as e:
            logger.error(f"Database error: {str(e)}")
            return jsonify({"error": f"Database error: {str(e)}"}), 500
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return jsonify({"error": f"Unexpected error: {str(e)}"}), 500
        
        
                    
    @app.route('/api/processed_texts', methods=['GET'])
    def get_processed_texts_from_db():
            """
            1) Fetch the 'Title' column from raw_patents.
            2) Preprocess each title with preprocess_text(...).
            3) Return JSON: { "processed": [ "...", "...", ... ] }.
            """
            try:
                # 1) Read only the Title column from raw_patents
                query = 'SELECT "Title" FROM raw_patents'
                df = pd.read_sql(query, engine)

                if df.empty:
                    return jsonify({"error": "No records found in raw_patents"}), 404

                # 2) Apply preprocess_text to every Title, skipping NaN
                processed_list = []
                for raw_title in df['Title'].fillna(""):
                    proc = preprocess_text(raw_title, use_stemming=True)
                    if proc: 
                        processed_list.append(proc)

                if not processed_list:
                    return jsonify({"error": "No valid text to process"}), 400

                # 3) Return JSON array
                return jsonify({"processed": processed_list}), 200

            except sqlalchemy.exc.SQLAlchemyError as e:
                # Database‐level errors
                logger.error(f"Database error in /api/processed_texts: {e}")
                return jsonify({"error": f"Database error: {str(e)}"}), 500

            except Exception as e:
                # Any other unexpected errors
                logger.error(f"Unexpected error in /api/processed_texts: {e}")
                return jsonify({"error": f"Unexpected error: {str(e)}"}), 500
            
            
            
    @app.route('/api/patents/first_filing_years', methods=['GET'])
    def get_first_filing_years():
    
        try:
            # Simple query using the existing first_filing_year column
            query = text('''
                SELECT first_filing_year AS year, COUNT(*) AS count
                FROM raw_patents
                WHERE first_filing_year IS NOT NULL
                GROUP BY first_filing_year
                ORDER BY year
            ''')

            with engine.connect() as conn:
                result = conn.execute(query)
                rows = result.fetchall()

            if not rows:
                return jsonify({"message": "No patent data available"}), 404

            # Prepare data for chart
            years = []
            counts = []
            for row in rows:
                years.append(int(row[0]))
                counts.append(row[1])

            return jsonify({
                "labels": years,
                "datasets": [{
                    "label": "Number of Patents",
                    "data": counts,
                    "borderColor": "rgb(75, 192, 192)",
                    "tension": 0.1
                }]
            }), 200

        except Exception as e:
            logger.error(f"Error fetching filing years: {str(e)}")
            return jsonify({"error": f"Failed to retrieve filing years: {str(e)}"}), 500
        


    def get_top_keywords(tfidf_matrix, feature_names, top_n=5):
        """
        Get top N keywords for each document in the TF-IDF matrix.
        """
        top_keywords = []
        for row in tfidf_matrix:
            top_indices = row.toarray().argsort()[0][-top_n:][::-1]
            top_words = [feature_names[i] for i in top_indices]
            top_keywords.append(top_words)
        return top_keywords

    @app.route('/api/topic_evolution', methods=['GET'])
    def topic_evolution():
        """
        Endpoint to analyze topic evolution in patent data.
        Stores data in patent_keywords only if keyword_df is not empty,
        analyzes topic evolution, and returns results as JSON.
        """
        try:
            # Query database for required columns
            patents = db.session.query(
                RawPatent.first_publication_number.label('first publication number'),
                RawPatent.title.label('Title'),
                RawPatent.first_filing_year.label('first filing year')
            ).all()
        
            if not patents:
                logger.info("No patents found in the database")
                return jsonify({"message": "No patents found in the database"}), 404
        
            # Create DataFrame from query results
            keyword_df = pd.DataFrame([row._asdict() for row in patents])
        
            # Drop rows where 'Title' is null
            keyword_df = keyword_df.dropna(subset=['Title'])
        
            if keyword_df.empty:
                logger.info("No patents with valid titles after filtering")
                return jsonify({"message": "No patents with valid titles"}), 404
        
            # Preprocess titles
            keyword_df['processed_title'] = keyword_df['Title'].apply(preprocess_text)
        
            # Compute TF-IDF
            vectorizer = TfidfVectorizer(max_df=0.85, min_df=2, ngram_range=(1, 3))
            tfidf_matrix = vectorizer.fit_transform(keyword_df['processed_title'])
            feature_names = vectorizer.get_feature_names_out()
        
            # Get top keywords for each patent
            top_keywords_list = get_top_keywords(tfidf_matrix, feature_names, top_n=5)
        
            # Store in patent_keywords only if keyword_df is not empty
            for i, row in keyword_df.iterrows():
                patent_keyword = PatentKeyword(
                    first_publication_number=row['first publication number'],
                    title=row['Title'],
                    first_filing_year=row['first filing year'],
                    keywords=top_keywords_list[i]
                )
                db.session.add(patent_keyword)
            db.session.commit()
        
            # Analyze topic evolution
            topic_evolution_data, windows_data , divergences, valid_years = analyze_topic_evolution(keyword_df)
        
            # Delete existing Window and Topic records
            db.session.query(Topic).delete()
            db.session.query(Window).delete()
            db.session.query(Divergence).delete()
            db.session.commit()
        
            # Insert new windows
            window_objects = []
            for win in windows_data:
                window = Window(start_year=win['start'], end_year=win['end'])
                db.session.add(window)
                window_objects.append(window)
            db.session.commit()
        
            # Map (start, end) to window objects
            window_dict = {(w.start_year, w.end_year): w for w in window_objects}
        
            # Insert topics
            for topic_data in topic_evolution_data:
                start = topic_data['start']
                end = topic_data['end']
                window = window_dict.get((start, end))
                if window:
                    topic_id_str = topic_data['topic_id']
                    topic_number = int(topic_id_str.split('-')[-1])
                    topic = Topic(
                        window_id=window.id,
                        topic_number=topic_number,
                        words=topic_data['words'],
                        weights=topic_data['weights']
                    )
                    db.session.add(topic)
            db.session.commit()
            
                   # Insert divergences
            for i in range(len(divergences)):
                from_year = int(valid_years[i])
                to_year = int(valid_years[i + 1])
                divergence_value = float(divergences[i])
                divergence_record = Divergence(from_year=from_year, to_year=to_year, divergence=divergence_value)
                db.session.add(divergence_record)
            db.session.commit()
        
            # Query stored data
            windows = Window.query.order_by(Window.start_year).all()
            topics = Topic.query.join(Window).order_by(Window.start_year, Topic.topic_number).all()
            divergences_query = Divergence.query.order_by(Divergence.from_year).all()

            window_list = [
                {
                    'start': w.start_year,
                    'end': w.end_year,
                    'years': list(range(w.start_year, w.end_year + 1))
                } for w in windows
            ]
        
            topic_evolution_list = [
                {
                    'start': t.window.start_year,
                    'end': t.window.end_year,
                    'topic_id': f"{t.window.start_year}-{t.topic_number}",
                    'words': t.words,
                    'weights': t.weights
                } for t in topics
            ]
            divergences_list = [
            {
                'from_year': d.from_year,
                'to_year': d.to_year,
                'divergence': d.divergence
            } for d in divergences_query
        ]
        
            # Query patent_keywords for response
            patent_keywords = PatentKeyword.query.all()
            patent_keywords_list = [
                {
                    'first_publication_number': pk.first_publication_number,
                    'title': pk.title,
                    'first_filing_year': pk.first_filing_year,
                    'keywords': pk.keywords
                } for pk in patent_keywords
            ]
        
            logger.info("Topic evolution analysis and keyword extraction completed successfully")
            return jsonify({
                "topic_evolution": topic_evolution_list,
                "windows": window_list,
                "patent_keywords": patent_keywords_list,
                "divergences": divergences_list
            }), 200
    
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error in topic_evolution endpoint: {str(e)}")
            return jsonify({"error": f"Failed to process topic evolution: {str(e)}"}), 500


    
    @app.route('/api/automatic_topic_shift', methods=['GET'])
    def automatic_topic_shift():
        try:
            # Query all divergences, ordered by from_year
            divergences = Divergence.query.order_by(Divergence.from_year).all()
            if not divergences:
                return jsonify({"message": "No divergence data available"}), 200

            # Calculate threshold as the 80th percentile of divergence values
            divergence_values = [d.divergence for d in divergences]
            threshold = np.percentile(divergence_values, 80) if divergence_values else 0

            # Query all windows, ordered by start_year
            windows = Window.query.order_by(Window.start_year).all()

            # Format divergence data
            divergence_data = [
                {
                    "from_year": d.from_year,
                    "to_year": d.to_year,
                    "divergence": d.divergence
                }   
                for d in divergences
            ]

            # Format windows data
            window_list = [
                {
                    "start": w.start_year,
                    "end": w.end_year
                }
                for w in windows
            ]

            # Construct and return the JSON response
            response = {
                "divergence_data": divergence_data,
                "threshold": threshold,
                "windows": window_list
            }
            return jsonify(response), 200

        except Exception as e:
            return jsonify({"error": str(e)}), 500
        
        
    # @app.route('/api/evolving_word_clouds', methods=['GET'])
    # def evolving_word_clouds():
    #     try:
    #         # Query all windows, ordered by start year
    #         windows = Window.query.order_by(Window.start_year).all()
    #         result = []

    #         for window in windows:
    #             # Query topics associated with this window
    #             topics = Topic.query.filter_by(window_id=window.id).all()
            
    #             # Collect all words from topics in this window
    #             all_words = [word for topic in topics for word in topic.words]
            
    #             if not all_words:
    #                 continue
            
    #             # Count frequency of each word (replicates WordCloud's default behavior)
    #             word_freq = Counter(all_words)
            
    #             # Format word frequencies as a list of dictionaries
    #             word_freq_list = [{'word': word, 'frequency': freq} for word, freq in word_freq.items()]
            
    #             # Append window data to result
    #             result.append({
    #                 'start': window.start_year,
    #                 'end': window.end_year,
    #                 'word_frequencies': word_freq_list
    #             })
        
    #         return jsonify(result), 200
    
    #     except Exception as e:
    #         return jsonify({"error": str(e)}), 500

    @app.route('/api/weighted_word_clouds', methods=['GET'])
    def weighted_word_clouds():
        """
        Endpoint to generate word cloud data for each time window.
        Aggregates word weights across all topics within each window.
        Returns a JSON response with start/end years and word-weight pairs.
        """
        try:
            # Set up logging
            logger = logging.getLogger(__name__)

            # Query all windows, ordered by start year
            windows = Window.query.order_by(Window.start_year).all()
            result = []

            for window in windows:
                # Use defaultdict to accumulate weights for each word
                word_weight_dict = defaultdict(float)

                # Query all topics associated with this window
                topics = Topic.query.filter_by(window_id=window.id).all()

                for topic in topics:
                    # Validate that words and weights arrays have the same length
                    if len(topic.words) != len(topic.weights):
                        logger.warning(
                            f"Skipping topic {topic.id}: words and weights lengths mismatch "
                            f"({len(topic.words)} words, {len(topic.weights)} weights)"
                        )
                        continue

                    # Aggregate weights by summing for each word
                    for word, weight in zip(topic.words, topic.weights):
                        processed_word = preprocess_text(word.lower())
                        if processed_word and processed_word.strip():
                            word_weight_dict[word] += weight

                # Skip if no words were aggregated
                if not word_weight_dict:
                    logger.info(f"No valid words found for window {window.start_year}-{window.end_year}")
                    continue

                # Convert aggregated weights to a sorted list of dictionaries
                word_list = [
                    {"word": word, "weight": weight}
                    for word, weight in sorted(word_weight_dict.items(), key=lambda x: x[1], reverse=True)
                ]

                # Append window data to the result
                result.append({
                    "start": window.start_year,
                    "end": window.end_year,
                    "words": word_list
                })

            # Log successful completion
            logger.info(f"Generated word cloud data for {len(result)} windows")
            return jsonify(result), 200

        except Exception as e:
            logger.error(f"Error in weighted_word_clouds endpoint: {str(e)}")
            return jsonify({"error": str(e)}), 500
        
        
    @app.route('/classify_patents', methods=['POST'])
    def classify_patents():
        try:
            # 1) If the IPCClassification table is empty, load from CSV
            if IPCClassification.query.count() == 0:
                csv_path = 'classification_df.csv'  

               
                df = pd.read_csv(
                    csv_path,
                    sep=';',
                    usecols=['CPC Symbol', 'Classification Title']
                )

                # Rename columns so they match the model field names
                df.columns = ['cpc_symbol', 'classification_title']

                # Use cpc_symbol as the DataFrame index for easy lookup
                df.set_index('cpc_symbol', inplace=True)

                # Insert into the database
                for code, row in df.iterrows():
                    new_class = IPCClassification(
                        cpc_symbol=code,
                        classification_title=row['classification_title']
                    )
                    db.session.add(new_class)
                db.session.commit()
                
            if IPCFieldOfStudy.query.count() == 0:
                csv_path = 'IPC_to_fieldOfStudy.csv'  

                df = pd.read_csv(
                    csv_path,
                    sep=',',
                    usecols=lambda x: x in ['IPC', 'description', 'fields'] 
                )

                # Rename columns so they match the model field names
                df.columns = ['ipc','description', 'fields']

                # Use cpc_symbol as the DataFrame index for easy lookup
                df.set_index('ipc', inplace=True)

                # Insert into the database
                for code, row in df.iterrows():
                    new_field = IPCFieldOfStudy(
                        ipc=code,
                        fields=row['fields'],
                        description=row['description']
                    )
                    db.session.add(new_field)
                db.session.commit()

            # 2) Build a lookup DataFrame from whatever is currently in the table
            all_rows = IPCClassification.query.all()
            classification_dict = {
                row.cpc_symbol: row.classification_title
                for row in all_rows
            }
            classification_df = pd.DataFrame.from_dict(
                classification_dict,
                orient='index',
                columns=['classification_title']
            )
            classification_df.index.name = 'cpc_symbol'
            
            all_rows2 = IPCFieldOfStudy.query.all()
            field_of_study_dict = {
                row.ipc: {
                    'ipc': row.ipc,
                    'fields': row.fields,
                    'description': row.description
                }
                for row in all_rows2
            }

            field_of_study_df = pd.DataFrame.from_dict(
                field_of_study_dict,
                orient='index',
                columns=['ipc','fields', 'description']
            )
            field_of_study_df.index.name = 'ipc'
            field_of_study_df['fields'] = field_of_study_df['fields'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            field_of_study_df['fields'] = field_of_study_df['fields'].apply(normalize_engineering)
            field_of_study_df['ipc'] = field_of_study_df['ipc'].str[:3].str.strip().str.upper()
            #ipc_mapping = field_of_study_df.set_index('IPC')['fields'].to_dict()
            # 3) Fetch unclassified patents and assign the “meaning” based on IPC codes
            patents = RawPatent.query.all()
            if not patents:
                return jsonify({"message": "No patents found to classify"}), 200

            updated_count = 0
            for patent in patents:
                if not patent.ipc:
                    continue
                # Cleanup step #1: remove leading/trailing braces & quotes:

                cleaned = patent.ipc.strip('"{}')
                # Cleanup step #2: split on commas, drop any empty strings
                
                ipc_list = [code.strip() for code in cleaned.split(',') if code.strip()]
                if not ipc_list:
                    continue
                
                meaning = get_ipc_meaning(ipc_list, classification_df)
                fields = get_patent_fields(ipc_list, field_of_study_df)  
                if isinstance(fields, str):
                    try:
                        fields = ast.literal_eval(fields)
                    except Exception:
                        fields = [fields]
                if meaning:
                # Check if this patent already exists in classified_patents
                    existing = ClassifiedPatent.query.get(patent.id)
                    if existing:
                        # Update existing record
                        existing.ipc_meaning = meaning
                        existing.fields = fields 
                    else:
                        # Create new classified patent record
                        classified_patent = ClassifiedPatent(
                            id=patent.id,
                            title=patent.title,
                            inventors=patent.inventors,
                            applicants=patent.applicants,
                            publication_number=patent.publication_number,
                            earliest_priority=patent.earliest_priority,
                            earliest_publication=patent.earliest_publication,
                            ipc=patent.ipc,
                            cpc=patent.cpc,
                            publication_date=patent.publication_date,
                            first_publication_date=patent.first_publication_date,
                            second_publication_date=patent.second_publication_date,
                            first_filing_year=patent.first_filing_year,
                            earliest_priority_year=patent.earliest_priority_year,
                            applicant_country=patent.applicant_country,
                            family_number=patent.family_number,
                            family_jurisdictions=patent.family_jurisdictions,
                            family_members=patent.family_members,
                            first_publication_number=patent.first_publication_number,
                            second_publication_number=patent.second_publication_number,
                            first_publication_country=patent.first_publication_country,
                            second_publication_country=patent.second_publication_country,
                            ipc_meaning=meaning,
                            fields=fields
                        )
                        db.session.add(classified_patent)
                    updated_count += 1

            db.session.commit()
            return jsonify({
                "message": f"Successfully processed {updated_count} patents",
                "classified_count": updated_count
            }), 200
        except Exception as e:
                db.session.rollback()
                return jsonify({"error": f"Failed to classify patents: {str(e)}"}), 500
        
    @app.route('/api/patent_field_trends', methods=['GET'])
    def patent_field_trends():
        """
        Returns JSON for plotting field trends over time.
        Query params:
        - top_n: number of top fields to return (default 5)
        - smoothing: rolling window size (default 3)
        """
        try:
            top_n = int(request.args.get('top_n', 5))
            smoothing = int(request.args.get('smoothing', 3))

            # Query classified patents with year and fields
            patents = ClassifiedPatent.query.with_entities(
                ClassifiedPatent.first_filing_year,
                ClassifiedPatent.fields
            ).filter(ClassifiedPatent.first_filing_year.isnot(None)).all()

            # Build DataFrame
            rows = []
            for year, fields in patents:
                # Ensure fields is a list
                if isinstance(fields, str):
                    try:
                        fields = ast.literal_eval(fields)
                    except Exception:
                        fields = [fields]
                for field in fields:
                    rows.append({'year': int(year), 'field': field})

            if not rows:
                return jsonify({"labels": [], "datasets": []})

            df = pd.DataFrame(rows)
            # Count papers per year/field
            field_counts = (
                df.groupby(['year', 'field'])
                .size()
                .reset_index(name='count')
            )
            # Pivot for trend lines
            pivot = field_counts.pivot(index='year', columns='field', values='count').fillna(0)
            # Smooth with rolling average
            pivot_smooth = pivot.rolling(window=smoothing, min_periods=1).mean()

            # Pick top N fields by total count
            top_fields = pivot.sum().sort_values(ascending=False).head(top_n).index
            labels = list(map(int, pivot_smooth.index))

            datasets = []
            for field in top_fields:
                datasets.append({
                    "label": field,
                    "data": [int(v) for v in pivot_smooth[field].values]
                })

            return jsonify({
                "labels": labels,
                "datasets": datasets
            }), 200

        except Exception as e:
            return jsonify({"error": str(e)}), 500
        

    @app.route('/api/cooccurrence_trends', methods=['GET'])
    def cooccurrence_trends():
        """
        Returns JSON data for plotting co-occurrence trends of technology keyword pairs.
        Removes English and French stopwords from titles and ensures term pairs are not identical.
        """
        try:
            # Fetch patent data from the database
            query = 'SELECT "Publication number", "Title", "first_filing_year" FROM raw_patents'
            df = pd.read_sql(query, engine)
            df = df.dropna(subset=['Title', 'first_filing_year'])

            # Remove stopwords from Title
            df['Title'] = df['Title'].apply(clean_text_remove_stopwords)

            # Remove empty titles after cleaning
            df = df[df['Title'].str.strip().astype(bool)]

            # Prepare the grouped DataFrame as required by track_cooccurrence_trends
            grouped = df.groupby("first_filing_year")["Title"].apply(lambda texts: " ".join(texts)).reset_index()

            # Run the co-occurrence trend analysis
            cooc_trends = track_cooccurrence_trends(
                grouped,
                time_col='first_filing_year',
                text_col='Title',
                window_size=5,
                min_count=10
            )

            # Filter out pairs where both terms are the same (case-insensitive)
            cooc_trends = cooc_trends[cooc_trends['term1'].str.lower() != cooc_trends['term2'].str.lower()]

            # Get top emerging and declining combinations
            emerging_tech = cooc_trends[
                (cooc_trends.slope > 0) & (cooc_trends.p_value < 0.05)
            ].sort_values('slope', ascending=False).head(5)

            declining_tech = cooc_trends[
                (cooc_trends.slope < 0) & (cooc_trends.p_value < 0.05)
            ].sort_values('slope').head(5)

            # Prepare data for plotting
            def prepare_plot_data(df):
                plot_data = []
                for _, row in df.iterrows():
                    for year, freq in row['frequency_history']:
                        plot_data.append({
                            'year': int(year),
                            'frequency': int(freq),
                            'term_pair': f"{row['term1']} & {row['term2']}"
                        })
                return plot_data

            response = {
                "emerging": prepare_plot_data(emerging_tech),
                "declining": prepare_plot_data(declining_tech),
                "emerging_pairs": [
                    {
                        "term1": row['term1'],
                        "term2": row['term2'],
                        "slope": row['slope'],
                        "total_count": row['total_count']
                    }
                    for _, row in emerging_tech.iterrows()
                ],
                "declining_pairs": [
                    {
                        "term1": row['term1'],
                        "term2": row['term2'],
                        "slope": row['slope'],
                        "total_count": row['total_count']
                    }
                    for _, row in declining_tech.iterrows()
                ]
            }
            return jsonify(response), 200

        except Exception as e:
            return jsonify({"error": str(e)}), 500
        
        
        
    @app.route('/api/analyze_applicants', methods=['POST'])
    def analyze_applicants():
        # Load patent data from DB
        patents = RawPatent.query.all()
        data = []
        for patent in patents:
            data.append({
                'first applicant': patent.applicants.split('\n')[0] if patent.applicants else None,
                'second applicant': patent.applicants.split('\n')[1] if patent.applicants and '\n' in patent.applicants else None,
                'Inventors': patent.inventors
            })
        df = pd.DataFrame(data)
        # Fill missing columns if needed
        if 'Inventors' not in df.columns:
            df['Inventors'] = None

        # Generate DataFrames
        applicants_df = get_applicants_df(df)
        applicant_type_df = get_applicant_type_df(applicants_df)

        # Store applicants_df
        Applicant.query.delete()
        for _, row in applicants_df.iterrows():
            db.session.add(Applicant(applicant_name=row['Applicants'], applicant_type=row['Applicant Type']))
        db.session.commit()

        # Store applicant_type_df
        ApplicantType.query.delete()
        for _, row in applicant_type_df.iterrows():
            db.session.add(ApplicantType(applicant_type=row['Applicant Type'], percentage=row['Percentage']))
        db.session.commit()

        return jsonify({"status": "success"})
    
    
    @app.route('/api/applicant_type_summary', methods=['GET'])
    def applicant_type_summary():
        """
        Returns applicant type distribution for visualization.
    Example response:
    {
        "labels": ["Company - Incorporated/Corporation", "University/Research Institution", ...],
        "percentages": [45.2, 30.1, ...],
        "details": [
            {"applicant_name": "Toyota Motor Corp", "applicant_type": "Company - Incorporated/Corporation"},
            ...
        ]
    }
     """
        # Query summary table
        summary = ApplicantType.query.order_by(ApplicantType.percentage.desc()).all()
        labels = [row.applicant_type for row in summary]
        percentages = [row.percentage for row in summary]

        # Optionally, provide a detailed list (for tooltips, drill-down, etc.)
        applicants = Applicant.query.all()
        details = [
            {"applicant_name": a.applicant_name, "applicant_type": a.applicant_type}
            for a in applicants
        ]

        return jsonify({
            "labels": labels,
            "percentages": percentages,
            "details": details
        })

    def load_applicants_df() -> pd.DataFrame:
        """
        Reads the entire `applicants` table into a DataFrame.
        """
        # this issues: SELECT * FROM applicants
        df = pd.read_sql_table(
            table_name='applicants',
            con=db.engine,
            columns=['id', 'applicant_name', 'applicant_type']
        )
        return df


    @app.route('/api/innovation_cycle', methods=['GET'])
    def innovation_cycle():
        # 1) Pull all “Applicants” strings from the DB
        df = load_applicants_df()
        df = df.rename(columns={'applicant_name': 'Applicants'})
        cycle = get_innovation_cycle(df, top_n=10)
        # 5) Return as JSON
        return jsonify(f"{cycle:.2f}"), 200
    
    
    @app.route('/api/coapplicant_rate', methods=['GET'])
    def coapplicant_rate():
        try:
        
            # Load applicant data
            query = """
            SELECT "Applicants" 
            FROM raw_patents 
            WHERE "Applicants" IS NOT NULL
            """
            df = pd.read_sql(query, engine)
            if df.empty:
                return jsonify({
                    "coapplicant_rate": 0.0,
                    "total_applications": 0,
                    "coapplicant_count": 0
                })
        
            # Split applicants into first and second
            split_applicants = df['Applicants'].str.split('\n', n=1, expand=True)
            df['first_applicant'] = split_applicants[0]
            df['second_applicant'] = split_applicants[1]
        
            # Calculate co-applicant rate
            total_applications = len(df)
            coapplicant_count = df['second_applicant'].notnull().sum()
            coapplicant_rate = (coapplicant_count / total_applications) * 100
        
            return jsonify({
                "coapplicant_rate": float(round(coapplicant_rate, 2)),
                "total_applications": int(total_applications),
                "coapplicant_count": int(coapplicant_count)
            })
        
        except Exception as e:
            app.logger.error(f"Error in coapplicant_rate: {str(e)}")
            return jsonify({"error": str(e)}), 500
        
        
        
    @app.route('/api/growth_rate', methods=['GET'])  
    def growth_rate():
        df = pd.read_sql_table('raw_patents', con=engine, columns=['first_filing_year'])
        growth_rate = patent_growth_summary(df)
        return jsonify({"growth_rate": growth_rate}), 200

                   
            
    @app.route('/api/market_cost', methods=['POST'])
    def update_costs():
        try:
            # Step 1: Ensure Cost table is populated
            if Cost.query.count() == 0:
                csv_path = os.path.join(os.path.dirname(__file__), 'corrected-patent-cost-data.csv')
                if not os.path.exists(csv_path):
                    return jsonify({"error": "CSV file not found"}), 404
                cost = pd.read_csv(csv_path)
                cost.columns = [
                'Country',   
                'Years 0.0-1.5',
                'Years 2.0-4.5',
                'Years 5.0-9.5',
                'Years 10.0-14.5',
                'Years 15.0-20.0',
                'Total Cost (US$)'
                ]
                updated_cost = update_cost_df(cost)
                rename_dict = {
                'Years 0.0-1.5': 'Years_0_1_5',
                'Years 2.0-4.5': 'Years_2_4_5',
                'Years 5.0-9.5': 'Years_5_9_5',
                'Years 10.0-14.5': 'Years_10_14_5',
                'Years 15.0-20.0': 'Years_15_20',
                'Total Cost (US$)': 'Total_cost'
                }
                updated_cost.rename(columns=rename_dict, inplace=True)
                for _, row in updated_cost.iterrows():
                    new_cost = Cost(
                        Country=row['Country'],
                        Years_0_1_5=row['Years_0_1_5'],
                        Years_2_4_5=row['Years_2_4_5'],
                        Years_5_9_5=row['Years_5_9_5'],
                        Years_10_14_5=row['Years_10_14_5'],
                        Years_15_20=row['Years_15_20'],
                        Total_cost=row['Total_cost']
                    )
                    db.session.add(new_cost)
                db.session.commit()

            # Step 2: Load cost data
            cost_df = pd.read_sql('SELECT * FROM costs', db.engine)

            # Step 3: Fetch patent data into a DataFrame
            query = 'SELECT "earliest_priority_year", "first_publication_country" FROM raw_patents'
            patent_df = pd.read_sql(query, db.engine)

            # Step 4: Calculate patent age
            patent_df = calculate_age(patent_df)

            # Step 5: Assign cost to each patent
            patent_df['cost'] = patent_df.apply(lambda row: assign_cost(row, cost_df), axis=1)

            # Step 6: Clear existing patent_costs and save new data
            db.session.query(PatentCost).delete()
            db.session.commit()
            for _, row in patent_df.iterrows():
                if pd.notnull(row['Patent Age']) and pd.notnull(row['cost']):
                    patent_cost = PatentCost(
                        patent_age=str(row['Patent Age']),
                        country=row['first_publication_country'],
                        cost=row['cost']
                    )
                    db.session.add(patent_cost)
            db.session.commit()

            return jsonify({"message": "Patent costs updated successfully"}), 200

        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Error in update_costs: {str(e)}")
            return jsonify({"error": str(e)}), 500
        
        
        






    @app.route('/api/originality_rate/fetch', methods=['POST'])
    def fetch_originality_data():
        request_id = str(uuid.uuid4())
        print(f"[{datetime.utcnow()}] Request ID: {request_id} - Fetch started")

        try:
            limit = request.args.get('limit', default=50, type=int)
            with app.app_context():
                db.create_all()
            patents_added = fetch_and_store_originality_data(limit=limit)
            # How many patents are in the DB now (valid for originality-rate calc)?
            _, _, valid_patents = calculate_originality_rate_from_db()

            print(f"[{datetime.utcnow()}] Request ID: {request_id} - Fetch completed")
            return jsonify({
                "status": "success",
                "message": f"Originality data fetched and stored (limit={limit}).",
                "patents_added": patents_added,
                "processed_patents": valid_patents, 
                "request_id": request_id
            }), 200
        except Exception as e:
            print(f"[{datetime.utcnow()}] Request ID: {request_id} - Error: {str(e)}")
            return jsonify({"error": str(e), "request_id": request_id}), 500

    @app.route('/api/originality_rate', methods=['GET'])
    def originality_rate():
        try:
            originality_rate, total_patents, n_valid = calculate_originality_rate_from_db()
            if originality_rate is None:
                return jsonify({
                    "error": f"Insufficient data: only {n_valid} valid patents. Need at least 30."
                }), 400
            return jsonify({
                "originality_rate": originality_rate,
                "total_patents": total_patents,
                "valid_patents": n_valid
            }), 200
        except Exception as e:
            app.logger.error(f"Error in originality_rate: {str(e)}")
            return jsonify({"error": str(e)}), 500
        
        
        
        
    @app.route('/api/market_metrics', methods=['GET'])
    def market_metrics():
        try:
            # Load all patents and cost data from the database
            patents = [p.to_dict() for p in RawPatent.query.all()]
            costs = [c.to_dict() for c in Cost.query.all()]
            import pandas as pd
            patents_df = pd.DataFrame(patents)
            cost_df = pd.DataFrame(costs)
            # Defensive: If either is empty, return zeros
            if patents_df.empty or cost_df.empty:
                return jsonify({
                'market_value': 0.0,
                'market_rate': 0.0,
                'mean_value': 0.0
            }), 200
            # Ensure 'Patent Age' column exists
            if 'Patent Age' not in patents_df.columns:
                from datetime import datetime
                current_year = datetime.now().year
                patents_df['Patent Age'] = current_year - patents_df['earliest_priority_year']
            # Compute metrics
            market_value, market_rate, mean_value = get_market_metrics(patents_df, cost_df)
            return jsonify({
                'market_value': float(market_value),
                'market_rate': float(market_rate),
                'mean_value': float(mean_value)
            }), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500



            
            
            
            
            


    # @app.route('/fetch_research_data', methods=['POST'])
    # def fetch_research_data():
    #     # Get query from the request
    #     data = request.get_json()
    #     query = data.get('query')
    
    #     if not query:
    #         return jsonify({'error': 'Query is required'}), 400
    
    #     # Load impact factors from the database for Semantic Scholar processing
    #     impact_factors = db.session.query(ImpactFactor).all()
    
    #     # Fetch and process Semantic Scholar data
    #     # Note: Assumes fetch_research_data2 includes 'externalIds' in fields
    #     papers = fetch_research_data3(query)
    #     if not papers:
    #         return jsonify({'error': 'No papers found from Semantic Scholar or API error occurred'}), 500
    #     sem_df = process_research_data3(papers, impact_factors)
    
    #     # Add DOI cleaning for Semantic Scholar DataFrame
    #     # Assumes 'DOI' is extracted in process_research_data2 from externalIds
    #     sem_df['doi_clean'] = sem_df['DOI'].apply(clean_doi)
    
    #     # Fetch and process OpenAlex data
    #     openalex_works = fetch_openalex_works(max_docs=300, per_page=200)
    #     if not openalex_works:
    #         return jsonify({'error': 'No works found from OpenAlex or API error occurred'}), 500
    #     openalex_df = process_documents(openalex_works)
    
    #     # Clean ISSN and DOI for OpenAlex DataFrame
    #     openalex_df['journal_issn_l_clean'] = openalex_df['journal_issn_l'].apply(clean_issn)
    #     openalex_df['doi_clean'] = openalex_df['DOI'].apply(clean_doi)
    
    #     # Rename columns to ensure consistency before merging
    #     sem_df, openalex_df = renaming_columns(sem_df, openalex_df)
    
    #     # Merge the two DataFrames using merge_unique_by_doi
    #     final_df = merge_unique_by_doi(sem_df, openalex_df)
    
    #     # Store the merged DataFrame
    #     store_research_data3(final_df)
    
    # # Return a success response
    #     return jsonify({
    #         'message': 'Research data fetched, merged, and stored successfully',
    #         'papers_processed': len(final_df)
    #     }), 200









#@app.route('/api/family_members/API', methods=['POST']) 

    
    # @app.route('/api/family', methods=['POST'])
    # def populate_family_members():
    # # Get JSON data from the request
    #     CONSUMER_KEY = os.getenv("CONSUMER_KEY_3").strip()
    #     CONSUMER_SECRET = os.getenv("CONSUMER_SECRET_3").strip()
    #     CONSUMER_KEY1 = os.getenv("CONSUMER_KEY_2").strip()
    #     CONSUMER_SECRET1 = os.getenv("CONSUMER_SECRET_2").strip()

    #     try:
    #     # Fetch data from database
    #         query = 'SELECT * FROM raw_patents'
    #         df = pd.read_sql(query, engine)

    #     # Data cleaning
    #         df.rename(columns={
    #             'Titre': 'Title',
    #             'Inventeurs': 'Inventors',
    #             'Demandeurs': 'Applicants',
    #             'Numéro de publication': 'Publication number',
    #             'Priorité la plus ancienne': 'Earliest priority',
    #             'CIB': 'IPC',
    #             'CPC': 'CPC',
    #             'Date de publication': 'Publication date',
    #             'Publication la plus ancienne': 'Earliest publication',
    #             'Numéro de famille': 'Family number'
    #         }, inplace=True)

    #         # Split 'Publication date' into two columns using regex
    #         df['first publication date'] = df['Publication date'].str.extract(r'^(\S+)', expand=False)
    #         df['second publication date'] = df['Publication date'].str.extract(r'^\S+\s+(.*)', expand=False)


    #         df['second publication date'] = df['second publication date'].str.strip('\r\n')
        
    #         df[['first publication number', 'second publication number']] = df['Publication number'].str.split(' ', n=1, expand=True)
    #         df['second publication number'] = df['second publication number'].str.strip('\r\n')
        
    #         if 'Unnamed: 11' in df.columns:
    #             df.drop(columns=['Unnamed: 11', 'Publication date'], inplace=True)
        
    #         df['family number'] = pd.to_numeric(df['Family number'], errors='coerce')
    #         df.rename(columns={'Family number': 'family number'}, inplace=True)
            
    #         # Calculate the number of rows for each part
    #         n = len(df) // 3

    #     # Split the DataFrame into three parts
    #         df1 = df.iloc[:n].copy()       # First part
    #         df2 = df.iloc[n:2*n].copy()    # Second part
    #         df3 = df.iloc[2*n:].copy()     # Third part

    #     # Process df1
    #         df1 = process_dataframe_parallel(df1, 'first publication number', max_workers=4)
    #         print('num of null values df1 :', df1['family_members'].isnull().sum(), 'number of empty arrays : ', df1['family_jurisdictions'].apply(lambda x: isinstance(x, list) and len(x) == 0).sum())
    #     #process df2
    #         search_keywords = fetch_last_search_keywords_from_db()
    #         if 'family_members' not in df2.columns: 
    #             df['family_members'] = None
    #         #split the dataframe into 3 parts 
    #         indices = df2.index.tolist()
    #         n = len(indices)
    #         part_size = n // 3
    #         remainder = n % 3
    #         parts= []
    #         start = 0
    #         for i in range(3) : 
    #             if i < remainder:
    #                 end = start + part_size + 1
    #             else:
    #                 end = start + part_size 
    #             parts.append(indices[start:end])
    #             start = end 
    #         #create three threads , each with ist own patentsSearch instance 
    #         threads = []
    #         for part in parts : 
    #             thread = threading.Thread(target=process_rows, args=(df2, part,search_keywords,False))
    #             threads.append(thread)
    #         for thread in threads:
    #             thread.start()
    #         for thread in threads:
    #             thread.join()
    #         print('num of null values df2 :', df2['family_members'].isnull().sum(), 'number of empty arrays : ', df2['family_jurisdictions'].apply(lambda x: isinstance(x, list) and len(x) == 0).sum())
    #         #add the processing of the rows of family members here

    #     #process df3
    #         CONSUMER_KEY = os.getenv("CONSUMER_KEY").strip()
    #         CONSUMER_SECRET = os.getenv("CONSUMER_SECRET").strip()
    #         CONSUMER_KEY1 = os.getenv("CONSUMER_KEY_1").strip()
    #         CONSUMER_SECRET1 = os.getenv("CONSUMER_SECRET_1").strip()
            
    #         df3 = process_dataframe_parallel(df3, 'first publication number', max_workers=4)
    #         print('num of null values df3 :', df3['family_members'].isnull().sum(), 'number of empty arrays : ', df3['family_jurisdictions'].apply(lambda x: isinstance(x, list) and len(x) == 0).sum())
        
    #     #merging the dataframes
    #         df = pd.concat([df1, df2, df3], ignore_index=True)
    #         df['family_members'] = df['family_members'].apply(lambda x: x if isinstance(x, list) else [])
    #         df['family_jurisdictions'] = df['family_jurisdictions'].apply(lambda x: x if isinstance(x, list) else [])
    #         # Ensure the columns exist in the DataFrame
    #         ensure_columns_exist(df, ['family_members', 'family_jurisdictions'])
    #         print('num of null values df :', df['family_members'].isnull().sum(), 'number of empty arrays : ', df['family_jurisdictions'].apply(lambda x: isinstance(x, list) and len(x) == 0).sum())
            
    #     # prepare updates for the database 
    #         updates = [
    #             {'id':row['id'],'first publication number' : row['first publication number'], 'jurisdictions' : json.dumps(row['family_jurisdictions']), 'members' : json.dumps(row['family_members'])
    #              } for index, row in df.iterrows()
    #         ]
    #         #update the database with the new columns 
    #         try:
    #             engine.execute(
    #                 text("UPDATE raw_patents SET family_jurisdictions = :jurisdictions, family_members = :members WHERE id = :id"),
    #                 updates
    #             )
    #         except Exception as e:
    #             return jsonify({"error": f"failed to update database: {str(e)}"}), 500


    #     # Prepare response
    #         results = df[['first publication number', 'family_jurisdictions', 'family_members']].to_dict(orient='records')
    #         empty_arrays_count = df['family_jurisdictions'].apply(lambda x: isinstance(x, list) and len(x) == 0).sum()
    #         null_count = df['family_jurisdictions'].isnull().sum()

    #         return jsonify({
    #             "results": results,
    #             "statistics": {
    #             "empty_jurisdictions_count": int(empty_arrays_count),
    #             "null_jurisdictions_count": int(null_count),
    #             "total_processed": len(df)
    #         }
    #     })

    #     except sqlalchemy.exc.SQLAlchemyError as e:
    #         return jsonify({"error": f"Database error: {str(e)}"}), 500
    #     except Exception as e:
    #         return jsonify({"error": f"Processing failed: {str(e)}"}), 500
    
    
    
    @app.route('/api/family', methods=['POST'])
    def populate_family_members():
        try:
            # Load API credentials from environment variables
            CONSUMER_KEY = os.getenv("CONSUMER_KEY")
            CONSUMER_SECRET = os.getenv("CONSUMER_SECRET")
            CONSUMER_KEY1 = os.getenv("CONSUMER_KEY1")
            CONSUMER_SECRET1 = os.getenv("CONSUMER_SECRET1")

            # Fetch data from database
            query = 'SELECT * FROM raw_patents'
            df = pd.read_sql(query, engine)

            # Data cleaning
            df.rename(columns={
                'Titre': 'Title',
                'Inventeurs': 'Inventors',
                'Demandeurs': 'Applicants',
                'Numéro de publication': 'Publication number',
                'Priorité la plus ancienne': 'Earliest priority',
                'CIB': 'IPC',
                'CPC': 'CPC',
                'Date de publication': 'Publication date',
                'Publication la plus ancienne': 'Earliest publication',
                'Numéro de famille': 'Family number'
            }, inplace=True)

            df['first publication date'] = df['Publication date'].str.extract(r'^(\S+)', expand=False)
            df['second publication date'] = df['Publication date'].str.extract(r'^\S+\s+(.*)', expand=False)
            df['second publication date'] = df['second publication date'].str.strip('\r\n')
            df[['first publication number', 'second publication number']] = df['Publication number'].str.split(' ', n=1, expand=True)
            df['second publication number'] = df['second publication number'].str.strip('\r\n')
            if 'Unnamed: 11' in df.columns:
                df.drop(columns=['Unnamed: 11', 'Publication date'], inplace=True)
            df['family number'] = pd.to_numeric(df['Family number'], errors='coerce')
            df.rename(columns={'Family number': 'family number'}, inplace=True)

            # Split the DataFrame into three parts
            n = len(df) // 3
            df1 = df.iloc[:n].copy()
            df2 = df.iloc[n:2*n].copy()
            df3 = df.iloc[2*n:].copy()

            # Process df1 using API with parallel processing
            df1 = process_dataframe_parallel(df1, 'first publication number', CONSUMER_KEY, CONSUMER_SECRET, max_workers=4)

            # Process df2 using web scraping with threading
            search_keywords = fetch_last_search_keywords_from_db()
            if 'family_members' not in df2.columns:
                df2['family_members'] = None
            indices = df2.index.tolist()
            part_size = len(indices) // 3
            parts = [indices[i:i + part_size] for i in range(0, len(indices), part_size)]
            threads = []
            for part in parts:
                thread = threading.Thread(target=process_rows, args=(df2, part, search_keywords, True))
                threads.append(thread)
                time.sleep(random.uniform(0.5, 1.5))  # Stagger thread starts
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            # Process df3 using API with parallel processing and different credentials
            df3 = process_dataframe_parallel(df3, 'first publication number', CONSUMER_KEY1, CONSUMER_SECRET1, max_workers=4)

            # Merge the DataFrames
            df = pd.concat([df1, df2, df3], ignore_index=True)
            df['family_members'] = df['family_members'].apply(lambda x: x if isinstance(x, list) else [])
            df['family_jurisdictions'] = df['family_jurisdictions'].apply(lambda x: x if isinstance(x, list) else [])

            # Prepare updates for the database
            updates = [
                {'id': row['id'], 'jurisdictions': json.dumps(row['family_jurisdictions']), 'members': json.dumps(row['family_members'])}
                for _, row in df.iterrows()
            ]

            # Update the database
            with engine.connect() as connection:
                connection.execute(
                    text("UPDATE raw_patents SET family_jurisdictions = :jurisdictions, family_members = :members WHERE id = :id"),
                updates
            )
            connection.commit()

            # Prepare response
            results = df[['first publication number', 'family_jurisdictions', 'family_members']].to_dict(orient='records')
            empty_arrays_count = df['family_jurisdictions'].apply(lambda x: isinstance(x, list) and len(x) == 0).sum()
            null_count = df['family_jurisdictions'].isnull().sum()

            return jsonify({
                "results": results,
                "statistics": {
                    "empty_jurisdictions_count": int(empty_arrays_count),
                    "null_jurisdictions_count": int(null_count),
                    "total_processed": len(df)
                }
            })
        except Exception as e:
            return jsonify({"error": f"Processing failed: {str(e)}"}), 500

    API_CREDENTIALS = load_api_credentials()
    TOKEN_URL = "https://ops.epo.org/3.2/auth/accesstoken"
    BASE_URL = "https://ops.epo.org/3.2/rest-services"
    # One token per credential
    TOKEN_CACHE = [{'token': None, 'expiry': 0} for _ in API_CREDENTIALS]
    TOKEN_LOCKS = [threading.Lock() for _ in API_CREDENTIALS]
    

    @app.route('/api/family_members/ops', methods=['POST'])
    def update_family_members_ops():
        """
        Update family_members and family_jurisdictions in raw_patents using EPO OPS API,
        with batch commits, key rotation, and optional scraping fallback.
        """
        # Query all patents (adjust query as needed, e.g. filter incomplete only)
        patents = RawPatent.query.all()
        batch_size = 100
        updates, failed = [], []
        progress = tqdm(total=len(patents), desc="Updating Patents", unit="patent")
        db_session = scoped_session(db.session)

        for idx, patent in enumerate(patents):
            pub_number = getattr(patent, 'first_publication_number', None)
            fam_number = getattr(patent, 'family_number', None)
            if not validate_patent_number(pub_number):
                failed.append({'id': patent.id, 'reason': 'invalid publication number'})
                progress.update(1)
                continue

            cred_idx = idx % len(API_CREDENTIALS)  # rotate API keys
            fam, err = fetch_family_data_api(pub_number, cred_idx)

            # Optional: Scraping fallback
            if fam is None:
                fam, err2 = fetch_family_data_scrape(pub_number, fam_number)
                if fam is None:
                    failed.append({'id': patent.id, 'reason': f"{err} | {err2}"})
                    progress.update(1)
                    continue

            # Update model (assuming JSON fields)
            patent.family_jurisdictions = fam['jurisdictions']
            patent.family_members = fam['family_members']
            updates.append({'id': patent.id, 'jurisdictions': fam['jurisdictions'], 'members': fam['family_members']})

            # Batch commit
            if len(updates) % batch_size == 0:
                try:
                    db_session.commit()
                    updates.clear()  # Only clear on successful commit
                except Exception as e:
                    db_session.rollback()
                    failed.append({'batch_failed_at': idx, 'error': str(e)})
            progress.update(1)
            time.sleep(1.2)  # Respect EPO rate limits

        # Final commit for any remaining updates
        try:
            db_session.commit()
        except Exception as e:
            db_session.rollback()
            return jsonify({'success': False, 'error': f'Database commit failed: {str(e)}'}), 500
        finally:
            db_session.remove()

        progress.close()
        return jsonify({
            'success': True,
            'updated': len(patents) - len(failed),
            'failed': failed[:100],  # truncate in case of huge jobs
            'sample_update': updates[0] if updates else None
        })
    
    
    
    # @app.route('/api/family_members/ops', methods=['POST'])
    # def update_family_members_ops():
    #     """
    #     Update family_members and family_jurisdictions in raw_patents using EPO OPS API.
    #     """
    #     import requests
    #     import time
    #     from urllib.parse import quote
    #     import os
    #     from dotenv import load_dotenv
    #     load_dotenv()
    #     CONSUMER_KEY = os.getenv("CONSUMER_KEY").strip()
    #     CONSUMER_SECRET = os.getenv("CONSUMER_SECRET").strip()
    #     TOKEN_URL = "https://ops.epo.org/3.2/auth/accesstoken"
    #     BASE_URL = "https://ops.epo.org/3.2/rest-services"
    #     TOKEN = None
    #     TOKEN_EXPIRY = 0

    #     def get_access_token():
    #         nonlocal TOKEN, TOKEN_EXPIRY
    #         if TOKEN and time.time() < TOKEN_EXPIRY:
    #             return TOKEN
    #         data = {
    #             "grant_type": "client_credentials",
    #             "client_id": CONSUMER_KEY,
    #             "client_secret": CONSUMER_SECRET
    #         }
    #         headers = {"Content-Type": "application/x-www-form-urlencoded"}
    #         response = requests.post(TOKEN_URL, data=data, headers=headers, timeout=15)
    #         response.raise_for_status()
    #         TOKEN = response.json()["access_token"]
    #         TOKEN_EXPIRY = time.time() + 3500  # ~58 min
    #         return TOKEN

    #     def validate_patent_number(patent):
    #         if not patent or len(str(patent).strip()) < 4:
    #             return False
    #         return True

    #     def extract_jurisdictions_and_members(data):
    #         try:
    #             jurisdictions = set()
    #             family_members = []
    #             world_data = data.get('ops:world-patent-data', {})
    #             patent_family = world_data.get('ops:patent-family', {})
    #             members = patent_family.get('ops:family-member', [])
    #             if isinstance(members, dict):
    #                 members = [members]
    #             for member in members:
    #                 pub_ref = member.get('publication-reference', {})
    #                 docs = pub_ref.get('document-id', [])
    #                 if isinstance(docs, dict):
    #                     docs = [docs]
    #                 for doc in docs:
    #                     if doc.get('@document-id-type') == 'docdb':
    #                         country = doc.get('country')
    #                         if isinstance(country, dict):
    #                             country = country.get('$')
    #                         doc_number = doc.get('doc-number')
    #                         if isinstance(doc_number, dict):
    #                             doc_number = doc_number.get('$')
    #                         kind = doc.get('kind')
    #                         if isinstance(kind, dict):
    #                             kind = kind.get('$')
    #                         if country and doc_number and kind:
    #                             jurisdictions.add(country)
    #                             family_members.append(f"{country}{doc_number}{kind}")
    #             return {
    #                 'jurisdictions': sorted(jurisdictions),
    #                 'family_members': sorted(set(family_members))
    #             }
    #         except Exception as e:
    #             print(f"Error parsing response: {e}")
    #             return {'jurisdictions': None, 'family_members': None}

    #     # Query all patents
    #     patents = RawPatent.query.all()
    #     updates = []
    #     failed = []
    #     request_delay = 1.2
    #     for patent in patents:
    #         pub_number = patent.first_publication_number
    #         if not validate_patent_number(pub_number):
    #             failed.append({'id': patent.id, 'reason': 'invalid publication number'})
    #             continue
    #         try:
    #             token = get_access_token()
    #             url = f"{BASE_URL}/family/publication/docdb/{quote(str(pub_number))}"
    #             headers = {
    #                 "Authorization": f"Bearer {token}",
    #                 "Accept": "application/json"
    #             }
    #             response = requests.get(url, headers=headers, timeout=15)
    #             if response.status_code == 403:
    #                 failed.append({'id': patent.id, 'reason': '403 forbidden'})
    #                 continue
    #             if response.status_code == 404:
    #                 failed.append({'id': patent.id, 'reason': '404 not found'})
    #                 continue
    #             response.raise_for_status()
    #             data = response.json()
    #             fam = extract_jurisdictions_and_members(data)
    #             # Update ORM object
    #             patent.family_jurisdictions = fam['jurisdictions']
    #             patent.family_members = fam['family_members']
    #             updates.append({'id': patent.id, 'jurisdictions': fam['jurisdictions'], 'members': fam['family_members']})
    #         except Exception as e:
    #             failed.append({'id': patent.id, 'reason': str(e)})
    #         time.sleep(request_delay)
    #     try:
    #         db.session.commit()
    #     except Exception as e:
    #         db.session.rollback()
    #         return jsonify({'success': False, 'error': f'Database commit failed: {str(e)}'}), 500
    #     return jsonify({
    #         'success': True,
    #         'updated': len(updates),
    #         'failed': failed,
    #         'sample_update': updates[0] if updates else None
    #     })









    @app.route('/api/family_member_by_country', methods=['GET'])
    def family_member_by_country():
        try:
            # Query all family_members from RawPatent
            patents = RawPatent.query.with_entities(RawPatent.family_members).all()
            # Convert to DataFrame
            import pandas as pd
            family_df = pd.DataFrame([{'family_members': p[0]} for p in patents])
            # Defensive: drop missing or empty
            family_df = family_df.dropna(subset=['family_members'])
            family_df = family_df[family_df['family_members'].apply(lambda x: isinstance(x, list) and len(x) > 0)]
            # Extract country codes for each family
            from cleaners import extract_country_codes_from_list
            family_df['country_codes'] = family_df['family_members'].apply(extract_country_codes_from_list)
            # Use analysis function
            family_counts_df = get_family_counts_by_country(family_df)
            # Prepare JSON for plotting
            result = {
                'labels': family_counts_df['country_code'].tolist(),
                'datasets': [{
                    'label': 'Number of Family Members',
                    'data': family_counts_df['member_count'].tolist(),
                    'backgroundColor': 'rgba(93, 164, 214, 0.7)'
                }]
            }
            return jsonify(result), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500



    @app.route('/api/top_10_patent_applicants', methods=['GET'])
    def top_10_patent_applicants():
        try:
            # Load applicants table into DataFrame
            df = pd.read_sql_table(
                table_name='applicants',
                con=db.engine,
                columns=['id', 'applicant_name']
            )
            # Count top 10 applicants
            top10 = df['applicant_name'].value_counts().head(10)
            result = {
                'labels': top10.index.tolist(),
                'datasets': [{
                    'label': 'Patent Count',
                    'data': top10.values.tolist(),
                    'backgroundColor': 'rgba(255, 99, 132, 0.7)'
                }]
            }
            return jsonify(result), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/applicant_collaboration_network', methods=['GET'])
    def applicant_collaboration_network():
        try:
            # Load patent data (Applicants column) from RawPatent
            patents = RawPatent.query.with_entities(RawPatent.applicants).all()
            import pandas as pd
            patent_df = pd.DataFrame([{'Applicants': p[0]} for p in patents])
            # Load applicant data from applicants table
            applicant_df = pd.read_sql_table(
                table_name='applicants',
                con=db.engine,
                columns=['applicant_name', 'applicant_type']
            )
            
            network_data = extract_applicant_collaboration_network(patent_df, applicant_df)
            return jsonify(network_data), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/top_ipc_codes', methods=['GET'])
    def top_ipc_codes():
        try:
            # Load IPC data from RawPatent
            patents = RawPatent.query.with_entities(RawPatent.ipc).all()

            ipc_list = []
            for p in patents:
                if p[0]:
                    # IPC may be a string with multiple codes separated by ';' or ','
                    codes = str(p[0]).replace(',', ';').split(';')
                    ipc_list.extend([c.strip() for c in codes if c.strip()])
            # Extract main IPC (before '/')
            main_ipc = [code.split('/')[0] for code in ipc_list if '/' in code]
            main_ipc_counts = Counter(main_ipc).most_common(10)
            main_ipc_df = pd.DataFrame(main_ipc_counts, columns=['Main IPC', 'Count'])

            # --- IPC meaning integration ---
            
            # Use the actual top IPC codes (main_ipc_df['Main IPC'])
            ipc_meanings = get_ipc_codes_meanings(main_ipc_df['Main IPC'].tolist())
            # Build a mapping for the frontend: list of dicts with code, count, title, explanation
            ipc_info = []
            for _, row in main_ipc_df.iterrows():
                code = row['Main IPC']
                count = row['Count']
                meaning = ipc_meanings.get(code, {})
                ipc_info.append({
                    'ipc_code': code,
                    'count': count,
                    'symbol': meaning.get('symbol', ''),
                    'title': meaning.get('title', ''),
                    'explanation': meaning.get('explanation', '')
                })
            result = {
                'labels': main_ipc_df['Main IPC'].tolist(),
                'datasets': [{
                    'label': 'Patent Count',
                    'data': main_ipc_df['Count'].tolist(),
                    'backgroundColor': 'rgba(153, 50, 204, 0.7)'
                }],
                'ipc_info': ipc_info
            }
            return jsonify(result), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        
       
  
    @app.route('/api/family_member_counts', methods=['GET'])
    def family_member_counts():
        """
        Returns a bar-chart payload of family-member counts per country.

        Extra logic:
        1️⃣ Read the most recent search keywords (same SQL used by
           /api/last_search_keywords).
        2️⃣ If none of those keywords appear in any patent title stored
           in raw_patents, we call the EPO-OPS updater
           (update_family_members_ops) to refresh the DB.
        3️⃣ After the optional refresh, aggregate family_jurisdictions
           exactly as before and shape the JSON for Chart.js.
        """
        try:
    # ───── 1️⃣  Read the column we care about ──────────────────────────
            query = text(
    'SELECT family_jurisdictions FROM raw_patents '
    'WHERE family_jurisdictions IS NOT NULL '
    '  AND array_length(family_jurisdictions, 1) > 0'
)
            df = pd.read_sql(query, engine)

    # ───── 2️⃣  If nothing there, run the heavy updater once ───────────
            if df.empty:
                app.logger.info("family_jurisdictions empty → running updater")
                update_family_members_ops()
                df = pd.read_sql(query, engine)      # retry

    # ───── 3️⃣  Still empty? Just return blanks ────────────────────────
            if df.empty:
                return jsonify({"labels": [], "datasets": []}), 200

    # ───── 4️⃣  Aggregate for the bar-chart ────────────────────────────
            df['country_codes'] = df['family_jurisdictions'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
                    )

            from family_analysis import get_family_counts_by_country
            counts_df = get_family_counts_by_country(df[['country_codes']])

            return jsonify({
        "labels":   counts_df['country_code'].tolist(),
        "datasets": [{
            "label": "Family Member Count",
            "data":  counts_df['member_count'].tolist()
        }]
    }), 200

        except Exception as e:
            app.logger.error(f"Error in /api/family_member_counts: {e}")
            return jsonify({"error": str(e)}), 500
        
        
        
        
    @app.route('/api/family_size_distribution', methods=['GET'])
    def family_size_distribution():
        """
        Return a histogram-ready payload for the distribution of patent-family sizes
        stored in raw_patents.

        JSON shape
        ----------
        {
            "labels":   [1, 2, 3, ...],      # family size buckets
            "datasets": [{
                "label": "Number of Families",
                "data":  [432, 97, 26, ...]  # counts per bucket (same length)
            }]
        }
        """
        try:
            # ── 1️⃣  Grab non-empty family_members arrays ───────────────────
            query = text(
                'SELECT family_members FROM raw_patents '
                'WHERE family_members IS NOT NULL '
                '  AND array_length(family_members, 1) > 0'
            )
            df = pd.read_sql(query, engine)

            # ── 2️⃣  If DB has none, populate via the heavy updater ─────────
            if df.empty:
                app.logger.info("family_members empty → running updater")
                update_family_members_ops()
                df = pd.read_sql(query, engine)

            # ── 3️⃣  Still empty? return blanks so UI can handle gracefully ─
            if df.empty:
                return jsonify({"labels": [], "datasets": []}), 200

            # ── 4️⃣  Compute family_size per row and build histogram data ───
            def _to_list(x):
                # handle TEXT-stored arrays like '[US,EP]'
                return ast.literal_eval(x) if isinstance(x, str) else x

            df['family_size'] = df['family_members'].apply(_to_list).apply(len)

            counts = (
                df['family_size']
                .value_counts()
                .sort_index()       # ascending bucket order
            )

            return jsonify({
                "labels":   counts.index.tolist(),   # [1,2,3,...]
                "datasets": [{
                    "label": "Number of Families",
                    "data":  counts.values.tolist()
                }]
            }), 200

        except Exception as e:
            app.logger.error(f"Error in /api/family_size_distribution: {e}")
            return jsonify({"error": str(e)}), 500
        
        
        
        
        
        
        
        

    @app.route('/api/international_protection_matrix', methods=['GET'])
    def international_protection_matrix():
        """
        Returns a JSON payload describing, for each origin country
        (first-publication country), how many family members were filed
        in every other jurisdiction.

        Response shape
        --------------
        {
            "origins": ["US", "CN", "JP", ...],          # y-axis order (bottom → top)
            "filings": ["US", "CN", "EP", ...],          # x-axis order (largest → smallest)
            "matrix":  [                                 # counts[y][x]
        [123, 45,  9, ...],     # US row
        [ 67, 98, 10, ...],     # CN row
        ...
      ]
        }
    """
        try:
            # ── 1️⃣  Pull the two columns we need ───────────────────────────
            query = text(
            'SELECT first_publication_country AS origin_country, '
            '       family_jurisdictions                       '
            'FROM   raw_patents                                '
            'WHERE  first_publication_country IS NOT NULL      '
            '  AND  family_jurisdictions IS NOT NULL           '
            '  AND  array_length(family_jurisdictions, 1) > 0'
            )

            df = pd.read_sql(query, engine)

        # ── 2️⃣  If empty, run the heavy updater once & retry ───────────
            if df.empty:
                app.logger.info("family_jurisdictions empty → running updater")
                update_family_members_ops()
                df = pd.read_sql(query, engine)

            if df.empty:
                return jsonify({"origins": [], "filings": [], "matrix": []}), 200

        # ── 3️⃣  Build relationships origin → filing ─────────────────────
            df['family_list'] = df['family_jurisdictions'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )

            records = []
            for idx, row in df.iterrows():
                for filing in row['family_list']:
                    records.append({
                        'origin_country': row['origin_country'],
                        'filing_country': filing
                    })

            rel_df = pd.DataFrame(records)

            grouped = (
                rel_df
                .groupby(['origin_country', 'filing_country'])
                .size()
                .reset_index(name='count')
            )

            pivot = (
                grouped
                .pivot(index='origin_country', columns='filing_country', values='count')
                .fillna(0)
            )

        # ── 4️⃣  Sort filing columns by total desc & origin rows by total asc
            pivot = pivot[pivot.sum(axis=0).sort_values(ascending=False).index]
            pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=True).index]

            origins = pivot.index.tolist()
            filings = pivot.columns.tolist()
            matrix  = pivot.astype(int).values.tolist()

            return jsonify({
                "origins": origins,
                "filings": filings,
                "matrix":  matrix
            }), 200

        except Exception as e:
            app.logger.error(f"Error in /api/international_protection_matrix: {e}")
            return jsonify({"error": str(e)}), 500
        
        
        
        
        
        #  >>> NEW ────────────────────────────────────────────────────────────────
    @app.route('/api/international_patent_flow', methods=['GET'])
    def international_patent_flow():
        """
        Deliver a receiver-vs-origin matrix for the stacked horizontal
        “International Patent Flow” bar-chart.

        JSON shape
        ----------
        {
            "receivers": ["US", "CN", "EP", ...],     # y-axis order (top→bottom)
            "origins":   ["US", "CN", "JP", ...],     # legend / stack order
            "matrix":    [                            # counts[row][col]
                [731, 212,  65, ...],   # patents each receiver got from US, CN, JP…
                [452, 530, 110, ...],   # row for CN, etc.
                ...
            ]
        }
    """
        try:
            # ── 1️⃣  Pull origin + family_jurisdictions ────────────────────
            query = text(
                'SELECT first_publication_country AS origin_country, '
                '       family_jurisdictions                       '
                'FROM   raw_patents                                '
                'WHERE  first_publication_country IS NOT NULL      '
                '  AND  family_jurisdictions IS NOT NULL           '
                '  AND  array_length(family_jurisdictions, 1) > 0'
            )
            df = pd.read_sql(query, engine)

        # ── 2️⃣  Auto-populate if column empty ──────────────────────────
            if df.empty:
                app.logger.info("family_jurisdictions empty → running updater")
                update_family_members_ops()
                df = pd.read_sql(query, engine)

            if df.empty:
                return jsonify({"receivers": [], "origins": [], "matrix": []}), 200

        # ── 3️⃣  Build long-form receiver←origin relationships ─────────
            df['family_list'] = df['family_jurisdictions'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )

            rel_records = []
            for idx, row in df.iterrows():
                for receiver in row['family_list']:
                    rel_records.append({
                        'receiver_country': receiver,
                        'origin_country':   row['origin_country']
                    })

            rel_df = pd.DataFrame(rel_records)

        # ── 4️⃣  Aggregate & pivot  receiver × origin  ─────────────────
            grouped = (
                rel_df
                .groupby(['receiver_country', 'origin_country'])
                .size()
                .reset_index(name='count')
            )

            pivot = (
                grouped
                .pivot(index='receiver_country', columns='origin_country', values='count')
                .fillna(0)
            )

        # ── 5️⃣  Row order: receivers sorted by total (desc)  ──────────
            pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

        #      Column order: origins sorted by overall total (desc) so
        #      biggest source stacks appear first (bottom of each bar)
            pivot = pivot[pivot.sum(axis=0).sort_values(ascending=False).index]

            receivers = pivot.index.tolist()
            origins   = pivot.columns.tolist()
            matrix    = pivot.astype(int).values.tolist()

            return jsonify({
                "receivers": receivers,
                "origins":   origins,
                "matrix":    matrix
            }), 200

        except Exception as e:
            app.logger.error(f"Error in /api/international_patent_flow: {e}")
            return jsonify({"error": str(e)}), 500
        
        
        
    @app.route('/api/geographical_distribution', methods=['GET'])
    def geographical_distribution():
        """
        Returns the patent count per first_publication_country.

        Example (top-3):
            {
              "labels": ["US", "JP", "CN"],
              "datasets": [{
                  "label": "Patent Count",
                  "data": [120, 83, 57],
                  "backgroundColor": "rgba(54, 162, 235, 0.7)"
              }]
            }
        """
        try:
            # 1️⃣  Pull the country column from raw_patents
            patents = RawPatent.query.with_entities(
                RawPatent.first_publication_country
            ).all()                     # -> list of tuples [(“US”,), (“JP”,)…]

            import pandas as pd
            df = pd.DataFrame(
                [c for (c,) in patents if c]  # drop NULL / empty
                , columns=['country']
            )

            if df.empty:                       # defensive – avoid div/0 on fresh DB
                return jsonify({
                    "labels": [],
                    "datasets": [{"label": "Patent Count", "data": []}]
                }), 200

            # 2️⃣  Aggregate & format for the card
            counts = df['country'].value_counts()
            result = {
                "labels": counts.index.tolist(),
                "datasets": [{
                    "label": "Patent Count",
                    "data": counts.values.tolist(),
                    "backgroundColor": "rgba(54, 162, 235, 0.7)"
                }]
            }
            return jsonify(result), 200

        except Exception as e:                 # ❸ standardised error handling
            app.logger.error(f"Error in geographical_distribution: {e}")
            return jsonify({"error": str(e)}), 500




        
        
        



        
        
        
        
        
        
        
        



    @app.route('/api/report/generate-pptx', methods=['POST'])
    def generate_pptx():
        data = request.get_json()
        images = data.get('images', [])      # list of { id, data }
        comments = data.get('comments', {})  # dict id→comment

        prs = Presentation()
        for img_obj in images:
            img_b64 = img_obj.get('data')
            chart_id = img_obj.get('id')
            # slide layout 5 is blank
            slide = prs.slides.add_slide(prs.slide_layouts[5])

            # decode and add image
            image_stream = io.BytesIO(
                base64.b64decode(img_b64.split(',', 1)[1])
            )
            slide.shapes.add_picture(
                image_stream,
                Inches(1),
                Inches(1.2),
                width=Inches(7)
            )

            # look up comment by the same id
            comment_text = comments.get(chart_id, "")
            if comment_text:
                txBox = slide.shapes.add_textbox(
                    Inches(1), Inches(5), Inches(7), Inches(1)
                )
                tf = txBox.text_frame
                tf.text = comment_text

        # write out PPTX
        output = io.BytesIO()
        prs.save(output)
        output.seek(0)
        return send_file(
            output,
            as_attachment=True,
            download_name="report.pptx",
            mimetype=(
                "application/"
                "vnd.openxmlformats-officedocument.presentationml.presentation"
            )
        )
        
        
        
        
        


    return app


if __name__ == '__main__':
    app = create_app() 
    try:
        with app.app_context():
            db.engine.connect()
            print('database conection successful!')
            db.create_all()
            print('database tables created successfully!')
    except Exception as e:
        print(f"Database connection failed: {e}")
        print(app.url_map)
        

    def find_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    port = find_free_port()

    # Update or add PROT in .env
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    try:
        with open(env_path, 'r') as f:
            lines = f.readlines()
        found = False
        for i, line in enumerate(lines):
            if re.match(r'^NEXT_PUBLIC_BACKEND_PORT\s*=.*', line):
                lines[i] = f'\n NEXT_PUBLIC_BACKEND_PORT={port}\n'
                found = True
                break
        if not found:
            lines.append(f'NEXT_PUBLIC_BACKEND_PORT={port}\n')
        with open(env_path, 'w') as f:
            f.writelines(lines)
        # Write port number to shared file for front and back access
        shared_path_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'shared_path.txt')
        with open(shared_path_file, 'w') as spf:
            spf.write(str(port))
        # Also copy port number to frontend/public/backend_port.txt for frontend access
        frontend_port_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frontend', 'public', 'backend_port.txt')
        try:
            with open(frontend_port_file, 'w') as fpf:
                fpf.write(str(port))
        except Exception as fe:
            print(f"Failed to write frontend port file: {fe}")
    except Exception as e:
        print(f"Failed to update .env: {e}")

    server = ServerThread(app, port)
    server.start()

    def handle_exit(signum, frame):
        print("\nShutting down server...")
        server.shutdown()
        exit(0)

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)
