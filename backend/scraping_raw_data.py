# technology-trend-analysis/backend/scraping_raw_data.py
import os
import glob
import time
import random
import pandas as pd
from sqlalchemy import text
from sqlalchemy import create_engine
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, ElementClickInterceptedException, SessionNotCreatedException
from cleaners import clean_espacenet_data 
import re

# Load environment variables
load_dotenv()

class DatabaseManager:
    """Handles database connection and operations"""
    def __init__(self):
        self.engine = create_engine(os.getenv('DATABASE_URL'))
        
    def store_patents(self, df):
        """Store patent data in PostgreSQL"""
        try:
            with self.engine.connect() as connection : 
                connection.execute(text("tRUNCATE TABLE raw_patents;"))
                connection.commit()
            df.to_sql('raw_patents', self.engine, 
                      if_exists='append', 
                      index=False,
                      method='multi')
            print(f"Successfully stored {len(df)} patents in database")
            return True
        except Exception as e:
            print(f"Error storing data: {e}")
            return False



# class EspacenetScraper:
#     def __init__(self, search_pairs, operators=None, headless=True, options_args=None):
#         """Initialize the scraper with configurable options and search keywords."""
#         self.search_keywords = search_pairs
#         self.operators = operators if operators is not None else []
#         options = uc.ChromeOptions()
#         if headless:
#             options.add_argument('--headless')  # Run in headless mode
#         if options_args:
#           for arg in options_args:
#             options.add_argument(arg)

#         options.add_argument('--disable-blink-features=AutomationControlled')
#         self.driver = uc.Chrome(options=options)
#         self.driver.set_page_load_timeout(30)
#         self.driver.set_window_size(1600, 1300)

#     def construct_search_url(self):
#         base_url = 'https://worldwide.espacenet.com/patent/search?q='
#         field_mapping = {
#             'title': 'ti',
#             'abstract': 'ab',
#             'claims': 'cl',
#             'title,abstract or claims': 'ctxt',
#             'all text fields': 'ftxt',
#             'title or abstract': 'ta',
#             'description': 'desc',
#             'all text fields or names': 'nftxt',
#             'title , abstract or names': 'ntxt'
#         }
#         query_parts = []
#         for field, keyword in self.search_keywords:
#             field_param = field_mapping.get(field, 'ctxt')
#             query_parts.append(f'{field_param} = "{keyword}"')
#         query = query_parts[0]
#         for op, part in zip(self.operators, query_parts[1:]):
#             query += f' {op} {part}'
#         query += '&queryLang=en%3Ade%3Afr'
#         return base_url + query

#     def add_random_delay(self, min_seconds=1, max_seconds=3):
#         """Add a random delay to mimic human behavior."""
#         time.sleep(random.uniform(min_seconds, max_seconds))

#     def get_page_html(self, retries=3):
#         """
#         Navigate to the constructed URL and return the page HTML.
#         Retry the operation if a timeout occurs.

#         Args:
#             retries (int): Number of retry attempts.

#         Returns:
#             str: The page HTML, or None if all retries fail.
#         """
#         url = self.construct_search_url()
#         for attempt in range(retries):
#             try:
#                 print(f"Navigating to: {url} (Attempt {attempt + 1})")
#                 self.driver.get(url)
#                 WebDriverWait(self.driver, 30).until(
#                     EC.presence_of_element_located((By.TAG_NAME, "body"))
#                 )

#                 # Add a random delay to mimic human behavior
#                 self.add_random_delay(3, 5)

#                 # Return the page HTML
#                 return self.driver.page_source

#             except TimeoutException:
#                 print(f"Timed out waiting for the page to load. Retrying ({attempt + 1}/{retries})...")
#                 if attempt == retries - 1:
#                     print("Max retries reached. Unable to load the page.")
#                     return None
#             except Exception as e:
#                 print(f"An error occurred: {e}")
#                 return None

#     def download_csv(self, retries=3, max_results=500):
#         """
#         Complete the sequence of clicking:
#         1. More Options button
#         2. Download dropdown
#         3. List (CSV) option
#         4. Handle download dialog by:
#            - Setting the "To" value to max_results (e.g., 500)
#            - Clicking the Download button
        
#         Args:
#             retries (int): Number of retry attempts for the entire sequence.
#             max_results (int): Maximum number of results to download (1-500).

#         Returns:
#             bool: True if the download sequence was successful, False otherwise.
#         """
#         for attempt in range(retries):
#             try:
#                 print(f"Attempting download sequence (Attempt {attempt + 1})...")
                
#                 # Step 1: Click "More Options" button
#                 print("Looking for More Options button...")
#                 more_options_selector = "#more-options-selector--publication-list-header"
#                 more_options_button = WebDriverWait(self.driver, 30).until(
#                     EC.element_to_be_clickable((By.CSS_SELECTOR, more_options_selector))
#                 )
                
#                 # Try to click, but handle intercepted clicks
#                 try:
#                     print("More Options button found. Clicking...")
#                     more_options_button.click()
#                 except ElementClickInterceptedException:
#                     print("Click intercepted, trying JavaScript click...")
#                     self.driver.execute_script('document.querySelector("#more-options-selector--publication-list-header").click()', more_options_button)
                    
#                 self.add_random_delay(2, 3)
#                 print('More Options clicked successfully')
                
#                 # Step 2: Click "Download" section in the dropdown
#                 print("Looking for Download section...")
#                 # Use a more general selector to find the Download section
#                 # This uses contains() to match the text rather than a fixed CSS path
#                 download_section_xpath = "/html/body/div[2]/div[3]/ul/section[1]"
#                 download_section = WebDriverWait(self.driver, 10).until(
#                     EC.element_to_be_clickable((By.XPATH, download_section_xpath))
#                 )
                
#                 try:
#                     print("Download section found. Clicking...")
#                     download_section.click()
#                 except ElementClickInterceptedException:
#                     print("Click intercepted, trying JavaScript click...")
#                     self.driver.execute_script('document.querySelector("#simple-dropdown > div.prod-jss1034.prod-jss966.prod-jss969.prod-jss1045 > ul > section:nth-child(1)").click()', download_section)
                    
#                 self.add_random_delay(1, 2)
#                 print('Download section clicked successfully')
                
#                 # Step 3: Click "List (CSV)" option
#                 print("Looking for List (CSV) option...")
#                 # Use contains() with the XPATH to find the CSV option based on text
#                 csv_option_xpath = "/html/body/div[2]/div[3]/ul/li[2]"
#                 csv_option = WebDriverWait(self.driver, 10).until(
#                     EC.element_to_be_clickable((By.XPATH, csv_option_xpath))
#                 )
                
#                 try:
#                     print("List (CSV) option found. Clicking...")
#                     csv_option.click()
#                 except ElementClickInterceptedException:
#                     print("Click intercepted, trying JavaScript click...")
#                     self.driver.execute_script('document.querySelector("#simple-dropdown > div.prod-jss1034.prod-jss966.prod-jss969.prod-jss1045 > ul > li:nth-child(3)").click()', csv_option)
                    
#                 self.add_random_delay(2, 3)
#                 print('List (CSV) option clicked successfully')
                
#                 # Step 4: Handle the download dialog
#                 print("Waiting for download dialog to appear...")
                
#                 # Wait for the dialog to appear
#                 download_dialog_xpath = "/html/body/div[2]/div[3]/div/div"
#                 WebDriverWait(self.driver, 10).until(
#                     EC.presence_of_element_located((By.XPATH, download_dialog_xpath))
#                 )
#                 print("Download dialog appeared")
                
#                 # Find the "To" input field
#                 to_input_xpath = "/html/body/div[2]/div[3]/div/div/div/div[1]/input[2]"
#                 to_input = WebDriverWait(self.driver, 10).until(
#                     EC.presence_of_element_located((By.XPATH, to_input_xpath))
#                 )
                
#                 # Clear the input and set it to max_results
#                 print(f"Setting maximum results to {max_results}...")
#                 to_input.clear()
#                 to_input.send_keys(str(max_results))
#                 self.add_random_delay(1, 2)
                
#                 # Click the Download button in the dialog
#                 download_button_xpath = "/html/body/div[2]/div[3]/div/div/div/button"
#                 download_button = WebDriverWait(self.driver, 10).until(
#                     EC.element_to_be_clickable((By.XPATH, download_button_xpath))
#                 )
                
#                 try:
#                     print("Download button found. Clicking...")
#                     download_button.click()
#                 except ElementClickInterceptedException:
#                     print("Click intercepted, trying JavaScript click...")
#                     self.driver.execute_script('document.querySelector("body > div.prod-jss12 > div.prod-jss15.prod-jss13 > div > div > div > button").click()', download_button)
                
#                 print("Download button clicked")
                
#                 # Wait for a moment to ensure the download starts
#                 self.add_random_delay(3, 5)
                
#                 # Check if there are any error messages
#                 try:
#                     error_message = self.driver.find_element(By.XPATH, "//div[contains(@class, 'download-modal__validation')]//span")
#                     if error_message.is_displayed() and error_message.text.strip():
#                         print(f"Error in download dialog: {error_message.text}")
#                         return False
#                 except:
#                     # No error message found, continue
#                     pass
                
#                 print("Download sequence completed successfully")
#                 return True
                
#             except TimeoutException as e:
#                 print(f"Timeout during download sequence: {e}")
#                 if attempt == retries - 1:
#                     print("Max retries reached. Download sequence failed.")
#                     return False
#             except Exception as e:
#                 print(f"Error during download sequence: {e}")
#                 if attempt == retries - 1:
#                     print("Max retries reached. Download sequence failed.")
#                     return False
                
#             # If we reach here, there was an error and we need to try again
#             # Refresh the page before the next attempt
#             try:
#                 self.driver.refresh()
#                 WebDriverWait(self.driver, 30).until(
#                     EC.presence_of_element_located((By.TAG_NAME, "body"))
#                 )
#                 self.add_random_delay(3, 5)
#             except Exception as e:
#                 print(f"Error refreshing page: {e}")

#         return False

#     def close(self):
#         """Close the browser when done."""
#         if self.driver:
#             self.driver.quit()

from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, ElementClickInterceptedException
from selenium.webdriver.common.keys import Keys
def extract_field_keyword_pairs(node):
                pairs = []
                if isinstance(node, dict):
                    if node.get("type") == "keyword":
                        field = node.get("field")
                        keyword = node.get("word")
                        if field and keyword:
                            pairs.append((field.strip().lower(), keyword.strip()))
                    elif node.get("type") == "group":
                        for child in node.get("keywords", []):
                            pairs.extend(extract_field_keyword_pairs(child))
                elif isinstance(node, list):
                    for item in node:
                        pairs.extend(extract_field_keyword_pairs(item))
                return pairs
class EspacenetScraper:
    def __init__(self, search_keywords, headless=True):
        """Initialize the scraper with configurable options and search keywords."""
        self.search_keywords = search_keywords
        options = uc.ChromeOptions()
        
        if headless:
            options.add_argument('--headless=new')             # modern headless mode
            options.add_argument('--window-size=1920,1080')    # wide viewport so UI loads fully
            options.add_argument('--disable-blink-features=AutomationControlled')
            options.add_argument('--no-sandbox')               # needed in some CI / container envs
            options.add_argument('--disable-dev-shm-usage')    # avoid shared-memory issues
            options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
            options.add_argument('--disable-extensions')       # drop automation extensions
            options.add_argument('--disable-gpu')              # optional – saves GPU resources
        
        def _fresh_options():
            o = uc.ChromeOptions()
            if headless:
                o.add_argument('--headless=new')
                o.add_argument('--window-size=1920,1080')
                o.add_argument('--disable-blink-features=AutomationControlled')
                o.add_argument('--no-sandbox')
                o.add_argument('--disable-dev-shm-usage')
                o.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
                o.add_argument('--disable-extensions')
                o.add_argument('--disable-gpu')
            return o

        try:
            self.driver = uc.Chrome(options=options)
        except SessionNotCreatedException as e:
            # Typical mismatch: "This version of ChromeDriver only supports Chrome version 140"
            # and "Current browser version is 139.x"
            msg = str(e)
            cur = re.search(r"Current browser version is (\d+)", msg)
            sup = re.search(r"only supports Chrome version (\d+)", msg)
            version_hint = None
            if cur:
                version_hint = int(cur.group(1))
            elif sup:
                # If only driver-supported version is shown, try that
                version_hint = int(sup.group(1))
            if version_hint:
                print(f"[undetected_chromedriver] Retrying with version_main={version_hint} due to mismatch")
                self.driver = uc.Chrome(options=_fresh_options(), version_main=version_hint)
            else:
                raise
        except Exception as e:
            # Some environments raise a generic WebDriver error; attempt the same fallback
            msg = str(e)
            cur = re.search(r"Current browser version is (\d+)", msg)
            sup = re.search(r"only supports Chrome version (\d+)", msg)
            version_hint = None
            if cur:
                version_hint = int(cur.group(1))
            elif sup:
                version_hint = int(sup.group(1))
            if version_hint:
                print(f"[undetected_chromedriver] Generic error, retrying with version_main={version_hint}")
                self.driver = uc.Chrome(options=_fresh_options(), version_main=version_hint)
            else:
                raise
        self.driver.set_page_load_timeout(30)
        self.driver.set_window_size(1600, 1300)
        # Will be populated from the results header like "844 244 résultats trouvés"
        self.total_results = None
        
    field_mapping = {
        'title': 'ti',
        'abstract': 'ab',
        'claims': 'cl',
        'title,abstract or claims': 'ctxt',
        'all text fields': 'ftxt',
        'title or abstract': 'ta',
        'description': 'desc',
        'all text fields or names': 'nftxt',
        'title , abstract or names': 'ntxt'
    }

    def build_query(self ,node):
        """
        Recursively build an Espacenet query string from a keyword or group node,
        using the node’s rule_op for single keywords.
        """
        # Determine node type (allow auto‑detection for groups if you like)
        node_type = node.get("type")
        if not node_type:
            if "operator" in node and "keywords" in node:
                node_type = "group"
            else:
                raise ValueError("Node must have 'type' key")

        if node_type == "keyword":
            # Validate
            for k in ("word","field","rule_op"):
                if k not in node:
                    raise ValueError(f"Keyword node must have '{k}' key")
            word, field, rule_op = node["word"], node["field"], node["rule_op"].lower()
            if field not in self.field_mapping:
                raise ValueError(f"Invalid field: {field}")
            if rule_op not in ("all","any"):
                raise ValueError("rule_op must be 'all' or 'any'")
            code = self.field_mapping[field]
            # Now include rule_op directly instead of '='
            return f'{code} {rule_op} "{word}"'

        elif node_type == "group":
            # Validate
            operator = node.get("operator", "").upper()
            if operator not in ("AND","OR"):
                raise ValueError("Operator must be 'AND' or 'OR'")
            keywords = node.get("keywords")
            if not isinstance(keywords, list) or not keywords:
                raise ValueError("Keywords must be a non-empty list")
            # Recurse
            parts = [self.build_query(kw) for kw in keywords]
            return "(" + f" {operator} ".join(parts) + ")"

        else:
            raise ValueError("Invalid node type: must be 'keyword' or 'group'")

    def create_espacenet_query(self,query_input):
        """
        Generate an Espacenet query string from the input dictionary.
        """
        if "query" not in query_input:
            raise ValueError("Input must have 'query' key")
        q = query_input["query"].get("group1")
        if not q:
            raise ValueError("Query must contain a group labeled 'group1'")
        return self.build_query(q)
        
    def enter_query_in_search_bar(self, query ):
        """
        Navigates to the Espacenet search page, finds the search bar, enters the given query, and submits it.
        """
        search_page_url = 'https://worldwide.espacenet.com/patent/search'
        self.driver.get(search_page_url)
        try:
            WebDriverWait(self.driver, 25).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, '#application-content > div > nav > div > div.nav__bar__section--center--6Wk6v59n > div > form'))
            )
            search_input = self.driver.find_element(By.CSS_SELECTOR, '#application-content > div > nav > div > div.nav__bar__section--center--6Wk6v59n > div > form > input')
            search_input.clear()
            search_input.send_keys(query)
            search_input.send_keys(Keys.ENTER)  # Submit the search
            print(f"Query '{query}' entered and search submitted.")
        except Exception as e:
            print(f"Failed to enter query and submit search: {e}")

    def add_random_delay(self, min_seconds=1, max_seconds=3):
        """Add a random delay to mimic human behavior."""
        time.sleep(random.uniform(min_seconds, max_seconds))

    def get_page_html(self, query_input, retries=3):
        """
        Generate the Espacenet query string from query_input, perform the search, and return the page HTML.
        Retry the operation if a timeout occurs.

        Args:
            query_input (dict): The input dictionary for creating the query.
            retries (int): Number of retry attempts.

        Returns:
            str: The page HTML, or None if all retries fail.
        """
        try:
            query_string = self.create_espacenet_query(query_input)
        except ValueError as e:
            print(f"Error creating query: {e}")
            return None

        for attempt in range(retries):
            try:
                print(f"Performing search with query: {query_string} (Attempt {attempt + 1})")
                self.enter_query_in_search_bar(query_string)
                # Wait for search results to load (adjust locator as needed)
                WebDriverWait(self.driver, 30).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "#result-list > div.publications-list-header__footer--vGECUEz4"))  # Placeholder; inspect actual page
                )
                # Add a random delay to mimic human behavior
                self.add_random_delay(3, 5)
                # Return the page HTML
                return self.driver.page_source
            except TimeoutException:
                print(f"Timed out waiting for search results to load. Retrying ({attempt + 1}/{retries})...")
                if attempt == retries - 1:
                    print("Max retries reached. Unable to load the search results.")
                    return None
            except Exception as e:
                print(f"An error occurred: {e}")
                return None

    def download_csv(self, retries=3, max_results=500):

        for attempt in range(retries):
            try:
                print(f"Attempting download sequence (Attempt {attempt + 1})...")
                
                # Step 1: Click "More Options" button
                print("Looking for More Options button...")
                more_options_selector = "#more-options-selector--publication-list-header"
                more_options_path = "/html/body/div/div/div[3]/div/div[4]/div[2]/div[1]/div[2]/div[4]/button"
                try: 
                    more_options_button = WebDriverWait(self.driver, 35).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, more_options_selector))
                    )
                    print(f"More Options button found by css selector: {more_options_button}")
                except TimeoutException:
                    print("More Options button not found by CSS selector, trying XPath...")
                    more_options_button = WebDriverWait(self.driver, 35).until(
                        EC.element_to_be_clickable((By.XPATH, more_options_path))
                    )
                    print(f"More Options button found by XPath: {more_options_button}")
                    
                
                # Right before clicking More Options, try to read the total results from header
                try:
                    # First try CSS selector for the header h1
                    header_h1 = None
                    try:
                        header_h1 = WebDriverWait(self.driver, 15).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, "#result-list > h1"))
                        )
                    except TimeoutException:
                        # Fallback to absolute XPath
                        header_h1 = WebDriverWait(self.driver, 10).until(
                            EC.presence_of_element_located((By.XPATH, "/html/body/div/div/div[3]/div/div[4]/div/h1"))
                        )

                    # Find the nested div that contains the text like "844 244 résultats trouvés"
                    text_div = None
                    try:
                        text_div = header_h1.find_element(By.CSS_SELECTOR, "div > span > div > div > div")
                    except Exception:
                        try:
                            text_div = self.driver.find_element(By.XPATH, "/html/body/div/div/div[3]/div/div[4]/div/h1/div/span/div/div/div")
                        except Exception:
                            text_div = None

                    header_text = None
                    if text_div and text_div.text:
                        header_text = text_div.text.strip()
                    else:
                        # Fallback: use entire h1 text
                        header_text = header_h1.text.strip() if header_h1 else None

                    if header_text:
                        # Extract digits only to handle thousands separators and any locale-specific text
                        digits_only = re.sub(r"\D", "", header_text)
                        if digits_only:
                            self.total_results = int(digits_only)
                            print(f"Detected total results from header: {self.total_results}")
                        else:
                            print(f"Header present but no digits parsed from: '{header_text}'")
                    else:
                        print("Results header not found or empty before clicking More Options.")
                except Exception as e:
                    print(f"Failed to parse total results from header: {e}")
                
                
                # Try to click, but handle intercepted clicks
                try:
                    print("More Options button found. Clicking...")
                    more_options_button.click()
                except ElementClickInterceptedException:
                    print("Click intercepted, trying JavaScript click...")
                    self.driver.execute_script('document.querySelector("#more-options-selector--publication-list-header").click()', more_options_button)
                    
                self.add_random_delay(2, 3)
                print('More Options clicked successfully')
                
                # Step 2: Click "Download" section in the dropdown
                print("Looking for Download section...")
                # Use a more general selector to find the Download section
                # This uses contains() to match the text rather than a fixed CSS path
                download_section_xpath = "/html/body/div[2]/div[3]/ul/section[1]"
                download_section = WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, download_section_xpath))
                )
                
                try:
                    print("Download section found. Clicking...")
                    download_section.click()
                except ElementClickInterceptedException:
                    print("Click intercepted, trying JavaScript click...")
                    self.driver.execute_script('document.querySelector("#simple-dropdown > div.prod-jss1034.prod-jss966.prod-jss969.prod-jss1045 > ul > section:nth-child(1)").click()', download_section)
                    
                self.add_random_delay(1, 2)
                print('Download section clicked successfully')
                
                # Step 3: Click "List (CSV)" option
                print("Looking for List (CSV) option...")
                # Use contains() with the XPATH to find the CSV option based on text
                csv_option_xpath = "/html/body/div[2]/div[3]/ul/li[2]"
                csv_option = WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, csv_option_xpath))
                )
                
                try:
                    print("List (CSV) option found. Clicking...")
                    csv_option.click()
                except ElementClickInterceptedException:
                    print("Click intercepted, trying JavaScript click...")
                    self.driver.execute_script('document.querySelector("#simple-dropdown > div.prod-jss1034.prod-jss966.prod-jss969.prod-jss1045 > ul > li:nth-child(3)").click()', csv_option)
                    
                self.add_random_delay(2, 3)
                print('List (CSV) option clicked successfully')
                
                # Step 4: Handle the download dialog
                print("Waiting for download dialog to appear...")
                
                # Wait for the dialog to appear
                download_dialog_xpath = "/html/body/div[2]/div[3]/div/div"
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, download_dialog_xpath))
                )
                print("Download dialog appeared")
                
                # Find the "To" input field
                to_input_xpath = "/html/body/div[2]/div[3]/div/div/div/div[1]/input[2]"
                to_input = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, to_input_xpath))
                )
                
                # Clear the input and set it to max_results
                print(f"Setting maximum results to {max_results}...")
                to_input.clear()
                to_input.send_keys(str(max_results))
                self.add_random_delay(1, 2)
                
                # Click the Download button in the dialog
                download_button_xpath = "/html/body/div[2]/div[3]/div/div/div/button"
                download_button = WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, download_button_xpath))
                )
                
                try:
                    print("Download button found. Clicking...")
                    download_button.click()
                except ElementClickInterceptedException:
                    print("Click intercepted, trying JavaScript click...")
                    self.driver.execute_script('document.querySelector("body > div.prod-jss12 > div.prod-jss15.prod-jss13 > div > div > div > button").click()', download_button)
                
                print("Download button clicked")
                
                # Wait for a moment to ensure the download starts
                self.add_random_delay(3, 5)
                
                # Check if there are any error messages
                try:
                    error_message = self.driver.find_element(By.XPATH, "//div[contains(@class, 'download-modal__validation')]//span")
                    if error_message.is_displayed() and error_message.text.strip():
                        print(f"Error in download dialog: {error_message.text}")
                        return False
                except:
                    # No error message found, continue
                    pass
                
                print("Download sequence completed successfully")
                return True
                
            except TimeoutException as e:
                print(f"Timeout during download sequence: {e}")
                if attempt == retries - 1:
                    print("Max retries reached. Download sequence failed.")
                    return False
            except Exception as e:
                print(f"Error during download sequence: {e}")
                if attempt == retries - 1:
                    print("Max retries reached. Download sequence failed.")
                    return False
                
            # If we reach here, there was an error and we need to try again
            # Refresh the page before the next attempt
            try:
                self.driver.refresh()
                WebDriverWait(self.driver, 30).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                self.add_random_delay(3, 5)
            except Exception as e:
                print(f"Error refreshing page: {e}")

        return False

    def close(self):
        """Close the browser when done."""
        if self.driver:
            self.driver.quit()





def process_downloaded_data(downloads_path):
    """Process latest downloaded CSV file"""
    try:
        # Find latest CSV
        list_of_files = glob.glob(os.path.join(downloads_path, "*.csv"))
        if not list_of_files:
            return None
            
        latest_file = max(list_of_files, key=os.path.getmtime)
        print(f"Processing file: {latest_file}")

        # Read and clean data
        df = pd.read_csv(latest_file, delimiter=';', skiprows=7)
        df = clean_espacenet_data(df)
        # if 'family_jurisdictions' not in df.columns:
        #     df['family_jurisdictions'] = None
        # if 'family_members' not in df.columns:
        #     df['family_members'] = None

    

          
        return df
    
    except Exception as e:
        print(f"Error processing data: {e}")
        return None

def main(search_keywords, max_results=500):
    # Initialize components
    scraper = EspacenetScraper(search_keywords, headless=False)
    db_manager = DatabaseManager()
    
    try:
        # Execute scraping
        if scraper.get_page_html(retries=3):
            if scraper.download_csv(max_results=max_results):
                # Process downloaded data
                time.sleep(10)  # Wait for download completion
                df = process_downloaded_data(os.path.expanduser("~/Downloads"))
                
                if df is not None:
                    # Store in database
                    if db_manager.store_patents(df):
                        print("Data pipeline completed successfully")
                        return True
        return False
    finally:
      if scraper: 
        scraper.close()
        

# if __name__ == '__main__':
#     # Configuration
#     SEARCH_KEYWORDS = {
#         "quantum": "title,abstract or claims",
#         "tunneling": "title,abstract or claims"
#     }
    
#     # Run pipeline
#     success = main(SEARCH_KEYWORDS, max_results=500)
    
#     if success:
#         print("✅ Data scraped and stored successfully")
#     else:
#         print("❌ Pipeline failed")
if __name__ == '__main__':
    # Configuration
    query_input = {
    "query": {
        "group1": {
            "type": "group",
            "operator": "AND",
            "keywords": [
                {
                    "type": "keyword",
                    "word": "radar",
                    "rule_op": "all",
                    "field": "title"
                },
                {
                    "type": "group",
                    "operator": "OR",
                    "keywords": [
                        {
                            "type": "keyword",
                            "word": "operating",
                            "rule_op": "all",
                            "field": "title"
                        },
                        {
                            "type": "keyword",
                            "word": "system",
                            "rule_op": "any",
                            "field": "title"
                        }
                    ]
                }
            ]
        }
    }
}

# Initialize the scraper with headless mode enabled
# 'search_keywords' is not used here since we provide a query_input, so set it to None
    scraper = EspacenetScraper(search_keywords=None, headless=False)

# Perform the search and retrieve the HTML of the search results page
    html = scraper.get_page_html(query_input)
    if html:
        print("Search performed successfully. HTML retrieved.")
    # Optionally, you could parse 'html' with BeautifulSoup here
    else:
        print("Failed to perform search.")

# Download the first 500 search results as a CSV file
    success = scraper.download_csv(max_results=500)
    if success:
        print("CSV downloaded successfully.")
    else:
        print("Failed to download CSV.")

# Close the browser instance
    scraper.close()