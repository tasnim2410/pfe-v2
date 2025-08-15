

import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import string
import unicodedata
import contractions
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.stats import entropy
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import defaultdict
import seaborn as sns
from db import db ,ResearchData3
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

_stop_words = set(stopwords.words("english")) | set(stopwords.words("french"))
_stemmer    = PorterStemmer()
_lemmatizer = WordNetLemmatizer()

def research_preprocess_text(text: str, use_stemming: bool = True) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = contractions.fix(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    filtered_tokens = [token for token in tokens if token.isalpha() and len(token) > 2 and token not in _stop_words]
    if use_stemming:
        processed = [_stemmer.stem(tok) for tok in filtered_tokens]
    else:
        processed = [_lemmatizer.lemmatize(tok) for tok in filtered_tokens]
    return " ".join(processed)

def extract_keywords(texts, max_df=0.85, min_df=2, ngram_range=(1, 3), top_n=10):
    vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df, ngram_range=ngram_range)
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    avg_scores = tfidf_matrix.mean(axis=0).tolist()[0]
    keywords = sorted(zip(feature_names, avg_scores), key=lambda x: x[1], reverse=True)
    return keywords[:top_n]

def research_analyze_topic_evolution(keyword_df):
    """
    Analyzer for research (title+abstract) topic evolution WITHOUT route fallbacks.
    - Uses NumPy masks to avoid pandas BooleanArray quirks.
    - Detects change windows via Jensen–Shannon divergence.
    - For topics per window:
        * If docs >= 15 → LDA using CountVectorizer with dynamic max_df per window.
        * Else → TF-IDF aggregate (still inside analyzer, not a route fallback).
      (Both paths are part of the analyzer by design.)
    Returns: topic_evolution, windows, divergences, valid_years
    """
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.decomposition import LatentDirichletAllocation

    # --------- Tunables (lenient to ensure output) ----------
    max_topics          = 5
    top_words_per_topic = 10
    min_docs_per_year   = 1            # allow sparse years
    divergence_pct      = 80
    min_window_span     = 0            # allow single-year windows
    global_max_df       = 0.99         # looser to keep vocab across corpus
    global_min_df       = 1

    # --------- Clean inputs & build base arrays ----------
    years = pd.to_numeric(keyword_df['year'], errors='coerce').values
    keep  = ~np.isnan(years)
    keyword_df = keyword_df.loc[keep].copy()
    years = years[keep].astype(int)

    # text is already preprocessed in the route; just ensure str & non-empty
    docs = keyword_df['title'].astype(str).tolist()
    nonempty_mask = np.array([bool(d.strip()) for d in docs])
    keyword_df = keyword_df.loc[nonempty_mask].copy()
    years      = years[nonempty_mask]
    arr_docs   = np.array([d.strip().lower() for d in keyword_df['title'].astype(str).tolist()], dtype=object)

    if arr_docs.size == 0:
        return [], [], [], []

    # --------- Global vocabulary (for divergence only) ----------
    global_vec = CountVectorizer(min_df=global_min_df, max_df=global_max_df, ngram_range=(1, 2), stop_words='english')
    try:
        global_vec.fit(arr_docs)
    except ValueError:
        # If even global vocab fails, nothing to analyze
        return [], [], [], []

    # --------- Yearly distributions with NumPy masks ----------
    yearly_counts = []
    valid_years   = []
    uniq_years = np.unique(years)
    for y in uniq_years:
        ymask = (years == int(y))              # NumPy bool mask
        n = int(ymask.sum())
        if n >= min_docs_per_year:
            try:
                counts = global_vec.transform(arr_docs[ymask]).sum(axis=0).A1
                yearly_counts.append((int(y), counts))
                valid_years.append(int(y))
            except Exception:
                continue

    if not yearly_counts:
        return [], [], [], []

    # --------- Jensen–Shannon divergence on consecutive years ----------
    divergences = []
    for i in range(1, len(yearly_counts)):
        p = yearly_counts[i-1][1].astype(float) + 1e-12
        q = yearly_counts[i][1].astype(float) + 1e-12
        p /= p.sum()
        q /= q.sum()
        m = 0.5 * (p + q)
        jsd = 0.5 * (np.sum(p * np.log2(p / m)) + np.sum(q * np.log2(q / m)))
        divergences.append(float(jsd))

    # --------- Change points → windows ----------
    if len(valid_years) == 1:
        change_points = [0, 1]   # one window
    else:
        thr = np.percentile(divergences, divergence_pct) if divergences else 0.0
        change_points = [0] + [i for i, d in enumerate(divergences) if d > thr] + [len(yearly_counts)]
        change_points = sorted(set(change_points))
        if change_points[-1] != len(yearly_counts):
            change_points.append(len(yearly_counts))

    windows = []
    for i in range(1, len(change_points)):
        s = change_points[i-1]
        e = change_points[i]
        if e <= s:
            continue
        yrs = [int(yearly_counts[j][0]) for j in range(s, e)]
        if not yrs:
            continue
        w_start, w_end = int(min(yrs)), int(max(yrs))
        if (w_end - w_start) < min_window_span:
            # allow zero-span windows (single-year)
            pass
        windows.append({'start': w_start, 'end': w_end, 'years': yrs})

    if not windows:
        return [], [], divergences, valid_years

    # --------- Topics per window (dynamic pruning to avoid empty vocab) ----------
    topic_evolution = []
    for w in windows:
        wmask  = np.isin(years, np.array(w['years'], dtype=int))
        docs_w = arr_docs[wmask]
        docs_w = np.array([d for d in docs_w if d and d.strip()], dtype=object)
        n_docs = docs_w.size
        if n_docs == 0:
            continue

        # Dynamic max_df: for small windows don't prune by document frequency
        max_df_win = 1.0 if n_docs <= 20 else 0.99
        min_df_win = 1

        if n_docs >= 15:
            cv = CountVectorizer(min_df=min_df_win, max_df=max_df_win, ngram_range=(1, 2), stop_words='english')
            X  = cv.fit_transform(docs_w)
            terms = np.array(cv.get_feature_names_out())
            if terms.size == 0:
                continue  # nothing meaningful for this window

            n_topics = min(max_topics, max(1, min(6, X.shape[0] // 5)))
            lda = LatentDirichletAllocation(n_components=n_topics, random_state=0, learning_method='batch')
            lda.fit(X)
            for k, comp in enumerate(lda.components_):
                idx = np.argsort(comp)[::-1][:top_words_per_topic]
                topic_evolution.append({
                    'start': w['start'],
                    'end':   w['end'],
                    'topic_id': f"{w['start']}-{k+1}",
                    'words':   terms[idx].tolist(),
                    'weights': comp[idx].tolist()
                })
        else:
            tfv = TfidfVectorizer(min_df=min_df_win, max_df=max_df_win, ngram_range=(1, 2))
            X   = tfv.fit_transform(docs_w)
            terms   = tfv.get_feature_names_out()
            if terms.size == 0:
                continue
            weights = X.sum(axis=0).A1
            idx     = np.argsort(weights)[::-1][:top_words_per_topic]
            topic_evolution.append({
                'start': w['start'],
                'end':   w['end'],
                'topic_id': f"{w['start']}-1",
                'words':   [terms[i] for i in idx],
                'weights': [float(weights[i]) for i in idx]
            })

    return topic_evolution, windows, divergences, valid_years



def build_research_keyword_df():
    """
    Pulls research_data3 (title, abstract, year), concatenates title+abstract,
    maps to columns 'Title' and 'first filing year', and preprocesses text.
    Returns a DataFrame ready for analyze_topic_evolution(...).
    """
    # 1) Load needed columns from research_data3
    rows = db.session.query(
        ResearchData3.title,
        ResearchData3.abstract,
        ResearchData3.year
    ).all()

    # To DataFrame with clear names
    df = pd.DataFrame(rows, columns=['title', 'abstract', 'year'])

    # 2) Drop rows with no title (abstract may be empty)
    df = df.dropna(subset=['title'])
    if df.empty:
        return df

    # 3) Concatenate title + abstract (abstract can be empty)
    df['abstract'] = df['abstract'].fillna('')
    df['title'] = (df['title'].astype(str) + ' ' + df['abstract'].astype(str)).str.strip()

    # 4) Map 'year' -> 'first filing year' expected by analyze_topic_evolution
    df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')

    # 5) Preprocess Title (same function you use for patents)
    df['processed_title'] = df['title'].apply(lambda t: research_preprocess_text(t, use_stemming=True))

    # Keep only needed columns for downstream functions
    return df[['title', 'year', 'processed_title']]
