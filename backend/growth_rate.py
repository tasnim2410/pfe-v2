import pandas as pd
from datetime import date
import numpy as np

def patent_growth_summary(df):
    current_year = date.today().year
    start_year = current_year - 2
    end_year = start_year - 5
    # Group by year and count patents
    patent_counts = df.groupby('first_filing_year').size().reset_index(name='Patent Count')
    
    # Ensure the DataFrame is sorted by year in ascending order for cumulative calculations
    patent_counts = patent_counts.sort_values('first_filing_year')
    
    # Calculate cumulative patent count
    patent_counts['Cumulative Count'] = patent_counts['Patent Count'].cumsum()
    
    # Calculate growth rate
    X = patent_counts['Patent Count']
    T = patent_counts['Cumulative Count']
    patent_counts['GR'] = ((X - X.shift(1)) / ((T + T.shift(1)) / 2)).fillna(0)
    patent_counts['GR'] = patent_counts['GR'].fillna(0)
    
    # Sort by year descending and select top 10
    patent_counts_sorted = patent_counts.sort_values('first_filing_year', ascending=False).head(10)

    df_2018_2023 = patent_counts[(patent_counts['first_filing_year'] >= end_year) & (patent_counts['first_filing_year'] <= start_year)]

    # Sum the annual growth rates (GR) for the period
    GR = df_2018_2023['GR'].sum()*100
    if GR >=50:
      print ("the technology is Booming")
    elif 20 <= GR < 50:
      print ("the technology is Trending")
    elif 10 <= GR < 20:
      print ("the technology is Quite_Trending")
    elif 0 <= GR < 10:
      print ("the technology is Steady")
    elif GR < 0:
      print ("the technology is Declining")
    

    # Return selected columns
    return GR , start_year, end_year
  
  
def patent_current_growth_rate(df: pd.DataFrame):
    """
    Compute summed growth rate over a 5-year window centered on current year:
      window = [current_year - 2, ..., current_year + 2]

    Returns:
        GR_percent, window_start_year, window_end_year
    """
    current_year = date.today().year
    start_year = current_year - 2
    end_year = current_year + 2

    # Count patents per year
    patent_counts = (
        df.groupby('first_filing_year')
          .size()
          .reset_index(name='Patent Count')
          .sort_values('first_filing_year', ascending=True)
          .reset_index(drop=True)
    )

    # Cumulative count
    patent_counts['Cumulative Count'] = patent_counts['Patent Count'].cumsum()

    # Growth rate per year: (ΔX) / avg cumulative
    X = patent_counts['Patent Count'].astype(float)
    T = patent_counts['Cumulative Count'].astype(float)
    denom = (T + T.shift(1)) / 2.0
    # avoid division by zero / NaN
    gr = (X - X.shift(1)) / denom
    gr = gr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    patent_counts['GR'] = gr

    # 5-year window centered on current year
    mask = (patent_counts['first_filing_year'] >= start_year) & \
           (patent_counts['first_filing_year'] <= end_year)
    GR = float(patent_counts.loc[mask, 'GR'].sum() * 100.0)  # percentage

    # quick label
    if GR >= 50:
        print("the technology is Booming")
    elif 20 <= GR < 50:
        print("the technology is Trending")
    elif 10 <= GR < 20:
        print("the technology is Quite_Trending")
    elif 0 <= GR < 10:
        print("the technology is Steady")
    else:
        print("the technology is Declining")

    return GR, start_year, end_year

def compute_patent_growth_from_counts(pats_counts_df: pd.DataFrame):
    """
    pats_counts_df: columns ['year', 'patent_count'] (ascending or not; we'll sort)
    Implements the same GR formula as growth_rate.py over the past window
      [current_year-7 .. current_year-2] (inclusive) and the current window
      [current_year-2 .. current_year+2] (inclusive).
    Returns: (past_GR_percent: float, past_start_year: int, past_end_year: int, past_label: str,
              curr_GR_percent: float, curr_start_year: int, curr_end_year: int, curr_label: str)
    """
    if pats_counts_df is None or len(pats_counts_df) == 0:
        return float("nan"), None, None, "unknown", float("nan"), None, None, "unknown"

    df = pats_counts_df.rename(columns={"year": "first_filing_year", "patent_count": "Patent Count"}) \
                       .loc[:, ["first_filing_year", "Patent Count"]] \
                       .sort_values("first_filing_year")

    # cumulative + per-year GR = (ΔX) / avg cumulative
    df["Cumulative Count"] = df["Patent Count"].cumsum()
    X = df["Patent Count"].astype(float)
    T = df["Cumulative Count"].astype(float)
    denom = (T + T.shift(1)) / 2.0
    gr = (X - X.shift(1)) / denom
    gr = gr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["GR"] = gr

    current_year = date.today().year

    # Past window: [cur-7 to cur-2]
    past_end_y = current_year - 2
    past_start_y = past_end_y - 5
    past_y0 = max(past_start_y, int(df["first_filing_year"].min()))
    past_y1 = min(past_end_y, int(df["first_filing_year"].max()))
    past_mask = (df["first_filing_year"] >= past_y0) & (df["first_filing_year"] <= past_y1)
    past_GR = float(df.loc[past_mask, "GR"].sum() * 100.0)

    if past_GR >= 50:
        past_label = "Booming"
    elif 20 <= past_GR < 50:
        past_label = "Trending"
    elif 10 <= past_GR < 20:
        past_label = "Quite_Trending"
    elif 0 <= past_GR < 10:
        past_label = "Steady"
    else:
        past_label = "Declining"

    # Current window: [cur-2 to cur+2] (actual + forecast)
    curr_start_y = current_year - 2
    curr_end_y = current_year + 2
    curr_y0 = max(curr_start_y, int(df["first_filing_year"].min()))
    curr_y1 = min(curr_end_y, int(df["first_filing_year"].max()))
    curr_mask = (df["first_filing_year"] >= curr_y0) & (df["first_filing_year"] <= curr_y1)
    curr_GR = float(df.loc[curr_mask, "GR"].sum() * 100.0)

    if curr_GR >= 50:
        curr_label = "Booming"
    elif 20 <= curr_GR < 50:
        curr_label = "Trending"
    elif 10 <= curr_GR < 20:
        curr_label = "Quite_Trending"
    elif 0 <= curr_GR < 10:
        curr_label = "Steady"
    else:
        curr_label = "Declining"

    return past_GR, past_y0, past_y1, past_label, curr_GR, curr_y0, curr_y1, curr_label