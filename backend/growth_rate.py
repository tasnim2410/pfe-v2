import pandas as pd
from datetime import date
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