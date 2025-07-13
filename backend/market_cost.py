import pandas as pd
from datetime import datetime
def update_cost_df(cost: pd.DataFrame) -> pd.DataFrame:
  cost_updated = cost.copy()
  reference_countries = ['FR', 'EP', 'US', 'CA', 'CN', 'IN', 'KR']
  economic_groups = {
    # China-like economies 
    'BR': 'CN', 'RU': 'CN', 'VN': 'CN', 'ZA': 'CN', 'MX': 'CN', 
    'ID': 'CN', 'TR': 'CN', 'TH': 'CN', 'SA': 'CN', 'AR': 'CN',
    'CL': 'CN', 'CO': 'CN', 'PE': 'CN', 'PH': 'CN', 'EG': 'CN',
    'PK': 'CN', 'BD': 'CN', 'MA': 'CN', 'VE': 'CN',
    
    # France-like economies 
    'DE': 'FR', 'GB': 'FR', 'IT': 'FR', 'ES': 'FR', 'NL': 'FR',
    'SE': 'FR', 'CH': 'FR', 'BE': 'FR', 'AT': 'FR', 'DK': 'FR',
    'FI': 'FR', 'NO': 'FR', 'PT': 'FR', 'IE': 'FR', 'GR': 'FR',
    'CZ': 'FR', 'HU': 'FR', 'SK': 'FR', 'PL': 'FR',
    
    # Canada-like economies 
    'AU': 'CA', 'JP': 'KR', 'SG': 'KR', 'TW': 'KR', 'IL': 'KR',
    'NZ': 'CA', 'MY': 'KR', 'HK': 'KR',
    
}
  for idx, row in cost_updated.iterrows():
    country = row['Country']
    if country in reference_countries:
      reference = economic_groups.get(country, "CN")
      ref_row = cost_updated[cost_updated['Country'] == reference].iloc[0]
      cost_updated.loc[idx, 'Years 0.0-1.5':'Total Cost (US$)'] = ref_row['Years 0.0-1.5':'Total Cost (US$)']

  return cost_updated


def calculate_age(df: pd.DataFrame,) -> pd.DataFrame:
  current_year = datetime.now().year
  df['Patent Age'] = current_year - df['earliest_priority_year']
  df['Patent Age'] = df['Patent Age'].apply(lambda x: max(0, x) if pd.notnull(x) else x)
  return df

def assign_cost(row, cost_df):
    country = row['first_publication_country']
    age = row['Patent Age']
    
    cost_row = cost_df[cost_df['Country'] == country]
    
    if not cost_row.empty:
        
        cost_columns = [
            'Years_0_1_5',
            'Years_2_4_5', 
            'Years_5_9_5',
            'Years_10_14_5',
            'Years_15_20'
        ]
        
        if age > 20:
            return cost_row['Total_cost'].values[0]
        elif age <= 1.5:
            return cost_row[cost_columns[0]].values[0]
        else:
            # Calculate cumulative sum up to current age bracket
            bracket_index = next(i for i, limit in enumerate([1.5, 4.5, 9.5, 14.5, 20.0]) if age <= limit)
            return cost_row[cost_columns[:bracket_index+1]].sum(axis=1).values[0]
    else:
        return None


def get_market_metrics(patents_df, cost_df):
    """
    Calculate:
      - Market value: sum of 'patent cost' (computed per patent using assign_cost)
      - Market rate: total family members / number of patents
      - Mean value of patents: market value / number of patents
    Args:
        patents_df: DataFrame with patent data (must include 'first_publication_country', 'Patent Age', 'family_members')
        cost_df: DataFrame with cost data (must include country and cost columns)
    Returns:
        (market_value, market_rate, mean_value)
    """
    # Compute patent cost for each patent
    patents_df = patents_df.copy()
    patents_df['patent cost'] = patents_df.apply(lambda row: assign_cost(row, cost_df), axis=1)
    # Market value
    market_value = patents_df['patent cost'].dropna().sum()
    # Family members count
    def fam_count(fam):
        if isinstance(fam, list):
            return len(fam)
        elif isinstance(fam, str):
            # Try comma-separated string
            return len([x for x in fam.split(',') if x.strip()])
        return 0
    patents_df['family_members_count'] = patents_df['family_members'].apply(fam_count)
    total_family_members = patents_df['family_members_count'].sum()
    num_patents = len(patents_df)
    market_rate = total_family_members / num_patents if num_patents else 0.0
    mean_value = market_value / num_patents if num_patents else 0.0
    return market_value, market_rate, mean_value