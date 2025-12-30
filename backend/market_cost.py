import pandas as pd
import re
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

def assign_cost(country, age, cost_df):
    """Calculate cost for a single country/jurisdiction based on age."""
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
            bracket_index = next(i for i, limit in enumerate([1.5, 4.5, 9.5, 14.5, 20.0]) if age <= limit)
            return cost_row[cost_columns[:bracket_index+1]].sum(axis=1).values[0]
    else:
        return 0.0


def calculate_family_cost(row, cost_df):
    """Calculate total cost for all family members based on their jurisdictions."""
    age = row['Patent Age']
    family_jurisdictions = row.get('family_jurisdictions', [])
    
    if isinstance(family_jurisdictions, str):
        family_jurisdictions = [x.strip() for x in family_jurisdictions.split(",") if x.strip()]
    elif not isinstance(family_jurisdictions, list):
        family_jurisdictions = []
    
    total_cost = 0.0
    for country in family_jurisdictions:
        cost = assign_cost(country, age, cost_df)
        if cost:
            total_cost += cost
    
    return total_cost


def get_market_metrics(patents_df,market_strategy_df, cost_df):
    """
    Calculate:
      - Market value: sum of costs of all family members across all alive patents
      - Market rate: total family members / number of alive patents
      - Mean value: market value / number of alive patents
    Args:
        patents_df: DataFrame (must include 'alive_any', 'family_jurisdictions', 'Patent Age')
        cost_df: DataFrame with costing rules
    Returns:
        (market_value, market_rate, mean_value)
    """
    patents_df = patents_df.copy()
    market_strategy_df = market_strategy_df.copy()

    def _normalize_pub(v) -> str:
        s = str(v or "").upper().strip()
        return re.sub(r"[^A-Z0-9]", "", s)

    if "publication_number" in patents_df.columns:
        patents_df["publication_number"] = patents_df["publication_number"].fillna("").astype(str).str.strip()
    if "first_publication_number" in patents_df.columns:
        patents_df["first_publication_number"] = patents_df["first_publication_number"].fillna("").astype(str).str.strip()

    if "publication_number" in market_strategy_df.columns:
        market_strategy_df["publication_number"] = market_strategy_df["publication_number"].fillna("").astype(str).str.strip()

    # Prefer joining on first_publication_number (patents), fallback to publication_number
    join_src = "first_publication_number" if "first_publication_number" in patents_df.columns else "publication_number"
    patents_df["_join_pub"] = patents_df[join_src]
    if "publication_number" in patents_df.columns:
        patents_df.loc[patents_df["_join_pub"].fillna("").astype(str).str.strip().eq(""), "_join_pub"] = patents_df["publication_number"]
    patents_df["_join_key"] = patents_df["_join_pub"].apply(_normalize_pub)
    market_strategy_df["_join_key"] = market_strategy_df["publication_number"].apply(_normalize_pub)

    cols = [c for c in ["_join_key", "publication_number", "legal_status", "family_jurisdictions", "family_members"] if c in market_strategy_df.columns]
    market_strategy_df = market_strategy_df[cols]

    market_value_df = pd.merge(
        patents_df,
        market_strategy_df,
        on="_join_key",
        how="left",
        suffixes=("", "_ms"),
    )

    def _coalesce_family_field(row, base_col: str):
        v = row.get(base_col)
        if isinstance(v, list):
            if len(v) > 0:
                return v
        elif isinstance(v, str):
            if v.strip():
                return v
        else:
            if pd.notnull(v):
                return v

        v2 = row.get(f"{base_col}_ms")
        return v2

    if "family_jurisdictions_ms" in market_value_df.columns:
        market_value_df["family_jurisdictions"] = market_value_df.apply(
            lambda r: _coalesce_family_field(r, "family_jurisdictions"), axis=1
        )
    if "family_members_ms" in market_value_df.columns:
        market_value_df["family_members"] = market_value_df.apply(
            lambda r: _coalesce_family_field(r, "family_members"), axis=1
        )

    # ---- 1. Filter alive patents only ----
    alive_statuses = {"ALIVE", "GRANTED", "PENDING"}
    legal_status_norm = market_value_df["legal_status"].fillna("").astype(str).str.upper().str.strip()
    alive_mask = legal_status_norm.isin(alive_statuses)
    alive_count = int(alive_mask.sum())
    print(f"[get_market_metrics] Total patents: {len(patents_df)}, Alive: {alive_count}")
    
    alive_df = market_value_df[alive_mask].copy()

    if alive_df.empty:
        print(f"[get_market_metrics] No alive patents found")
        return 0.0, 0.0, 0.0

    # ---- 2. Compute total family cost for each alive patent ----
    alive_df["family_cost"] = alive_df.apply(lambda row: calculate_family_cost(row, cost_df), axis=1)
    
    # Debug: Check family_jurisdictions
    has_family_jurs = alive_df["family_jurisdictions"].notna().sum()
    empty_family_jurs = (alive_df["family_jurisdictions"].apply(lambda x: len(x) if isinstance(x, (list, str)) else 0) == 0).sum()
    print(f"[get_market_metrics] Alive patents with family_jurisdictions: {has_family_jurs}, Empty: {empty_family_jurs}")
    print(f"[get_market_metrics] Sample family_jurisdictions: {alive_df['family_jurisdictions'].iloc[0] if len(alive_df) > 0 else 'N/A'}")
    
    # Debug: Check family_cost
    non_zero_costs = (alive_df["family_cost"] > 0).sum()
    print(f"[get_market_metrics] Alive patents with non-zero cost: {non_zero_costs}")
    print(f"[get_market_metrics] Sample family_cost: {alive_df['family_cost'].iloc[0] if len(alive_df) > 0 else 'N/A'}")

    # ---- 3. Count family members ----
    def fam_count(fam):
        if isinstance(fam, list):
            return len(fam)
        elif isinstance(fam, str):
            return len([x for x in fam.split(",") if x.strip()])
        return 0

    base_for_count = "family_members" if "family_members" in alive_df.columns else "family_jurisdictions"
    alive_df["family_members_count"] = alive_df[base_for_count].apply(fam_count)

    # ---- 4. Compute metrics ----
    market_value = alive_df["family_cost"].dropna().sum()
    total_family_members = alive_df["family_members_count"].sum()
    num_alive = len(alive_df)

    print(f"[get_market_metrics] Market value: {market_value}, Total family members: {total_family_members}, Num alive: {num_alive}")

    market_rate = total_family_members / num_alive if num_alive else 0.0
    mean_value = market_value / num_alive if num_alive else 0.0

    return market_value, market_rate, mean_value
