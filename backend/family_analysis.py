"""Takes a DataFrame with country_codes column and counts member families by country."""

import pandas as pd
from typing import List

def get_family_counts_by_country(patents_df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a DataFrame of patents with a 'country_codes' column (list of country codes per row),
    and returns a DataFrame with the counts of family members by country code.

    Parameters:
        patents_df (pd.DataFrame): DataFrame containing at least a 'country_codes' column (list of country codes per row).

    Returns:
        pd.DataFrame: DataFrame with columns ['country_code', 'member_count']
    """
    # Flatten all country_codes into a single list
    all_country_codes = [code for sublist in patents_df['country_codes'] for code in sublist]
    # Count occurrences
    family_counts_df = pd.Series(all_country_codes).value_counts().reset_index()
    family_counts_df.columns = ['country_code', 'member_count']
    return family_counts_df
