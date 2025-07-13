# df[['first applicant', 'second applicant']] = df['Applicants'].str.split('\n' , n=1 , expand=True)

import pandas as pd

def classify_applicant(applicant, inventors):
    if pd.isna(applicant):
        return "Unknown"
    applicant_lower = applicant.lower()
    
    # Corporations/Companies
    if any(keyword in applicant_lower for keyword in ["corp", "inc", "ltd", "co.", "llc", "ag", "gmbh", "co", "holdings", "ventures"]):
        if ("inc" or "corp" or "holdings" or "ventures")  in applicant_lower or "incorporated" in applicant_lower:
            return "Company - Incorporated/Corporation"
        elif ("ltd" or "llc" or "gmbh" or "kk" or "bv") in applicant_lower or "limited" in applicant_lower:
            return "Company - Limited"
        elif any(keyword in applicant_lower for keyword in ["s.a.", "sociedad anónima", "société anonyme"]):
            return "Company - Anonymous (S.A.)"
        else:
            return "Company - General"
    
    # Automotive manufacturers
    if any(keyword in applicant_lower for keyword in ["automobile", "motor", "vehicle", "auto" , "mobility","motors"]):
        return "Automotive Manufacturer"
    
    # Energy companies
    if any(keyword in applicant_lower for keyword in ["power", "energy", "fuel cell", "hydrogen"]):
        return "Energy Company"
    
    # Technology companies
    if any(keyword in applicant_lower for keyword in ["tech", "technology", "creative", "innovation" , "engineering" , "systems" , "digital" , "solutions"]):
        return "Technology Company"
    
    # Material Science/Nanotechnology companies
    if any(keyword in applicant_lower for keyword in ["nano", "material"]):
        return "Material Science/Nanotechnology Company"
    
    # Environmental protection companies
    if any(keyword in applicant_lower for keyword in ["environmental protection", "green air"]):
        return "Environmental Protection Company"
    
    # Universities/Research Institutions
    if any(keyword in applicant_lower for keyword in ["univ", "university", "college", "polytechnic", "institute", "school", "academia", "laboratory", "research"]):
        return "University/Research Institution"
    
    # Technical Universities
    if any(keyword in applicant_lower for keyword in ["teknik", "technical", "polytechnic"]):
        return "Technical University"
    
    # Research Laboratories
    if any(keyword in applicant_lower for keyword in ["laboratory", "institute"]):
        return "Research Laboratory"
    
    # Government/Public Institutions
    if any(keyword in applicant_lower for keyword in ["national", "government", "ministry", "agency"]):
        return "Government/Public Institution"
    
    # Individual Inventors 
    if applicant in inventors.values:
        return "Individual Inventor"
    if "[" in applicant and "]" in applicant:
        return "Individual Inventor"
    
    return "Individual Inventor"

# df['first applicant type'] = df.apply(lambda row: classify_applicant(row['first applicant'], df['Inventors']), axis=1)
# df['second applicant type'] = df.apply(lambda row: classify_applicant(row['second applicant'], df['Inventors']), axis=1)

def get_applicants_df(df):
    applicants = []
    for index, row in df.iterrows():
        first_applicant = row['first applicant']
        second_applicant = row['second applicant']
        if pd.notna(first_applicant):
            applicants.append(first_applicant)
        if pd.notna(second_applicant):
            applicants.append(second_applicant)
    applicants_df = pd.DataFrame(applicants, columns=['Applicants'])
    applicants_df['Applicant Type'] = applicants_df.apply(
        lambda row: classify_applicant(row['Applicants'], df['Inventors']), axis=1)
    return applicants_df

def get_applicant_type_df(applicants_df):
    applicant_type_counts = applicants_df['Applicant Type'].value_counts()
    applicant_type_percentages = (applicant_type_counts / applicant_type_counts.sum()) * 100
    applicant_type_df = applicant_type_percentages.reset_index()
    applicant_type_df.columns = ['Applicant Type', 'Percentage']
    applicant_type_df = applicant_type_df.sort_values(by='Percentage', ascending=False)
    return applicant_type_df



def get_top_10_applicants(applicants_df):
    """
    Returns a DataFrame of the top 10 applicants by patent count,
    including their contribution percentage.
    
    Parameters:
        applicants_df (pd.DataFrame): DataFrame with an 'Applicants' column.
        
    Returns:
        pd.DataFrame: Top 10 applicants with columns ['Applicant', 'Patent Count', 'Contribution (%)']
    """
    applicant_counts = applicants_df['Applicants'].value_counts()
    top_10_applicants_df = applicant_counts.head(10).reset_index()
    top_10_applicants_df.columns = ['Applicant', 'Patent Count']
    total_patents = applicants_df['Applicants'].count()
    top_10_applicants_df['Contribution (%)'] = (top_10_applicants_df['Patent Count'] / total_patents) * 100
    return top_10_applicants_df



import pandas as pd

def get_innovation_cycle(applicants, top_n: int = 10) -> str:
    if isinstance(applicants,list):
        series = pd.Series(applicants, name='Applicants')
    elif isinstance(applicants, pd.DataFrame):
        series = applicants['Applicants']
    else:
        series = applicants
        
    applicant_counts = series.value_counts()


    top_10_applicants_df = applicant_counts.head(10).reset_index()
    top_10_applicants_df.columns = ['Applicant', 'Patent Count']  
    
    total_patents = applicants['Applicants'].count()

    total_top_10_patents = top_10_applicants_df['Patent Count'].sum()

    pct = (total_top_10_patents / total_patents) * 100
    
    
    # Map percentage to cycle phase
    if pct >= 50:
        return pct #the innovation cycle is Ending
    elif pct >= 30:
        return pct #the innovation cycle is Slowing
    elif pct >= 20:
        return pct #the innovation cycle is Ongoing
    elif pct >= 10:
        return pct #the innovation cycle is Beginning
    else:
        return pct #the innovation cycle is Emerging

import pandas as pd





def extract_applicant_collaboration_network(patent_df, applicant_df, split_on='\n'):
    """
    Extracts first and second applicants from the patent DataFrame, matches them with their types from applicant_df,
    returns data for a network graph and the main type of collaboration.
    No normalization: company types are preserved as-is.

    Args:
        patent_df (pd.DataFrame): DataFrame with at least an 'Applicants' column.
        applicant_df (pd.DataFrame): DataFrame with columns ['applicant_name', 'applicant_type'].
        split_on (str): Delimiter for splitting applicants (default: '\n').
    Returns:
        dict: {
            'edges': list of (type1, type2, weight),
            'network_type': str (main type),
            'type_counts': dict of collaboration type counts
        }
    """
    # Extract first and second applicants
    if isinstance(patent_df, pd.DataFrame):
        series = patent_df['Applicants']
    else:
        series = patent_df
    first = series.str.split(split_on, n=1, expand=True).iloc[:, 0]
    second = series.str.split(split_on, n=1, expand=True).iloc[:, 1]

    # Prepare lookup for applicant type
    applicant_type_lookup = applicant_df.set_index('applicant_name')['applicant_type'].to_dict()

    # Build edges and count types
    edges = {}
    type_pair_counts = {}
    for a1, a2 in zip(first, second):
        if pd.isna(a1) or pd.isna(a2):
            continue
        t1 = str(applicant_type_lookup.get(a1, 'Unknown')).lower()
        t2 = str(applicant_type_lookup.get(a2, 'Unknown')).lower()
        if t1 == 'unknown' or t2 == 'unknown' or t1 == t2:
            continue
        edge = tuple(sorted([t1, t2]))
        edges[edge] = edges.get(edge, 0) + 1
        type_pair_counts[edge] = type_pair_counts.get(edge, 0) + 1

    # Determine main network type
    if not type_pair_counts:
        network_type = 'none'
    else:
        main_edge = max(type_pair_counts.items(), key=lambda x: x[1])[0]
        network_type = '-'.join(main_edge)

    # Prepare output for frontend
    edge_list = [{'source': k[0], 'target': k[1], 'weight': v} for k, v in edges.items()]
    serializable_type_counts = {f"{k[0]}-{k[1]}": v for k, v in type_pair_counts.items()}
    return {
        'edges': edge_list,
        'network_type': network_type,
        'type_counts': serializable_type_counts
    }






def get_co_applicant_rate(applicants, split_on: str = '\n') -> float:
    """
    Calculate the percentage of patents that have more than one listed applicant.
    
    Parameters
    ----------
    applicants : pd.DataFrame or pd.Series
        If DataFrame, must have a column named 'Applicants'.
        If Series, should contain the applicant strings directly.
    split_on : str, optional
        The delimiter to split multiple applicants on (default is newline).
    
    Returns
    -------
    float
        Percentage of entries with at least two applicants.
    """
    # Extract the raw Series
    if isinstance(applicants, pd.DataFrame):
        series = applicants['Applicants']
    else:
        series = applicants

    # Split on the delimiter, expand into two columns
    first, second = series.str.split(split_on, n=1, expand=True).iloc[:,0], series.str.split(split_on, n=1, expand=True).iloc[:,1]

    # Count how many have a non-null second applicant
    num_multi = second.notna().sum()
    total = len(series)

    # Compute percentage
    return (num_multi / total) * 100

