import re, os
import pandas as pd
from typing import Dict, Tuple, List
from collections import defaultdict

# ============================
# CONFIG — base folder and specific files
# ============================
BASE_DIR = "backend/outputs"   # your files live here
PATENTS_FILE = os.path.join(BASE_DIR, "patents.csv")
PUBLICATIONS_FILE = os.path.join(BASE_DIR, "publications.csv")
TECHS_ORDER_FILE = os.path.join(BASE_DIR, "techs_order.txt")

# ============================
# Helpers
# ============================
def slugify(name: str) -> str:
    s = name.lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"_+", "_", s)
    return s

def parse_order_file(path: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Lines like:
      pat3; Health Information Systems
      pub5; Smart Grid Technology
    """
    pat_map, pub_map = {}, {}
    if not path or not os.path.exists(path):
        return pat_map, pub_map
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if ";" not in line:
                continue
            key, name = [x.strip() for x in line.split(";", 1)]
            key_norm = key.lower().replace("_", "")  # 'pat_3' -> 'pat3'
            if key_norm.startswith("pat") and re.match(r"^pat\d+$", key_norm):
                pat_map[key_norm] = name
            elif key_norm.startswith("pub") and re.match(r"^pub\d+$", key_norm):
                pub_map[key_norm] = name
    return pat_map, pub_map

def load_kind(csv_path: str, mapping: Dict[str, str], kind: str) -> pd.DataFrame:
    """
    kind in {'pat','pub'}. Accepts column names like pat3 or pat_3 (case-insensitive).
    Renames to 'pat_<slug>' / 'pub_<slug>'.
    """
    if not csv_path or not os.path.exists(csv_path):
        return pd.DataFrame(columns=["year"])

    df = pd.read_csv(csv_path)

    year_col = next((c for c in df.columns if c.lower() == "year"), None)
    if not year_col:
        raise ValueError(f"{csv_path}: missing 'year' column")
    df = df.rename(columns={year_col: "year"})

    pattern = re.compile(rf"^{kind}_?\d+$", re.IGNORECASE)
    val_cols = [c for c in df.columns if pattern.match(c)]
    if not val_cols:
        return df[["year"]].copy()

    rename = {}
    for c in val_cols:
        key_norm = c.lower().replace("_", "")  # pat_3 -> pat3
        tech_name = mapping.get(key_norm, key_norm)  # fallback if mapping missing
        rename[c] = f"{kind}_{slugify(tech_name)}"

    df = df[["year"] + val_cols].rename(columns=rename)

    for c in df.columns:
        if c == "year": continue
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if df["year"].duplicated().any():
        df = df.groupby("year", as_index=False).sum(numeric_only=True)

    return df

def outer_merge_by_year(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    if not dfs:
        return pd.DataFrame(columns=["year"])
    out = dfs[0]
    for d in dfs[1:]:
        out = out.merge(d, on="year", how="outer")

    # collapse _x/_y duplicates by summing
    def collapse(df):
        base_map = defaultdict(list)
        for c in df.columns:
            if c == "year": continue
            base = re.sub(r"(_x|_y|\.\d+)$", "", c)
            base_map[base].append(c)
        res = df[["year"]].copy()
        for base, cols in base_map.items():
            res[base] = df[cols].sum(axis=1, skipna=True)
        return res

    return collapse(out)

# ============================
# Merge patents & publications
# ============================
# Parse the technology order mapping
pat_map, pub_map = parse_order_file(TECHS_ORDER_FILE)

# Load and process the specific files
pat_df = load_kind(PATENTS_FILE, pat_map, "pat")
pub_df = load_kind(PUBLICATIONS_FILE, pub_map, "pub")

# Since we only have one file each, no need to merge multiple DataFrames
pat_merged = pat_df
pub_merged = pub_df

def minmax_year(df):
    if df.empty or df["year"].dropna().empty:
        return None, None
    return int(df["year"].min()), int(df["year"].max())

mins = [x for x in [minmax_year(pat_merged)[0], minmax_year(pub_merged)[0]] if x is not None]
maxs = [x for x in [minmax_year(pat_merged)[1], minmax_year(pub_merged)[1]] if x is not None]
if not mins or not maxs:
    raise RuntimeError("No data found in CSVs under outputs/.")

global_min, global_max = min(mins), max(maxs)
all_years = pd.DataFrame({"year": list(range(global_min, global_max + 1))})

pat_merged = all_years.merge(pat_merged, on="year", how="left").fillna(0)
pub_merged = all_years.merge(pub_merged, on="year", how="left").fillna(0)

for df in (pat_merged, pub_merged):
    for c in df.columns:
        if c == "year": continue
        s = df[c]
        if s.notna().all() and ((s % 1) == 0).all():
            df[c] = s.astype(int)

# Ensure outputs folder exists (it does, but just in case)
os.makedirs(BASE_DIR, exist_ok=True)

out_pat = os.path.join(BASE_DIR, "patents_merged.csv")
out_pub = os.path.join(BASE_DIR, "publications_merged.csv")

pat_merged.to_csv(out_pat, index=False)
pub_merged.to_csv(out_pub, index=False)

print(f"✅ wrote {out_pat}  shape={pat_merged.shape}  years={pat_merged.year.min()}–{pat_merged.year.max()}")
print(f"✅ wrote {out_pub}  shape={pub_merged.shape}  years={pub_merged.year.min()}–{pub_merged.year.max()}")
