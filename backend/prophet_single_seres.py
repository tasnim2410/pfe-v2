"""
Prophet single-series pipeline (current tech only) — with patent autocorrelation (Option A)

- Pulls the CURRENT search's data from Postgres:
    * publications: research_data3.year
    * patents: raw_patents.first_filing_year
- Truncates tail years to avoid reporting delays (patents: last 3y, pubs: last 1y by default)
- Fills missing years with zeros
- Auto-detects publication->patent lag via cross-correlation (detrended)
- Forecasts publications (univariate Prophet)
- Forecasts patents with:
    * baseline (no regressors)
    * lagged publications regressor (auto-selected lag)
    * lagged patents (AR) as regressors: pat_lag1, pat_lag2  <-- Option A
- Time-based evaluation on last N test years (after truncation)
- Produces CSV artifacts and prints a concise summary

Env:
    DATABASE_URL=postgresql+psycopg2://user:pass@host:port/dbname

Requires:
    pip install prophet sqlalchemy psycopg2-binary pandas numpy scikit-learn python-dotenv
"""

import os
import math
import warnings
from dataclasses import dataclass
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

# Prophet import (new name 'prophet', fallback to 'fbprophet')
try:
    from prophet import Prophet
except Exception:
    from fbprophet import Prophet  # type: ignore

from sklearn.metrics import mean_absolute_error, mean_squared_error

# ----------------------------
# Config defaults
# ----------------------------
PUB_TAIL_TRUNC_DEFAULT = 1   # drop last 1 y of publications (incomplete year)
PAT_TAIL_TRUNC_DEFAULT = 3   # drop last 3 y of patents (filing delays)
MAX_LAG_DEFAULT = 5          # search lags 0..5 for pubs->patents
TEST_YEARS_DEFAULT = 5       # holdout length for evaluation
HORIZON_DEFAULT = 10         # forecast years ahead
TECH_LABEL_DEFAULT = "current"  # label for outputs when single series
AR_LAGS = (1, 2)             # patent AR terms to include as regressors

load_dotenv()

# ----------------------------
# Helpers
# ----------------------------
def _require_engine() -> Engine:
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL not set. Example: postgresql+psycopg2://user:pass@host:5432/db")
    return create_engine(db_url)

def _to_year_end(years: pd.Series) -> pd.Series:
    """Map integer years to Dec-31 timestamps."""
    return pd.PeriodIndex(years.astype(int), freq="Y").to_timestamp(how="end")

def _fill_missing_years(df: pd.DataFrame, year_col: str, count_col: str) -> pd.DataFrame:
    if df.empty:
        return df
    y_min, y_max = int(df[year_col].min()), int(df[year_col].max())
    full = pd.DataFrame({year_col: list(range(y_min, y_max + 1))})
    out = full.merge(df[[year_col, count_col]], on=year_col, how="left").fillna({count_col: 0})
    out[count_col] = out[count_col].astype(int)
    return out

def _truncate_tail_years(df: pd.DataFrame, tail: int, year_col="year") -> pd.DataFrame:
    if df.empty or tail <= 0:
        return df.copy()
    max_year = int(df[year_col].max())
    cutoff = max_year - tail
    return df[df[year_col] <= cutoff].copy()

def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 3 or len(b) < 3:
        return np.nan
    try:
        return float(a.corr(b))
    except Exception:
        return np.nan

def infer_best_pub_to_patent_lag(
    pubs_df: pd.DataFrame,
    patents_df: pd.DataFrame,
    max_lag: int = MAX_LAG_DEFAULT,
    detrend: str = "diff",
    min_overlap: int = 8,
    prefer_nonnegative: bool = True
) -> Tuple[int, float]:
    """
    Choose lag L in [0..max_lag] maximizing corr( pubs[t], patents[t+L] ), after optional detrending.
    Returns (best_lag, corr_at_best). If unreliable, returns (1, nan).
    """
    pub = pubs_df.set_index("year")["pub_count"].astype(float)
    pat = patents_df.set_index("year")["patent_count"].astype(float)
    years = pub.index.intersection(pat.index)
    if len(years) < min_overlap:
        return 1, np.nan

    best_lag, best_corr = 0, -np.inf
    for L in range(0, max_lag + 1):
        # Align pubs[t] with patents[t+L]  => shift patents backward by L
        aligned = pat.shift(-L)
        df = pd.concat([pub, aligned], axis=1).dropna()
        if len(df) < min_overlap:
            continue

        x = df.iloc[:, 0].copy()
        y = df.iloc[:, 1].copy()
        if detrend == "diff":
            x, y = x.diff().dropna(), y.diff().dropna()
            idx = x.index.intersection(y.index)
            x, y = x.loc[idx], y.loc[idx]
        elif detrend == "pct":
            x = x.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
            y = y.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
            idx = x.index.intersection(y.index)
            x, y = x.loc[idx], y.loc[idx]
        elif detrend == "zscore":
            if x.std(ddof=0) > 0:
                x = (x - x.mean()) / x.std(ddof=0)
            if y.std(ddof=0) > 0:
                y = (y - y.mean()) / y.std(ddof=0)

        r = _safe_corr(x, y)
        if np.isnan(r):
            continue
        if prefer_nonnegative and r < 0:
            continue
        if r > best_corr:
            best_corr = r
            best_lag = L

    if best_corr == -np.inf:
        return 1, np.nan  # fallback
    return best_lag, best_corr

# ----------------------------
# Data access
# ----------------------------
def fetch_current_series(
    engine,
    pub_tail_trunc: int = PUB_TAIL_TRUNC_DEFAULT,
    pat_tail_trunc: int = PAT_TAIL_TRUNC_DEFAULT
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pull the CURRENT tech series stored by your app:
      - Publications from research_data3.year
      - Patents from raw_patents.first_filing_year
    Apply tail truncation and fill missing years.
    Returns: pubs_df[year,pub_count], patents_df[year,patent_count]
    """
    pubs_sql = """
        SELECT year::int AS year, COUNT(*)::int AS pub_count
        FROM research_data3
        WHERE year IS NOT NULL
        GROUP BY year
        ORDER BY year
    """
    pats_sql = """
        SELECT first_filing_year::int AS year, COUNT(*)::int AS patent_count
        FROM raw_patents
        WHERE first_filing_year IS NOT NULL
        GROUP BY first_filing_year
        ORDER BY year
    """
    pubs = pd.read_sql(pubs_sql, engine)
    pats = pd.read_sql(pats_sql, engine)

    # Truncate tails
    pubs = _truncate_tail_years(pubs, pub_tail_trunc, "year")
    pats = _truncate_tail_years(pats, pat_tail_trunc, "year")

    # Keep common span (helps stability)
    if not pubs.empty and not pats.empty:
        y_min = max(int(pubs["year"].min()), int(pats["year"].min()))
        y_max = min(int(pubs["year"].max()), int(pats["year"].max()))
        pubs = pubs[(pubs["year"] >= y_min) & (pubs["year"] <= y_max)]
        pats = pats[(pats["year"] >= y_min) & (pats["year"] <= y_max)]

    # Fill gaps with zeros
    pubs = _fill_missing_years(pubs, "year", "pub_count")
    pats = _fill_missing_years(pats, "year", "patent_count")
    return pubs, pats

# ----------------------------
# Prophet pieces
# ----------------------------
def prophet_forecast_publications(
    pubs_df: pd.DataFrame,
    horizon: int
) -> pd.DataFrame:
    """Univariate Prophet on publications. Returns future rows (years, pub_count_hat)."""
    if pubs_df.empty:
        return pd.DataFrame(columns=["year", "pub_count_hat", "yhat_lower", "yhat_upper"])

    df = pubs_df.copy()
    df["ds"] = _to_year_end(df["year"])
    df = df.rename(columns={"pub_count": "y"})

    m = Prophet(yearly_seasonality=False, daily_seasonality=False)
    m.fit(df[["ds", "y"]])

    # Build future years explicitly
    last_year = int(df["ds"].dt.year.max())
    future_years = [last_year + i for i in range(1, horizon + 1)]
    future = pd.DataFrame({"ds": _to_year_end(pd.Series(future_years))})

    fcst = m.predict(future)
    out = pd.DataFrame({
        "year": fcst["ds"].dt.year.astype(int),
        "pub_count_hat": fcst["yhat"],
        "yhat_lower": fcst["yhat_lower"],
        "yhat_upper": fcst["yhat_upper"],
    })
    return out

def build_pub_reg(
    pubs_hist: pd.DataFrame,
    pubs_future: pd.DataFrame,
    lag: int
) -> pd.DataFrame:
    """
    Build a unified publications series (history + future hats) and
    compute pub_reg at each year y as pubs[y - lag].
    Returns table with columns: year, pub_reg
    """
    hist = pubs_hist.rename(columns={"pub_count": "pub"}).copy()
    fut  = pubs_future.rename(columns={"pub_count_hat": "pub"}).copy()
    both = pd.concat([hist[["year", "pub"]], fut[["year", "pub"]]], ignore_index=True).drop_duplicates("year")
    both = both.sort_values("year").reset_index(drop=True)

    reg = both.copy()
    reg["year"] = reg["year"].astype(int) + lag  # shift forward so that reg[t] = pub[t - lag]
    reg = reg.rename(columns={"pub": "pub_reg"})
    reg = reg[["year", "pub_reg"]]
    return reg

def build_patent_lags(pats_df: pd.DataFrame, lags: tuple = (1, 2)) -> pd.DataFrame:
    """
    Add pat_lagK columns to a patents dataframe (cols: year, patent_count).
    Returns a copy with added lag columns.
    """
    out = pats_df.copy()
    for k in lags:
        out[f"pat_lag{k}"] = out["patent_count"].shift(k)
    return out

def prophet_patents_baseline_predict(
    pats_train: pd.DataFrame,
    predict_years: List[int]
) -> np.ndarray:
    df = pats_train.copy()
    df["ds"] = _to_year_end(df["year"])
    df = df.rename(columns={"patent_count": "y"})
    m = Prophet(yearly_seasonality=False, daily_seasonality=False)
    m.fit(df[["ds", "y"]])

    fut = pd.DataFrame({"ds": _to_year_end(pd.Series(predict_years))})
    fc = m.predict(fut)
    return fc["yhat"].values

def _iterative_prophet_predict_with_regressors(
    model: Prophet,
    years: List[int],
    pub_reg_all: pd.DataFrame,       # cols: year, pub_reg for all needed future years
    seed_hist: dict,                  # {year: patent_count or predicted}
    lags: tuple = (1, 2),
) -> np.ndarray:
    """
    Iteratively predict year-by-year using Prophet with multiple regressors.
    For each year y, we fetch pub_reg[y] and pat_lag1 = seed_hist[y-1], pat_lag2 = seed_hist[y-2], ...
    Append the new prediction to seed_hist so subsequent steps can use it.
    Returns an array of predictions aligned to 'years'.
    """
    preds = []
    for y in years:
        row = {"year": y, "ds": _to_year_end(pd.Series([y]))[0]}
        # publication regressor
        pr = pub_reg_all.loc[pub_reg_all["year"] == y, "pub_reg"]
        row["pub_reg"] = float(pr.iloc[0]) if len(pr) else np.nan
        # patent lag regressors
        for k in lags:
            row[f"pat_lag{k}"] = seed_hist.get(y - k, np.nan)

        fut = pd.DataFrame([row])
        # ensure regressor columns are present
        fut = fut[["ds", "pub_reg"] + [f"pat_lag{k}" for k in lags]]
        fc = model.predict(fut)
        yhat = float(fc["yhat"].iloc[0])
        preds.append(yhat)
        seed_hist[y] = yhat  # update for next step
    return np.array(preds)

def prophet_patents_with_pub_and_ar_predict(
    pats_train: pd.DataFrame,          # cols: year, patent_count
    pub_reg_train: pd.DataFrame,       # cols: year, pub_reg (train years only)
    predict_years: List[int],          # years to predict
    pub_reg_all: pd.DataFrame,         # cols: year, pub_reg (for all needed future years)
    seed_hist: dict,                   # {year: patent_count} includes at least last max(AR_LAGS) train years
    lags: tuple = AR_LAGS,
) -> np.ndarray:
    """
    Train Prophet with regressors: pub_reg + patent lags (AR terms), then predict iteratively.
    """
    # Build TRAIN with AR lags
    train = build_patent_lags(pats_train, lags=lags)
    train = train.merge(pub_reg_train, on="year", how="left")
    needed_cols = ["year", "patent_count", "pub_reg"] + [f"pat_lag{k}" for k in lags]
    train = train.dropna(subset=needed_cols).copy()
    if train.empty or len(train) < 4:
        warnings.warn("Too few rows after adding lagged regressors; falling back to baseline.")
        return prophet_patents_baseline_predict(pats_train, predict_years)

    # Prophet-ready columns
    train["ds"] = _to_year_end(train["year"])
    train = train.rename(columns={"patent_count": "y"})

    # Fit Prophet with all regressors
    m = Prophet(yearly_seasonality=False, daily_seasonality=False)
    m.add_regressor("pub_reg")
    for k in lags:
        m.add_regressor(f"pat_lag{k}")
    m.fit(train[["ds", "y", "pub_reg"] + [f"pat_lag{k}" for k in lags]])

    # Iterative future predictions
    preds = _iterative_prophet_predict_with_regressors(
        model=m,
        years=predict_years,
        pub_reg_all=pub_reg_all,
        seed_hist=dict(seed_hist),  # copy to avoid side effects
        lags=lags
    )
    return preds

# ----------------------------
# Evaluation + Orchestration
# ----------------------------
@dataclass
class EvalResults:
    tech: str
    best_lag: int
    xcorr: float
    mae_pubs: float
    rmse_pubs: float
    mae_pat_no_reg: float
    rmse_pat_no_reg: float
    mae_pat_with_reg: float
    rmse_pat_with_reg: float

def run_for_current_series(
    engine=None,
    tech_label: str = TECH_LABEL_DEFAULT,
    pub_tail_trunc: int = PUB_TAIL_TRUNC_DEFAULT,
    pat_tail_trunc: int = PAT_TAIL_TRUNC_DEFAULT,
    max_lag: int = MAX_LAG_DEFAULT,
    test_years: int = TEST_YEARS_DEFAULT,
    horizon: int = HORIZON_DEFAULT,
    pubs_error_threshold_rmse: Optional[float] = None,  # if set, can switch to scenario
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, EvalResults]:
    """
    Full pipeline on the current DB contents:
      - fetch series
      - infer lag (train-only)
      - eval (last test_years)
      - final forecasts for horizon years
    Returns: publications_forecast_df, patents_forecast_df, metrics_df, eval_summary
    """
    engine = engine or _require_engine()
    pubs, pats = fetch_current_series(engine, pub_tail_trunc, pat_tail_trunc)

    if pubs.empty or pats.empty:
        raise RuntimeError("Empty series after truncation; ensure the app populated research_data3 and raw_patents.")

    # Split TRAIN/TEST (no leakage)
    y_max = min(int(pubs["year"].max()), int(pats["year"].max()))
    cutoff = y_max - max(1, test_years)
    pub_train, pub_test = pubs[pubs["year"] <= cutoff].copy(), pubs[pubs["year"] > cutoff].copy()
    pat_train, pat_test = pats[pats["year"] <= cutoff].copy(), pats[pats["year"] > cutoff].copy()

    # 1) Publications forecast for TEST horizon (evaluation)
    pub_fc_test = prophet_forecast_publications(pub_train, horizon=len(pub_test))
    # Align to test years
    pub_pred_test = pub_fc_test[pub_fc_test["year"].isin(pub_test["year"])].copy()
    if len(pub_pred_test) and len(pub_test):
        mae_pubs = mean_absolute_error(pub_test["pub_count"].values, pub_pred_test["pub_count_hat"].values)
        rmse_pubs = np.sqrt(mean_squared_error(pub_test["pub_count"].values, pub_pred_test["pub_count_hat"].values))
    else:
        mae_pubs = rmse_pubs = np.nan

    # 2) Auto-lag (train-only)
    best_lag, xcorr = infer_best_pub_to_patent_lag(pub_train, pat_train, max_lag=max_lag)

    # 3) Patents baseline (no regressor)
    pat_base_pred = prophet_patents_baseline_predict(pat_train, predict_years=pat_test["year"].tolist())
    mae_base = mean_absolute_error(pat_test["patent_count"].values, pat_base_pred) if len(pat_test) else np.nan
    rmse_base = np.sqrt(mean_squared_error(pat_test["patent_count"].values, pat_base_pred)) if len(pat_test) else np.nan

    # 4) Patents with lagged pubs + AR (Option A)
    # Build pub_reg for TRAIN and TEST
    pub_reg_train = build_pub_reg(pub_train, pubs_future=pd.DataFrame(columns=["year","pub_count_hat"]), lag=best_lag)
    pub_reg_test_full = build_pub_reg(pub_train, pub_fc_test, lag=best_lag)

    # seed history for iterative prediction: use TRAIN actuals
    seed_hist_train = pat_train.set_index("year")["patent_count"].to_dict()

    pat_reg_pred = prophet_patents_with_pub_and_ar_predict(
        pats_train=pat_train,
        pub_reg_train=pub_reg_train,
        predict_years=pat_test["year"].tolist(),
        pub_reg_all=pub_reg_test_full,
        seed_hist=seed_hist_train,
        lags=AR_LAGS
    )
    mae_reg = mean_absolute_error(pat_test["patent_count"].values, pat_reg_pred) if len(pat_test) else np.nan
    rmse_reg = np.sqrt(mean_squared_error(pat_test["patent_count"].values, pat_reg_pred)) if len(pat_test) else np.nan

    eval_summary = EvalResults(
        tech=tech_label,
        best_lag=best_lag,
        xcorr=xcorr if xcorr is not None else np.nan,
        mae_pubs=mae_pubs, rmse_pubs=rmse_pubs,
        mae_pat_no_reg=mae_base, rmse_pat_no_reg=rmse_base,
        mae_pat_with_reg=mae_reg, rmse_pat_with_reg=rmse_reg,
    )

    # ========= Final production forecasts (full data) =========
    # Publications: forecast H years from latest (after truncation)
    pub_fc_full = prophet_forecast_publications(pubs, horizon=horizon)

    # Patents with lagged pubs + AR on FULL history
    pub_reg_hist_full = build_pub_reg(pubs, pubs_future=pd.DataFrame(columns=["year","pub_count_hat"]), lag=best_lag)
    pub_reg_all_full  = build_pub_reg(pubs, pub_fc_full, lag=best_lag)

    # seed with ALL observed patents
    seed_hist_full = pats.set_index("year")["patent_count"].to_dict()

    future_years = list(range(y_max + 1, y_max + 1 + horizon))
    pat_reg_future = prophet_patents_with_pub_and_ar_predict(
        pats_train=pats,
        pub_reg_train=pub_reg_hist_full,
        predict_years=future_years,
        pub_reg_all=pub_reg_all_full,
        seed_hist=seed_hist_full,
        lags=AR_LAGS
    )

    # Also baseline (optional, for comparison)
    pat_base_future = prophet_patents_baseline_predict(pats, predict_years=future_years)

    # Assemble outputs
    pubs_forecasts = pub_fc_full.copy()
    pubs_forecasts.insert(0, "tech", tech_label)

    patents_forecasts = pd.DataFrame({
        "tech": tech_label,
        "year": future_years,
        "yhat_with_pub_reg": pat_reg_future,
        "yhat_baseline": pat_base_future
    })

    metrics_df = pd.DataFrame([{
        "tech": tech_label,
        "best_lag": best_lag,
        "xcorr": xcorr,
        "mae_pubs": mae_pubs,
        "rmse_pubs": rmse_pubs,
        "mae_patents_with_reg": mae_reg,
        "rmse_patents_with_reg": rmse_reg,
        "mae_patents_baseline": mae_base,
        "rmse_patents_baseline": rmse_base
    }])

    return pubs_forecasts, patents_forecasts, metrics_df, eval_summary
  
  
  

# ----------------------------
# CLI
# ----------------------------
def _fmt2(x):
    return "nan" if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x:.2f}"

if __name__ == "__main__":
    pubs_fc, pats_fc, metrics_df, summary = run_for_current_series()

    pubs_fc.to_csv("publications_forecasts.csv", index=False)
    pats_fc.to_csv("patents_forecasts_with_pub_reg.csv", index=False)
    metrics_df.to_csv("prophet_eval_metrics.csv", index=False)

    print("\n=== Evaluation (last test years) ===")
    print(f"tech: {summary.tech}")

    xc = summary.xcorr
    xc_str = "nan" if (xc is None or (isinstance(xc, float) and math.isnan(xc))) else f"{xc:.3f}"
    print(f"auto-selected lag (pubs→patents): {summary.best_lag} years (xcorr={xc_str})")

    print(f"Publications  MAE={_fmt2(summary.mae_pubs)}  RMSE={_fmt2(summary.rmse_pubs)}")
    print(f"Patents base  MAE={_fmt2(summary.mae_pat_no_reg)}  RMSE={_fmt2(summary.rmse_pat_no_reg)}")
    print(f"Patents reg   MAE={_fmt2(summary.mae_pat_with_reg)}  RMSE={_fmt2(summary.rmse_pat_with_reg)}")

    print("\nArtifacts written:")
    print("  - publications_forecasts.csv")
    print("  - patents_forecasts_with_pub_reg.csv")
    print("  - prophet_eval_metrics.csv")
