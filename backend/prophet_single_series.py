


# # """
# # Prophet single-series pipeline (current tech only) — with patent autocorrelation (Option A)

# # - Pulls the CURRENT search's data from Postgres:
# #     * publications: research_data3.year
# #     * patents: raw_patents.first_filing_year
# # - Truncates tail years to avoid reporting delays (patents: last 3y, pubs: last 1y by default)
# # - Fills missing years with zeros
# # - Auto-detects publication->patent lag via cross-correlation (detrended)
# # - Forecasts publications (univariate Prophet)
# # - Forecasts patents with:
# #     * baseline (no regressors)
# #     * lagged publications regressor (auto-selected lag)
# #     * lagged patents (AR) as regressors: pat_lag1, pat_lag2  <-- Option A
# # - Time-based evaluation on last N test years (after truncation)
# # - Produces CSV artifacts and prints a concise summary

# # Env:
# #     DATABASE_URL=postgresql+psycopg2://user:pass@host:port/dbname

# # Requires:
# #     pip install prophet sqlalchemy psycopg2-binary pandas numpy scikit-learn python-dotenv
# # """

# # import os
# # import math
# # import warnings
# # from dataclasses import dataclass
# # from typing import Tuple, Optional, List
# # from datetime import date
# # import numpy as np
# # import pandas as pd
# # from dotenv import load_dotenv
# # from sqlalchemy import create_engine
# # from sqlalchemy.engine import Engine

# # # Prophet import (new name 'prophet', fallback to 'fbprophet')
# # try:
# #     from prophet import Prophet
# # except Exception:
# #     from fbprophet import Prophet  # type: ignore

# # from sklearn.metrics import mean_absolute_error, mean_squared_error
# # import matplotlib.pyplot as plt
# # from statsmodels.tsa.arima.model import ARIMA 
# # # ----------------------------
# # # Config defaults
# # # ----------------------------
# # PUB_TAIL_TRUNC_DEFAULT = 1   # drop last 1 y of publications (incomplete year)
# # PAT_TAIL_TRUNC_DEFAULT = 3   # drop last 3 y of patents (filing delays)
# # MAX_LAG_DEFAULT = 5          # search lags 0..5 for pubs->patents
# # TEST_YEARS_DEFAULT = 5       # holdout length for evaluation
# # HORIZON_MIN = 3              # Minimum forecast years
# # HORIZON_MAX = 20             # Maximum forecast years
# # HORIZON_DEFAULT = 5         # forecast years ahead
# # TECH_LABEL_DEFAULT = "current"  # label for outputs when single series
# # AR_LAGS = (1, 2)             # patent AR terms to include as regressors

# # load_dotenv()

# # # ----------------------------
# # # Helpers
# # # ----------------------------
# # def _require_engine() -> Engine:
# #     db_url = os.getenv("DATABASE_URL")
# #     if not db_url:
# #         raise RuntimeError("DATABASE_URL not set. Example: postgresql+psycopg2://user:pass@host:5432/db")
# #     return create_engine(db_url)

# # def _to_year_end(years: pd.Series) -> pd.Series:
# #     """Map integer years to Dec-31 timestamps."""
# #     return pd.PeriodIndex(years.astype(int), freq="Y").to_timestamp(how="end")

# # def _fill_missing_years(df: pd.DataFrame, year_col: str, count_col: str) -> pd.DataFrame:
# #     if df.empty:
# #         return df
# #     y_min, y_max = int(df[year_col].min()), int(df[year_col].max())
# #     full = pd.DataFrame({year_col: list(range(y_min, y_max + 1))})
# #     out = full.merge(df[[year_col, count_col]], on=year_col, how="left").fillna({count_col: 0})
# #     out[count_col] = out[count_col].astype(int)
# #     return out

# # def _truncate_tail_years(df: pd.DataFrame, tail: int, year_col="year") -> pd.DataFrame:
# #     if df.empty or tail <= 0:
# #         return df.copy()
# #     max_year = int(df[year_col].max())
# #     cutoff = max_year - tail
# #     return df[df[year_col] <= cutoff].copy()

# # def _safe_corr(a: pd.Series, b: pd.Series) -> float:
# #     if len(a) < 3 or len(b) < 3:
# #         return np.nan
# #     try:
# #         return float(a.corr(b))
# #     except Exception:
# #         return np.nan


# # def _ols_aic(x: pd.Series, y: pd.Series) -> float:
# #     """
# #     AIC for simple OLS: y = b0 + b1*x + e (Gaussian).
# #     Returns +inf if not enough data.
# #     """
# #     x, y = x.dropna(), y.dropna()
# #     idx = x.index.intersection(y.index)
# #     x, y = x.loc[idx], y.loc[idx]
# #     n = len(x)
# #     if n < 3:
# #         return float("inf")
# #     X = np.column_stack([np.ones(n), x.values])
# #     beta, *_ = np.linalg.lstsq(X, y.values, rcond=None)
# #     resid = y.values - X.dot(beta)
# #     rss = float(np.sum(resid ** 2))
# #     k = 2  # intercept + slope
# #     if rss <= 0:
# #         return float("-inf")
# #     sigma2 = rss / n
# #     return n * np.log(sigma2) + 2 * k

# # def _prewhiten_series_ar1(s: pd.Series) -> pd.Series:
# #     """
# #     AR(1) prewhitening: e_t = s_t - phi * s_{t-1}, where phi = corr(s_t, s_{t-1}).
# #     """
# #     s1 = s.shift(1).dropna()
# #     s2 = s.loc[s1.index]
# #     phi = _safe_corr(s2, s1)
# #     if np.isnan(phi):
# #         phi = 0.0
# #     e = s2 - phi * s1
# #     return e.dropna()

# # def _prewhiten_series_arima(s: pd.Series) -> pd.Series:
# #     """
# #     ARIMA(1,1,0) residuals as prewhitened signal. Falls back to diff2 if statsmodels is unavailable.
# #     """
# #     if ARIMA is None or len(s) < 6:
# #         return s.diff().diff().dropna()
# #     try:
# #         model = ARIMA(s.astype(float), order=(1, 1, 0))
# #         res = model.fit(method_kwargs={"warn_convergence": False})
# #         # ARIMA(1,1,0) residuals align to differenced series length
# #         e = pd.Series(res.resid, index=res.resid.index)
# #         return e.dropna()
# #     except Exception:
# #         return s.diff().diff().dropna()

# # def _transform_for_xcorr(x: pd.Series, y: pd.Series, detrend: str = "diff", prewhiten: str = "none") -> Tuple[pd.Series, pd.Series]:
# #     """
# #     Apply detrending and/or prewhitening consistently to x and y, then align indexes.
# #     detrend in {"none","diff","diff2","pct","zscore"}
# #     prewhiten in {"none","ar1","arima"}
# #     """
# #     x = x.copy()
# #     y = y.copy()

# #     # Detrend / difference
# #     if detrend == "diff":
# #         x, y = x.diff(), y.diff()
# #     elif detrend == "diff2":
# #         x, y = x.diff().diff(), y.diff().diff()
# #     elif detrend == "pct":
# #         x = x.pct_change().replace([np.inf, -np.inf], np.nan)
# #         y = y.pct_change().replace([np.inf, -np.inf], np.nan)
# #     elif detrend == "zscore":
# #         if x.std(ddof=0) > 0:
# #             x = (x - x.mean()) / x.std(ddof=0)
# #         if y.std(ddof=0) > 0:
# #             y = (y - y.mean()) / y.std(ddof=0)
# #     # else "none": leave as-is

# #     x, y = x.dropna(), y.dropna()
# #     idx = x.index.intersection(y.index)
# #     x, y = x.loc[idx], y.loc[idx]

# #     # Prewhiten
# #     if prewhiten == "ar1":
# #         x = _prewhiten_series_ar1(x)
# #         y = _prewhiten_series_ar1(y)
# #     elif prewhiten == "arima":
# #         x = _prewhiten_series_arima(x)
# #         y = _prewhiten_series_arima(y)

# #     # Final align
# #     x, y = x.dropna(), y.dropna()
# #     idx = x.index.intersection(y.index)
# #     return x.loc[idx], y.loc[idx]


# # def infer_best_pub_to_patent_lag(
# #     pubs_df: pd.DataFrame,
# #     patents_df: pd.DataFrame,
# #     max_lag: int = MAX_LAG_DEFAULT,
# #     detrend: str = "diff",
# #     min_overlap: int = 8,
# #     prefer_nonnegative: bool = True,
# #     prewhiten: str = "ar1",          # "none", "ar1", or "arima"
# #     allow_zero_lag: bool = True,     # allow contemporaneous link in search
# #     zero_margin: float = 0.05,       # r(0) must beat r(1) by > margin to prefer 0
# #     ic_tiebreak: bool = True         # if r0≈r1, use AIC to choose; otherwise prefer lag>=1
# # ) -> Tuple[int, float, dict]:
# #     """
# #     Choose lag L in [0..max_lag] maximizing corr( pubs[t], patents[t+L] ), after optional
# #     detrending + prewhitening. Returns (best_lag, corr_at_best, corr_map).

# #     - If allow_zero_lag is False, searches L>=1 only.
# #     - If r(0) and r(1) are close (within zero_margin), prefer lag=1 via information criterion
# #       (AIC) or by heuristic preference for causal lag.
# #     """
# #     pub = pubs_df.set_index("year")["pub_count"].astype(float)
# #     pat = patents_df.set_index("year")["patent_count"].astype(float)
# #     years = pub.index.intersection(pat.index)
# #     if len(years) < min_overlap:
# #         return 1, np.nan, {}

# #     # search range
# #     start_L = 0 if allow_zero_lag else 1
# #     corr_at: dict = {}
# #     best_lag, best_corr = start_L, -np.inf

# #     for L in range(start_L, max_lag + 1):
# #         # Align pubs[t] with patents[t+L]  => shift patents backward by L
# #         aligned = pat.shift(-L)
# #         df = pd.concat([pub, aligned], axis=1).dropna()
# #         if len(df) < min_overlap:
# #             continue

# #         x_raw = df.iloc[:, 0]
# #         y_raw = df.iloc[:, 1]

# #         x, y = _transform_for_xcorr(x_raw, y_raw, detrend=detrend, prewhiten=prewhiten)
# #         if len(x) < min_overlap or len(y) < min_overlap:
# #             continue

# #         r = _safe_corr(x, y)
# #         if np.isnan(r):
# #             continue
# #         if prefer_nonnegative and r < 0:
# #             continue

# #         corr_at[L] = r
# #         if r > best_corr:
# #             best_corr = r
# #             best_lag = L

# #     if not corr_at:
# #         return 1, np.nan, {}

# #     # Zero-lag tie-breaks vs lag=1
# #     if allow_zero_lag and 0 in corr_at and 1 in corr_at:
# #         r0, r1 = corr_at[0], corr_at[1]
# #         # if close, prefer causal (>=1) using AIC if enabled
# #         if (not np.isnan(r0)) and (not np.isnan(r1)) and (r0 <= r1 + zero_margin):
# #             if ic_tiebreak:
# #                 # recompute aligned & transformed series for AIC tie-break
# #                 aligned0 = pat.shift(0)
# #                 df0 = pd.concat([pub, aligned0], axis=1).dropna()
# #                 x0, y0 = _transform_for_xcorr(df0.iloc[:, 0], df0.iloc[:, 1], detrend=detrend, prewhiten=prewhiten)

# #                 aligned1 = pat.shift(-1)
# #                 df1 = pd.concat([pub, aligned1], axis=1).dropna()
# #                 x1, y1 = _transform_for_xcorr(df1.iloc[:, 0], df1.iloc[:, 1], detrend=detrend, prewhiten=prewhiten)

# #                 aic0 = _ols_aic(x0, y0)
# #                 aic1 = _ols_aic(x1, y1)
# #                 # prefer lower AIC; if tie or noisy, prefer causal lag=1
# #                 if aic1 <= aic0 + 1e-6:
# #                     best_lag, best_corr = 1, r1
# #                 else:
# #                     best_lag, best_corr = 0, r0
# #             else:
# #                 best_lag, best_corr = 1, r1

# #     return best_lag, best_corr, corr_at


# # # ----------------------------
# # # Data access
# # # ----------------------------
# # def fetch_current_series(
# #     engine,
# #     pub_tail_trunc: int = PUB_TAIL_TRUNC_DEFAULT,
# #     pat_tail_trunc: int = PAT_TAIL_TRUNC_DEFAULT
# # ) -> Tuple[pd.DataFrame, pd.DataFrame]:
# #     """
# #     Pull the CURRENT tech series stored by your app:
# #       - Publications from research_data3.year
# #       - Patents from raw_patents.first_filing_year
# #     Apply tail truncation and fill missing years.
# #     Returns: pubs_df[year,pub_count], patents_df[year,patent_count]
# #     """
# #     pubs_sql = """
# #         SELECT year::int AS year, COUNT(*)::int AS pub_count
# #         FROM research_data3
# #         WHERE year IS NOT NULL
# #         GROUP BY year
# #         ORDER BY year
# #     """
# #     pats_sql = """
# #         SELECT first_filing_year::int AS year, COUNT(*)::int AS patent_count
# #         FROM raw_patents
# #         WHERE first_filing_year IS NOT NULL
# #         GROUP BY first_filing_year
# #         ORDER BY year
# #     """
# #     pubs = pd.read_sql(pubs_sql, engine)
# #     pats = pd.read_sql(pats_sql, engine)

# #     # Truncate tails
# #     pubs = _truncate_tail_years(pubs, pub_tail_trunc, "year")
# #     pats = _truncate_tail_years(pats, pat_tail_trunc, "year")

# #     # Keep common span (helps stability)
# #     if not pubs.empty and not pats.empty:
# #         y_min = max(int(pubs["year"].min()), int(pats["year"].min()))
# #         y_max = min(int(pubs["year"].max()), int(pats["year"].max()))
# #         pubs = pubs[(pubs["year"] >= y_min) & (pubs["year"] <= y_max)]
# #         pats = pats[(pats["year"] >= y_min) & (pats["year"] <= y_max)]

# #     # Fill gaps with zeros
# #     pubs = _fill_missing_years(pubs, "year", "pub_count")
# #     pats = _fill_missing_years(pats, "year", "patent_count")
# #     return pubs, pats

# # # ----------------------------
# # # Prophet pieces
# # # ----------------------------

# # def _build_pub_lags(pubs_df: pd.DataFrame, lags: tuple = (1, 2)) -> pd.DataFrame:
# #     out = pubs_df.copy()
# #     for k in lags:
# #         out[f"pub_lag{k}"] = out["pub_count"].shift(k)
# #     return out

# # def prophet_forecast_publications_ar(
# #     pubs_df: pd.DataFrame,
# #     horizon: int,
# #     lags: tuple = (1, 2)
# # ) -> pd.DataFrame:
# #     """
# #     Prophet with publication self-lags as extra regressors.
# #     Returns future rows: [year, pub_count_hat, yhat_lower, yhat_upper].
# #     Falls back to univariate Prophet if too few rows remain after lagging.
# #     """
# #     if pubs_df.empty or horizon <= 0:
# #         return pd.DataFrame(columns=["year", "pub_count_hat", "yhat_lower", "yhat_upper"])

# #     train = _build_pub_lags(pubs_df, lags=lags).dropna().copy()
# #     if len(train) < 4:
# #         return prophet_forecast_publications(pubs_df, horizon=horizon)

# #     train["ds"] = _to_year_end(train["year"])
# #     train = train.rename(columns={"pub_count": "y"})

# #     m = Prophet(yearly_seasonality=False, daily_seasonality=False)
# #     for k in lags:
# #         m.add_regressor(f"pub_lag{k}")
# #     m.fit(train[["ds", "y"] + [f"pub_lag{k}" for k in lags]])

# #     # iterative future, feed-forward yhat to build future lags
# #     last_year = int(pubs_df["year"].max())
# #     hist = pubs_df.set_index("year")["pub_count"].astype(float).to_dict()

# #     years, preds, lows, ups = [], [], [], []
# #     for i in range(1, horizon + 1):
# #         y = last_year + i
# #         row = {"ds": _to_year_end(pd.Series([y]))[0]}
# #         for k in lags:
# #             row[f"pub_lag{k}"] = hist.get(y - k, np.nan)
# #         fut = pd.DataFrame([row])[["ds"] + [f"pub_lag{k}" for k in lags]]
# #         fc = m.predict(fut)
# #         yhat = float(fc["yhat"].iloc[0])
# #         years.append(y); preds.append(yhat)
# #         lows.append(float(fc["yhat_lower"].iloc[0])); ups.append(float(fc["yhat_upper"].iloc[0]))
# #         hist[y] = yhat  # feed-forward

# #     return pd.DataFrame({
# #         "year": years,
# #         "pub_count_hat": preds,
# #         "yhat_lower": lows,
# #         "yhat_upper": ups,
# #     })



# # def prophet_forecast_publications(
# #     pubs_df: pd.DataFrame,
# #     horizon: int
# # ) -> pd.DataFrame:
# #     """Univariate Prophet on publications. Returns future rows (years, pub_count_hat)."""
# #     if pubs_df.empty:
# #         return pd.DataFrame(columns=["year", "pub_count_hat", "yhat_lower", "yhat_upper"])

# #     df = pubs_df.copy()
# #     df["ds"] = _to_year_end(df["year"])
# #     df = df.rename(columns={"pub_count": "y"})

# #     m = Prophet(yearly_seasonality=False, daily_seasonality=False)
# #     m.fit(df[["ds", "y"]])

# #     # Build future years explicitly
# #     last_year = int(df["ds"].dt.year.max())
# #     future_years = [last_year + i for i in range(1, horizon + 1)]
# #     future = pd.DataFrame({"ds": _to_year_end(pd.Series(future_years))})

# #     fcst = m.predict(future)
# #     out = pd.DataFrame({
# #         "year": fcst["ds"].dt.year.astype(int),
# #         "pub_count_hat": fcst["yhat"],
# #         "yhat_lower": fcst["yhat_lower"],
# #         "yhat_upper": fcst["yhat_upper"],
# #     })
# #     return out

# # def build_pub_reg(
# #     pubs_hist: pd.DataFrame,
# #     pubs_future: pd.DataFrame,
# #     lag: int,
# #     scale: float = 1.0
# # ) -> pd.DataFrame:
# #     """
# #     Build a unified publications series (history + future hats) and
# #     compute pub_reg at each year y as pubs[y - lag], then optionally scale.
# #     Returns: [year, pub_reg]
# #     """
# #     hist = pubs_hist.rename(columns={"pub_count": "pub"}).copy()
# #     fut  = pubs_future.rename(columns={"pub_count_hat": "pub"}).copy()
# #     both = pd.concat([hist[["year", "pub"]], fut[["year", "pub"]]], ignore_index=True).drop_duplicates("year")
# #     both = both.sort_values("year").reset_index(drop=True)

# #     reg = both.copy()
# #     reg["year"] = reg["year"].astype(int) + lag  # reg[t] = pub[t - lag]
# #     reg = reg.rename(columns={"pub": "pub_reg"})
# #     reg = reg[["year", "pub_reg"]]
# #     if scale != 1.0:
# #         reg["pub_reg"] = reg["pub_reg"].astype(float) * float(scale)
# #     return reg


# # def build_patent_lags(pats_df: pd.DataFrame, lags: tuple = (1, 2)) -> pd.DataFrame:
# #     """
# #     Add pat_lagK columns to a patents dataframe (cols: year, patent_count).
# #     Returns a copy with added lag columns.
# #     """
# #     out = pats_df.copy()
# #     for k in lags:
# #         out[f"pat_lag{k}"] = out["patent_count"].shift(k)
# #     return out

# # def prophet_patents_baseline_predict(
# #     pats_train: pd.DataFrame,
# #     predict_years: List[int]
# # ) -> np.ndarray:
# #     df = pats_train.copy()
# #     df["ds"] = _to_year_end(df["year"])
# #     df = df.rename(columns={"patent_count": "y"})
# #     m = Prophet(yearly_seasonality=False, daily_seasonality=False)
# #     m.fit(df[["ds", "y"]])

# #     fut = pd.DataFrame({"ds": _to_year_end(pd.Series(predict_years))})
# #     fc = m.predict(fut)
# #     return fc["yhat"].values

# # def _iterative_prophet_predict_with_regressors(
# #     model: Prophet,
# #     years: List[int],
# #     pub_reg_all: pd.DataFrame,       # cols: year, pub_reg for all needed future years
# #     seed_hist: dict,                  # {year: patent_count or predicted}
# #     lags: tuple = (1, 2),
# # ) -> np.ndarray:
# #     """
# #     Iteratively predict year-by-year using Prophet with multiple regressors.
# #     For each year y, we fetch pub_reg[y] and pat_lag1 = seed_hist[y-1], pat_lag2 = seed_hist[y-2], ...
# #     Append the new prediction to seed_hist so subsequent steps can use it.
# #     Returns an array of predictions aligned to 'years'.
# #     """
# #     preds = []
# #     for y in years:
# #         row = {"year": y, "ds": _to_year_end(pd.Series([y]))[0]}
# #         # publication regressor
# #         pr = pub_reg_all.loc[pub_reg_all["year"] == y, "pub_reg"]
# #         row["pub_reg"] = float(pr.iloc[0]) if len(pr) else np.nan
# #         # patent lag regressors
# #         for k in lags:
# #             row[f"pat_lag{k}"] = seed_hist.get(y - k, np.nan)

# #         fut = pd.DataFrame([row])
# #         # ensure regressor columns are present
# #         fut = fut[["ds", "pub_reg"] + [f"pat_lag{k}" for k in lags]]
# #         fc = model.predict(fut)
# #         yhat = float(fc["yhat"].iloc[0])
# #         preds.append(yhat)
# #         seed_hist[y] = yhat  # update for next step
# #     return np.array(preds)

# # def prophet_patents_with_pub_and_ar_predict(
# #     pats_train: pd.DataFrame,          # cols: year, patent_count
# #     pub_reg_train: pd.DataFrame,       # cols: year, pub_reg (train years only)
# #     predict_years: List[int],          # years to predict
# #     pub_reg_all: pd.DataFrame,         # cols: year, pub_reg (for all needed future years)
# #     seed_hist: dict,                   # {year: patent_count} includes at least last max(AR_LAGS) train years
# #     lags: tuple = AR_LAGS,
# # ) -> np.ndarray:
# #     """
# #     Train Prophet with regressors: pub_reg + patent lags (AR terms), then predict iteratively.
# #     """
# #     # Build TRAIN with AR lags
# #     train = build_patent_lags(pats_train, lags=lags)
# #     train = train.merge(pub_reg_train, on="year", how="left")
# #     needed_cols = ["year", "patent_count", "pub_reg"] + [f"pat_lag{k}" for k in lags]
# #     train = train.dropna(subset=needed_cols).copy()
# #     if train.empty or len(train) < 4:
# #         warnings.warn("Too few rows after adding lagged regressors; falling back to baseline.")
# #         return prophet_patents_baseline_predict(pats_train, predict_years)

# #     # Prophet-ready columns
# #     train["ds"] = _to_year_end(train["year"])
# #     train = train.rename(columns={"patent_count": "y"})

# #     # Fit Prophet with all regressors
# #     m = Prophet(yearly_seasonality=False, daily_seasonality=False)
# #     m.add_regressor("pub_reg")
# #     for k in lags:
# #         m.add_regressor(f"pat_lag{k}")
# #     m.fit(train[["ds", "y", "pub_reg"] + [f"pat_lag{k}" for k in lags]])
# #     print(train.shape)  # after `dropna` on lagged regressors

# #     # Iterative future predictions
# #     preds = _iterative_prophet_predict_with_regressors(
# #         model=m,
# #         years=predict_years,
# #         pub_reg_all=pub_reg_all,
# #         seed_hist=dict(seed_hist),  # copy to avoid side effects
# #         lags=lags
# #     )
# #     return preds

# # def mean_abs_pct_error(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
# #     """
# #     AMPE (%): 100 * mean( |y - yhat| / max(eps, |y|) )
# #     Only calculates mean for data points that have valid prediction values (non-NaN).
# #     eps avoids blow-ups when y==0 (common in sparse years).
# #     """
# #     y_true = np.asarray(y_true, dtype=float)
# #     y_pred = np.asarray(y_pred, dtype=float)
    
# #     # Create mask for valid predictions (non-NaN and finite)
# #     valid_mask = np.isfinite(y_pred) & np.isfinite(y_true)
    
# #     if not np.any(valid_mask):
# #         return np.nan
    
# #     # Filter to only valid data points
# #     y_true_valid = y_true[valid_mask]
# #     y_pred_valid = y_pred[valid_mask]
    
# #     denom = np.maximum(np.abs(y_true_valid), eps)
# #     return float(np.mean(np.abs(y_true_valid - y_pred_valid) / denom) * 100.0)


# # # ----------------------------
# # # Evaluation + Orchestration
# # # ----------------------------
# # @dataclass
# # class EvalResults:
# #     tech: str
# #     best_lag: int
# #     xcorr: float
# #     mae_pubs: float
# #     rmse_pubs: float
# #     mae_pat_no_reg: float
# #     rmse_pat_no_reg: float
# #     mae_pat_with_reg: float
# #     rmse_pat_with_reg: float
# #     ampe_pubs: float
# #     ampe_pat_no_reg: float
# #     ampe_pat_with_reg: float


# # def validate_horizon(horizon: int) -> int:
# #     """Validate and constrain forecast horizon."""
# #     if not isinstance(horizon, int):
# #         try:
# #             horizon = int(horizon)
# #         except (TypeError, ValueError):
# #             return HORIZON_DEFAULT
# #     return max(HORIZON_MIN, min(horizon, HORIZON_MAX))

# # def run_for_current_series(
# #     engine=None,
# #     tech_label: str = TECH_LABEL_DEFAULT,
# #     pub_tail_trunc: int = PUB_TAIL_TRUNC_DEFAULT,
# #     pat_tail_trunc: int = PAT_TAIL_TRUNC_DEFAULT,
# #     max_lag: int = MAX_LAG_DEFAULT,
# #     test_years: int = TEST_YEARS_DEFAULT,
# #     horizon: int = HORIZON_DEFAULT,
# #     pubs_error_threshold_rmse: Optional[float] = None,
# #     # flexible evaluation controls:
# #     split_year: Optional[int] = None,            # train ≤ split_year; test = next `test_years` years
# #     eval_start_year: Optional[int] = None,       # explicit eval window start (train ≤ start-1)
# #     eval_end_year: Optional[int] = None          # explicit eval window end (inclusive)
# # ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, EvalResults,
# #            pd.DataFrame, pd.DataFrame]:
# #     """
# #     Returns:
# #       pubs_forecasts (future horizon),
# #       patents_forecasts (future horizon),
# #       metrics_df (eval metrics),
# #       eval_summary (EvalResults),
# #       pubs_test_eval  (year, actual, yhat, yhat_lower, yhat_upper),
# #       pats_test_eval  (year, actual, yhat_baseline, yhat_with_pub_reg)

# #     Evaluation split priority:
# #       1) If eval_start_year (and optional eval_end_year) provided:
# #             train ≤ (eval_start_year - 1), test = [eval_start_year .. eval_end_year]
# #       2) Else if split_year provided:
# #             train ≤ split_year, test = next `test_years` consecutive years
# #       3) Else (fallback):
# #             test = last `test_years` years after tail truncation
# #     """
# #     # Validate horizon parameter
# #     horizon = validate_horizon(horizon)

# #     engine = engine or _require_engine()
# #     pubs, pats = fetch_current_series(engine, pub_tail_trunc, pat_tail_trunc)
# #     if pubs.empty or pats.empty:
# #         raise RuntimeError("Empty series after truncation; ensure the app populated research_data3 and raw_patents.")

# #     # Common span bounds
# #     y_min = max(int(pubs["year"].min()), int(pats["year"].min()))
# #     y_max = min(int(pubs["year"].max()), int(pats["year"].max()))

# #     def _clip_year(y: int) -> int:
# #         return max(y_min, min(y, y_max))

# #     # ----- Build TRAIN / TEST split -----
# #     if eval_start_year is not None:
# #         es = _clip_year(int(eval_start_year))
# #         ee = _clip_year(int(eval_end_year)) if (eval_end_year is not None) else _clip_year(es + max(1, int(test_years)) - 1)
# #         if es > ee:
# #             raise RuntimeError(f"Invalid eval range: start {es} > end {ee}")
# #         cutoff = _clip_year(es - 1)
# #         pub_train = pubs[pubs["year"] <= cutoff].copy()
# #         pat_train = pats[pats["year"] <= cutoff].copy()
# #         pub_test  = pubs[(pubs["year"] >= es) & (pubs["year"] <= ee)].copy()
# #         pat_test  = pats[(pats["year"] >= es) & (pats["year"] <= ee)].copy()
# #     elif split_year is not None:
# #         cutoff = _clip_year(int(split_year))
# #         next_years = list(range(cutoff + 1, min(y_max, cutoff + int(test_years)) + 1))
# #         pub_train = pubs[pubs["year"] <= cutoff].copy()
# #         pat_train = pats[pats["year"] <= cutoff].copy()
# #         pub_test  = pubs[pubs["year"].isin(next_years)].copy()
# #         pat_test  = pats[pats["year"].isin(next_years)].copy()
# #     else:
# #         cutoff = y_max - max(1, int(test_years))
# #         pub_train, pub_test = pubs[pubs["year"] <= cutoff].copy(), pubs[pubs["year"] > cutoff].copy()
# #         pat_train, pat_test = pats[pats["year"] <= cutoff].copy(), pats[pats["year"] > cutoff].copy()

# #     if pub_train.empty or pat_train.empty:
# #         raise RuntimeError("Training set is empty for the chosen split; try an earlier split_year or a shorter eval range.")

# #     # ----- (1) Publications forecast for TEST (evaluation path) -----
# #     if len(pub_test):
# #         pub_fc_test = prophet_forecast_publications_ar(pub_train, horizon=len(pub_test))
# #         pub_pred_test = pub_fc_test[pub_fc_test["year"].isin(pub_test["year"])].copy()
# #         mae_pubs  = mean_absolute_error(pub_test["pub_count"].values, pub_pred_test["pub_count_hat"].values)
# #         rmse_pubs = np.sqrt(mean_squared_error(pub_test["pub_count"].values, pub_pred_test["pub_count_hat"].values))
# #         ampe_pubs = mean_abs_pct_error(pub_test["pub_count"].values, pub_pred_test["pub_count_hat"].values)
# #         # Build test eval table for publications
# #         pubs_test_eval = (
# #             pub_test.merge(pub_pred_test, on="year", how="inner")
# #                     .rename(columns={"pub_count": "actual",
# #                                      "pub_count_hat": "yhat"})
# #                     .loc[:, ["year", "actual", "yhat", "yhat_lower", "yhat_upper"]]
# #                     .sort_values("year")
# #                     .reset_index(drop=True)
# #         )
# #     else:
# #         pub_fc_test = pd.DataFrame(columns=["year", "pub_count_hat", "yhat_lower", "yhat_upper"])
# #         mae_pubs = rmse_pubs = ampe_pubs = np.nan
# #         pubs_test_eval = pd.DataFrame(columns=["year", "actual", "yhat", "yhat_lower", "yhat_upper"])

# #     # ----- (2) Auto-lag selection (pubs -> patents) -----
# #     best_lag, xcorr, corr_map = infer_best_pub_to_patent_lag(
# #         pub_train, pat_train,
# #         max_lag=max_lag,
# #         detrend="diff",
# #         prewhiten="ar1",
# #         allow_zero_lag=True,
# #         zero_margin=0.05,
# #         ic_tiebreak=True
# #     )

# #     # small shrink if zero-lag narrowly wins
# #     pub_reg_scale = 1.0
# #     r0 = corr_map.get(0, np.nan); r1 = corr_map.get(1, np.nan)
# #     if best_lag == 0 and np.isfinite(r0) and np.isfinite(r1) and (r0 - r1) < 0.05:
# #         pub_reg_scale = 0.7

# #     # ----- (3) Patents baseline (no regressors) -----
# #     if len(pat_test):
# #         predict_years = pat_test["year"].tolist()
# #         pat_base_pred = prophet_patents_baseline_predict(pat_train, predict_years=predict_years)
# #         mae_base  = mean_absolute_error(pat_test["patent_count"].values, pat_base_pred)
# #         rmse_base = np.sqrt(mean_squared_error(pat_test["patent_count"].values, pat_base_pred))
# #         ampe_base = mean_abs_pct_error(pat_test["patent_count"].values, pat_base_pred)
# #     else:
# #         pat_base_pred = np.array([])
# #         mae_base = rmse_base = ampe_base = np.nan

# #     # ----- (4) Patents with lagged pubs + AR -----
# #     pub_reg_train     = build_pub_reg(pub_train, pubs_future=pd.DataFrame(columns=["year","pub_count_hat"]), lag=best_lag, scale=pub_reg_scale)
# #     pub_reg_test_full = build_pub_reg(pub_train, pub_fc_test, lag=best_lag, scale=pub_reg_scale)
# #     seed_hist_train   = pat_train.set_index("year")["patent_count"].to_dict()

# #     if len(pat_test):
# #         predict_years = pat_test["year"].tolist()
# #         pat_reg_pred = prophet_patents_with_pub_and_ar_predict(
# #             pats_train=pat_train,
# #             pub_reg_train=pub_reg_train,
# #             predict_years=predict_years,
# #             pub_reg_all=pub_reg_test_full,
# #             seed_hist=seed_hist_train,
# #             lags=AR_LAGS
# #         )
# #         mae_reg  = mean_absolute_error(pat_test["patent_count"].values, pat_reg_pred)
# #         rmse_reg = np.sqrt(mean_squared_error(pat_test["patent_count"].values, pat_reg_pred))
# #         ampe_reg = mean_abs_pct_error(pat_test["patent_count"].values, pat_reg_pred)
# #         # Build test eval table for patents
# #         pats_test_eval = pd.DataFrame({
# #             "year": predict_years,
# #             "actual": pat_test["patent_count"].values.astype(float),
# #             "yhat_baseline": pat_base_pred.astype(float),
# #             "yhat_with_pub_reg": pat_reg_pred.astype(float),
# #         }).sort_values("year").reset_index(drop=True)
# #     else:
# #         pat_reg_pred = np.array([])
# #         mae_reg = rmse_reg = ampe_reg = np.nan
# #         pats_test_eval = pd.DataFrame(columns=["year", "actual", "yhat_baseline", "yhat_with_pub_reg"])

# #     # ----- Summary for metrics -----
# #     eval_summary = EvalResults(
# #         tech=tech_label,
# #         best_lag=best_lag,
# #         xcorr=xcorr if xcorr is not None else np.nan,
# #         mae_pubs=mae_pubs, rmse_pubs=rmse_pubs,
# #         mae_pat_no_reg=mae_base, rmse_pat_no_reg=rmse_base,
# #         mae_pat_with_reg=mae_reg, rmse_pat_with_reg=rmse_reg,
# #         ampe_pubs=ampe_pubs, ampe_pat_no_reg=ampe_base, ampe_pat_with_reg=ampe_reg
# #     )

# #     # ========= Final production forecasts (full data) =========
# #     pub_fc_full = prophet_forecast_publications_ar(pubs, horizon=horizon)

# #     pub_reg_hist_full = build_pub_reg(pubs, pubs_future=pd.DataFrame(columns=["year","pub_count_hat"]), lag=best_lag, scale=pub_reg_scale)
# #     pub_reg_all_full  = build_pub_reg(pubs, pub_fc_full, lag=best_lag, scale=pub_reg_scale)
# #     seed_hist_full    = pats.set_index("year")["patent_count"].to_dict()
# #     future_years      = list(range(y_max + 1, y_max + 1 + horizon))

# #     pat_reg_future = prophet_patents_with_pub_and_ar_predict(
# #         pats_train=pats,
# #         pub_reg_train=pub_reg_hist_full,
# #         predict_years=future_years,
# #         pub_reg_all=pub_reg_all_full,
# #         seed_hist=seed_hist_full,
# #         lags=AR_LAGS
# #     )
# #     pat_base_future = prophet_patents_baseline_predict(pats, predict_years=future_years)

# #     pubs_forecasts = pub_fc_full.copy()
# #     pubs_forecasts.insert(0, "tech", tech_label)

# #     patents_forecasts = pd.DataFrame({
# #         "tech": tech_label,
# #         "year": future_years,
# #         "yhat_with_pub_reg": pat_reg_future,
# #         "yhat_baseline": pat_base_future
# #     })

# #     metrics_df = pd.DataFrame([{
# #         "tech": tech_label,
# #         "best_lag": best_lag,
# #         "xcorr": xcorr,
# #         "mae_pubs": mae_pubs,
# #         "rmse_pubs": rmse_pubs,
# #         "ampe_pubs": ampe_pubs,
# #         "mae_patents_with_reg": mae_reg,
# #         "rmse_patents_with_reg": rmse_reg,
# #         "ampe_patents_with_reg": ampe_reg,
# #         "mae_patents_baseline": mae_base,
# #         "rmse_patents_baseline": rmse_base,
# #         "ampe_patents_baseline": ampe_base
# #     }])

# #     return pubs_forecasts, patents_forecasts, metrics_df, eval_summary, pubs_test_eval, pats_test_eval




# # # ----------------------------
# # # CLI
# # # ----------------------------
# # def _fmt2(x):
# #     return "nan" if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x:.2f}"

# # if __name__ == "__main__":
# #     pubs_fc, pats_fc, metrics_df, summary, pubs_test_eval, pats_test_eval  = run_for_current_series()

# #     # pubs_fc.to_csv("publications_forecasts.csv", index=False)
# #     # pats_fc.to_csv("patents_forecasts_with_pub_reg.csv", index=False)
# #     # metrics_df.to_csv("prophet_eval_metrics.csv", index=False)
    
# #     print("publications forecsts :")
# #     print( pubs_fc)
# #     print('patents forecsts : ')
# #     print(pats_fc)
# #     print('metrics_df : ')
# #     print(metrics_df)
    

# #     print("\n=== Evaluation (last test years) ===")
# #     print(f"tech: {summary.tech}")

# #     xc = summary.xcorr
# #     xc_str = "nan" if (xc is None or (isinstance(xc, float) and math.isnan(xc))) else f"{xc:.3f}"
# #     print(f"auto-selected lag (pubs→patents): {summary.best_lag} years (xcorr={xc_str})")

# #     print(f"Publications  MAE={_fmt2(summary.mae_pubs)}  RMSE={_fmt2(summary.rmse_pubs)}")
# #     print(f"Patents base  MAE={_fmt2(summary.mae_pat_no_reg)}  RMSE={_fmt2(summary.rmse_pat_no_reg)}")
# #     print(f"Patents reg   MAE={_fmt2(summary.mae_pat_with_reg)}  RMSE={_fmt2(summary.rmse_pat_with_reg)}")




















# """
# Prophet single-series pipeline (current tech only) — with patent autocorrelation (Option A)

# - Pulls the CURRENT search's data from Postgres:
#     * publications: research_data3.year
#     * patents: raw_patents.first_filing_year
# - Truncates tail years to avoid reporting delays (patents: last 3y, pubs: last 1y by default)
# - Fills missing years with zeros
# - Auto-detects publication->patent lag via cross-correlation (detrended)
# - Forecasts publications (univariate Prophet)
# - Forecasts patents with:
#     * baseline (no regressors)
#     * lagged publications regressor (auto-selected lag)
#     * lagged patents (AR) as regressors: pat_lag1, pat_lag2  <-- Option A
# - Time-based evaluation on last N test years (after truncation)
# - Produces CSV artifacts and prints a concise summary

# Env:
#     DATABASE_URL=postgresql+psycopg2://user:pass@host:port/dbname

# Requires:
#     pip install prophet sqlalchemy psycopg2-binary pandas numpy scikit-learn python-dotenv
# """

# import os
# import math
# import warnings
# from dataclasses import dataclass
# from typing import Tuple, Optional, List
# from datetime import date
# import numpy as np
# import pandas as pd
# from dotenv import load_dotenv
# from sqlalchemy import create_engine
# from sqlalchemy.engine import Engine

# # Prophet import (new name 'prophet', fallback to 'fbprophet')
# try:
#     from prophet import Prophet
# except Exception:
#     from fbprophet import Prophet  # type: ignore

# from sklearn.metrics import mean_absolute_error, mean_squared_error
# import matplotlib.pyplot as plt
# from statsmodels.tsa.arima.model import ARIMA 
# # ----------------------------
# # Config defaults
# # ----------------------------
# PUB_TAIL_TRUNC_DEFAULT = 1   # drop last 1 y of publications (incomplete year)
# PAT_TAIL_TRUNC_DEFAULT = 3   # drop last 3 y of patents (filing delays)
# MAX_LAG_DEFAULT = 5          # search lags 0..5 for pubs->patents
# TEST_YEARS_DEFAULT = 5       # holdout length for evaluation
# HORIZON_MIN = 3              # Minimum forecast years
# HORIZON_MAX = 20             # Maximum forecast years
# HORIZON_DEFAULT = 5         # forecast years ahead
# TECH_LABEL_DEFAULT = "current"  # label for outputs when single series
# AR_LAGS = (1, 2)             # patent AR terms to include as regressors

# load_dotenv()

# # ----------------------------
# # Helpers
# # ----------------------------
# def _require_engine() -> Engine:
#     db_url = os.getenv("DATABASE_URL")
#     if not db_url:
#         raise RuntimeError("DATABASE_URL not set. Example: postgresql+psycopg2://user:pass@host:5432/db")
#     return create_engine(db_url)

# def _to_year_end(years: pd.Series) -> pd.Series:
#     """Map integer years to Dec-31 timestamps."""
#     return pd.PeriodIndex(years.astype(int), freq="Y").to_timestamp(how="end")

# def _fill_missing_years(df: pd.DataFrame, year_col: str, count_col: str) -> pd.DataFrame:
#     if df.empty:
#         return df
#     y_min, y_max = int(df[year_col].min()), int(df[year_col].max())
#     full = pd.DataFrame({year_col: list(range(y_min, y_max + 1))})
#     out = full.merge(df[[year_col, count_col]], on=year_col, how="left").fillna({count_col: 0})
#     out[count_col] = out[count_col].astype(int)
#     return out

# def _truncate_tail_years(df: pd.DataFrame, tail: int, year_col="year") -> pd.DataFrame:
#     if df.empty or tail <= 0:
#         return df.copy()
#     max_year = int(df[year_col].max())
#     cutoff = max_year - tail
#     return df[df[year_col] <= cutoff].copy()

# def _safe_corr(a: pd.Series, b: pd.Series) -> float:
#     if len(a) < 3 or len(b) < 3:
#         return np.nan
#     try:
#         return float(a.corr(b))
#     except Exception:
#         return np.nan


# def _ols_aic(x: pd.Series, y: pd.Series) -> float:
#     """
#     AIC for simple OLS: y = b0 + b1*x + e (Gaussian).
#     Returns +inf if not enough data.
#     """
#     x, y = x.dropna(), y.dropna()
#     idx = x.index.intersection(y.index)
#     x, y = x.loc[idx], y.loc[idx]
#     n = len(x)
#     if n < 3:
#         return float("inf")
#     X = np.column_stack([np.ones(n), x.values])
#     beta, *_ = np.linalg.lstsq(X, y.values, rcond=None)
#     resid = y.values - X.dot(beta)
#     rss = float(np.sum(resid ** 2))
#     k = 2  # intercept + slope
#     if rss <= 0:
#         return float("-inf")
#     sigma2 = rss / n
#     return n * np.log(sigma2) + 2 * k

# def _prewhiten_series_ar1(s: pd.Series) -> pd.Series:
#     """
#     AR(1) prewhitening: e_t = s_t - phi * s_{t-1}, where phi = corr(s_t, s_{t-1}).
#     """
#     s1 = s.shift(1).dropna()
#     s2 = s.loc[s1.index]
#     phi = _safe_corr(s2, s1)
#     if np.isnan(phi):
#         phi = 0.0
#     e = s2 - phi * s1
#     return e.dropna()

# def _prewhiten_series_arima(s: pd.Series) -> pd.Series:
#     """
#     ARIMA(1,1,0) residuals as prewhitened signal. Falls back to diff2 if statsmodels is unavailable.
#     """
#     if ARIMA is None or len(s) < 6:
#         return s.diff().diff().dropna()
#     try:
#         model = ARIMA(s.astype(float), order=(1, 1, 0))
#         res = model.fit(method_kwargs={"warn_convergence": False})
#         # ARIMA(1,1,0) residuals align to differenced series length
#         e = pd.Series(res.resid, index=res.resid.index)
#         return e.dropna()
#     except Exception:
#         return s.diff().diff().dropna()

# def _transform_for_xcorr(x: pd.Series, y: pd.Series, detrend: str = "diff", prewhiten: str = "none") -> Tuple[pd.Series, pd.Series]:
#     """
#     Apply detrending and/or prewhitening consistently to x and y, then align indexes.
#     detrend in {"none","diff","diff2","pct","zscore"}
#     prewhiten in {"none","ar1","arima"}
#     """
#     x = x.copy()
#     y = y.copy()

#     # Detrend / difference
#     if detrend == "diff":
#         x, y = x.diff(), y.diff()
#     elif detrend == "diff2":
#         x, y = x.diff().diff(), y.diff().diff()
#     elif detrend == "pct":
#         x = x.pct_change().replace([np.inf, -np.inf], np.nan)
#         y = y.pct_change().replace([np.inf, -np.inf], np.nan)
#     elif detrend == "zscore":
#         if x.std(ddof=0) > 0:
#             x = (x - x.mean()) / x.std(ddof=0)
#         if y.std(ddof=0) > 0:
#             y = (y - y.mean()) / y.std(ddof=0)
#     # else "none": leave as-is

#     x, y = x.dropna(), y.dropna()
#     idx = x.index.intersection(y.index)
#     x, y = x.loc[idx], y.loc[idx]

#     # Prewhiten
#     if prewhiten == "ar1":
#         x = _prewhiten_series_ar1(x)
#         y = _prewhiten_series_ar1(y)
#     elif prewhiten == "arima":
#         x = _prewhiten_series_arima(x)
#         y = _prewhiten_series_arima(y)

#     # Final align
#     x, y = x.dropna(), y.dropna()
#     idx = x.index.intersection(y.index)
#     return x.loc[idx], y.loc[idx]


# def infer_best_pub_to_patent_lag(
#     pubs_df: pd.DataFrame,
#     patents_df: pd.DataFrame,
#     max_lag: int = MAX_LAG_DEFAULT,
#     detrend: str = "diff",
#     min_overlap: int = 8,
#     prefer_nonnegative: bool = True,
#     prewhiten: str = "ar1",          # "none", "ar1", or "arima"
#     allow_zero_lag: bool = True,     # allow contemporaneous link in search
#     zero_margin: float = 0.05,       # r(0) must beat r(1) by > margin to prefer 0
#     ic_tiebreak: bool = True         # if r0≈r1, use AIC to choose; otherwise prefer lag>=1
# ) -> Tuple[int, float, dict]:
#     """
#     Choose lag L in [0..max_lag] maximizing corr( pubs[t], patents[t+L] ), after optional
#     detrending + prewhitening. Returns (best_lag, corr_at_best, corr_map).

#     - If allow_zero_lag is False, searches L>=1 only.
#     - If r(0) and r(1) are close (within zero_margin), prefer lag=1 via information criterion
#       (AIC) or by heuristic preference for causal lag.
#     """
#     pub = pubs_df.set_index("year")["pub_count"].astype(float)
#     pat = patents_df.set_index("year")["patent_count"].astype(float)
#     years = pub.index.intersection(pat.index)
#     if len(years) < min_overlap:
#         return 1, np.nan, {}

#     # search range
#     start_L = 0 if allow_zero_lag else 1
#     corr_at: dict = {}
#     best_lag, best_corr = start_L, -np.inf

#     for L in range(start_L, max_lag + 1):
#         # Align pubs[t] with patents[t+L]  => shift patents backward by L
#         aligned = pat.shift(-L)
#         df = pd.concat([pub, aligned], axis=1).dropna()
#         if len(df) < min_overlap:
#             continue

#         x_raw = df.iloc[:, 0]
#         y_raw = df.iloc[:, 1]

#         x, y = _transform_for_xcorr(x_raw, y_raw, detrend=detrend, prewhiten=prewhiten)
#         if len(x) < min_overlap or len(y) < min_overlap:
#             continue

#         r = _safe_corr(x, y)
#         if np.isnan(r):
#             continue
#         if prefer_nonnegative and r < 0:
#             continue

#         corr_at[L] = r
#         if r > best_corr:
#             best_corr = r
#             best_lag = L

#     if not corr_at:
#         return 1, np.nan, {}

#     # Zero-lag tie-breaks vs lag=1
#     if allow_zero_lag and 0 in corr_at and 1 in corr_at:
#         r0, r1 = corr_at[0], corr_at[1]
#         # if close, prefer causal (>=1) using AIC if enabled
#         if (not np.isnan(r0)) and (not np.isnan(r1)) and (r0 <= r1 + zero_margin):
#             if ic_tiebreak:
#                 # recompute aligned & transformed series for AIC tie-break
#                 aligned0 = pat.shift(0)
#                 df0 = pd.concat([pub, aligned0], axis=1).dropna()
#                 x0, y0 = _transform_for_xcorr(df0.iloc[:, 0], df0.iloc[:, 1], detrend=detrend, prewhiten=prewhiten)

#                 aligned1 = pat.shift(-1)
#                 df1 = pd.concat([pub, aligned1], axis=1).dropna()
#                 x1, y1 = _transform_for_xcorr(df1.iloc[:, 0], df1.iloc[:, 1], detrend=detrend, prewhiten=prewhiten)

#                 aic0 = _ols_aic(x0, y0)
#                 aic1 = _ols_aic(x1, y1)
#                 # prefer lower AIC; if tie or noisy, prefer causal lag=1
#                 if aic1 <= aic0 + 1e-6:
#                     best_lag, best_corr = 1, r1
#                 else:
#                     best_lag, best_corr = 0, r0
#             else:
#                 best_lag, best_corr = 1, r1

#     return best_lag, best_corr, corr_at


# # ----------------------------
# # Data access
# # ----------------------------
# def fetch_current_series(
#     engine,
#     pub_tail_trunc: int = PUB_TAIL_TRUNC_DEFAULT,
#     pat_tail_trunc: int = PAT_TAIL_TRUNC_DEFAULT
# ) -> Tuple[pd.DataFrame, pd.DataFrame]:
#     """
#     Pull the CURRENT tech series stored by your app:
#       - Publications from research_data3.year
#       - Patents from raw_patents.first_filing_year
#     Apply tail truncation and fill missing years.
#     Returns: pubs_df[year,pub_count], patents_df[year,patent_count]
#     """
#     pubs_sql = """
#         SELECT year::int AS year, COUNT(*)::int AS pub_count
#         FROM research_data3
#         WHERE year IS NOT NULL
#         GROUP BY year
#         ORDER BY year
#     """
#     pats_sql = """
#         SELECT first_filing_year::int AS year, COUNT(*)::int AS patent_count
#         FROM raw_patents
#         WHERE first_filing_year IS NOT NULL
#         GROUP BY first_filing_year
#         ORDER BY year
#     """
#     pubs = pd.read_sql(pubs_sql, engine)
#     pats = pd.read_sql(pats_sql, engine)

#     # Truncate tails
#     pubs = _truncate_tail_years(pubs, pub_tail_trunc, "year")
#     pats = _truncate_tail_years(pats, pat_tail_trunc, "year")

#     # Keep common span (helps stability)
#     if not pubs.empty and not pats.empty:
#         y_min = max(int(pubs["year"].min()), int(pats["year"].min()))
#         y_max = min(int(pubs["year"].max()), int(pats["year"].max()))
#         pubs = pubs[(pubs["year"] >= y_min) & (pubs["year"] <= y_max)]
#         pats = pats[(pats["year"] >= y_min) & (pats["year"] <= y_max)]

#     # Fill gaps with zeros
#     pubs = _fill_missing_years(pubs, "year", "pub_count")
#     pats = _fill_missing_years(pats, "year", "patent_count")
#     return pubs, pats

# # ----------------------------
# # Prophet pieces
# # ----------------------------

# def _build_pub_lags(pubs_df: pd.DataFrame, lags: tuple = (1, 2)) -> pd.DataFrame:
#     out = pubs_df.copy()
#     for k in lags:
#         out[f"pub_lag{k}"] = out["pub_count"].shift(k)
#     return out

# def prophet_forecast_publications_ar(
#     pubs_df: pd.DataFrame,
#     horizon: int,
#     lags: tuple = (1, 2)
# ) -> pd.DataFrame:
#     """
#     Prophet with publication self-lags as extra regressors.
#     Returns future rows: [year, pub_count_hat, yhat_lower, yhat_upper].
#     Falls back to univariate Prophet if too few rows remain after lagging.
#     """
#     if pubs_df.empty or horizon <= 0:
#         return pd.DataFrame(columns=["year", "pub_count_hat", "yhat_lower", "yhat_upper"])

#     train = _build_pub_lags(pubs_df, lags=lags).dropna().copy()
#     if len(train) < 4:
#         return prophet_forecast_publications(pubs_df, horizon=horizon)

#     train["ds"] = _to_year_end(train["year"])
#     train = train.rename(columns={"pub_count": "y"})

#     m = Prophet(yearly_seasonality=False, daily_seasonality=False)
#     for k in lags:
#         m.add_regressor(f"pub_lag{k}")
#     m.fit(train[["ds", "y"] + [f"pub_lag{k}" for k in lags]])

#     # iterative future, feed-forward yhat to build future lags
#     last_year = int(pubs_df["year"].max())
#     hist = pubs_df.set_index("year")["pub_count"].astype(float).to_dict()

#     years, preds, lows, ups = [], [], [], []
#     for i in range(1, horizon + 1):
#         y = last_year + i
#         row = {"ds": _to_year_end(pd.Series([y]))[0]}
#         for k in lags:
#             row[f"pub_lag{k}"] = hist.get(y - k, np.nan)
#         fut = pd.DataFrame([row])[["ds"] + [f"pub_lag{k}" for k in lags]]
#         fc = m.predict(fut)
#         yhat = float(fc["yhat"].iloc[0])
#         years.append(y); preds.append(yhat)
#         lows.append(float(fc["yhat_lower"].iloc[0])); ups.append(float(fc["yhat_upper"].iloc[0]))
#         hist[y] = yhat  # feed-forward

#     return pd.DataFrame({
#         "year": years,
#         "pub_count_hat": preds,
#         "yhat_lower": lows,
#         "yhat_upper": ups,
#     })



# def prophet_forecast_publications(
#     pubs_df: pd.DataFrame,
#     horizon: int
# ) -> pd.DataFrame:
#     """Univariate Prophet on publications. Returns future rows (years, pub_count_hat)."""
#     if pubs_df.empty:
#         return pd.DataFrame(columns=["year", "pub_count_hat", "yhat_lower", "yhat_upper"])

#     df = pubs_df.copy()
#     df["ds"] = _to_year_end(df["year"])
#     df = df.rename(columns={"pub_count": "y"})

#     m = Prophet(yearly_seasonality=False, daily_seasonality=False)
#     m.fit(df[["ds", "y"]])

#     # Build future years explicitly
#     last_year = int(df["ds"].dt.year.max())
#     future_years = [last_year + i for i in range(1, horizon + 1)]
#     future = pd.DataFrame({"ds": _to_year_end(pd.Series(future_years))})

#     fcst = m.predict(future)
#     out = pd.DataFrame({
#         "year": fcst["ds"].dt.year.astype(int),
#         "pub_count_hat": fcst["yhat"],
#         "yhat_lower": fcst["yhat_lower"],
#         "yhat_upper": fcst["yhat_upper"],
#     })
#     return out

# def build_pub_reg(
#     pubs_hist: pd.DataFrame,
#     pubs_future: pd.DataFrame,
#     lag: int,
#     scale: float = 1.0
# ) -> pd.DataFrame:
#     """
#     Build a unified publications series (history + future hats) and
#     compute pub_reg at each year y as pubs[y - lag], then optionally scale.
#     Returns: [year, pub_reg]
#     """
#     hist = pubs_hist.rename(columns={"pub_count": "pub"}).copy()
#     fut  = pubs_future.rename(columns={"pub_count_hat": "pub"}).copy()
#     both = pd.concat([hist[["year", "pub"]], fut[["year", "pub"]]], ignore_index=True).drop_duplicates("year")
#     both = both.sort_values("year").reset_index(drop=True)

#     reg = both.copy()
#     reg["year"] = reg["year"].astype(int) + lag  # reg[t] = pub[t - lag]
#     reg = reg.rename(columns={"pub": "pub_reg"})
#     reg = reg[["year", "pub_reg"]]
#     if scale != 1.0:
#         reg["pub_reg"] = reg["pub_reg"].astype(float) * float(scale)
#     return reg


# def build_patent_lags(pats_df: pd.DataFrame, lags: tuple = (1, 2)) -> pd.DataFrame:
#     """
#     Add pat_lagK columns to a patents dataframe (cols: year, patent_count).
#     Returns a copy with added lag columns.
#     """
#     out = pats_df.copy()
#     for k in lags:
#         out[f"pat_lag{k}"] = out["patent_count"].shift(k)
#     return out

# def prophet_patents_baseline_predict(
#     pats_train: pd.DataFrame,
#     predict_years: List[int]
# ) -> np.ndarray:
#     df = pats_train.copy()
#     df["ds"] = _to_year_end(df["year"])
#     df = df.rename(columns={"patent_count": "y"})
#     m = Prophet(yearly_seasonality=False, daily_seasonality=False)
#     m.fit(df[["ds", "y"]])

#     fut = pd.DataFrame({"ds": _to_year_end(pd.Series(predict_years))})
#     fc = m.predict(fut)
#     return fc["yhat"].values

# def _iterative_prophet_predict_with_regressors(
#     model: Prophet,
#     years: List[int],
#     pub_reg_all: pd.DataFrame,       # cols: year, pub_reg for all needed future years
#     seed_hist: dict,                  # {year: patent_count or predicted}
#     lags: tuple = (1, 2),
# ) -> np.ndarray:
#     """
#     Iteratively predict year-by-year using Prophet with multiple regressors.
#     For each year y, we fetch pub_reg[y] and pat_lag1 = seed_hist[y-1], pat_lag2 = seed_hist[y-2], ...
#     Append the new prediction to seed_hist so subsequent steps can use it.
#     Returns an array of predictions aligned to 'years'.
#     """
#     preds = []
#     for y in years:
#         row = {"year": y, "ds": _to_year_end(pd.Series([y]))[0]}
#         # publication regressor
#         pr = pub_reg_all.loc[pub_reg_all["year"] == y, "pub_reg"]
#         row["pub_reg"] = float(pr.iloc[0]) if len(pr) else np.nan
#         # patent lag regressors
#         for k in lags:
#             row[f"pat_lag{k}"] = seed_hist.get(y - k, np.nan)

#         fut = pd.DataFrame([row])
#         # ensure regressor columns are present
#         fut = fut[["ds", "pub_reg"] + [f"pat_lag{k}" for k in lags]]
#         fc = model.predict(fut)
#         yhat = float(fc["yhat"].iloc[0])
#         preds.append(yhat)
#         seed_hist[y] = yhat  # update for next step
#     return np.array(preds)

# def prophet_patents_with_pub_and_ar_predict(
#     pats_train: pd.DataFrame,          # cols: year, patent_count
#     pub_reg_train: pd.DataFrame,       # cols: year, pub_reg (train years only)
#     predict_years: List[int],          # years to predict
#     pub_reg_all: pd.DataFrame,         # cols: year, pub_reg (for all needed future years)
#     seed_hist: dict,                   # {year: patent_count} includes at least last max(AR_LAGS) train years
#     lags: tuple = AR_LAGS,
# ) -> np.ndarray:
#     """
#     Train Prophet with regressors: pub_reg + patent lags (AR terms), then predict iteratively.
#     """
#     # Build TRAIN with AR lags
#     train = build_patent_lags(pats_train, lags=lags)
#     train = train.merge(pub_reg_train, on="year", how="left")
#     needed_cols = ["year", "patent_count", "pub_reg"] + [f"pat_lag{k}" for k in lags]
#     train = train.dropna(subset=needed_cols).copy()
#     if train.empty or len(train) < 4:
#         warnings.warn("Too few rows after adding lagged regressors; falling back to baseline.")
#         return prophet_patents_baseline_predict(pats_train, predict_years)

#     # Prophet-ready columns
#     train["ds"] = _to_year_end(train["year"])
#     train = train.rename(columns={"patent_count": "y"})

#     # Fit Prophet with all regressors
#     m = Prophet(yearly_seasonality=False, daily_seasonality=False)
#     m.add_regressor("pub_reg")
#     for k in lags:
#         m.add_regressor(f"pat_lag{k}")
#     m.fit(train[["ds", "y", "pub_reg"] + [f"pat_lag{k}" for k in lags]])
#     print(train.shape)  # after `dropna` on lagged regressors

#     # Iterative future predictions
#     preds = _iterative_prophet_predict_with_regressors(
#         model=m,
#         years=predict_years,
#         pub_reg_all=pub_reg_all,
#         seed_hist=dict(seed_hist),  # copy to avoid side effects
#         lags=lags
#     )
#     return preds

# def mean_abs_pct_error(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
#     """
#     AMPE (%): 100 * mean( |y - yhat| / max(eps, |y|) )
#     Only calculates mean for data points that have valid prediction values (non-NaN).
#     eps avoids blow-ups when y==0 (common in sparse years).
#     """
#     y_true = np.asarray(y_true, dtype=float)
#     y_pred = np.asarray(y_pred, dtype=float)
    
#     # Create mask for valid predictions (non-NaN and finite)
#     valid_mask = np.isfinite(y_pred) & np.isfinite(y_true)
    
#     if not np.any(valid_mask):
#         return np.nan
    
#     # Filter to only valid data points
#     y_true_valid = y_true[valid_mask]
#     y_pred_valid = y_pred[valid_mask]
    
#     denom = np.maximum(np.abs(y_true_valid), eps)
#     return float(np.mean(np.abs(y_true_valid - y_pred_valid) / denom) * 100.0)


# # ----------------------------
# # Evaluation + Orchestration
# # ----------------------------
# @dataclass
# class EvalResults:
#     tech: str
#     best_lag: int
#     xcorr: float
#     mae_pubs: float
#     rmse_pubs: float
#     mae_pat_no_reg: float
#     rmse_pat_no_reg: float
#     mae_pat_with_reg: float
#     rmse_pat_with_reg: float
#     ampe_pubs: float
#     ampe_pat_no_reg: float
#     ampe_pat_with_reg: float


# def validate_horizon(horizon: int) -> int:
#     """Validate and constrain forecast horizon."""
#     if not isinstance(horizon, int):
#         try:
#             horizon = int(horizon)
#         except (TypeError, ValueError):
#             return HORIZON_DEFAULT
#     return max(HORIZON_MIN, min(horizon, HORIZON_MAX))

# def run_for_current_series(
#     engine=None,
#     tech_label: str = TECH_LABEL_DEFAULT,
#     pub_tail_trunc: int = PUB_TAIL_TRUNC_DEFAULT,
#     pat_tail_trunc: int = PAT_TAIL_TRUNC_DEFAULT,
#     max_lag: int = MAX_LAG_DEFAULT,
#     test_years: int = TEST_YEARS_DEFAULT,
#     horizon: int = HORIZON_DEFAULT,
#     pubs_error_threshold_rmse: Optional[float] = None,
#     # flexible evaluation controls:
#     split_year: Optional[int] = None,            # train ≤ split_year; test = next `test_years` years
#     eval_start_year: Optional[int] = None,       # explicit eval window start (train ≤ start-1)
#     eval_end_year: Optional[int] = None          # explicit eval window end (inclusive)
# ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, EvalResults,
#            pd.DataFrame, pd.DataFrame, float, int, int, str, float, int, int, str]:
#     """
#     Returns:
#       pubs_forecasts (future horizon),
#       patents_forecasts (future horizon),
#       metrics_df (eval metrics),
#       eval_summary (EvalResults),
#       pubs_test_eval  (year, actual, yhat, yhat_lower, yhat_upper),
#       pats_test_eval  (year, actual, yhat_baseline, yhat_with_pub_reg),
#       past_GR_percent, past_start_year, past_end_year, past_label,
#       curr_GR_percent, curr_start_year, curr_end_year, curr_label

#     Evaluation split priority:
#       1) If eval_start_year (and optional eval_end_year) provided:
#             train ≤ (eval_start_year - 1), test = [eval_start_year .. eval_end_year]
#       2) Else if split_year provided:
#             train ≤ split_year, test = next `test_years` consecutive years
#       3) Else (fallback):
#             test = last `test_years` years after tail truncation
#     """
#     # Validate horizon parameter
#     horizon = validate_horizon(horizon)

#     engine = engine or _require_engine()
#     pubs, pats = fetch_current_series(engine, pub_tail_trunc, pat_tail_trunc)
#     if pubs.empty or pats.empty:
#         raise RuntimeError("Empty series after truncation; ensure the app populated research_data3 and raw_patents.")

#     # Compute growth rates on patents
#     past_GR, past_y0, past_y1, past_label, curr_GR, curr_y0, curr_y1, curr_label = compute_patent_growth_from_counts(pats)

#     # Common span bounds
#     y_min = max(int(pubs["year"].min()), int(pats["year"].min()))
#     y_max = min(int(pubs["year"].max()), int(pats["year"].max()))

#     def _clip_year(y: int) -> int:
#         return max(y_min, min(y, y_max))

#     # ----- Build TRAIN / TEST split -----
#     if eval_start_year is not None:
#         es = _clip_year(int(eval_start_year))
#         ee = _clip_year(int(eval_end_year)) if (eval_end_year is not None) else _clip_year(es + max(1, int(test_years)) - 1)
#         if es > ee:
#             raise RuntimeError(f"Invalid eval range: start {es} > end {ee}")
#         cutoff = _clip_year(es - 1)
#         pub_train = pubs[pubs["year"] <= cutoff].copy()
#         pat_train = pats[pats["year"] <= cutoff].copy()
#         pub_test  = pubs[(pubs["year"] >= es) & (pubs["year"] <= ee)].copy()
#         pat_test  = pats[(pats["year"] >= es) & (pats["year"] <= ee)].copy()
#     elif split_year is not None:
#         cutoff = _clip_year(int(split_year))
#         next_years = list(range(cutoff + 1, min(y_max, cutoff + int(test_years)) + 1))
#         pub_train = pubs[pubs["year"] <= cutoff].copy()
#         pat_train = pats[pats["year"] <= cutoff].copy()
#         pub_test  = pubs[pubs["year"].isin(next_years)].copy()
#         pat_test  = pats[pats["year"].isin(next_years)].copy()
#     else:
#         cutoff = y_max - max(1, int(test_years))
#         pub_train, pub_test = pubs[pubs["year"] <= cutoff].copy(), pubs[pubs["year"] > cutoff].copy()
#         pat_train, pat_test = pats[pats["year"] <= cutoff].copy(), pats[pats["year"] > cutoff].copy()

#     if pub_train.empty or pat_train.empty:
#         raise RuntimeError("Training set is empty for the chosen split; try an earlier split_year or a shorter eval range.")

#     # ----- (1) Publications forecast for TEST (evaluation path) -----
#     if len(pub_test):
#         pub_fc_test = prophet_forecast_publications_ar(pub_train, horizon=len(pub_test))
#         pub_pred_test = pub_fc_test[pub_fc_test["year"].isin(pub_test["year"])].copy()
#         mae_pubs  = mean_absolute_error(pub_test["pub_count"].values, pub_pred_test["pub_count_hat"].values)
#         rmse_pubs = np.sqrt(mean_squared_error(pub_test["pub_count"].values, pub_pred_test["pub_count_hat"].values))
#         ampe_pubs = mean_abs_pct_error(pub_test["pub_count"].values, pub_pred_test["pub_count_hat"].values)
#         # Build test eval table for publications
#         pubs_test_eval = (
#             pub_test.merge(pub_pred_test, on="year", how="inner")
#                     .rename(columns={"pub_count": "actual",
#                                      "pub_count_hat": "yhat"})
#                     .loc[:, ["year", "actual", "yhat", "yhat_lower", "yhat_upper"]]
#                     .sort_values("year")
#                     .reset_index(drop=True)
#         )
#     else:
#         pub_fc_test = pd.DataFrame(columns=["year", "pub_count_hat", "yhat_lower", "yhat_upper"])
#         mae_pubs = rmse_pubs = ampe_pubs = np.nan
#         pubs_test_eval = pd.DataFrame(columns=["year", "actual", "yhat", "yhat_lower", "yhat_upper"])

#     # ----- (2) Auto-lag selection (pubs -> patents) -----
#     best_lag, xcorr, corr_map = infer_best_pub_to_patent_lag(
#         pub_train, pat_train,
#         max_lag=max_lag,
#         detrend="diff",
#         prewhiten="ar1",
#         allow_zero_lag=True,
#         zero_margin=0.05,
#         ic_tiebreak=True
#     )

#     # small shrink if zero-lag narrowly wins
#     pub_reg_scale = 1.0
#     r0 = corr_map.get(0, np.nan); r1 = corr_map.get(1, np.nan)
#     if best_lag == 0 and np.isfinite(r0) and np.isfinite(r1) and (r0 - r1) < 0.05:
#         pub_reg_scale = 0.7

#     # ----- (3) Patents baseline (no regressors) -----
#     if len(pat_test):
#         predict_years = pat_test["year"].tolist()
#         pat_base_pred = prophet_patents_baseline_predict(pat_train, predict_years=predict_years)
#         mae_base  = mean_absolute_error(pat_test["patent_count"].values, pat_base_pred)
#         rmse_base = np.sqrt(mean_squared_error(pat_test["patent_count"].values, pat_base_pred))
#         ampe_base = mean_abs_pct_error(pat_test["patent_count"].values, pat_base_pred)
#     else:
#         pat_base_pred = np.array([])
#         mae_base = rmse_base = ampe_base = np.nan

#     # ----- (4) Patents with lagged pubs + AR -----
#     pub_reg_train     = build_pub_reg(pub_train, pubs_future=pd.DataFrame(columns=["year","pub_count_hat"]), lag=best_lag, scale=pub_reg_scale)
#     pub_reg_test_full = build_pub_reg(pub_train, pub_fc_test, lag=best_lag, scale=pub_reg_scale)
#     seed_hist_train   = pat_train.set_index("year")["patent_count"].to_dict()

#     if len(pat_test):
#         predict_years = pat_test["year"].tolist()
#         pat_reg_pred = prophet_patents_with_pub_and_ar_predict(
#             pats_train=pat_train,
#             pub_reg_train=pub_reg_train,
#             predict_years=predict_years,
#             pub_reg_all=pub_reg_test_full,
#             seed_hist=seed_hist_train,
#             lags=AR_LAGS
#         )
#         mae_reg  = mean_absolute_error(pat_test["patent_count"].values, pat_reg_pred)
#         rmse_reg = np.sqrt(mean_squared_error(pat_test["patent_count"].values, pat_reg_pred))
#         ampe_reg = mean_abs_pct_error(pat_test["patent_count"].values, pat_reg_pred)
#         # Build test eval table for patents
#         pats_test_eval = pd.DataFrame({
#             "year": predict_years,
#             "actual": pat_test["patent_count"].values.astype(float),
#             "yhat_baseline": pat_base_pred.astype(float),
#             "yhat_with_pub_reg": pat_reg_pred.astype(float),
#         }).sort_values("year").reset_index(drop=True)
#     else:
#         pat_reg_pred = np.array([])
#         mae_reg = rmse_reg = ampe_reg = np.nan
#         pats_test_eval = pd.DataFrame(columns=["year", "actual", "yhat_baseline", "yhat_with_pub_reg"])

#     # ----- Summary for metrics -----
#     eval_summary = EvalResults(
#         tech=tech_label,
#         best_lag=best_lag,
#         xcorr=xcorr if xcorr is not None else np.nan,
#         mae_pubs=mae_pubs, rmse_pubs=rmse_pubs,
#         mae_pat_no_reg=mae_base, rmse_pat_no_reg=rmse_base,
#         mae_pat_with_reg=mae_reg, rmse_pat_with_reg=rmse_reg,
#         ampe_pubs=ampe_pubs, ampe_pat_no_reg=ampe_base, ampe_pat_with_reg=ampe_reg
#     )

#     # ========= Final production forecasts (full data) =========
#     pub_fc_full = prophet_forecast_publications_ar(pubs, horizon=horizon)

#     pub_reg_hist_full = build_pub_reg(pubs, pubs_future=pd.DataFrame(columns=["year","pub_count_hat"]), lag=best_lag, scale=pub_reg_scale)
#     pub_reg_all_full  = build_pub_reg(pubs, pub_fc_full, lag=best_lag, scale=pub_reg_scale)
#     seed_hist_full    = pats.set_index("year")["patent_count"].to_dict()
#     future_years      = list(range(y_max + 1, y_max + 1 + horizon))

#     pat_reg_future = prophet_patents_with_pub_and_ar_predict(
#         pats_train=pats,
#         pub_reg_train=pub_reg_hist_full,
#         predict_years=future_years,
#         pub_reg_all=pub_reg_all_full,
#         seed_hist=seed_hist_full,
#         lags=AR_LAGS
#     )
#     pat_base_future = prophet_patents_baseline_predict(pats, predict_years=future_years)

#     pubs_forecasts = pub_fc_full.copy()
#     pubs_forecasts.insert(0, "tech", tech_label)

#     patents_forecasts = pd.DataFrame({
#         "tech": tech_label,
#         "year": future_years,
#         "yhat_with_pub_reg": pat_reg_future,
#         "yhat_baseline": pat_base_future
#     })

#     metrics_df = pd.DataFrame([{
#         "tech": tech_label,
#         "best_lag": best_lag,
#         "xcorr": xcorr,
#         "mae_pubs": mae_pubs,
#         "rmse_pubs": rmse_pubs,
#         "ampe_pubs": ampe_pubs,
#         "mae_patents_with_reg": mae_reg,
#         "rmse_patents_with_reg": rmse_reg,
#         "ampe_patents_with_reg": ampe_reg,
#         "mae_patents_baseline": mae_base,
#         "rmse_patents_baseline": rmse_base,
#         "ampe_patents_baseline": ampe_base
#     }])

#     return pubs_forecasts, patents_forecasts, metrics_df, eval_summary, pubs_test_eval, pats_test_eval, past_GR, past_y0, past_y1, past_label, curr_GR, curr_y0, curr_y1, curr_label




# # ----------------------------
# # CLI
# # ----------------------------
# def _fmt2(x):
#     return "nan" if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x:.2f}"

# if __name__ == "__main__":
#     pubs_fc, pats_fc, metrics_df, summary, pubs_test_eval, pats_test_eval, past_GR, past_y0, past_y1, past_label, curr_GR, curr_y0, curr_y1, curr_label  = run_for_current_series()

#     # pubs_fc.to_csv("publications_forecasts.csv", index=False)
#     # pats_fc.to_csv("patents_forecasts_with_pub_reg.csv", index=False)
#     # metrics_df.to_csv("prophet_eval_metrics.csv", index=False)
    
#     print("publications forecsts :")
#     print( pubs_fc)
#     print('patents forecsts : ')
#     print(pats_fc)
#     print('metrics_df : ')
#     print(metrics_df)
    

#     print("\n=== Evaluation (last test years) ===")
#     print(f"tech: {summary.tech}")

#     xc = summary.xcorr
#     xc_str = "nan" if (xc is None or (isinstance(xc, float) and math.isnan(xc))) else f"{xc:.3f}"
#     print(f"auto-selected lag (pubs→patents): {summary.best_lag} years (xcorr={xc_str})")

#     print(f"Publications  MAE={_fmt2(summary.mae_pubs)}  RMSE={_fmt2(summary.rmse_pubs)}")
#     print(f"Patents base  MAE={_fmt2(summary.mae_pat_no_reg)}  RMSE={_fmt2(summary.rmse_pat_no_reg)}")
#     print(f"Patents reg   MAE={_fmt2(summary.mae_pat_with_reg)}  RMSE={_fmt2(summary.rmse_pat_with_reg)}")

#     print(f"\nPast Growth Rate: {past_GR:.2f}% ({past_label}) from {past_y0} to {past_y1}")
#     print(f"Current Growth Rate: {curr_GR:.2f}% ({curr_label}) from {curr_y0} to {curr_y1}")

# def compute_patent_growth_from_counts(pats_counts_df: pd.DataFrame):
#     """
#     pats_counts_df: columns ['year', 'patent_count'] (ascending or not; we'll sort)
#     Implements the same GR formula as growth_rate.py over the past window
#       [current_year-7 .. current_year-2] (inclusive) and the current window
#       [current_year-2 .. current_year+2] (inclusive).
#     Returns: (past_GR_percent: float, past_start_year: int, past_end_year: int, past_label: str,
#               curr_GR_percent: float, curr_start_year: int, curr_end_year: int, curr_label: str)
#     """
#     if pats_counts_df is None or len(pats_counts_df) == 0:
#         return float("nan"), None, None, "unknown", float("nan"), None, None, "unknown"

#     df = pats_counts_df.rename(columns={"year": "first_filing_year", "patent_count": "Patent Count"}) \
#                        .loc[:, ["first_filing_year", "Patent Count"]] \
#                        .sort_values("first_filing_year")

#     # cumulative + per-year GR = (ΔX) / avg cumulative
#     df["Cumulative Count"] = df["Patent Count"].cumsum()
#     X = df["Patent Count"].astype(float)
#     T = df["Cumulative Count"].astype(float)
#     denom = (T + T.shift(1)) / 2.0
#     gr = (X - X.shift(1)) / denom
#     gr = gr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
#     df["GR"] = gr

#     current_year = date.today().year

#     # Past window: [cur-7 to cur-2]
#     past_end_y = current_year - 2
#     past_start_y = past_end_y - 5
#     past_y0 = max(past_start_y, int(df["first_filing_year"].min()))
#     past_y1 = min(past_end_y, int(df["first_filing_year"].max()))
#     past_mask = (df["first_filing_year"] >= past_y0) & (df["first_filing_year"] <= past_y1)
#     past_GR = float(df.loc[past_mask, "GR"].sum() * 100.0)

#     if past_GR >= 50:
#         past_label = "Booming"
#     elif 20 <= past_GR < 50:
#         past_label = "Trending"
#     elif 10 <= past_GR < 20:
#         past_label = "Quite_Trending"
#     elif 0 <= past_GR < 10:
#         past_label = "Steady"
#     else:
#         past_label = "Declining"

#     # Current window: [cur-2 to cur+2] (actual + forecast)
#     curr_start_y = current_year - 2
#     curr_end_y = current_year + 2
#     curr_y0 = max(curr_start_y, int(df["first_filing_year"].min()))
#     curr_y1 = min(curr_end_y, int(df["first_filing_year"].max()))
#     curr_mask = (df["first_filing_year"] >= curr_y0) & (df["first_filing_year"] <= curr_y1)
#     curr_GR = float(df.loc[curr_mask, "GR"].sum() * 100.0)

#     if curr_GR >= 50:
#         curr_label = "Booming"
#     elif 20 <= curr_GR < 50:
#         curr_label = "Trending"
#     elif 10 <= curr_GR < 20:
#         curr_label = "Quite_Trending"
#     elif 0 <= curr_GR < 10:
#         curr_label = "Steady"
#     else:
#         curr_label = "Declining"

#     return past_GR, past_y0, past_y1, past_label, curr_GR, curr_y0, curr_y1, curr_label

# def patent_growth_summary(df):
#     current_year = date.today().year
#     start_year = current_year - 2
#     end_year = start_year - 5
#     # Group by year and count patents
#     patent_counts = df.groupby('first_filing_year').size().reset_index(name='Patent Count')
    
#     # Ensure the DataFrame is sorted by year in ascending order for cumulative calculations
#     patent_counts = patent_counts.sort_values('first_filing_year')
    
#     # Calculate cumulative patent count
#     patent_counts['Cumulative Count'] = patent_counts['Patent Count'].cumsum()
    
#     # Calculate growth rate
#     X = patent_counts['Patent Count']
#     T = patent_counts['Cumulative Count']
#     patent_counts['GR'] = ((X - X.shift(1)) / ((T + T.shift(1)) / 2)).fillna(0)
#     patent_counts['GR'] = patent_counts['GR'].fillna(0)
    
#     # Sort by year descending and select top 10
#     patent_counts_sorted = patent_counts.sort_values('first_filing_year', ascending=False).head(10)

#     df_2018_2023 = patent_counts[(patent_counts['first_filing_year'] >= end_year) & (patent_counts['first_filing_year'] <= start_year)]

#     # Sum the annual growth rates (GR) for the period
#     GR = df_2018_2023['GR'].sum()*100
#     if GR >=50:
#       print ("the technology is Booming")
#     elif 20 <= GR < 50:
#       print ("the technology is Trending")
#     elif 10 <= GR < 20:
#       print ("the technology is Quite_Trending")
#     elif 0 <= GR < 10:
#       print ("the technology is Steady")
#     elif GR < 0:
#       print ("the technology is Declining")
    

#     # Return selected columns
#     return GR , start_year, end_year
  
  
  
  


# def patent_current_growth_rate(df: pd.DataFrame):
#     """
#     Compute summed growth rate over a 5-year window centered on current year:
#       window = [current_year - 2, ..., current_year + 2]

#     Returns:
#         GR_percent, window_start_year, window_end_year
#     """
#     current_year = date.today().year
#     start_year = current_year - 2
#     end_year = current_year + 2

#     # Count patents per year
#     patent_counts = (
#         df.groupby('first_filing_year')
#           .size()
#           .reset_index(name='Patent Count')
#           .sort_values('first_filing_year', ascending=True)
#           .reset_index(drop=True)
#     )

#     # Cumulative count
#     patent_counts['Cumulative Count'] = patent_counts['Patent Count'].cumsum()

#     # Growth rate per year: (ΔX) / avg cumulative
#     X = patent_counts['Patent Count'].astype(float)
#     T = patent_counts['Cumulative Count'].astype(float)
#     denom = (T + T.shift(1)) / 2.0
#     # avoid division by zero / NaN
#     gr = (X - X.shift(1)) / denom
#     gr = gr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
#     patent_counts['GR'] = gr

#     # 5-year window centered on current year
#     mask = (patent_counts['first_filing_year'] >= start_year) & \
#            (patent_counts['first_filing_year'] <= end_year)
#     GR = float(patent_counts.loc[mask, 'GR'].sum() * 100.0)  # percentage

#     # quick label
#     if GR >= 50:
#         print("the technology is Booming")
#     elif 20 <= GR < 50:
#         print("the technology is Trending")
#     elif 10 <= GR < 20:
#         print("the technology is Quite_Trending")
#     elif 0 <= GR < 10:
#         print("the technology is Steady")
#     else:
#         print("the technology is Declining")

#     return GR, start_year, end_year













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
from datetime import date
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
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA 
# ----------------------------
# Config defaults
# ----------------------------
PUB_TAIL_TRUNC_DEFAULT = 1   # drop last 1 y of publications (incomplete year)
PAT_TAIL_TRUNC_DEFAULT = 3   # drop last 3 y of patents (filing delays)
MAX_LAG_DEFAULT = 5          # search lags 0..5 for pubs->patents
TEST_YEARS_DEFAULT = 5       # holdout length for evaluation
HORIZON_MIN = 3              # Minimum forecast years
HORIZON_MAX = 20             # Maximum forecast years
HORIZON_DEFAULT = 5         # forecast years ahead
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


def _ols_aic(x: pd.Series, y: pd.Series) -> float:
    """
    AIC for simple OLS: y = b0 + b1*x + e (Gaussian).
    Returns +inf if not enough data.
    """
    x, y = x.dropna(), y.dropna()
    idx = x.index.intersection(y.index)
    x, y = x.loc[idx], y.loc[idx]
    n = len(x)
    if n < 3:
        return float("inf")
    X = np.column_stack([np.ones(n), x.values])
    beta, *_ = np.linalg.lstsq(X, y.values, rcond=None)
    resid = y.values - X.dot(beta)
    rss = float(np.sum(resid ** 2))
    k = 2  # intercept + slope
    if rss <= 0:
        return float("-inf")
    sigma2 = rss / n
    return n * np.log(sigma2) + 2 * k

def _prewhiten_series_ar1(s: pd.Series) -> pd.Series:
    """
    AR(1) prewhitening: e_t = s_t - phi * s_{t-1}, where phi = corr(s_t, s_{t-1}).
    """
    s1 = s.shift(1).dropna()
    s2 = s.loc[s1.index]
    phi = _safe_corr(s2, s1)
    if np.isnan(phi):
        phi = 0.0
    e = s2 - phi * s1
    return e.dropna()

def _prewhiten_series_arima(s: pd.Series) -> pd.Series:
    """
    ARIMA(1,1,0) residuals as prewhitened signal. Falls back to diff2 if statsmodels is unavailable.
    """
    if ARIMA is None or len(s) < 6:
        return s.diff().diff().dropna()
    try:
        model = ARIMA(s.astype(float), order=(1, 1, 0))
        res = model.fit(method_kwargs={"warn_convergence": False})
        # ARIMA(1,1,0) residuals align to differenced series length
        e = pd.Series(res.resid, index=res.resid.index)
        return e.dropna()
    except Exception:
        return s.diff().diff().dropna()

def _transform_for_xcorr(x: pd.Series, y: pd.Series, detrend: str = "diff", prewhiten: str = "none") -> Tuple[pd.Series, pd.Series]:
    """
    Apply detrending and/or prewhitening consistently to x and y, then align indexes.
    detrend in {"none","diff","diff2","pct","zscore"}
    prewhiten in {"none","ar1","arima"}
    """
    x = x.copy()
    y = y.copy()

    # Detrend / difference
    if detrend == "diff":
        x, y = x.diff(), y.diff()
    elif detrend == "diff2":
        x, y = x.diff().diff(), y.diff().diff()
    elif detrend == "pct":
        x = x.pct_change().replace([np.inf, -np.inf], np.nan)
        y = y.pct_change().replace([np.inf, -np.inf], np.nan)
    elif detrend == "zscore":
        if x.std(ddof=0) > 0:
            x = (x - x.mean()) / x.std(ddof=0)
        if y.std(ddof=0) > 0:
            y = (y - y.mean()) / y.std(ddof=0)
    # else "none": leave as-is

    x, y = x.dropna(), y.dropna()
    idx = x.index.intersection(y.index)
    x, y = x.loc[idx], y.loc[idx]

    # Prewhiten
    if prewhiten == "ar1":
        x = _prewhiten_series_ar1(x)
        y = _prewhiten_series_ar1(y)
    elif prewhiten == "arima":
        x = _prewhiten_series_arima(x)
        y = _prewhiten_series_arima(y)

    # Final align
    x, y = x.dropna(), y.dropna()
    idx = x.index.intersection(y.index)
    return x.loc[idx], y.loc[idx]


def infer_best_pub_to_patent_lag(
    pubs_df: pd.DataFrame,
    patents_df: pd.DataFrame,
    max_lag: int = MAX_LAG_DEFAULT,
    detrend: str = "diff",
    min_overlap: int = 8,
    prefer_nonnegative: bool = True,
    prewhiten: str = "ar1",          # "none", "ar1", or "arima"
    allow_zero_lag: bool = True,     # allow contemporaneous link in search
    zero_margin: float = 0.05,       # r(0) must beat r(1) by > margin to prefer 0
    ic_tiebreak: bool = True         # if r0≈r1, use AIC to choose; otherwise prefer lag>=1
) -> Tuple[int, float, dict]:
    """
    Choose lag L in [0..max_lag] maximizing corr( pubs[t], patents[t+L] ), after optional
    detrending + prewhitening. Returns (best_lag, corr_at_best, corr_map).

    - If allow_zero_lag is False, searches L>=1 only.
    - If r(0) and r(1) are close (within zero_margin), prefer lag=1 via information criterion
      (AIC) or by heuristic preference for causal lag.
    """
    pub = pubs_df.set_index("year")["pub_count"].astype(float)
    pat = patents_df.set_index("year")["patent_count"].astype(float)
    years = pub.index.intersection(pat.index)
    if len(years) < min_overlap:
        return 1, np.nan, {}

    # search range
    start_L = 0 if allow_zero_lag else 1
    corr_at: dict = {}
    best_lag, best_corr = start_L, -np.inf

    for L in range(start_L, max_lag + 1):
        # Align pubs[t] with patents[t+L]  => shift patents backward by L
        aligned = pat.shift(-L)
        df = pd.concat([pub, aligned], axis=1).dropna()
        if len(df) < min_overlap:
            continue

        x_raw = df.iloc[:, 0]
        y_raw = df.iloc[:, 1]

        x, y = _transform_for_xcorr(x_raw, y_raw, detrend=detrend, prewhiten=prewhiten)
        if len(x) < min_overlap or len(y) < min_overlap:
            continue

        r = _safe_corr(x, y)
        if np.isnan(r):
            continue
        if prefer_nonnegative and r < 0:
            continue

        corr_at[L] = r
        if r > best_corr:
            best_corr = r
            best_lag = L

    if not corr_at:
        return 1, np.nan, {}

    # Zero-lag tie-breaks vs lag=1
    if allow_zero_lag and 0 in corr_at and 1 in corr_at:
        r0, r1 = corr_at[0], corr_at[1]
        # if close, prefer causal (>=1) using AIC if enabled
        if (not np.isnan(r0)) and (not np.isnan(r1)) and (r0 <= r1 + zero_margin):
            if ic_tiebreak:
                # recompute aligned & transformed series for AIC tie-break
                aligned0 = pat.shift(0)
                df0 = pd.concat([pub, aligned0], axis=1).dropna()
                x0, y0 = _transform_for_xcorr(df0.iloc[:, 0], df0.iloc[:, 1], detrend=detrend, prewhiten=prewhiten)

                aligned1 = pat.shift(-1)
                df1 = pd.concat([pub, aligned1], axis=1).dropna()
                x1, y1 = _transform_for_xcorr(df1.iloc[:, 0], df1.iloc[:, 1], detrend=detrend, prewhiten=prewhiten)

                aic0 = _ols_aic(x0, y0)
                aic1 = _ols_aic(x1, y1)
                # prefer lower AIC; if tie or noisy, prefer causal lag=1
                if aic1 <= aic0 + 1e-6:
                    best_lag, best_corr = 1, r1
                else:
                    best_lag, best_corr = 0, r0
            else:
                best_lag, best_corr = 1, r1

    return best_lag, best_corr, corr_at


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

def _build_pub_lags(pubs_df: pd.DataFrame, lags: tuple = (1, 2)) -> pd.DataFrame:
    out = pubs_df.copy()
    for k in lags:
        out[f"pub_lag{k}"] = out["pub_count"].shift(k)
    return out

def prophet_forecast_publications_ar(
    pubs_df: pd.DataFrame,
    horizon: int,
    lags: tuple = (1, 2)
) -> pd.DataFrame:
    """
    Prophet with publication self-lags as extra regressors.
    Returns future rows: [year, pub_count_hat, yhat_lower, yhat_upper].
    Falls back to univariate Prophet if too few rows remain after lagging.
    """
    if pubs_df.empty or horizon <= 0:
        return pd.DataFrame(columns=["year", "pub_count_hat", "yhat_lower", "yhat_upper"])

    train = _build_pub_lags(pubs_df, lags=lags).dropna().copy()
    if len(train) < 4:
        return prophet_forecast_publications(pubs_df, horizon=horizon)

    train["ds"] = _to_year_end(train["year"])
    train = train.rename(columns={"pub_count": "y"})

    m = Prophet(yearly_seasonality=False, daily_seasonality=False)
    for k in lags:
        m.add_regressor(f"pub_lag{k}")
    m.fit(train[["ds", "y"] + [f"pub_lag{k}" for k in lags]])

    # iterative future, feed-forward yhat to build future lags
    last_year = int(pubs_df["year"].max())
    hist = pubs_df.set_index("year")["pub_count"].astype(float).to_dict()

    years, preds, lows, ups = [], [], [], []
    for i in range(1, horizon + 1):
        y = last_year + i
        row = {"ds": _to_year_end(pd.Series([y]))[0]}
        for k in lags:
            row[f"pub_lag{k}"] = hist.get(y - k, np.nan)
        fut = pd.DataFrame([row])[["ds"] + [f"pub_lag{k}" for k in lags]]
        fc = m.predict(fut)
        yhat = float(fc["yhat"].iloc[0])
        years.append(y); preds.append(yhat)
        lows.append(float(fc["yhat_lower"].iloc[0])); ups.append(float(fc["yhat_upper"].iloc[0]))
        hist[y] = yhat  # feed-forward

    return pd.DataFrame({
        "year": years,
        "pub_count_hat": preds,
        "yhat_lower": lows,
        "yhat_upper": ups,
    })



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
    lag: int,
    scale: float = 1.0
) -> pd.DataFrame:
    """
    Build a unified publications series (history + future hats) and
    compute pub_reg at each year y as pubs[y - lag], then optionally scale.
    Returns: [year, pub_reg]
    """
    hist = pubs_hist.rename(columns={"pub_count": "pub"}).copy()
    fut  = pubs_future.rename(columns={"pub_count_hat": "pub"}).copy()
    both = pd.concat([hist[["year", "pub"]], fut[["year", "pub"]]], ignore_index=True).drop_duplicates("year")
    both = both.sort_values("year").reset_index(drop=True)

    reg = both.copy()
    reg["year"] = reg["year"].astype(int) + lag  # reg[t] = pub[t - lag]
    reg = reg.rename(columns={"pub": "pub_reg"})
    reg = reg[["year", "pub_reg"]]
    if scale != 1.0:
        reg["pub_reg"] = reg["pub_reg"].astype(float) * float(scale)
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
    print(train.shape)  # after `dropna` on lagged regressors

    # Iterative future predictions
    preds = _iterative_prophet_predict_with_regressors(
        model=m,
        years=predict_years,
        pub_reg_all=pub_reg_all,
        seed_hist=dict(seed_hist),  # copy to avoid side effects
        lags=lags
    )
    return preds

def mean_abs_pct_error(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    """
    AMPE (%): 100 * mean( |y - yhat| / max(eps, |y|) )
    Only calculates mean for data points that have valid prediction values (non-NaN).
    eps avoids blow-ups when y==0 (common in sparse years).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    # Create mask for valid predictions (non-NaN and finite)
    valid_mask = np.isfinite(y_pred) & np.isfinite(y_true)
    
    if not np.any(valid_mask):
        return np.nan
    
    # Filter to only valid data points
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    
    denom = np.maximum(np.abs(y_true_valid), eps)
    return float(np.mean(np.abs(y_true_valid - y_pred_valid) / denom) * 100.0)


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
    ampe_pubs: float
    ampe_pat_no_reg: float
    ampe_pat_with_reg: float


def validate_horizon(horizon: int) -> int:
    """Validate and constrain forecast horizon."""
    if not isinstance(horizon, int):
        try:
            horizon = int(horizon)
        except (TypeError, ValueError):
            return HORIZON_DEFAULT
    return max(HORIZON_MIN, min(horizon, HORIZON_MAX))

def run_for_current_series(
    engine=None,
    tech_label: str = TECH_LABEL_DEFAULT,
    pub_tail_trunc: int = PUB_TAIL_TRUNC_DEFAULT,
    pat_tail_trunc: int = PAT_TAIL_TRUNC_DEFAULT,
    max_lag: int = MAX_LAG_DEFAULT,
    test_years: int = TEST_YEARS_DEFAULT,
    horizon: int = HORIZON_DEFAULT,
    pubs_error_threshold_rmse: Optional[float] = None,
    # flexible evaluation controls:
    split_year: Optional[int] = None,            # train ≤ split_year; test = next `test_years` years
    eval_start_year: Optional[int] = None,       # explicit eval window start (train ≤ start-1)
    eval_end_year: Optional[int] = None          # explicit eval window end (inclusive)
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, EvalResults,
           pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      pubs_forecasts (future horizon),
      patents_forecasts (future horizon),
      metrics_df (eval metrics),
      eval_summary (EvalResults),
      pubs_test_eval  (year, actual, yhat, yhat_lower, yhat_upper),
      pats_test_eval  (year, actual, yhat_baseline, yhat_with_pub_reg)

    Evaluation split priority:
      1) If eval_start_year (and optional eval_end_year) provided:
            train ≤ (eval_start_year - 1), test = [eval_start_year .. eval_end_year]
      2) Else if split_year provided:
            train ≤ split_year, test = next `test_years` consecutive years
      3) Else (fallback):
            test = last `test_years` years after tail truncation
    """
    # Validate horizon parameter
    horizon = validate_horizon(horizon)

    engine = engine or _require_engine()
    pubs, pats = fetch_current_series(engine, pub_tail_trunc, pat_tail_trunc)
    if pubs.empty or pats.empty:
        raise RuntimeError("Empty series after truncation; ensure the app populated research_data3 and raw_patents.")

    # Common span bounds
    y_min = max(int(pubs["year"].min()), int(pats["year"].min()))
    y_max = min(int(pubs["year"].max()), int(pats["year"].max()))

    def _clip_year(y: int) -> int:
        return max(y_min, min(y, y_max))

    # ----- Build TRAIN / TEST split -----
    if eval_start_year is not None:
        es = _clip_year(int(eval_start_year))
        ee = _clip_year(int(eval_end_year)) if (eval_end_year is not None) else _clip_year(es + max(1, int(test_years)) - 1)
        if es > ee:
            raise RuntimeError(f"Invalid eval range: start {es} > end {ee}")
        cutoff = _clip_year(es - 1)
        pub_train = pubs[pubs["year"] <= cutoff].copy()
        pat_train = pats[pats["year"] <= cutoff].copy()
        pub_test  = pubs[(pubs["year"] >= es) & (pubs["year"] <= ee)].copy()
        pat_test  = pats[(pats["year"] >= es) & (pats["year"] <= ee)].copy()
    elif split_year is not None:
        cutoff = _clip_year(int(split_year))
        next_years = list(range(cutoff + 1, min(y_max, cutoff + int(test_years)) + 1))
        pub_train = pubs[pubs["year"] <= cutoff].copy()
        pat_train = pats[pats["year"] <= cutoff].copy()
        pub_test  = pubs[pubs["year"].isin(next_years)].copy()
        pat_test  = pats[pats["year"].isin(next_years)].copy()
    else:
        cutoff = y_max - max(1, int(test_years))
        pub_train, pub_test = pubs[pubs["year"] <= cutoff].copy(), pubs[pubs["year"] > cutoff].copy()
        pat_train, pat_test = pats[pats["year"] <= cutoff].copy(), pats[pats["year"] > cutoff].copy()

    if pub_train.empty or pat_train.empty:
        raise RuntimeError("Training set is empty for the chosen split; try an earlier split_year or a shorter eval range.")

    # ----- (1) Publications forecast for TEST (evaluation path) -----
    if len(pub_test):
        pub_fc_test = prophet_forecast_publications_ar(pub_train, horizon=len(pub_test))
        pub_pred_test = pub_fc_test[pub_fc_test["year"].isin(pub_test["year"])].copy()
        mae_pubs  = mean_absolute_error(pub_test["pub_count"].values, pub_pred_test["pub_count_hat"].values)
        rmse_pubs = np.sqrt(mean_squared_error(pub_test["pub_count"].values, pub_pred_test["pub_count_hat"].values))
        ampe_pubs = mean_abs_pct_error(pub_test["pub_count"].values, pub_pred_test["pub_count_hat"].values)
        # Build test eval table for publications
        pubs_test_eval = (
            pub_test.merge(pub_pred_test, on="year", how="inner")
                    .rename(columns={"pub_count": "actual",
                                     "pub_count_hat": "yhat"})
                    .loc[:, ["year", "actual", "yhat", "yhat_lower", "yhat_upper"]]
                    .sort_values("year")
                    .reset_index(drop=True)
        )
    else:
        pub_fc_test = pd.DataFrame(columns=["year", "pub_count_hat", "yhat_lower", "yhat_upper"])
        mae_pubs = rmse_pubs = ampe_pubs = np.nan
        pubs_test_eval = pd.DataFrame(columns=["year", "actual", "yhat", "yhat_lower", "yhat_upper"])

    # ----- (2) Auto-lag selection (pubs -> patents) -----
    best_lag, xcorr, corr_map = infer_best_pub_to_patent_lag(
        pub_train, pat_train,
        max_lag=max_lag,
        detrend="diff",
        prewhiten="ar1",
        allow_zero_lag=True,
        zero_margin=0.05,
        ic_tiebreak=True
    )

    # small shrink if zero-lag narrowly wins
    pub_reg_scale = 1.0
    r0 = corr_map.get(0, np.nan); r1 = corr_map.get(1, np.nan)
    if best_lag == 0 and np.isfinite(r0) and np.isfinite(r1) and (r0 - r1) < 0.05:
        pub_reg_scale = 0.7

    # ----- (3) Patents baseline (no regressors) -----
    if len(pat_test):
        predict_years = pat_test["year"].tolist()
        pat_base_pred = prophet_patents_baseline_predict(pat_train, predict_years=predict_years)
        mae_base  = mean_absolute_error(pat_test["patent_count"].values, pat_base_pred)
        rmse_base = np.sqrt(mean_squared_error(pat_test["patent_count"].values, pat_base_pred))
        ampe_base = mean_abs_pct_error(pat_test["patent_count"].values, pat_base_pred)
    else:
        pat_base_pred = np.array([])
        mae_base = rmse_base = ampe_base = np.nan

    # ----- (4) Patents with lagged pubs + AR -----
    pub_reg_train     = build_pub_reg(pub_train, pubs_future=pd.DataFrame(columns=["year","pub_count_hat"]), lag=best_lag, scale=pub_reg_scale)
    pub_reg_test_full = build_pub_reg(pub_train, pub_fc_test, lag=best_lag, scale=pub_reg_scale)
    seed_hist_train   = pat_train.set_index("year")["patent_count"].to_dict()

    if len(pat_test):
        predict_years = pat_test["year"].tolist()
        pat_reg_pred = prophet_patents_with_pub_and_ar_predict(
            pats_train=pat_train,
            pub_reg_train=pub_reg_train,
            predict_years=predict_years,
            pub_reg_all=pub_reg_test_full,
            seed_hist=seed_hist_train,
            lags=AR_LAGS
        )
        mae_reg  = mean_absolute_error(pat_test["patent_count"].values, pat_reg_pred)
        rmse_reg = np.sqrt(mean_squared_error(pat_test["patent_count"].values, pat_reg_pred))
        ampe_reg = mean_abs_pct_error(pat_test["patent_count"].values, pat_reg_pred)
        # Build test eval table for patents
        pats_test_eval = pd.DataFrame({
            "year": predict_years,
            "actual": pat_test["patent_count"].values.astype(float),
            "yhat_baseline": pat_base_pred.astype(float),
            "yhat_with_pub_reg": pat_reg_pred.astype(float),
        }).sort_values("year").reset_index(drop=True)
    else:
        pat_reg_pred = np.array([])
        mae_reg = rmse_reg = ampe_reg = np.nan
        pats_test_eval = pd.DataFrame(columns=["year", "actual", "yhat_baseline", "yhat_with_pub_reg"])

    # ----- Summary for metrics -----
    eval_summary = EvalResults(
        tech=tech_label,
        best_lag=best_lag,
        xcorr=xcorr if xcorr is not None else np.nan,
        mae_pubs=mae_pubs, rmse_pubs=rmse_pubs,
        mae_pat_no_reg=mae_base, rmse_pat_no_reg=rmse_base,
        mae_pat_with_reg=mae_reg, rmse_pat_with_reg=rmse_reg,
        ampe_pubs=ampe_pubs, ampe_pat_no_reg=ampe_base, ampe_pat_with_reg=ampe_reg
    )

    # ========= Final production forecasts (full data) =========
    pub_fc_full = prophet_forecast_publications_ar(pubs, horizon=horizon)

    pub_reg_hist_full = build_pub_reg(pubs, pubs_future=pd.DataFrame(columns=["year","pub_count_hat"]), lag=best_lag, scale=pub_reg_scale)
    pub_reg_all_full  = build_pub_reg(pubs, pub_fc_full, lag=best_lag, scale=pub_reg_scale)
    seed_hist_full    = pats.set_index("year")["patent_count"].to_dict()
    future_years      = list(range(y_max + 1, y_max + 1 + horizon))

    pat_reg_future = prophet_patents_with_pub_and_ar_predict(
        pats_train=pats,
        pub_reg_train=pub_reg_hist_full,
        predict_years=future_years,
        pub_reg_all=pub_reg_all_full,
        seed_hist=seed_hist_full,
        lags=AR_LAGS
    )
    pat_base_future = prophet_patents_baseline_predict(pats, predict_years=future_years)

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
        "ampe_pubs": ampe_pubs,
        "mae_patents_with_reg": mae_reg,
        "rmse_patents_with_reg": rmse_reg,
        "ampe_patents_with_reg": ampe_reg,
        "mae_patents_baseline": mae_base,
        "rmse_patents_baseline": rmse_base,
        "ampe_patents_baseline": ampe_base
    }])

    return pubs_forecasts, patents_forecasts, metrics_df, eval_summary, pubs_test_eval, pats_test_eval




# ----------------------------
# CLI
# ----------------------------
def _fmt2(x):
    return "nan" if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x:.2f}"

if __name__ == "__main__":
    pubs_fc, pats_fc, metrics_df, summary, pubs_test_eval, pats_test_eval  = run_for_current_series()

    # pubs_fc.to_csv("publications_forecasts.csv", index=False)
    # pats_fc.to_csv("patents_forecasts_with_pub_reg.csv", index=False)
    # metrics_df.to_csv("prophet_eval_metrics.csv", index=False)
    
    print("publications forecsts :")
    print( pubs_fc)
    print('patents forecsts : ')
    print(pats_fc)
    print('metrics_df : ')
    print(metrics_df)
    

    print("\n=== Evaluation (last test years) ===")
    print(f"tech: {summary.tech}")

    xc = summary.xcorr
    xc_str = "nan" if (xc is None or (isinstance(xc, float) and math.isnan(xc))) else f"{xc:.3f}"
    print(f"auto-selected lag (pubs→patents): {summary.best_lag} years (xcorr={xc_str})")

    print(f"Publications  MAE={_fmt2(summary.mae_pubs)}  RMSE={_fmt2(summary.rmse_pubs)}")
    print(f"Patents base  MAE={_fmt2(summary.mae_pat_no_reg)}  RMSE={_fmt2(summary.rmse_pat_no_reg)}")
    print(f"Patents reg   MAE={_fmt2(summary.mae_pat_with_reg)}  RMSE={_fmt2(summary.rmse_pat_with_reg)}")



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