

# %% [setup]
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from outputs.scaler import GlobalScalers

# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = os.path.join(os.path.dirname(__file__), "outputs")
PATENTS_CSV = os.path.join(BASE_DIR, "patents_merged.csv")          # year + pat_*
PUBLICATIONS_CSV = os.path.join(BASE_DIR, "publications_merged.csv") # year + pub_*

PAT_PREFIX = "pat_"
PUB_PREFIX = "pub_"

# Split strategy:
#   "time"   -> per-technology tail splits (use VAL_YEARS / TEST_YEARS)
#   "series" -> hold out entire technologies for val/test (use VAL_TECHS/TEST_TECHS or fractions)
SPLIT_MODE = "time"          # "time" or "series"

# For SPLIT_MODE="time": small tails (your request)
N_STEPS = 5
VAL_YEARS = 2
TEST_YEARS = 2

# For SPLIT_MODE="series":
VAL_TECHS: List[str] = []    # optional explicit tech names; leave [] to sample by fraction
TEST_TECHS: List[str] = []
VAL_FRAC = 0.2               # used only if VAL_TECHS empty
TEST_FRAC = 0.2

# Tech identity feature:
INCLUDE_TECH_ID_TIME = False          # time splits: OK to include one-hot ID
INCLUDE_TECH_ID_SERIES = False   # series splits: disable (unseen tech = cold-start)

# Model/Training - Enhanced hyperparameters
LSTM_UNITS = 64
LSTM_UNITS_2 = 32  # Second LSTM layer
DROPOUT = 0.3
RECURRENT_DROPOUT = 0.2
L1_REG = 1e-5
L2_REG = 1e-4
EPOCHS = 300
BATCH_SIZE = 32
PATIENCE = 25
LEARNING_RATE = 0.001
MIN_LEARNING_RATE = 1e-6
GRADIENT_CLIP_NORM = 1.0
USE_BIDIRECTIONAL = True
USE_BATCH_NORM = True
USE_LAYER_NORM = False  # Alternative to batch norm

# Standardization guard
SCALE_EPS = 1e-6             # floor for scaler.scale_

# Target transform (helps scale skew)
USE_LOG_TARGET = True        # train on log1p(target), evaluate in original scale

# -----------------------------
# UTILITIES
# -----------------------------
@dataclass
class TechScalers:
    x_scaler: StandardScaler
    y_scaler: StandardScaler

@dataclass
class GlobalScalers:
    x_scaler: StandardScaler
    y_scaler: StandardScaler

def _y_fwd(y):
    return np.log1p(y) if USE_LOG_TARGET else y

def _y_inv(y):
    return np.expm1(y) if USE_LOG_TARGET else y

def load_wide_tables(pat_path: str, pub_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    # Try different separators for CSV files
    try:
        pat = pd.read_csv(pat_path, sep=';')
        if "year" not in pat.columns:
            pat = pd.read_csv(pat_path)
    except:
        pat = pd.read_csv(pat_path)
    
    try:
        pub = pd.read_csv(pub_path, sep=';')
        if "year" not in pub.columns:
            pub = pd.read_csv(pub_path)
    except:
        pub = pd.read_csv(pub_path)
    
    if "year" not in pat.columns or "year" not in pub.columns:
        raise ValueError("Both CSVs must contain a 'year' column.")
    
    # Handle case where patents file might have pub_ prefix instead of pat_
    pat_techs = []
    if any(c.startswith(PAT_PREFIX) for c in pat.columns):
        pat_techs = [c[len(PAT_PREFIX):] for c in pat.columns if c.startswith(PAT_PREFIX)]
    elif any(c.startswith(PUB_PREFIX) for c in pat.columns):
        # Patents file has pub_ prefix, rename columns
        rename_dict = {c: c.replace(PUB_PREFIX, PAT_PREFIX) for c in pat.columns if c.startswith(PUB_PREFIX)}
        pat = pat.rename(columns=rename_dict)
        pat_techs = [c[len(PAT_PREFIX):] for c in pat.columns if c.startswith(PAT_PREFIX)]
    
    pub_techs = [c[len(PUB_PREFIX):] for c in pub.columns if c.startswith(PUB_PREFIX)]
    techs = sorted(list(set(pat_techs).intersection(set(pub_techs))))
    if not techs:
        raise ValueError("No overlapping technologies found between patents and publications CSVs.")
    return pat, pub, techs

def _trim_leading_zeros(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    nz = (df[["patents", "publications"]] != 0).any(axis=1)
    if nz.any():
        first_idx = nz.idxmax()
        return df.loc[first_idx:].reset_index(drop=True)
    return df

def build_tech_dataframe(pat: pd.DataFrame, pub: pd.DataFrame, tech: str) -> pd.DataFrame:
    df_pat = pat[["year", f"{PAT_PREFIX}{tech}"]].rename(columns={f"{PAT_PREFIX}{tech}": "patents"})
    df_pub = pub[["year", f"{PUB_PREFIX}{tech}"]].rename(columns={f"{PUB_PREFIX}{tech}": "publications"})
    df = pd.merge(df_pat, df_pub, on="year", how="inner").dropna()
    df = df.sort_values("year").reset_index(drop=True)
    df = _trim_leading_zeros(df)
    return df

def years_split(years: List[int], val_years=VAL_YEARS, test_years=TEST_YEARS) -> Tuple[set, set, set]:
    yrs = sorted(years)
    test = set(yrs[-test_years:]) if len(yrs) >= test_years else set()
    remain = yrs[:len(yrs)-len(test)]
    val = set(remain[-val_years:]) if len(remain) >= val_years else set()
    train = set(remain[:len(remain)-len(val)])
    return set(train), set(val), set(test)

def create_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create enhanced features for better model performance with numerical stability.
    """
    df = df.copy()
    
    # Moving averages
    df['patents_ma3'] = df['patents'].rolling(window=3, min_periods=1).mean()
    df['publications_ma3'] = df['publications'].rolling(window=3, min_periods=1).mean()
    
    # Growth rates (year-over-year) with clipping
    df['patents_growth'] = df['patents'].pct_change().fillna(0)
    df['publications_growth'] = df['publications'].pct_change().fillna(0)
    
    # Clip extreme growth rates to prevent infinity
    df['patents_growth'] = np.clip(df['patents_growth'], -10, 10)
    df['publications_growth'] = np.clip(df['publications_growth'], -10, 10)
    
    # Ratios and interactions with safe division
    df['patent_pub_ratio'] = df['patents'] / np.maximum(df['publications'], 1e-6)
    df['patent_pub_ratio'] = np.clip(df['patent_pub_ratio'], 0, 1000)  # Clip extreme ratios
    df['total_activity'] = df['patents'] + df['publications']
    
    # Lagged features
    df['patents_lag1'] = df['patents'].shift(1).fillna(0)
    df['publications_lag1'] = df['publications'].shift(1).fillna(0)
    
    # Volatility (rolling standard deviation) with safe handling
    df['patents_volatility'] = df['patents'].rolling(window=3, min_periods=1).std().fillna(0)
    df['publications_volatility'] = df['publications'].rolling(window=3, min_periods=1).std().fillna(0)
    
    # Trend indicators with safe calculation
    def trend_slope(series, window=3):
        def calc_slope(x):
            if len(x) < 2:
                return 0
            try:
                slope = np.polyfit(range(len(x)), x, 1)[0]
                # Clip extreme slopes
                return np.clip(slope, -1000, 1000)
            except (np.linalg.LinAlgError, ValueError):
                return 0
        return series.rolling(window=window, min_periods=1).apply(calc_slope, raw=True)
    
    df['patents_trend'] = trend_slope(df['patents'])
    df['publications_trend'] = trend_slope(df['publications'])
    
    # Final safety check: replace any remaining inf/nan values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        df[col] = df[col].fillna(0)
        # Additional clipping for extreme values
        if col != 'year':
            df[col] = np.clip(df[col], -1e6, 1e6)
    
    return df

def create_sequences_for_tech(df: pd.DataFrame):
    """
    Returns X_all, y_all, target_years (no split). Splitting happens later by config.
    Enhanced with additional features.
    """
    # Create enhanced features
    df_enhanced = create_enhanced_features(df)
    
    # Select features for model input
    feature_cols = [
        'patents', 'publications',  # Original features
        'patents_ma3', 'publications_ma3',  # Moving averages
        'patents_growth', 'publications_growth',  # Growth rates
        'patent_pub_ratio', 'total_activity',  # Ratios and interactions
        'patents_lag1', 'publications_lag1',  # Lagged features
        'patents_volatility', 'publications_volatility',  # Volatility
        'patents_trend', 'publications_trend'  # Trend indicators
    ]
    
    X_all, y_all, target_years = [], [], []
    vals = df_enhanced[feature_cols].values.astype(float)
    yrs = df_enhanced["year"].values.astype(int)
    
    for i in range(len(df_enhanced) - N_STEPS):
        X_all.append(vals[i:i+N_STEPS])
        y_all.append(_y_fwd(df_enhanced.iloc[i+N_STEPS]['patents']))   # patents at t+1
        target_years.append(yrs[i+N_STEPS])
    
    return np.array(X_all), np.array(y_all), np.array(target_years), yrs

def fit_scaler_guarded(data_2d: np.ndarray) -> StandardScaler:
    # Additional safety checks for input data
    data_2d = np.asarray(data_2d, dtype=np.float64)
    
    # Replace inf and nan values
    data_2d = np.where(np.isfinite(data_2d), data_2d, 0)
    
    # Clip extreme values
    data_2d = np.clip(data_2d, -1e6, 1e6)
    
    sc = StandardScaler().fit(data_2d)
    sc.scale_ = np.maximum(sc.scale_, SCALE_EPS)
    sc.var_ = sc.scale_ ** 2
    return sc

def fit_scalers_for_tech(X_train: np.ndarray, y_train: np.ndarray) -> TechScalers:
    """Fit scalers with enhanced feature handling"""
    x_scaler = fit_scaler_guarded(X_train.reshape(-1, X_train.shape[-1]))
    y_scaler = fit_scaler_guarded(y_train.reshape(-1, 1))
    return TechScalers(x_scaler=x_scaler, y_scaler=y_scaler)

def fit_global_scalers(X_train_list: List[np.ndarray], y_train_list: List[np.ndarray]) -> GlobalScalers:
    """Fit global scalers with enhanced feature handling"""
    X_stack = np.concatenate([X.reshape(-1, X.shape[-1]) for X in X_train_list], axis=0)
    y_stack = np.concatenate([y.reshape(-1, 1) for y in y_train_list], axis=0)
    x_scaler = fit_scaler_guarded(X_stack)
    y_scaler = fit_scaler_guarded(y_stack)
    return GlobalScalers(x_scaler=x_scaler, y_scaler=y_scaler)

def scale_with_scalers(X: np.ndarray, y: np.ndarray, x_scaler: StandardScaler, y_scaler: StandardScaler):
    Xs = x_scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape[0], X.shape[1], X.shape[2])
    ys = y_scaler.transform(y.reshape(-1, 1)).ravel()
    return Xs, ys

def add_tech_one_hot(X: np.ndarray, tech_idx: int, n_techs: int, use_one_hot: bool) -> np.ndarray:
    if not use_one_hot:
        return X
    one_hot = np.zeros((X.shape[0], X.shape[1], n_techs), dtype=np.float32)
    one_hot[:, :, tech_idx] = 1.0
    return np.concatenate([X, one_hot], axis=-1)

def build_lstm_model(n_steps: int, n_features: int) -> tf.keras.Model:
    """
    Enhanced LSTM model with bidirectional layers, regularization, and normalization.
    """
    model = Sequential()
    model.add(Input(shape=(n_steps, n_features)))
    
    # First LSTM layer (bidirectional if enabled)
    if USE_BIDIRECTIONAL:
        model.add(Bidirectional(
            LSTM(LSTM_UNITS, 
                 return_sequences=True,
                 activation="tanh",
                 recurrent_activation="sigmoid",
                 dropout=DROPOUT,
                 recurrent_dropout=RECURRENT_DROPOUT,
                 kernel_regularizer=l1_l2(l1=L1_REG, l2=L2_REG),
                 recurrent_regularizer=l1_l2(l1=L1_REG, l2=L2_REG))
        ))
    else:
        model.add(LSTM(LSTM_UNITS, 
                      return_sequences=True,
                      activation="tanh",
                      recurrent_activation="sigmoid",
                      dropout=DROPOUT,
                      recurrent_dropout=RECURRENT_DROPOUT,
                      kernel_regularizer=l1_l2(l1=L1_REG, l2=L2_REG),
                      recurrent_regularizer=l1_l2(l1=L1_REG, l2=L2_REG)))
    
    # Normalization after first LSTM
    if USE_BATCH_NORM:
        model.add(BatchNormalization())
    elif USE_LAYER_NORM:
        model.add(LayerNormalization())
    
    # Second LSTM layer
    if USE_BIDIRECTIONAL:
        model.add(Bidirectional(
            LSTM(LSTM_UNITS_2,
                 activation="tanh",
                 recurrent_activation="sigmoid",
                 dropout=DROPOUT,
                 recurrent_dropout=RECURRENT_DROPOUT,
                 kernel_regularizer=l1_l2(l1=L1_REG, l2=L2_REG),
                 recurrent_regularizer=l1_l2(l1=L1_REG, l2=L2_REG))
        ))
    else:
        model.add(LSTM(LSTM_UNITS_2,
                      activation="tanh",
                      recurrent_activation="sigmoid",
                      dropout=DROPOUT,
                      recurrent_dropout=RECURRENT_DROPOUT,
                      kernel_regularizer=l1_l2(l1=L1_REG, l2=L2_REG),
                      recurrent_regularizer=l1_l2(l1=L1_REG, l2=L2_REG)))
    
    # Normalization after second LSTM
    if USE_BATCH_NORM:
        model.add(BatchNormalization())
    elif USE_LAYER_NORM:
        model.add(LayerNormalization())
    
    # Dense layers with dropout
    model.add(Dropout(DROPOUT))
    model.add(Dense(32, activation="relu", kernel_regularizer=l1_l2(l1=L1_REG, l2=L2_REG)))
    model.add(Dropout(DROPOUT * 0.5))
    model.add(Dense(1))
    
    # Enhanced optimizer with custom learning rate
    optimizer = Adam(
        learning_rate=LEARNING_RATE,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        clipnorm=GRADIENT_CLIP_NORM
    )
    
    # Compile with Huber loss for robustness to outliers
    model.compile(
        optimizer=optimizer, 
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=['mae']
    )
    
    return model

# -----------------------------
# Metrics (robust WAPE + robust sMAPE)
# -----------------------------
def wape_robust(y_true, y_pred, min_denom=10.0, eps=1e-8):
    """
    WAPE (%) = sum|err| / sum|y| * 100, but returns NaN if denom < min_denom to avoid nonsense on tiny series.
    """
    denom = float(np.sum(np.abs(y_true)))
    if denom < min_denom:
        return np.nan
    return 100.0 * float(np.sum(np.abs(y_true - y_pred))) / (denom + eps)

def _smape_floor_from_y(y_true, floor_quantile=0.20, eps=1e-8, min_floor=None) -> float:
    """
    Compute the denominator floor from NONZERO |y| only.
    Fallback: 0.5 * mean(|y|). Optional absolute minimum via min_floor.
    """
    yt = np.asarray(y_true, dtype=float)
    pos = np.abs(yt)[np.abs(yt) > 0]
    if pos.size:
        floor = float(np.quantile(pos, floor_quantile))
    else:
        floor = float(np.mean(np.abs(yt)) * 0.5 if yt.size else 0.0)
    if min_floor is not None:
        floor = max(floor, float(min_floor))
    return max(floor, eps)

def smape_robust(y_true, y_pred, eps=1e-8, floor_quantile=0.20, min_floor=1.0):
    """
    Symmetric MAPE (%) with a data-aware denominator floor:
      denom = max( (|y|+|ŷ|)/2 , floor_from_nonzero(|y|, q), min_floor, eps )
    Default min_floor=1.0 since we model counts.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    valid = np.isfinite(yt) & np.isfinite(yp)
    if not np.any(valid):
        return np.nan
    yt = yt[valid]; yp = yp[valid]

    denom = (np.abs(yt) + np.abs(yp)) / 2.0
    floor = _smape_floor_from_y(yt, floor_quantile=floor_quantile, eps=eps, min_floor=min_floor)
    denom = np.maximum(denom, floor)

    return 100.0 * float(np.mean(np.abs(yt - yp) / denom))

def smape_weighted(y_true, y_pred, eps=1e-8):
    """
    Weighted sMAPE (%) with weights=|y| (bounded version of WAPE-like behavior).
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(yt) & np.isfinite(yp)
    if not np.any(valid):
        return np.nan
    yt = yt[valid]; yp = yp[valid]
    denom = (np.abs(yt) + np.abs(yp)) / 2.0
    ratio = np.abs(yt - yp) / np.maximum(denom, eps)
    w = np.maximum(np.abs(yt), eps)
    return 100.0 * float(np.average(ratio, weights=w))

def fmt_num(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return round(float(x), nd)

def fmt_pct(x, nd=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return round(float(x), nd)

def _tiny_share(y, t=1.0):
    """
    % of NONZERO |y| that are ≤ t (diagnostic for sMAPE sensitivity on tiny scales).
    """
    y = np.asarray(y, float)
    nz = np.abs(y) > 0
    if not np.any(nz):
        return 100.0
    return 100.0 * float(np.mean(np.abs(y[nz]) <= t))

def _scale_diag(y):
    """
    Return basic scale diagnostics for |y| over NONZERO entries.
    """
    y = np.asarray(y, float)
    nz = np.abs(y) > 0
    if not np.any(nz):
        return dict(nonzero=False, q20=0.0, median=0.0, mean=0.0)
    a = np.abs(y[nz])
    return dict(
        nonzero=True,
        q20=float(np.quantile(a, 0.2)),
        median=float(np.median(a)),
        mean=float(np.mean(a))
    )

# %% [main execution]
if __name__ == "__main__":
    # %% [data prep]
    pat, pub, techs_all = load_wide_tables(PATENTS_CSV, PUBLICATIONS_CSV)

    # Build per-tech sequences
    per_tech = {}
    eligible_techs = []
    for t in techs_all:
        df_t = build_tech_dataframe(pat, pub, t)
        if len(df_t) < (N_STEPS + 1):
            print(f"[skip] {t}: too short after trimming.")
            continue
        X_all, y_all, target_years, yrs_full = create_sequences_for_tech(df_t)
        if len(X_all) == 0:
            print(f"[skip] {t}: no windows.")
            continue
        per_tech[t] = dict(df=df_t, X=X_all, y=y_all, target_years=target_years, yrs_full=yrs_full)
        eligible_techs.append(t)

    if len(eligible_techs) < 1:
        raise RuntimeError("No technology series met the minimum length.")

    print(f"Eligible technologies: {eligible_techs}")

    # -----------------------------
    # Build splits
    # -----------------------------
    tech_index_map = {t: i for i, t in enumerate(eligible_techs)}
    n_techs = len(eligible_techs)

    per_tech_splits: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}

    if SPLIT_MODE == "time":
        # Per-tech tail split with smaller val/test tails
        for t in eligible_techs:
            rows = per_tech[t]
            df_t = rows["df"]
            train_years, val_years, test_years = years_split(df_t["year"].tolist(), VAL_YEARS, TEST_YEARS)
            X_all, y_all, yrs_all = rows["X"], rows["y"], rows["target_years"]
            def mask(year_set): 
                return np.array([yy in year_set for yy in yrs_all])
            per_tech_splits[t] = {
                "train": (X_all[mask(train_years)], y_all[mask(train_years)], yrs_all[mask(train_years)]),
                "val":   (X_all[mask(val_years)],   y_all[mask(val_years)],   yrs_all[mask(val_years)]),
                "test":  (X_all[mask(test_years)],  y_all[mask(test_years)],  yrs_all[mask(test_years)]),
            }

    elif SPLIT_MODE == "series":
        rng = np.random.default_rng(SEED)
        techs = eligible_techs.copy()
        t_set = set(techs)
        val_set = set(VAL_TECHS) & t_set if VAL_TECHS else set()
        test_set = set(TEST_TECHS) & (t_set - val_set) if TEST_TECHS else set()
        remain = list(t_set - val_set - test_set)
        if not VAL_TECHS:
            n_val = max(1, int(np.floor(VAL_FRAC * len(techs))))
            val_set.update(rng.choice(remain, size=min(n_val, len(remain)), replace=False).tolist())
            remain = list(t_set - val_set - test_set)
        if not TEST_TECHS:
            n_test = max(1, int(np.floor(TEST_FRAC * len(techs))))
            test_set.update(rng.choice(remain, size=min(n_test, len(remain)), replace=False).tolist())
            remain = list(t_set - val_set - test_set)
        train_set = list(remain); val_set = list(val_set); test_set = list(test_set)

        print(f"SPLIT_MODE='series' → train techs: {train_set}")
        print(f"                       val techs:   {val_set}")
        print(f"                       test techs:  {test_set}")

        for t in eligible_techs:
            X_all, y_all, yrs_all = per_tech[t]["X"], per_tech[t]["y"], per_tech[t]["target_years"]
            if t in train_set:
                per_tech_splits[t] = {"train": (X_all, y_all, yrs_all), "val": (np.empty((0, N_STEPS, 2)), np.array([]), np.array([])), "test": (np.empty((0, N_STEPS, 2)), np.array([]), np.array([]))}
            elif t in val_set:
                per_tech_splits[t] = {"train": (np.empty((0, N_STEPS, 2)), np.array([]), np.array([])), "val": (X_all, y_all, yrs_all), "test": (np.empty((0, N_STEPS, 2)), np.array([]), np.array([]))}
            elif t in test_set:
                per_tech_splits[t] = {"train": (np.empty((0, N_STEPS, 2)), np.array([]), np.array([])), "val": (np.empty((0, N_STEPS, 2)), np.array([]), np.array([])), "test": (X_all, y_all, yrs_all)}
            else:
                per_tech_splits[t] = {"train": (X_all, y_all, yrs_all), "val": (np.empty((0, N_STEPS, 2)), np.array([]), np.array([])), "test": (np.empty((0, N_STEPS, 2)), np.array([]), np.array([]))}
    else:
        raise ValueError("SPLIT_MODE must be 'time' or 'series'.")

    # -----------------------------
    # Scaling & pooling
    # -----------------------------
    X_train_all, y_train_all = [], []
    X_val_all, y_val_all = [], []
    X_test_all, y_test_all = [], []

    train_labels, val_labels, test_labels = [], [], []
    train_years, val_years_list, test_years_list = [], [], []

    if SPLIT_MODE == "time":
        per_tech_scalers: Dict[str, TechScalers] = {}
        for t in eligible_techs:
            X_tr, y_tr, _ = per_tech_splits[t]["train"]
            if len(y_tr) == 0:
                continue
            per_tech_scalers[t] = fit_scalers_for_tech(X_tr, y_tr)

        for t in eligible_techs:
            if t not in per_tech_scalers:
                continue
            sc = per_tech_scalers[t]
            for split_name, (X, y, yrs) in per_tech_splits[t].items():
                if len(y) == 0:
                    continue
                Xs, ys = scale_with_scalers(X, y, sc.x_scaler, sc.y_scaler)
                use_one_hot = INCLUDE_TECH_ID_TIME
                Xs = add_tech_one_hot(Xs, tech_index_map[t], n_techs, use_one_hot)
                if split_name == "train":
                    X_train_all.append(Xs); y_train_all.append(ys)
                    train_labels.extend([t]*len(ys)); train_years.extend(yrs.tolist())
                elif split_name == "val":
                    X_val_all.append(Xs); y_val_all.append(ys)
                    val_labels.extend([t]*len(ys)); val_years_list.extend(yrs.tolist())
                elif split_name == "test":
                    X_test_all.append(Xs); y_test_all.append(ys)
                    test_labels.extend([t]*len(ys)); test_years_list.extend(yrs.tolist())

    else:
        # Global scalers fitted on training windows across *training* techs
        X_train_lists, y_train_lists = [], []
        for t in eligible_techs:
            X_tr, y_tr, _ = per_tech_splits[t]["train"]
            if len(y_tr) > 0:
                X_train_lists.append(X_tr); y_train_lists.append(y_tr)
        if not X_train_lists:
            raise RuntimeError("No training windows to fit global scalers.")
        global_scalers = fit_global_scalers(X_train_lists, y_train_lists)

        for t in eligible_techs:
            for split_name, (X, y, yrs) in per_tech_splits[t].items():
                if len(y) == 0:
                    continue
                Xs, ys = scale_with_scalers(X, y, global_scalers.x_scaler, global_scalers.y_scaler)
                use_one_hot = INCLUDE_TECH_ID_SERIES
                Xs = add_tech_one_hot(Xs, tech_index_map[t], n_techs, use_one_hot)
                if split_name == "train":
                    X_train_all.append(Xs); y_train_all.append(ys)
                    train_labels.extend([t]*len(ys)); train_years.extend(yrs.tolist())
                elif split_name == "val":
                    X_val_all.append(Xs); y_val_all.append(ys)
                    val_labels.extend([t]*len(ys)); val_years_list.extend(yrs.tolist())
                elif split_name == "test":
                    X_test_all.append(Xs); y_test_all.append(ys)
                    test_labels.extend([t]*len(ys)); test_years_list.extend(yrs.tolist())

    def concat_or_empty(parts, n_feat_base=12, n_techs=n_techs, use_one_hot=True):  # Updated to 12 features
        return np.concatenate(parts, axis=0) if len(parts) else np.empty((0, N_STEPS, n_feat_base + (n_techs if use_one_hot else 0)))

    # figure out final feature size based on mode
    use_one_hot_any = (INCLUDE_TECH_ID_TIME and SPLIT_MODE == "time") or (INCLUDE_TECH_ID_SERIES and SPLIT_MODE == "series")

    X_train = concat_or_empty(X_train_all, use_one_hot=use_one_hot_any)
    y_train = np.concatenate(y_train_all, axis=0) if len(y_train_all) else np.array([])
    X_val   = concat_or_empty(X_val_all, use_one_hot=use_one_hot_any)
    y_val   = np.concatenate(y_val_all, axis=0) if len(y_val_all) else np.array([])
    X_test  = concat_or_empty(X_test_all, use_one_hot=use_one_hot_any)
    y_test  = np.concatenate(y_test_all, axis=0) if len(y_test_all) else np.array([])

    print(f"Train samples: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    assert X_train.shape[1] == N_STEPS
    n_features = X_train.shape[2]
    print(f"Input features per timestep: {n_features}")

    # %% [model]
    model = build_lstm_model(N_STEPS, n_features)
    
    # Learning rate scheduler
    def lr_schedule(epoch, lr):
        """Exponential decay with warmup"""
        if epoch < 10:  # Warmup phase
            return LEARNING_RATE * (epoch + 1) / 10
        else:
            return max(lr * 0.995, MIN_LEARNING_RATE)  # Exponential decay
    
    # Enhanced callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss", 
            patience=PATIENCE, 
            restore_best_weights=True,
            min_delta=1e-4,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", 
            factor=0.7, 
            patience=max(8, PATIENCE//3), 
            min_lr=MIN_LEARNING_RATE, 
            min_delta=1e-4,
            verbose=1
        ),
        LearningRateScheduler(lr_schedule, verbose=0),
        ModelCheckpoint(
            filepath=os.path.join(BASE_DIR, "best_model_checkpoint.keras"),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
    ]

    # Enhanced training with validation split if no validation data
    if X_val.shape[0] == 0:
        print("No validation data provided, using 20% of training data for validation")
        validation_data = None
        validation_split = 0.2
    else:
        validation_data = (X_val, y_val)
        validation_split = 0.0
    
    print(f"\nTraining enhanced LSTM model with {model.count_params():,} parameters")
    print(f"Architecture: {'Bidirectional' if USE_BIDIRECTIONAL else 'Unidirectional'} LSTM")
    print(f"Normalization: {'Batch' if USE_BATCH_NORM else 'Layer' if USE_LAYER_NORM else 'None'}")
    print(f"Regularization: L1={L1_REG}, L2={L2_REG}, Dropout={DROPOUT}")
    
    history = model.fit(
        X_train, y_train,
        validation_data=validation_data,
        validation_split=validation_split,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=callbacks,
        verbose=1
    )
    
    # Load best model if checkpoint was saved
    checkpoint_path = os.path.join(BASE_DIR, "best_model_checkpoint.keras")
    if os.path.exists(checkpoint_path):
        print("Loading best model from checkpoint...")
        model = tf.keras.models.load_model(checkpoint_path)

    # %% [evaluation]
    def inverse_transform_predictions(
        X_split: np.ndarray, y_split: np.ndarray, tech_labels: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict, inverse-scale (per-tech or global), then inverse-target-transform.
        Clip negative predictions to 0 (counts can't be negative).
        """
        if X_split.shape[0] == 0:
            return np.array([]), np.array([])
        y_pred_scaled = model.predict(X_split, verbose=0).ravel()
        y_true_inv, y_pred_inv = [], []

        if SPLIT_MODE == "time":
            for y_s, yp_s, tech in zip(y_split, y_pred_scaled, tech_labels):
                sc = per_tech_scalers[tech]
                y_true_unscaled = sc.y_scaler.inverse_transform([[y_s]])[0, 0]
                y_pred_unscaled = sc.y_scaler.inverse_transform([[yp_s]])[0, 0]
                y_true_val = _y_inv(y_true_unscaled)
                y_pred_val = _y_inv(y_pred_unscaled)
                y_true_inv.append(y_true_val)
                y_pred_inv.append(y_pred_val)
        else:
            for y_s, yp_s in zip(y_split, y_pred_scaled):
                y_true_unscaled = global_scalers.y_scaler.inverse_transform([[y_s]])[0, 0]
                y_pred_unscaled = global_scalers.y_scaler.inverse_transform([[yp_s]])[0, 0]
                y_true_val = _y_inv(y_true_unscaled)
                y_pred_val = _y_inv(y_pred_unscaled)
                y_true_inv.append(y_true_val)
                y_pred_inv.append(y_pred_val)

        # Convert to arrays and clip predictions to be non-negative
        y_true_inv = np.asarray(y_true_inv, dtype=float)
        y_pred_inv = np.asarray(y_pred_inv, dtype=float)
        y_pred_inv = np.maximum(y_pred_inv, 0.0)

        return y_true_inv, y_pred_inv


    def metrics(y_true, y_pred):
        if len(y_true) == 0:
            return {"MAE": np.nan, "RMSE": np.nan}
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return {"MAE": mae, "RMSE": rmse}

    # Test metrics
    y_true_test, y_pred_test = inverse_transform_predictions(X_test, y_test, test_labels)
    overall = metrics(y_true_test, y_pred_test)

    # Diagnostics for test set
    test_diag = _scale_diag(y_true_test)
    print("\n=== Test (pooled) ===")
    print({
        "MAE": fmt_num(overall["MAE"], 3),
        "RMSE": fmt_num(overall["RMSE"], 3),
        "WAPE_%": fmt_pct(wape_robust(y_true_test, y_pred_test)),
        "sMAPE_%": fmt_pct(smape_robust(y_true_test, y_pred_test, floor_quantile=0.20, min_floor=1.0)),
        "sMAPE_w_%": fmt_pct(smape_weighted(y_true_test, y_pred_test)),
    })
    print(f"[diag] test: %|y|<=1 = {fmt_num(_tiny_share(y_true_test, 1.0), 1)}%, "
          f"%|y|<=5 = {fmt_num(_tiny_share(y_true_test, 5.0), 1)}%, "
          f"q20(|y|)={fmt_num(test_diag['q20'], 3)}, median(|y|)={fmt_num(test_diag['median'], 3)}")

    # Per-tech metrics
    print("\n=== Test per technology ===")
    for t in eligible_techs:
        idx = [i for i, lab in enumerate(test_labels) if lab == t]
        if not idx:
            continue
        y_t_true = y_true_test[idx]
        y_t_pred = y_pred_test[idx]
        m = metrics(y_t_true, y_t_pred)
        WAPEp  = wape_robust(y_t_true, y_t_pred)   # may be NaN for tiny series
        sMAPEp = smape_robust(y_t_true, y_t_pred, floor_quantile=0.20, min_floor=1.0)
        sMAPEw = smape_weighted(y_t_true, y_t_pred)
        nice = {
            "MAE": fmt_num(m["MAE"], 3),
            "RMSE": fmt_num(m["RMSE"], 3),
            "WAPE_%": fmt_pct(WAPEp, 2),
            "sMAPE_%": fmt_pct(sMAPEp, 2),
            "sMAPE_w_%": fmt_pct(sMAPEw, 2)
        }
        print(t, nice)

    # Validation metrics
    y_true_val, y_pred_val = inverse_transform_predictions(X_val, y_val, val_labels)
    val_overall = metrics(y_true_val, y_pred_val)

    # Diagnostics for val set
    val_diag = _scale_diag(y_true_val)
    print("\n=== Val (pooled) ===")
    print({
        "MAE": fmt_num(val_overall["MAE"], 3),
        "RMSE": fmt_num(val_overall["RMSE"], 3),
        "WAPE_%": fmt_pct(wape_robust(y_true_val, y_pred_val)),
        "sMAPE_%": fmt_pct(smape_robust(y_true_val, y_pred_val, floor_quantile=0.20, min_floor=1.0)),
        "sMAPE_w_%": fmt_pct(smape_weighted(y_true_val, y_pred_val)),
    })
    print(f"[diag] val:  %|y|<=1 = {fmt_num(_tiny_share(y_true_val, 1.0), 1)}%, "
          f"%|y|<=5 = {fmt_num(_tiny_share(y_true_val, 5.0), 1)}%, "
          f"q20(|y|)={fmt_num(val_diag['q20'], 3)}, median(|y|)={fmt_num(val_diag['median'], 3)}")

    # %% [save model and training history]
    os.makedirs(BASE_DIR, exist_ok=True)
    model_path = os.path.join(BASE_DIR, "lstm_patent_forecaster_enhanced_series.keras")
    model.save(model_path)
    
    # Save scalers based on split mode
    if SPLIT_MODE == "time":
        scaler_path = os.path.join(BASE_DIR, "per_tech_scalers_series.pkl")
        joblib.dump(per_tech_scalers, scaler_path)
        print(f"Saved per-tech scalers to {scaler_path}")
    else:
        scaler_path = os.path.join(BASE_DIR, "global_scalers.pkl")
        joblib.dump(global_scalers, scaler_path)
        print(f"Saved global scalers to {scaler_path}")
    
    # Save training history
    history_path = os.path.join(BASE_DIR, "training_history_series.pkl")
    joblib.dump(history.history, history_path)
    
    # Save model configuration
    config = {
        'N_STEPS': N_STEPS,
        'LSTM_UNITS': LSTM_UNITS,
        'LSTM_UNITS_2': LSTM_UNITS_2,
        'DROPOUT': DROPOUT,
        'RECURRENT_DROPOUT': RECURRENT_DROPOUT,
        'L1_REG': L1_REG,
        'L2_REG': L2_REG,
        'LEARNING_RATE': LEARNING_RATE,
        'USE_BIDIRECTIONAL': USE_BIDIRECTIONAL,
        'USE_BATCH_NORM': USE_BATCH_NORM,
        'USE_LAYER_NORM': USE_LAYER_NORM,
        'USE_LOG_TARGET': USE_LOG_TARGET,
        'SPLIT_MODE': SPLIT_MODE,
        'n_features': n_features,
        'eligible_techs': eligible_techs
    }
    config_path = os.path.join(BASE_DIR, "model_config_series.pkl")
    joblib.dump(config, config_path)
    
    print(f"\nSaved enhanced model to {model_path}")
    print(f"Saved training history to {history_path}")
    print(f"Saved model configuration to {config_path}")
    
    # Print final training summary
    final_train_loss = min(history.history['loss'])
    final_val_loss = min(history.history.get('val_loss', [np.inf]))
    print(f"\nTraining Summary:")
    print(f"Best training loss: {final_train_loss:.6f}")
    print(f"Best validation loss: {final_val_loss:.6f}")
    print(f"Total epochs trained: {len(history.history['loss'])}")
