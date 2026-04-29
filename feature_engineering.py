"""
Feature engineering for football injury prediction.
- ACWR (Acute:Chronic Workload Ratio)
- Monotony Index
- Joint angles (from motion/track data)
- Fatigue
- Sliding window features
"""
import numpy as np
import pandas as pd
from typing import Optional


# --- ACWR ---
def compute_acwr(
    load: np.ndarray,
    acute_days: int = 7,
    chronic_days: int = 28,
    ewma_alpha: Optional[float] = None,
) -> np.ndarray:
    """
    Acute:Chronic Workload Ratio.
    acute_days=7, chronic_days=28 → 1-week acute vs 4-week chronic.
    If ewma_alpha given, use EWMA instead of rolling mean for sensitivity.
    """
    n = len(load)
    acwr = np.full(n, np.nan)
    
    if ewma_alpha is not None:
        # EWMA: recent workloads weighted more
        acute = _ewma(load, span=acute_days)
        chronic = _ewma(load, span=chronic_days)
    else:
        acute = pd.Series(load).rolling(acute_days, min_periods=1).mean().values
        chronic = pd.Series(load).rolling(chronic_days, min_periods=chronic_days // 2).mean().values
    
    valid = chronic > 0
    acwr[valid] = acute[valid] / chronic[valid]
    return acwr


def _ewma(x: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1)
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


# --- Monotony Index ---
def compute_monotony(
    daily_load: np.ndarray,
    window_days: int = 7,
) -> np.ndarray:
    """
    Monotony = mean(daily load) / std(daily load).
    High monotony = little day-to-day variation → fatigue risk.
    """
    n = len(daily_load)
    mono = np.full(n, np.nan)
    for i in range(window_days - 1, n):
        w = daily_load[i - window_days + 1 : i + 1]
        std_w = np.nanstd(w)
        if std_w > 1e-9:
            mono[i] = np.nanmean(w) / std_w
        else:
            mono[i] = np.nanmean(w)
    return mono


# --- Fatigue ---
def compute_fatigue_proxy(
    load: np.ndarray,
    decay: float = 0.1,
) -> np.ndarray:
    """Exponential decay cumulative fatigue proxy."""
    n = len(load)
    fatigue = np.zeros(n)
    fatigue[0] = load[0]
    for i in range(1, n):
        fatigue[i] = decay * load[i] + (1 - decay) * fatigue[i - 1]
    return fatigue


# --- Joint Angles (proxy from track / pose) ---
def compute_joint_angle_features(
    df: pd.DataFrame,
    x_cols: list[str],
    y_cols: list[str],
    prefix: str = "angle_",
) -> pd.DataFrame:
    """
    Compute approximate joint angles from 2D positions.
    angle ≈ arctan2(dy, dx) between segments.
    """
    out = {}
    for xc, yc in zip(x_cols, y_cols):
        if xc in df.columns and yc in df.columns:
            dx = df[xc].diff()
            dy = df[yc].diff()
            out[f"{prefix}{xc}_{yc}"] = np.degrees(np.arctan2(dy, dx))
    return pd.DataFrame(out, index=df.index)


def infer_joint_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Infer x/y or lat/lon columns for joint-angle proxy."""
    x_candidates = [c for c in df.columns if any(k in c.lower() for k in ("x", "lon", "longitude", "pos_x"))]
    y_candidates = [c for c in df.columns if any(k in c.lower() for k in ("y", "lat", "latitude", "pos_y"))]
    if not x_candidates:
        x_candidates = [c for c in df.columns if "x" in c.lower()]
    if not y_candidates:
        y_candidates = [c for c in df.columns if "y" in c.lower()]
    return x_candidates[:3], y_candidates[:3]


# --- Sliding Window Features ---
def sliding_window_features(
    df: pd.DataFrame,
    value_cols: list[str],
    windows: list[int] = [3, 7, 14],
    agg: str = "mean",
) -> pd.DataFrame:
    """
    Rolling window aggregations over value_cols.
    """
    out = {}
    for c in value_cols:
        if c not in df.columns:
            continue
        s = pd.Series(df[c].values)
        for w in windows:
            out[f"{c}_{agg}_{w}d"] = s.rolling(w, min_periods=max(1, w // 2)).agg(agg).values
    return pd.DataFrame(out, index=df.index)


def build_load_column(df: pd.DataFrame, rpe_col: str = "RPE", duration_col: str = "Duration") -> np.ndarray:
    """
    Session load = RPE * Duration (minutes).
    Falls back to sRPE-like proxy if columns differ.
    """
    rpe = None
    dur = None
    for c in df.columns:
        if rpe_col.lower() in c.lower():
            rpe = df[c].values
            break
    for c in df.columns:
        if duration_col.lower() in c.lower() or "min" in c.lower() or "time" in c.lower():
            dur = df[c].values
            break
    if rpe is None:
        # Use intensity/load if present
        for c in df.columns:
            if "load" in c.lower() or "intensity" in c.lower():
                return df[c].fillna(0).values
        return np.ones(len(df)) * 100  # fallback
    if dur is None:
        dur = np.ones(len(df)) * 60
    return np.nan_to_num(rpe) * np.nan_to_num(dur)
