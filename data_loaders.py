"""
Data loaders for football injury datasets.
Supports:
  - NFL Playing Surface Analytics (InjuryRecord, PlayList, PlayerTrackData)
  - Multimodal Sports Injury Dataset (biometric, wellness, joint data)
"""
"""
XGBoost injury risk model with SHAP explainability.
Tabular biometric + wellness -> binary injury risk.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
)
import xgboost as xgb
import shap
from pathlib import Path
from typing import Optional

from feature_engineering import (
    compute_acwr,
    compute_monotony,
    compute_fatigue_proxy,
    compute_joint_angle_features,
    infer_joint_columns,
)


def prepare_features(
    df: pd.DataFrame,
    load_col: str | None = None,
    target_col: str = "injury",
) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    
    if load_col is None:
        candidates = [
            c for c in df.columns
            if "load" in c.lower() or "rpe" in c.lower()
            or "intensity" in c.lower() or "workload" in c.lower()
        ]
        load_col = candidates[0] if candidates else None
    
    if load_col is None:
        np.random.seed(42)
        load_col = "_synthetic_load"
        df[load_col] = np.random.exponential(100, len(df)) + 20
    
    load = df[load_col].fillna(df[load_col].median()).values
    
    acwr = compute_acwr(load, acute_days=7, chronic_days=28)
    df["acwr"] = acwr
    
    mono = compute_monotony(load, window_days=7)
    df["monotony_index"] = mono
    
    fatigue = compute_fatigue_proxy(load, decay=0.1)
    df["fatigue_proxy"] = fatigue
    
    for w in [3, 7, 14]:
        df[f"load_roll_mean_{w}d"] = pd.Series(load).rolling(w, min_periods=1).mean().values
    
    x_cols, y_cols = infer_joint_columns(df)
    if x_cols and y_cols:
        angles = compute_joint_angle_features(df, x_cols, y_cols)
        for c in angles.columns:
            df[c] = angles[c]
    
    if target_col not in df.columns:
        df[target_col] = (acwr > np.nanpercentile(acwr[~np.isnan(acwr)], 80)).astype(int)
    
    feature_cols = [
        c for c in df.columns
        if c not in (target_col, "PlayerKey", "player_id", "date", "Date")
        and df[c].dtype in (np.float64, np.int64, np.float32, np.int32)
    ]
    
    X = df[feature_cols].fillna(0)
    y = df[target_col].astype(int)
    
    return X, y


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if y.nunique() > 1 else None
    )
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        eval_metric="logloss",
    )
    clf.fit(X_train_s, y_train)
    
    return clf, scaler, X_train_s, X_test_s, y_train, y_test


def get_top3_risk_factors(
    clf: xgb.XGBClassifier,
    X: pd.DataFrame,
    feature_names: list[str],
    player_idx: int,
) -> list[tuple[str, float]]:
    explainer = shap.TreeExplainer(clf, X)
    shap_vals = explainer.shap_values(X)
    
    if len(shap_vals.shape) > 1:
        vals = shap_vals[player_idx, :] if shap_vals.ndim == 2 else shap_vals[player_idx]
    else:
        vals = shap_vals[player_idx]
    
    if vals.ndim > 1:
        vals = vals.sum(axis=0)
    
    contribs = list(zip(feature_names, vals))
    contribs.sort(key=lambda t: abs(t[1]), reverse=True)
    return contribs[:3]


def run_pipeline(
    df: pd.DataFrame,
    output_dir: str = "output",
    load_col: Optional[str] = None,
    target_col: str = "injury",
) -> dict:
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    X, y = prepare_features(df, load_col=load_col, target_col=target_col)
    feature_names = list(X.columns)
    
    clf, scaler, X_train_s, X_test_s, y_train, y_test = train_model(X, y)
    
    X_full_s = scaler.transform(X.fillna(0))
    y_pred = clf.predict(X_full_s)
    y_proba = clf.predict_proba(X_full_s)[:, 1] if clf.classes_[1] == 1 else clf.predict_proba(X_full_s)[:, 0]
    
    report = classification_report(y, y_pred)
    auc = roc_auc_score(y, y_proba) if y.nunique() > 1 else 0.0
    cm = confusion_matrix(y, y_pred)
    
    print("Classification Report:\n", report)
    print("ROC-AUC:", auc)
    print("Confusion Matrix:\n", cm)
    
    flagged = np.where(y_pred == 1)[0]
    explainer = shap.TreeExplainer(clf, X_full_s)
    shap_vals = explainer.shap_values(X_full_s)
    
    # Normalize SHAP values to always be 2D (n_samples, n_features)
    if hasattr(shap_vals, 'values'):
        shap_vals = shap_vals.values

    if shap_vals.ndim == 3:
        vals_use = shap_vals[:, :, 1]
    elif shap_vals.ndim == 2:
        vals_use = shap_vals
    else:
        vals_use = shap_vals.reshape(1, -1)

    results = []
    for idx in flagged[:20]:
        v = vals_use[idx]
        contribs = list(zip(feature_names, v))
        contribs.sort(key=lambda t: abs(t[1]), reverse=True)
        top3 = contribs[:3]
        results.append({
            "player_idx": int(idx),
            "risk_score": float(y_proba[idx]),
            "top3_risk_factors": top3,
        })
    
    return {
        "model": clf,
        "scaler": scaler,
        "report": report,
        "auc": auc,
        "confusion_matrix": cm,
        "flagged_explanations": results,
        "feature_names": feature_names,
    }