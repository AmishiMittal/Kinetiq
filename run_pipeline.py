"""
Run football injury prediction pipeline.
Downloads datasets from Kaggle (requires kaggle.json) and runs:
  - Feature engineering (ACWR, monotony, joint angles, fatigue, sliding window)
  - XGBoost (tabular biometric+wellness -> binary injury risk)
  - SHAP explainability (top 3 risk factors per flagged player)
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def download_kaggle_datasets():
    """Download NFL and Multimodal datasets from Kaggle."""
    try:
        import kaggle
    except ImportError:
        print("Install kaggle: pip install kaggle")
        return False

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # NFL Playing Surface Analytics
    nfl_dir = data_dir / "nfl"
    nfl_dir.mkdir(exist_ok=True)
    if not (nfl_dir / "InjuryRecord.csv").exists():
        try:
            kaggle.api.competition_download_files("nfl-playing-surface-analytics", path=str(nfl_dir))
            import zipfile
            for z in nfl_dir.glob("*.zip"):
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(nfl_dir)
        except Exception as e:
            print(f"Could not download NFL data: {e}")
            print("Download manually from: https://www.kaggle.com/competitions/nfl-playing-surface-analytics/data")

    # Multimodal Sports Injury
    multi_dir = data_dir / "multimodal"
    multi_dir.mkdir(exist_ok=True)
    if not any(multi_dir.glob("*.csv")) and not any(multi_dir.glob("*.parquet")):
        try:
            kaggle.api.dataset_download_files("anjalibhegam/multimodal-sports-injury-dataset", path=str(multi_dir), unzip=True)
        except Exception as e:
            print(f"Could not download Multimodal data: {e}")
            print("Download manually from: https://www.kaggle.com/datasets/anjalibhegam/multimodal-sports-injury-dataset")

    return True


def build_demo_df(n_samples: int = 2000):
    """Build demo dataframe when real data is unavailable."""
    np.random.seed(42)
    load = np.random.exponential(120, n_samples) + 30
    acwr = load / np.maximum(np.roll(load, 7) + 1e-6, 1)
    acwr[:20] = np.nan

    df = pd.DataFrame({
        "player_id": np.random.randint(1, 100, n_samples),
        "date": pd.date_range("2023-01-01", periods=n_samples, freq="D"),
        "load": load,
        "RPE": np.random.uniform(4, 9, n_samples),
        "Duration": np.random.uniform(30, 120, n_samples),
    })
    # Binary injury target (higher ACWR -> higher risk)
    acwr_valid = acwr[~np.isnan(acwr)]
    p80 = np.percentile(acwr_valid, 80)
    df["injury"] = (acwr > p80).astype(int)
    df.loc[np.isnan(acwr), "injury"] = 0

    return df


def main():
    parser = argparse.ArgumentParser(description="Football Injury Prediction Pipeline")
    parser.add_argument("--download", action="store_true", help="Download Kaggle datasets")
    parser.add_argument("--demo", action="store_true", help="Run on demo synthetic data")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    args = parser.parse_args()

    if args.download:
        download_kaggle_datasets()

    df = None

    # Try NFL data (InjuryRecord + PlayList)
    nfl_injury = Path("data/nfl/InjuryRecord.csv")
    nfl_play = Path("data/nfl/PlayList.csv")
    if nfl_injury.exists() and nfl_play.exists():
        injury = pd.read_csv(nfl_injury)
        playlist = pd.read_csv(nfl_play)
        # Find common key (PlayerKey, gsis_id, etc.)
        keys = [c for c in injury.columns if c in playlist.columns]
        if keys:
            df = playlist.merge(injury, on=keys[0], how="left")
        else:
            df = playlist.copy()
            df["injury"] = 0
        # Set binary injury target
        inj_col = next((c for c in df.columns if "injury" in c.lower()), None)
        if inj_col and inj_col in df.columns:
            df["injury"] = pd.to_numeric(df[inj_col], errors="coerce").fillna(0).astype(int)
        if "load" not in df.columns and not any("load" in c.lower() for c in df.columns):
            df["load"] = np.random.exponential(100, len(df)) + 20

    # Try Multimodal data
    if df is None:
        for p in Path("data/multimodal").rglob("*.csv"):
            try:
                d = pd.read_csv(p, nrows=5000)
                if len(d.columns) >= 3:
                    df = d
                    break
            except Exception:
                pass

    if df is None or args.demo:
        print("Using demo synthetic data (run with --download to fetch real datasets)")
        df = build_demo_df(3000)

    print(f"Running pipeline on {len(df)} rows, {df.shape[1]} columns")
    from model import run_pipeline

    result = run_pipeline(df, output_dir=args.output_dir)

    # Print SHAP top-3 for flagged players
    print("\n" + "=" * 60)
    print("SHAP TOP-3 RISK FACTORS (why each player was flagged)")
    print("=" * 60)
    for r in result["flagged_explanations"][:10]:
        print(f"\nPlayer index {r['player_idx']} | Risk score: {r['risk_score']:.3f}")
        for name, val in r["top3_risk_factors"]:
            print(f"  • {name}: {val:+.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
