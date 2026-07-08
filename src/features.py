import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT / "data"
OUTPUT_DIR  = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_data():
    """Load both CSVs from the data folder."""
    print("Loading impressions.csv ...")
    impressions = pd.read_csv(DATA_DIR / "impressions.csv")
    print(f"  Impressions shape: {impressions.shape}")

    print("Loading characters.csv ...")
    characters = pd.read_csv(DATA_DIR / "characters.csv")
    print(f"  Characters shape:  {characters.shape}")

    return impressions, characters


def merge_data(impressions, characters):
    """Join character metadata onto impressions."""
    print("Merging datasets ...")
    df = impressions.merge(characters, on="character_id", how="left")
    print(f"  Merged shape: {df.shape}")
    return df


def engineer_features(df):
    """Create all model features from raw columns."""
    print("Engineering features ...")

    # ── 1. Parse hour column (YYMMDDHH) ───────────────────────────────────────
    hour_str         = df["hour"].astype(str).str.zfill(8)
    df["year"]       = hour_str.str[:2].astype(int) + 2000
    df["month"]      = hour_str.str[2:4].astype(int)
    df["day"]        = hour_str.str[4:6].astype(int)
    df["hour_of_day"]= hour_str.str[6:8].astype(int)

    # ── 2. Conversation features ───────────────────────────────────────────────
    # How far into the session was the ad shown (0 = start, 1 = end)
    df["turn_ratio"] = df["conversation_turn"] / df["session_msg_count"].clip(lower=1)

    # ── 3. Character age in days at time of impression ─────────────────────────
    impression_date        = pd.to_datetime(
        df[["year","month","day"]].rename(columns={"year":"year","month":"month","day":"day"})
    )
    df["created_at"]       = pd.to_datetime(df["created_at"])
    df["character_age_days"] = (impression_date - df["created_at"]).dt.days.clip(lower=0)

    # ── 4. Log transform num_interactions (fixes power-law skew) ──────────────
    df["log_num_interactions"] = np.log1p(df["num_interactions"])

    # ── 5. Encode safety_tier as ordered number ────────────────────────────────
    safety_map = {"sfw": 0, "suggestive": 1, "mature": 2}
    df["safety_tier_enc"] = df["safety_tier"].map(safety_map).fillna(0).astype(int)

    # ── 6. Encode creator_type as binary ──────────────────────────────────────
    df["is_official"] = (df["creator_type"] == "official").astype(int)

    # ── 7. Encode string categoricals as integer codes ────────────────────────
    df["site_category"] = df["site_category"].astype("category").cat.codes
    df["app_category"]  = df["app_category"].astype("category").cat.codes

    # ── 8. Drop columns we will NOT use in the model ──────────────────────────
    drop_cols = [
        "id",           # unique impression ID - pure leakage
        "device_id",    # unique per device - will overfit
        "device_ip",    # unique per user - will overfit
        "character_name",       # text - not used in main model
        "character_description",# text - not used in main model
        "creator_type",         # replaced by is_official
        "safety_tier",          # replaced by safety_tier_enc
        "year", "month",        # redundant after day/hour_of_day
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    print(f"  Final feature set shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    return df


def split_data(df):
    """
    Time-based train/val/test split.
    Train  : days 21-28
    Val    : day 29
    Test   : day 30
    We never use random split because in production the model
    always predicts future impressions from past training data.
    """
    print("Splitting data by time ...")
    train = df[df["day"] <= 28].copy()
    val   = df[df["day"] == 29].copy()
    test  = df[df["day"] == 30].copy()

    print(f"  Train : {len(train):>7} rows  CTR={train['click'].mean():.4f}")
    print(f"  Val   : {len(val):>7} rows  CTR={val['click'].mean():.4f}")
    print(f"  Test  : {len(test):>7} rows  CTR={test['click'].mean():.4f}")
    return train, val, test


def get_feature_columns(df):
    """Return the list of columns to use as model input features."""
    exclude = ["click", "hour", "created_at", "character_id",
               "site_id", "site_domain", "app_id", "app_domain",
               "device_model"]
    return [c for c in df.columns if c not in exclude]


def run():
    impressions, characters = load_data()
    df                      = merge_data(impressions, characters)
    df                      = engineer_features(df)
    train, val, test        = split_data(df)

    # Save processed splits to outputs folder
    print("Saving processed splits ...")
    train.to_csv(OUTPUT_DIR / "train.csv", index=False)
    val.to_csv(OUTPUT_DIR  / "val.csv",   index=False)
    test.to_csv(OUTPUT_DIR / "test.csv",  index=False)
    print("Done. Files saved to outputs/")
    return train, val, test


if __name__ == "__main__":
    run()