import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"
DATA_DIR   = ROOT / "data"


def load_model():
    with open(OUTPUT_DIR / "ctr_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model


def load_training_data():
    """Load train split to compute fallback statistics."""
    return pd.read_csv(OUTPUT_DIR / "train.csv")


# ── Step 1: Classify how much data a character has ────────────────────────────

def classify_character(character_id, train_df):
    """
    Determine if a character is cold, warming, or established.

    Cold     : 0-49 impressions   → no reliable CTR signal
    Warming  : 50-499 impressions → some signal but not fully trusted
    Established: 500+ impressions → enough data to trust fully

    These thresholds are based on statistical reliability:
    - Need at least 5 clicks to estimate CTR (at 18% CTR, need ~28 impressions)
    - We use 50 as a conservative threshold for graduation from cold
    """
    char_data = train_df[train_df["character_id"] == character_id]
    n_impressions = len(char_data)
    n_clicks      = char_data["click"].sum() if n_impressions > 0 else 0

    if n_impressions == 0:
        status = "cold"
    elif n_impressions < 50:
        status = "cold"
    elif n_impressions < 500:
        status = "warming"
    else:
        status = "established"

    return {
        "character_id"  : character_id,
        "n_impressions" : n_impressions,
        "n_clicks"      : n_clicks,
        "actual_ctr"    : n_clicks / n_impressions if n_impressions > 0 else None,
        "status"        : status
    }


# ── Step 2: Build fallback CTR lookup table ───────────────────────────────────

def build_fallback_table(train_df):
    """
    For cold start characters, we need a fallback CTR.
    We compute average CTR grouped by:
    - safety_tier_enc  (most important grouping)
    - is_official      (second grouping)
    - banner_pos       (position matters for CTR)

    This gives us a specific fallback for each combination.
    Example: a new sfw community character at banner_pos=0
    gets the average CTR of all sfw community characters at pos=0.
    """
    print("Building fallback CTR table ...")

    fallback = (
        train_df
        .groupby(["safety_tier_enc", "is_official", "banner_pos"])
        .agg(
            fallback_ctr     = ("click", "mean"),
            n_impressions    = ("click", "count")
        )
        .reset_index()
    )

    # Also compute global fallback (when even the group has no data)
    global_ctr = train_df["click"].mean()

    print(f"  Global CTR fallback : {global_ctr:.4f}")
    print(f"  Fallback table rows : {len(fallback)}")
    print("\n  Sample fallback CTRs by safety tier:")

    sample = fallback.groupby("safety_tier_enc").agg(
        avg_fallback=("fallback_ctr", "mean")
    ).reset_index()
    tier_map = {0: "sfw", 1: "suggestive", 2: "mature"}
    for _, row in sample.iterrows():
        tier = tier_map.get(row["safety_tier_enc"], "unknown")
        print(f"    {tier:<12} avg CTR = {row['avg_fallback']:.4f}")

    return fallback, global_ctr


# ── Step 3: Get CTR estimate for any character ────────────────────────────────

def get_ctr_estimate(
    character_id,
    safety_tier_enc,
    is_official,
    banner_pos,
    train_df,
    fallback_table,
    global_ctr,
    model=None,
    feature_row=None
):
    """
    Returns a CTR estimate and explains how it was derived.

    Logic:
    1. If established  → use model prediction (full trust)
    2. If warming      → blend model prediction with fallback CTR
    3. If cold         → use fallback CTR from similar characters
    4. If fallback missing → use global CTR
    """
    info = classify_character(character_id, train_df)
    status = info["status"]

    # ── Established: trust the model fully ────────────────────────────────────
    if status == "established":
        if model is not None and feature_row is not None:
            pred = model.predict(feature_row)[0]
        else:
            pred = info["actual_ctr"]
        return {
            "character_id"  : character_id,
            "status"        : status,
            "n_impressions" : info["n_impressions"],
            "ctr_estimate"  : pred,
            "method"        : "model prediction (full trust)",
            "actual_ctr"    : info["actual_ctr"]
        }

    # ── Look up fallback CTR from similar characters ───────────────────────────
    match = fallback_table[
        (fallback_table["safety_tier_enc"] == safety_tier_enc) &
        (fallback_table["is_official"]     == is_official) &
        (fallback_table["banner_pos"]      == banner_pos)
    ]
    fallback_ctr = match["fallback_ctr"].values[0] if len(match) > 0 else global_ctr

    # ── Warming: blend model + fallback ───────────────────────────────────────
    if status == "warming":
        if model is not None and feature_row is not None:
            model_pred = model.predict(feature_row)[0]
        else:
            model_pred = info["actual_ctr"] or global_ctr

        # Blend weight: more impressions → trust model more
        # At 50 impressions: 10% model, 90% fallback
        # At 499 impressions: 99% model, 1% fallback
        blend_weight = (info["n_impressions"] - 50) / (500 - 50)
        blend_weight = np.clip(blend_weight, 0.1, 0.99)
        estimate     = blend_weight * model_pred + (1 - blend_weight) * fallback_ctr

        return {
            "character_id"  : character_id,
            "status"        : status,
            "n_impressions" : info["n_impressions"],
            "ctr_estimate"  : estimate,
            "method"        : f"blended (model {blend_weight:.0%} + fallback {1-blend_weight:.0%})",
            "actual_ctr"    : info["actual_ctr"]
        }

    # ── Cold: use fallback entirely ───────────────────────────────────────────
    return {
        "character_id"  : character_id,
        "status"        : status,
        "n_impressions" : info["n_impressions"],
        "ctr_estimate"  : fallback_ctr,
        "method"        : "fallback CTR from similar characters",
        "actual_ctr"    : info["actual_ctr"]
    }


# ── Step 4: Demo ──────────────────────────────────────────────────────────────

def run():
    print("Loading data and model ...")
    train_df       = load_training_data()
    model          = load_model()
    fallback_table, global_ctr = build_fallback_table(train_df)

    # Pick example characters from different status buckets
    impression_counts = train_df.groupby("character_id").size()

    cold_chars        = impression_counts[impression_counts < 50].index.tolist()
    warming_chars     = impression_counts[
        (impression_counts >= 50) & (impression_counts < 500)
    ].index.tolist()
    established_chars = impression_counts[impression_counts >= 500].index.tolist()

    print(f"\n=== Character Cold Start Status in Training Data ===")
    print(f"  Cold        (0-49 impressions)  : {len(cold_chars):>5} characters")
    print(f"  Warming  (50-499 impressions)   : {len(warming_chars):>5} characters")
    print(f"  Established (500+ impressions)  : {len(established_chars):>5} characters")

    # Demo: one character from each bucket + a brand new one
    examples = []

    # Brand new character — never seen before
    examples.append(("BRAND_NEW_999", 0, 1, 0))  # sfw, official, pos=0

    # Cold character
    if cold_chars:
        cid = cold_chars[0]
        row = train_df[train_df["character_id"] == cid].iloc[0]
        examples.append((cid, int(row["safety_tier_enc"]),
                         int(row["is_official"]), int(row["banner_pos"])))

    # Warming character
    if warming_chars:
        cid = warming_chars[0]
        row = train_df[train_df["character_id"] == cid].iloc[0]
        examples.append((cid, int(row["safety_tier_enc"]),
                         int(row["is_official"]), int(row["banner_pos"])))

    # Established character
    if established_chars:
        cid = established_chars[0]
        row = train_df[train_df["character_id"] == cid].iloc[0]
        examples.append((cid, int(row["safety_tier_enc"]),
                         int(row["is_official"]), int(row["banner_pos"])))

    print(f"\n=== CTR Estimates for Example Characters ===\n")
    print(f"  {'Character':<22} {'Status':<14} {'Impressions':<14} "
          f"{'CTR Estimate':<14} {'Method'}")
    print("  " + "-"*90)

    results = []
    for char_id, safety, official, pos in examples:
        result = get_ctr_estimate(
            character_id    = char_id,
            safety_tier_enc = safety,
            is_official     = official,
            banner_pos      = pos,
            train_df        = train_df,
            fallback_table  = fallback_table,
            global_ctr      = global_ctr
        )
        results.append(result)
        actual_str = f"{result['actual_ctr']:.4f}" if result["actual_ctr"] else "N/A"
        print(f"  {str(char_id):<22} {result['status']:<14} "
              f"{result['n_impressions']:<14} "
              f"{result['ctr_estimate']:.4f}         "
              f"{result['method']}")

    # Save results
    out = pd.DataFrame(results)
    out.to_csv(OUTPUT_DIR / "cold_start_estimates.csv", index=False)
    print(f"\n  Results saved to outputs/cold_start_estimates.csv")

    print(f"\n=== Graduation Thresholds (when to trust model fully) ===")
    print(f"  Cold → Warming     : 50 impressions")
    print(f"  Warming → Established : 500 impressions")
    print(f"  At 500 impressions, expected clicks seen = "
          f"{int(500 * global_ctr)} (statistically reliable)")


if __name__ == "__main__":
    run()