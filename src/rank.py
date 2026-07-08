import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"


def load_model():
    """Load the trained CTR model from disk."""
    model_path = OUTPUT_DIR / "ctr_model.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
    return model


def get_feature_columns():
    """Must match exactly what train.py used."""
    return [
        "banner_pos", "site_category", "app_category",
        "device_type", "device_conn_type",
        "C1", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21",
        "day", "hour_of_day",
        "conversation_turn", "session_msg_count", "turn_ratio",
        "log_num_interactions", "character_age_days",
        "safety_tier_enc", "is_official"
    ]


# ── Business Rules ─────────────────────────────────────────────────────────────

def apply_safety_gate(candidates, context_safety_tier):
    """
    Block mature ads from appearing on sfw or suggestive characters.
    Rule: ad safety must be <= character safety.
    Safety order: sfw=0, suggestive=1, mature=2
    """
    safety_map = {"sfw": 0, "suggestive": 1, "mature": 2}
    context_level = safety_map.get(context_safety_tier, 0)

    before = len(candidates)
    candidates = candidates[
        candidates["safety_tier_enc"] <= context_level
    ].copy()
    after = len(candidates)

    if before != after:
        print(f"  Safety gate: removed {before - after} ads "
              f"(mature ads blocked on {context_safety_tier} character)")
    return candidates


def apply_fatigue_penalty(candidates, recently_shown_c17s):
    """
    Reduce score of creatives the user has already seen recently.
    recently_shown_c17s: list of C17 values shown in last N turns.
    Penalty: multiply predicted CTR by 0.7 for repeated creatives.
    """
    if not recently_shown_c17s:
        return candidates

    fatigue_mask = candidates["C17"].isin(recently_shown_c17s)
    candidates = candidates.copy()
    candidates.loc[fatigue_mask, "predicted_ctr"] *= 0.7

    n_penalized = fatigue_mask.sum()
    if n_penalized > 0:
        print(f"  Fatigue penalty: applied 0.7x discount to "
              f"{n_penalized} recently-seen creatives")
    return candidates


def apply_uncertainty_floor(candidates, global_ctr=0.18):
    """
    When model is uncertain (prediction close to global average),
    blend with global CTR to avoid overconfident ranking.
    Blend weight increases as prediction gets closer to global_ctr.
    """
    candidates = candidates.copy()
    preds = candidates["predicted_ctr"].values
    uncertainty = 1 - np.abs(preds - global_ctr) / global_ctr
    uncertainty = np.clip(uncertainty, 0, 1)
    candidates["predicted_ctr"] = (
        (1 - uncertainty * 0.3) * preds +
        (uncertainty * 0.3) * global_ctr
    )
    return candidates


# ── Core Ranking Function ──────────────────────────────────────────────────────

def rank_candidates(
    model,
    candidates_df,
    context_safety_tier="sfw",
    recently_shown_c17s=None,
    global_ctr=0.18
):
    """
    Given N candidate ads, return them ranked best to worst.

    Parameters
    ----------
    model               : trained LightGBM model
    candidates_df       : DataFrame with one row per candidate ad
    context_safety_tier : safety tier of the character in this impression
    recently_shown_c17s : list of C17 creative IDs shown recently
    global_ctr          : fallback CTR for uncertainty blending

    Returns
    -------
    DataFrame sorted by final_score descending, with scores attached
    """
    feature_cols = get_feature_columns()

    # Make sure all feature columns exist
    for col in feature_cols:
        if col not in candidates_df.columns:
            candidates_df[col] = 0

    candidates = candidates_df.copy()

    # ── Step 1: Model prediction ───────────────────────────────────────────────
    X = candidates[feature_cols]
    candidates["predicted_ctr"] = model.predict(X)

    # ── Step 2: Safety gate — hard filter ─────────────────────────────────────
    candidates = apply_safety_gate(candidates, context_safety_tier)

    if len(candidates) == 0:
        print("  WARNING: All candidates blocked by safety gate. "
              "Returning empty ranking.")
        return candidates

    # ── Step 3: Fatigue penalty — soft discount ────────────────────────────────
    candidates = apply_fatigue_penalty(
        candidates,
        recently_shown_c17s or []
    )

    # ── Step 4: Uncertainty floor — blend toward global CTR ───────────────────
    candidates = apply_uncertainty_floor(candidates, global_ctr)

    # ── Step 5: Final ranking ──────────────────────────────────────────────────
    candidates["final_score"] = candidates["predicted_ctr"]
    candidates = candidates.sort_values(
        "final_score", ascending=False
    ).reset_index(drop=True)
    candidates["rank"] = candidates.index + 1

    return candidates


# ── Demo ───────────────────────────────────────────────────────────────────────

def run():
    """
    Demonstrate ranking on two realistic impression contexts.
    Context A: sfw character, early in conversation
    Context B: mature character, deep in conversation, with fatigue
    """
    model = load_model()

    print("\n" + "="*60)
    print("CONTEXT A: SFW character, turn 1 of 5, morning slot")
    print("="*60)

    # Simulate 5 candidate ads with different C14/C17 (campaign/creative)
    candidates_a = pd.DataFrame([
        # banner_pos, site_cat, app_cat, dev_type, dev_conn,
        # C1,  C14,   C15, C16,  C17,  C18, C19, C20, C21,
        # day, hour, conv_turn, sess_msg, turn_ratio,
        # log_num_inter, char_age, safety_enc, is_official
        dict(banner_pos=0, site_category=3,  app_category=2,
             device_type=1, device_conn_type=2,
             C1=1000, C14=1005, C15=50,  C16=50,  C17=10,
             C18=0, C19=35, C20=100013, C21=320,
             day=30, hour_of_day=9,
             conversation_turn=1, session_msg_count=5,
             turn_ratio=0.2,
             log_num_interactions=5.2, character_age_days=180,
             safety_tier_enc=0, is_official=1),

        dict(banner_pos=0, site_category=3,  app_category=2,
             device_type=1, device_conn_type=2,
             C1=1000, C14=1006, C15=50,  C16=50,  C17=15,
             C18=0, C19=35, C20=100013, C21=320,
             day=30, hour_of_day=9,
             conversation_turn=1, session_msg_count=5,
             turn_ratio=0.2,
             log_num_interactions=5.2, character_age_days=180,
             safety_tier_enc=0, is_official=1),

        dict(banner_pos=1, site_category=3,  app_category=2,
             device_type=1, device_conn_type=2,
             C1=1000, C14=1007, C15=50,  C16=50,  C17=20,
             C18=0, C19=35, C20=100013, C21=320,
             day=30, hour_of_day=9,
             conversation_turn=1, session_msg_count=5,
             turn_ratio=0.2,
             log_num_interactions=5.2, character_age_days=180,
             safety_tier_enc=0, is_official=1),

        dict(banner_pos=0, site_category=3,  app_category=2,
             device_type=1, device_conn_type=2,
             C1=1000, C14=1008, C15=50,  C16=50,  C17=25,
             C18=1, C19=35, C20=100013, C21=320,
             day=30, hour_of_day=9,
             conversation_turn=1, session_msg_count=5,
             turn_ratio=0.2,
             log_num_interactions=5.2, character_age_days=180,
             safety_tier_enc=2, is_official=0),   # ← mature ad!

        dict(banner_pos=0, site_category=3,  app_category=2,
             device_type=1, device_conn_type=2,
             C1=1000, C14=1009, C15=50,  C16=50,  C17=30,
             C18=0, C19=35, C20=100013, C21=320,
             day=30, hour_of_day=9,
             conversation_turn=1, session_msg_count=5,
             turn_ratio=0.2,
             log_num_interactions=5.2, character_age_days=180,
             safety_tier_enc=0, is_official=1),
    ])

    ranked_a = rank_candidates(
        model,
        candidates_a,
        context_safety_tier="sfw",
        recently_shown_c17s=None
    )

    print("\n  Final Ranking:")
    print(f"  {'Rank':<6} {'C14 (Campaign)':<18} "
          f"{'C17 (Creative)':<18} {'Score':<10} {'Notes'}")
    print("  " + "-"*65)
    for _, row in ranked_a.iterrows():
        notes = "BLOCKED" if row["rank"] == 999 else ""
        print(f"  {int(row['rank']):<6} {int(row['C14']):<18} "
              f"{int(row['C17']):<18} {row['final_score']:.4f}    {notes}")

    print("\n" + "="*60)
    print("CONTEXT B: Mature character, turn 8 of 10, with fatigue")
    print("="*60)

    candidates_b = pd.DataFrame([
        dict(banner_pos=0, site_category=3,  app_category=2,
             device_type=1, device_conn_type=2,
             C1=1000, C14=2001, C15=50,  C16=50,  C17=50,
             C18=0, C19=35, C20=100013, C21=320,
             day=30, hour_of_day=22,
             conversation_turn=8, session_msg_count=10,
             turn_ratio=0.8,
             log_num_interactions=6.1, character_age_days=30,
             safety_tier_enc=2, is_official=0),

        dict(banner_pos=0, site_category=3,  app_category=2,
             device_type=1, device_conn_type=2,
             C1=1000, C14=2002, C15=50,  C16=50,  C17=55,
             C18=0, C19=35, C20=100013, C21=320,
             day=30, hour_of_day=22,
             conversation_turn=8, session_msg_count=10,
             turn_ratio=0.8,
             log_num_interactions=6.1, character_age_days=30,
             safety_tier_enc=1, is_official=0),

        dict(banner_pos=1, site_category=3,  app_category=2,
             device_type=1, device_conn_type=2,
             C1=1000, C14=2003, C15=50,  C16=50,  C17=60,
             C18=1, C19=35, C20=100013, C21=320,
             day=30, hour_of_day=22,
             conversation_turn=8, session_msg_count=10,
             turn_ratio=0.8,
             log_num_interactions=6.1, character_age_days=30,
             safety_tier_enc=2, is_official=1),

        dict(banner_pos=0, site_category=3,  app_category=2,
             device_type=1, device_conn_type=2,
             C1=1000, C14=2004, C15=50,  C16=50,  C17=50,
             C18=0, C19=35, C20=100013, C21=320,
             day=30, hour_of_day=22,
             conversation_turn=8, session_msg_count=10,
             turn_ratio=0.8,
             log_num_interactions=6.1, character_age_days=30,
             safety_tier_enc=0, is_official=1),
    ])

    # C17=50 was already shown recently — fatigue applies
    ranked_b = rank_candidates(
        model,
        candidates_b,
        context_safety_tier="mature",
        recently_shown_c17s=[50]
    )

    print("\n  Final Ranking:")
    print(f"  {'Rank':<6} {'C14 (Campaign)':<18} "
          f"{'C17 (Creative)':<18} {'Score':<10} {'Fatigue?'}")
    print("  " + "-"*65)
    for _, row in ranked_b.iterrows():
        fatigue = "YES - discounted" if row["C17"] == 50 else ""
        print(f"  {int(row['rank']):<6} {int(row['C14']):<18} "
              f"{int(row['C17']):<18} {row['final_score']:.4f}    {fatigue}")

    # Save sample output
    output_path = OUTPUT_DIR / "sample_rankings.csv"
    pd.concat([ranked_a, ranked_b]).to_csv(output_path, index=False)
    print(f"\n  Sample rankings saved to {output_path}")


if __name__ == "__main__":
    run()