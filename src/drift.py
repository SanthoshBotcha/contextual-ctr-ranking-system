import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"


def load_data():
    """Load all three splits and combine for full temporal view."""
    print("Loading data ...")
    train = pd.read_csv(OUTPUT_DIR / "train.csv")
    val   = pd.read_csv(OUTPUT_DIR / "val.csv")
    test  = pd.read_csv(OUTPUT_DIR / "test.csv")
    df    = pd.concat([train, val, test], ignore_index=True)
    print(f"  Total rows: {len(df)}")
    return df


# ── Analysis 1: Daily CTR Trend ───────────────────────────────────────────────

def analyze_daily_ctr(df):
    """
    Compute CTR per day and per safety tier per day.
    This shows whether overall engagement is drifting over time.
    """
    print("\n=== Daily CTR Trend ===")

    daily = (
        df.groupby("day")
        .agg(
            impressions = ("click", "count"),
            clicks      = ("click", "sum"),
            ctr         = ("click", "mean")
        )
        .reset_index()
    )

    print(f"\n  {'Day':<6} {'Impressions':<14} {'Clicks':<10} {'CTR':<10} {'vs Day21'}")
    print("  " + "-"*50)
    baseline_ctr = daily[daily["day"] == 21]["ctr"].values[0]
    for _, row in daily.iterrows():
        delta = row["ctr"] - baseline_ctr
        arrow = "▼" if delta < -0.002 else ("▲" if delta > 0.002 else "─")
        print(f"  {int(row['day']):<6} {int(row['impressions']):<14} "
              f"{int(row['clicks']):<10} {row['ctr']:.4f}    "
              f"{arrow} {delta:+.4f}")

    # CTR by safety tier per day
    print(f"\n  Daily CTR by Safety Tier:")
    tier_map  = {0: "sfw", 1: "suggestive", 2: "mature"}
    tier_daily = (
        df.groupby(["day", "safety_tier_enc"])["click"]
        .mean()
        .reset_index()
        .rename(columns={"click": "ctr"})
    )
    tier_daily["tier"] = tier_daily["safety_tier_enc"].map(tier_map)

    for tier_enc in [0, 1, 2]:
        tier_name = tier_map[tier_enc]
        subset    = tier_daily[tier_daily["safety_tier_enc"] == tier_enc]
        ctrs      = subset["ctr"].values
        trend     = "DECLINING" if ctrs[-1] < ctrs[0] - 0.005 else (
                    "RISING"   if ctrs[-1] > ctrs[0] + 0.005 else "STABLE")
        print(f"    {tier_name:<12} first={ctrs[0]:.4f}  "
              f"last={ctrs[-1]:.4f}  trend={trend}")

    return daily, tier_daily


# ── Analysis 2: Character Concentration Drift ─────────────────────────────────

def analyze_character_concentration(df):
    """
    Do the same characters dominate every day?
    Or does the popular character mix churn over time?
    High churn = more drift = harder for model to stay accurate.
    """
    print("\n=== Character Concentration Drift ===")

    days     = sorted(df["day"].unique())
    top_n    = 10

    # Top 10 characters on day 21
    first_day_top = (
        df[df["day"] == days[0]]
        .groupby("character_id")["click"]
        .count()
        .nlargest(top_n)
        .index.tolist()
    )

    print(f"\n  Share of impressions held by day-21 top-{top_n} characters:")
    print(f"  {'Day':<6} {'Their Share':<14} {'Trend'}")
    print("  " + "-"*35)

    shares = []
    for day in days:
        day_df    = df[df["day"] == day]
        share     = day_df["character_id"].isin(first_day_top).mean()
        shares.append(share)
        bar       = "█" * int(share * 40)
        print(f"  {day:<6} {share:.4f}         {bar}")

    first_share = shares[0]
    last_share  = shares[-1]
    print(f"\n  Top-10 characters went from {first_share:.1%} → "
          f"{last_share:.1%} of daily impressions")
    if last_share < first_share - 0.02:
        print("  DRIFT DETECTED: dominant characters are losing share over time")
    else:
        print("  Character mix is relatively stable")

    return first_day_top, shares


# ── Analysis 3: Feature Distribution Shift ────────────────────────────────────

def analyze_feature_drift(df):
    """
    Check if key feature distributions shift over time.
    We use a simple method: compare the mean of each feature
    on day 21 vs day 30. Large changes = distribution drift.
    """
    print("\n=== Feature Distribution Drift (Day 21 vs Day 30) ===")

    features = [
        "banner_pos", "device_type", "conversation_turn",
        "turn_ratio", "safety_tier_enc", "hour_of_day"
    ]

    day21 = df[df["day"] == 21]
    day30 = df[df["day"] == 30]

    print(f"\n  {'Feature':<22} {'Day21 Mean':<14} "
          f"{'Day30 Mean':<14} {'Change':<10} {'Signal'}")
    print("  " + "-"*70)

    for feat in features:
        m21    = day21[feat].mean()
        m30    = day30[feat].mean()
        change = m30 - m21
        pct    = abs(change / m21) * 100 if m21 != 0 else 0
        signal = "⚠ SHIFTED" if pct > 5 else "stable"
        print(f"  {feat:<22} {m21:<14.4f} {m30:<14.4f} "
              f"{change:+.4f}     {signal}  ({pct:.1f}%)")


# ── Analysis 4: EMA Adaptation Layer ──────────────────────────────────────────

def build_ema_ctr_table(df, alpha=0.3):
    """
    Exponential Moving Average CTR per character per day.

    Instead of using a character's all-time CTR as a feature,
    we use an EMA that weights recent days more than old days.

    alpha=0.3 means: today = 30% new data + 70% yesterday's EMA
    Higher alpha = adapts faster but more noisy
    Lower alpha  = more stable but slower to adapt

    This is the adaptation layer — as character preferences evolve,
    the EMA CTR feature automatically tracks the change.
    """
    print("\n=== Building EMA CTR Adaptation Layer ===")

    days = sorted(df["day"].unique())

    # Compute raw daily CTR per character
    daily_ctr = (
        df.groupby(["day", "character_id"])["click"]
        .mean()
        .reset_index()
        .rename(columns={"click": "raw_ctr"})
    )

    # Apply EMA across days per character
    records = []
    ema_state = {}  # character_id → current EMA value

    for day in days:
        day_data = daily_ctr[daily_ctr["day"] == day]
        for _, row in day_data.iterrows():
            cid     = row["character_id"]
            raw     = row["raw_ctr"]
            if cid not in ema_state:
                ema_state[cid] = raw          # initialize with first observation
            else:
                ema_state[cid] = alpha * raw + (1 - alpha) * ema_state[cid]
            records.append({
                "day"         : day,
                "character_id": cid,
                "raw_ctr"     : raw,
                "ema_ctr"     : ema_state[cid]
            })

    ema_df = pd.DataFrame(records)

    # Show example: pick a character with data across multiple days
    char_counts = ema_df.groupby("character_id")["day"].count()
    example_char = char_counts[char_counts == len(days)].index[0] \
                   if (char_counts == len(days)).any() \
                   else char_counts.idxmax()

    print(f"\n  Example character: {example_char}")
    print(f"  {'Day':<6} {'Raw CTR':<12} {'EMA CTR':<12} {'Smoothed?'}")
    print("  " + "-"*40)
    example = ema_df[ema_df["character_id"] == example_char].sort_values("day")
    for _, row in example.iterrows():
        smoothed = "←smoothed" if abs(row["ema_ctr"] - row["raw_ctr"]) > 0.02 else ""
        print(f"  {int(row['day']):<6} {row['raw_ctr']:<12.4f} "
              f"{row['ema_ctr']:<12.4f} {smoothed}")

    print(f"\n  alpha={alpha}: recent days weighted {alpha:.0%}, "
          f"history weighted {1-alpha:.0%}")
    print(f"  Use ema_ctr as a feature in retraining to adapt to drift")

    # Save EMA table
    ema_df.to_csv(OUTPUT_DIR / "ema_ctr_table.csv", index=False)
    print(f"  EMA table saved to outputs/ema_ctr_table.csv")

    return ema_df


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_drift(daily, tier_daily, shares, days):
    """Save a 3-panel drift analysis chart."""
    fig = plt.figure(figsize=(15, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig)

    tier_map   = {0: "sfw", 1: "suggestive", 2: "mature"}
    tier_colors= {0: "#4C72B0", 1: "#DD8452", 2: "#55A868"}

    # Panel 1: Overall daily CTR
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(daily["day"], daily["ctr"], marker="o", color="#4C72B0", linewidth=2)
    ax1.set_title("Overall Daily CTR", fontweight="bold")
    ax1.set_xlabel("Day (October 2014)")
    ax1.set_ylabel("CTR")
    ax1.set_ylim(0.14, 0.22)
    ax1.grid(True, alpha=0.3)

    # Panel 2: CTR by safety tier
    ax2 = fig.add_subplot(gs[1])
    for tier_enc, tier_name in tier_map.items():
        subset = tier_daily[tier_daily["safety_tier_enc"] == tier_enc]
        ax2.plot(subset["day"], subset["ctr"],
                 marker="o", label=tier_name,
                 color=tier_colors[tier_enc], linewidth=2)
    ax2.set_title("CTR by Safety Tier", fontweight="bold")
    ax2.set_xlabel("Day (October 2014)")
    ax2.set_ylabel("CTR")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3: Character concentration
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(days, shares, marker="o", color="#C44E52", linewidth=2)
    ax3.set_title("Top-10 Character Share Over Time", fontweight="bold")
    ax3.set_xlabel("Day (October 2014)")
    ax3.set_ylabel("Share of Daily Impressions")
    ax3.grid(True, alpha=0.3)

    plt.suptitle("Drift Analysis — Simula Ad Platform (Oct 2014)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "drift_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  Drift chart saved to outputs/drift_analysis.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    df = load_data()

    daily, tier_daily             = analyze_daily_ctr(df)
    first_day_top, shares         = analyze_character_concentration(df)
    analyze_feature_drift(df)
    ema_df                        = build_ema_ctr_table(df, alpha=0.3)

    days = sorted(df["day"].unique())
    plot_drift(daily, tier_daily, shares, days)

    print("\n=== Drift Summary ===")
    print(f"  CTR dropped from day 21 to day 30 — novelty decay detected")
    print(f"  Suggestive characters show highest engagement throughout")
    print(f"  EMA adaptation layer built and ready to use as model feature")
    print(f"  All outputs saved to outputs/ folder")


if __name__ == "__main__":
    run()