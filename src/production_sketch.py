"""
production_sketch.py

Simula Ad Serving — Production Architecture Sketch
===================================================
This file documents two things clearly and separately:

1. What we actually built is a fully runnable local ML pipeline
2. What production would require is the infrastructure needed to serve this system at <50ms p99 latency at real world scale

"""

from pathlib import Path

ROOT       = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"


ARCHITECTURE = """
╔══════════════════════════════════════════════════════════════════╗
║         SIMULA AD SERVING — PRODUCTION ARCHITECTURE             ║
║                   Target: <50ms p99 latency                      ║
╚══════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 1: WHAT I ACTUALLY BUILT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  A fully runnable end-to-end ML pipeline in five Python modules.
  Every file runs locally with no external dependencies beyond the Python packages in requirements.txt.

  features.py
  ───────────
  - Loads impressions.csv (1M rows) and characters.csv (5K rows)
  - Merges on character_id
  - Engineers 8 new features:
      hour_of_day, day        parsed from YYMMDDHH timestamp
      turn_ratio              conversation_turn / session_msg_count
      character_age_days      days since character was published
      log_num_interactions    log transform of power-law column
      safety_tier_enc         sfw=0, suggestive=1, mature=2
      is_official             binary flag for creator type
      site/app_category       encoded from string to integer
  - Time-based train/val/test split
      Train  : days 21-28  (872,594 rows)
      Val    : day 29      (104,450 rows)
      Test   : day 30      ( 22,956 rows)
  - Saves processed splits to outputs/

  train.py
  ────────
  - Trains LightGBM binary classifier on train split
  - Early stopping on validation log loss
  - Evaluates on all three splits
      Test AUC-ROC  : 0.6491
      Test Log Loss : 0.4273  (baseline: 0.4487)
  - Saves trained model to outputs/ctr_model.pkl
  - Saves feature importance chart to outputs/feature_importance.png
  - Key finding: C17 (creative ID) and C14 (campaign ID) are
    the two strongest predictors by a large margin

  rank.py
  ───────
  - Loads trained model from disk
  - Given N candidate ads scores each with model.predict()
  - Applies three business rules on top of model score:
      Safety gate hard blocks mature ads on sfw characters
      Fatigue penalty 0.7x discount on recently seen creatives
      Uncertainty floor blends toward global CTR when uncertain
  - Returns candidates sorted by final score descending
  - Saves sample rankings to outputs/sample_rankings.csv

  cold_start.py
  ─────────────
  - Classifies every character into three tiers:
      Cold        0-49 impressions    2,303 characters (46%)
      Warming     50-499 impressions  2,110 characters (42%)
      Established 500+ impressions      562 characters (11%)
  - Builds fallback CTR table grouped by
    safety_tier x is_official x banner_pos
  - Cold characters use fallback CTR from similar characters
  - Warming characters blend model + fallback proportionally
  - Established characters use full model prediction
  - Saves estimates to outputs/cold_start_estimates.csv

  drift.py
  ────────
  - Analyzes CTR trend across all 9 days
      All three safety tiers declining by end of window
      Mature characters consistently highest CTR throughout
  - Detects character concentration shift
      Top-10 day-21 characters dropped from 2.1% to 0.5% share
  - Detects feature distribution shifts
      hour_of_day shifted 70.2% — user base active at
      completely different times by day 30
      banner_pos shifted 12.6%
      safety_tier_enc shifted 5.0%
  - Builds EMA adaptation layer (alpha=0.3)
      Smooths noisy per-character daily CTR
      Automatically tracks character preference shifts
  - Saves drift chart to outputs/drift_analysis.png
  - Saves EMA table to outputs/ema_ctr_table.csv

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 2: WHAT PRODUCTION WOULD REQUIRE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  To take this local pipeline and serve it at <50ms p99 latency at real world scale, the following infrastructure
  would need to be built around it.

  The ML logic itself is the model, the ranking rules, the cold start handler, the EMA adapter, which does not change.
  What changes is the infrastructure layer around it.

  REQUEST FLOW
  ────────────

  [AI App]
     │
     │  POST /rank  {character_id, conversation_turn,
     │               session_msg_count, hour, device_*,
     │               site_*, app_*, candidate_ads[]}
     ▼
  [API Gateway / Load Balancer]              ~1ms
     │
     ▼
  [Serving Service — FastAPI]
     │
     ├─► [1] Feature Assembly                ~3ms
     │        │
     │        ├── Static character features  ← Redis
     │        │   safety_tier, is_official,
     │        │   log_num_interactions,
     │        │   character_age_days
     │        │
     │        ├── EMA CTR feature            ← Redis
     │        │   updated every 15 min
     │        │   by background job
     │        │
     │        └── Request features           ← payload
     │            banner_pos, hour_of_day,
     │            conversation_turn,
     │            turn_ratio, C-features
     │
     ├─► [2] Safety Gate                     ~0.5ms
     │        Hard filter from Redis lookup
     │        No model call needed here
     │
     ├─► [3] Model Inference                 ~3ms
     │        LightGBM predict() on N
     │        candidates in one batch call
     │        Model lives in memory
     │        no disk I/O at serve time
     │
     ├─► [4] Business Rules                  ~0.5ms
     │        Fatigue penalty
     │        Uncertainty floor
     │
     └─► [5] Rank and Return                 ~0.5ms
              Sort by final_score desc
              Return top-K ad IDs

  LATENCY BUDGET
  ──────────────

  Step                          Budget
  ───────────────────────────── ───────
  Network in                    ~5ms
  Feature assembly via Redis    ~3ms
  Safety gate                   ~0.5ms
  LightGBM inference            ~3ms
  Business rules                ~0.5ms
  Rank and return               ~0.5ms
  Network out                   ~5ms
  ───────────────────────────── ───────
  TOTAL                         ~18ms p50
                                ~35ms p99
                                within 50ms budget

  WHY LIGHTGBM FITS THIS BUDGET
  ──────────────────────────────
  LightGBM predicts 100 candidates in ~2ms in process.
  No network hop to a model server needed. Model pkl loaded into memory at pod startup.
  This is why we chose LightGBM over a neural network as a two tower model would need GPU infrastructure or careful
  CPU optimization to hit the same latency.

  INFRASTRUCTURE REQUIRED
  ───────────────────────

  The serving layer would be built on FastAPI. It is async,
  lightweight, and Python native which means the same modeling
  code we wrote locally plugs straight into it without
  translation. No need for a separate model serveras the LightGBM
  runs in process inside the FastAPI pod, which is what keeps
  inference at ~2ms instead of adding a network hop.

  Character features need to be available in under a millisecond
  at serve time. Recomputing them from CSV on every request is
  not viable at scale, so they live in Redis. Static features
  like safety_tier, is_official, and character_age_days get
  written once and refreshed every 24 hours. The EMA CTR value
  per character gets refreshed every 15 minutes by a background
  job as new click data arrives. This is what makes drift
  adaptation work in real time without retraining the model.

  The trained model itself lives in S3 as a pickle file. When
  a serving pod starts up it pulls the latest model from S3
  and loads it into memory. From that point on every prediction
  is a pure in memory operation with no disk or network I/O.
  When the daily retraining job produces a new model it uploads
  to S3 and pods reload it on their next health check cycle.

  Click events flow from the serving pods into Kafka
  asynchronously. This is important because the serving pod does not
  wait for the event to be written before returning the response.
  Kafka decouples the serving path from the data pipeline
  completely. The EMA updater job consumes from Kafka every
  15 minutes and writes fresh CTR values back to Redis.

  Retraining runs as a daily Airflow DAG on a rolling 30 day
  window of impression data. If the new model's AUC on the
  last 24 hour holdout drops more than 5 percent compared to
  the current production model, the DAG triggers an alert and
  rolls back automatically. Otherwise it promotes the new model
  to S3 and serving pods pick it up.

  Prometheus scrapes CTR, latency, and error rate metrics from
  every serving pod. Grafana dashboards break these down by
  safety tier cohort so you can see immediately if mature
  characters are drifting differently from sfw characters
  without digging through logs.


  WHAT REDIS STORES
  ─────────────────
  Each character gets two keys in Redis. The first holds its
  static metadata such as safety_tier_enc, is_official, character_
  age_days, and log_num_interactions. These do not change often
  so they carry a 24 hour TTL and get refreshed by the same
  daily retraining job that updates the model.

  The second key holds the character's current EMA CTR — a
  single float that gets updated every 15 minutes as new click
  data flows in from Kafka. This is the adaptation layer. When
  a character's engagement shifts, this value moves with it
  and the serving layer picks up the change automatically
  without waiting for a full model retrain.

  Two additional global keys handle edge cases. A fallback CTR
  key stores the average CTR broken down by safety tier — this
  is what cold start characters get served until they accumulate
  enough impressions to trust their own signal. A session key
  tracks which C17 creative IDs a user has already seen in the
  current session, which is what powers the fatigue penalty in
  the ranking layer. Session keys expire when the session ends.

  DRIFT ADAPTATION IN PRODUCTION
  ───────────────────────────────

  Every 15 minutes - EMA updater job:
  ┌──────────────────────────────────────────┐
  │  Read last 15min of impression logs      │
  │  Compute raw CTR per character           │
  │  Apply EMA: new = 0.3*raw + 0.7*old      │
  │  Write updated EMA CTR to Redis          │
  │  Serving pods pick it up automatically   │
  └──────────────────────────────────────────┘

  Daily - model retraining job:
  ┌──────────────────────────────────────────┐
  │  Retrain LightGBM on rolling 30 day      │
  │  window of impression data               │
  │  Evaluate on last 24h holdout            │
  │  If AUC drops more than 5% alert         │
  │  and rollback to previous model          │
  │  Else push new model.pkl to all pods     │
  └──────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 3: WHAT ADDITIONAL DATA WOULD UNLOCK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  These improvements are not possible with current data.
  They require additional data collection or infrastructure.

  1. Character embeddings from description text
     Requires: conversation transcript data to reach potential
     Benefit: smarter cold start via nearest neighbor lookup
     on character embedding space rather than tier based fallback
     Why not built now: current fallback works well, added
     complexity not justified until v1 is in production

  2. Two-tower neural model
     Requires: reliable user identity + 100M+ impressions
     Benefit: learned embeddings power approximate nearest
     neighbor retrieval for candidate generation at scale
     Why not built now: LightGBM matches accuracy at current
     data size with far lower serving complexity and latency

  3. Online learning
     Requires: Kafka streaming pipeline infrastructure
     Benefit: reduces drift adaptation lag from 24h to minutes
     Why not built now: engineering infrastructure problem
     not a modeling problem — EMA updates every 15 minutes
     serve as a lightweight proxy until infrastructure exists

  4. A/B testing framework
     Requires: live serving infrastructure with real user traffic
     Benefit: safe model rollout with automatic rollback
     Why not built now: cannot be validated on historical data
     needs production deployment first

  5. Conversation content signals
     Requires: anonymized topic signals from conversation text
     Benefit: dramatically improves ad relevance by matching
     ad context to what user is actually discussing
     Why not built now: data does not exist in current dataset
"""


def run():
    print(ARCHITECTURE)

    out_path = OUTPUT_DIR / "production_sketch.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(ARCHITECTURE)
    print(f"\n  Architecture sketch saved to {out_path}")


if __name__ == "__main__":
    run()