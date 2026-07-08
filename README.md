# Contextual CTR Ranking System

## What This Is

This project builds an end-to-end machine learning system for contextual ad ranking. It predicts the probability that a user clicks an ad, ranks candidate ads for a given impression opportunity, handles entities with little or no history, detects engagement drift over time, and sketches how the system can run in production under a low-latency serving budget.

The focus is not only model training. The project covers the full ML system flow: feature engineering, time-based validation, CTR prediction, business-rule-aware ranking, cold-start fallback logic, drift monitoring, and production serving design.

---

## The Data

The project uses two CSV files covering a multi-day impression window.

`impressions.csv` contains impression-level rows. Each row represents an ad impression shown in an interactive digital environment. The target column is `click`, where 1 means the user clicked and 0 means they did not. The dataset includes device signals, surface metadata, anonymized campaign and creative attributes, and session-level interaction signals.

`entities.csv` contains metadata for the contextual entity associated with each impression. It includes safety/category tier, creator type, historical interaction count, and publish timestamp.

The two files join on `entity_id`.

---

## Running the Pipeline

There are five scripts, and they build on each other.

`features.py` must run first because it creates the processed train, validation, and test splits used by the rest of the pipeline.

After that, the remaining scripts can run independently:

* `train.py`
* `rank.py`
* `cold_start.py`
* `drift.py`
* `production_sketch.py`

All outputs are written to the `outputs/` folder, including processed CSV files, trained model artifacts, charts, metrics, and production architecture notes.

---

## What Each File Does

### features.py

Loads the impression and entity metadata files, joins them on `entity_id`, engineers model features, and saves time-based train, validation, and test splits.

The main feature engineering decisions include dropping near-unique identifiers such as device-level IDs and impression IDs because they can cause memorization and poor generalization. Free-text fields are also excluded from the first version to keep the model focused on structured metadata and avoid unnecessary complexity.

Created features include:

* `hour_of_day`
* `day`
* `turn_ratio`
* `entity_age_days`
* log-transformed interaction count

The split is time-based rather than random. Earlier days are used for training, the next day for validation, and the final day for testing. This better reflects production behavior, where the model is trained on historical impressions and used to predict future impressions.

---

### train.py

Trains a LightGBM binary classifier to predict click probability for each ad impression.

LightGBM is a strong fit for this problem because CTR prediction is a tabular, high-cardinality, categorical-heavy ML problem. It handles categorical features efficiently, trains quickly on large datasets, does not require feature scaling, and supports low-latency inference.

The primary metric is log loss because the system needs calibrated click probabilities, not just class labels. A ranking system depends on probability quality, especially when predicted CTR is used to order candidate ads.

The model also reports AUC and feature importance to understand ranking quality and which signals drive prediction behavior.

---

### rank.py

Scores and ranks candidate ads for a given impression opportunity using the trained model.

The ranking layer applies model predictions along with business rules:

1. **Safety gate**
   Removes candidates that are not eligible for the current context before scoring.

2. **Fatigue penalty**
   Penalizes creatives already shown in the current session to reduce repetition and improve user experience.

3. **Uncertainty floor**
   Blends uncertain predictions back toward the global average CTR when the model has weak signal.

This makes the ranking system more production-oriented because the model does not operate blindly. It works inside constraints that protect quality, safety, and user experience.

---

### cold_start.py

Handles new or low-history entities where the model has limited signal.

The system classifies each entity into three stages:

**Cold**
Entities with very few impressions. These use fallback CTR estimates based on similar entities grouped by metadata such as safety/category tier, creator type, and placement.

**Warming**
Entities with moderate history. These use a blended estimate that gradually shifts from fallback CTR toward the model prediction as more impressions accumulate.

**Established**
Entities with enough historical data. These rely mostly on the trained model.

This is important because cold start is not an edge case in recommendation and ad-ranking systems. New entities, new creatives, and sparse-history contexts appear constantly, so fallback logic needs to be part of the core system design.

---

### drift.py

Analyzes how engagement patterns shift over time.

The script tracks CTR movement, feature distribution changes, entity popularity churn, and time-of-day shifts. These signals help identify when the model may become stale or less aligned with current user behavior.

The system uses an exponential moving average CTR feature to capture recent engagement changes while still preserving historical signal. This allows the ranking layer to adapt between full retraining cycles.

In production, this type of signal could be updated frequently and served from a fast feature store or cache.

---

### production_sketch.py

Documents how the local ML pipeline could be moved into a production serving environment.

The proposed serving design includes:

* FastAPI service for inference
* LightGBM model loaded in process
* Redis for low-latency feature lookup
* Kafka for asynchronous click and impression event logging
* Scheduled retraining pipeline using a rolling training window
* Automated model promotion based on validation metrics
* Monitoring for CTR drift, feature drift, latency, and prediction quality

The goal is to keep the serving path fast while still allowing the system to learn from new behavior over time.

---

## Key Findings

Creative and campaign-level features are among the strongest predictors of click behavior. This suggests that candidate retrieval and creative selection may be just as important as the final ranking model.

Contextual metadata still matters. Safety/category tier, placement, session depth, and entity history all contribute useful signal.

Temporal drift is a major concern. CTR patterns, active user behavior, and entity popularity can shift quickly over time, so the system needs monitoring and adaptive features rather than relying only on static training data.

Cold start is also a first-class problem. A large portion of entities can have limited history, so fallback CTR estimates and gradual trust-building are necessary for stable ranking performance.

---

## What Additional Data Would Improve the System

Conversation or content-topic signals would likely improve relevance by helping the model understand what kind of context the ad is being shown in.

User-level history would unlock stronger personalization and could support two-tower or embedding-based ranking architectures at larger scale.

Text embeddings from entity descriptions could improve cold-start handling by finding similar established entities and borrowing their historical CTR behavior.

These additions would make the ranking system more personalized, adaptive, and semantically aware, but they should be introduced only after validating that the added complexity improves production metrics.

---

## Tech Stack

Python, pandas, NumPy, scikit-learn, LightGBM, matplotlib, feature engineering, CTR prediction, ranking systems, time-based validation, drift monitoring, cold-start handling, FastAPI architecture, Redis, Kafka, and production ML system design.
