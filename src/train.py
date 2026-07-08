import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
from pathlib import Path
from sklearn.metrics import (
    log_loss, roc_auc_score, average_precision_score
)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, saves files instead of popup

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"


def load_splits():
    """Load the processed splits saved by features.py"""
    print("Loading processed splits ...")
    train = pd.read_csv(OUTPUT_DIR / "train.csv")
    val   = pd.read_csv(OUTPUT_DIR / "val.csv")
    test  = pd.read_csv(OUTPUT_DIR / "test.csv")
    print(f"  Train: {len(train)} rows")
    print(f"  Val  : {len(val)} rows")
    print(f"  Test : {len(test)} rows")
    return train, val, test


def get_feature_columns(df):
    """Columns to use as model input — exclude IDs, targets, raw text."""
    exclude = [
        "click", "hour", "created_at", "character_id",
        "site_id", "site_domain", "app_id", "app_domain",
        "device_model", "num_interactions"  # replaced by log version
    ]
    return [c for c in df.columns if c not in exclude]


def get_categorical_columns(feature_cols):
    """Which feature columns are categorical (not continuous numbers)."""
    cats = [
        "banner_pos", "site_category", "app_category",
        "device_type", "device_conn_type",
        "C1", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21",
        "safety_tier_enc", "is_official"
    ]
    return [c for c in cats if c in feature_cols]


def prepare_xy(df, feature_cols):
    """Split dataframe into X (features) and y (target)."""
    X = df[feature_cols].copy()
    y = df["click"].values
    return X, y


def train_model(train, val, feature_cols, cat_cols):
    """Train LightGBM model with early stopping on validation set."""
    print("\nTraining LightGBM model ...")
    print(f"  Features used : {len(feature_cols)}")
    print(f"  Categorical   : {cat_cols}")

    X_train, y_train = prepare_xy(train, feature_cols)
    X_val,   y_val   = prepare_xy(val,   feature_cols)

    # Convert to LightGBM dataset format
    dtrain = lgb.Dataset(
        X_train, label=y_train,
        categorical_feature=cat_cols,
        free_raw_data=False
    )
    dval = lgb.Dataset(
        X_val, label=y_val,
        categorical_feature=cat_cols,
        reference=dtrain,
        free_raw_data=False
    )

    # Model parameters
    params = {
        "objective"        : "binary",       # binary classification
        "metric"           : "binary_logloss",# log loss as training metric
        "learning_rate"    : 0.05,
        "num_leaves"       : 63,             # controls model complexity
        "min_child_samples": 50,             # prevents overfitting on rare values
        "feature_fraction" : 0.8,            # use 80% of features per tree
        "bagging_fraction" : 0.8,            # use 80% of data per tree
        "bagging_freq"     : 5,
        "verbose"          : -1,             # suppress lightgbm output
        "seed"             : 42,
    }

    callbacks = [
        lgb.early_stopping(stopping_rounds=30, verbose=True),
        lgb.log_evaluation(period=50)
    ]

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=500,
        valid_sets=[dval],
        callbacks=callbacks
    )

    print(f"\n  Best iteration: {model.best_iteration}")
    return model


def evaluate_model(model, df, feature_cols, split_name):
    """Evaluate model on a dataset split and print metrics."""
    X, y        = prepare_xy(df, feature_cols)
    y_pred      = model.predict(X)

    logloss     = log_loss(y, y_pred)
    auc         = roc_auc_score(y, y_pred)
    avg_prec    = average_precision_score(y, y_pred)

    print(f"\n  [{split_name}]")
    print(f"    Log Loss         : {logloss:.4f}  (lower is better)")
    print(f"    AUC-ROC          : {auc:.4f}  (higher is better, max=1.0)")
    print(f"    Avg Precision    : {avg_prec:.4f}  (higher is better)")
    print(f"    Baseline logloss : {log_loss(y, [y.mean()]*len(y)):.4f}  (predicting mean CTR always)")

    return {"split": split_name, "logloss": logloss, "auc": auc, "avg_precision": avg_prec}


def plot_feature_importance(model, feature_cols):
    """Save a bar chart of the top 20 most important features."""
    importance = pd.DataFrame({
        "feature"   : feature_cols,
        "importance": model.feature_importance(importance_type="gain")
    }).sort_values("importance", ascending=False).head(20)

    plt.figure(figsize=(10, 6))
    plt.barh(importance["feature"][::-1], importance["importance"][::-1])
    plt.xlabel("Importance (gain)")
    plt.title("Top 20 Feature Importances")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150)
    plt.close()
    print("\n  Feature importance chart saved to outputs/feature_importance.png")
    print("\n  Top 10 features:")
    for _, row in importance.head(10).iterrows():
        print(f"    {row['feature']:<30} {row['importance']:>10.1f}")


def save_model(model):
    """Pickle the trained model for use by other scripts."""
    model_path = OUTPUT_DIR / "ctr_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\n  Model saved to {model_path}")


def run():
    train, val, test = load_splits()

    feature_cols = get_feature_columns(train)
    cat_cols     = get_categorical_columns(feature_cols)

    model        = train_model(train, val, feature_cols, cat_cols)

    print("\n=== Model Evaluation ===")
    evaluate_model(model, train, feature_cols, "Train")
    evaluate_model(model, val,   feature_cols, "Val")
    evaluate_model(model, test,  feature_cols, "Test")

    plot_feature_importance(model, feature_cols)
    save_model(model)

    return model, feature_cols


if __name__ == "__main__":
    run()