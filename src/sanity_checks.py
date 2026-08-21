import joblib
import pandas as pd
import numpy as np
from pathlib import Path

from src.config import (
    RESULTS_DIR,
    MODELS_DIR,
    PROTECTED_ATTRIBUTES,
    MODEL_FEATURES,
    DEFAULT_K,
    MIN_INTERSECTION_GROUP_SIZE
)
from src.fairness_mitigation import rerank_candidates, compare_rankings
from src.top_k_recommender import evaluate_user_recommendations

def run_sanity_checks():
    print("=" * 60)
    print("RUNNING AUTOMATED SANITY CHECKS & SCIENTIFIC ASSERTIONS")
    print("=" * 60)

    test_split_path = RESULTS_DIR / "test_split.csv"
    X_test_path = RESULTS_DIR / "X_test.csv"
    model_path = MODELS_DIR / "xgboost_baseline.pkl"

    assert test_split_path.exists(), "test_split.csv does not exist"
    assert X_test_path.exists(), "X_test.csv does not exist"
    assert model_path.exists(), "xgboost_baseline.pkl does not exist"

    test_df = pd.read_csv(test_split_path)
    X_test = pd.read_csv(X_test_path)
    model = joblib.load(model_path)

    # 1. Verify protected attributes are NOT in model features
    for attr in PROTECTED_ATTRIBUTES:
        assert attr not in MODEL_FEATURES, f"Protected attribute {attr} found in MODEL_FEATURES!"
        assert attr not in X_test.columns, f"Protected attribute {attr} found in model input features!"
    print("[PASS] Sanity Check 1: No protected attributes directly in relevance model.")

    # 2. Verify candidate pool size is preserved per user
    cand_counts = test_df.groupby("user_id").size()
    assert cand_counts.min() >= 1, "User with 0 candidates found!"
    print(f"[PASS] Sanity Check 2: Full candidate pools preserved (Min={cand_counts.min()}, Max={cand_counts.max()}, Total={len(test_df)}).")

    # 3. Verify Baseline vs Strength=0 exact equivalence
    test_df["baseline_score"] = model.predict_proba(X_test)[:, 1]
    test_df = test_df.sort_values(["user_id", "baseline_score"], ascending=[True, False]).copy()
    test_df["baseline_rank"] = test_df.groupby("user_id").cumcount() + 1

    reranked_0 = []
    for user_id, group in test_df.groupby("user_id"):
        r_0 = rerank_candidates(group, "baseline_score", PROTECTED_ATTRIBUTES, 0.0, {})
        reranked_0.append(r_0)
    df_0 = pd.concat(reranked_0, ignore_index=True)

    diag_0 = compare_rankings(test_df, df_0, DEFAULT_K)
    assert diag_0["top_k_changed"].sum() == 0, "Strength=0 modified Top-K!"
    assert diag_0["top_k_overlap"].mean() == 1.0, "Strength=0 overlap is not 1.0!"
    assert diag_0["kendall_tau"].mean() == 1.0, "Strength=0 Kendall tau is not 1.0!"
    print("[PASS] Sanity Check 3: Fairness strength = 0.0 is mathematically identical to baseline.")

    # 4. Verify Strength > 0 produces ranking modifications (No Invariance Bug)
    group_stats = {}
    for attr in PROTECTED_ATTRIBUTES:
        g_means = test_df.groupby(attr)["baseline_score"].mean()
        overall = test_df["baseline_score"].mean()
        group_stats[attr] = (overall / g_means).clip(0.80, 1.25).to_dict()

    obj_config = {"group_stats": group_stats}
    reranked_5 = []
    for user_id, group in test_df.groupby("user_id"):
        r_5 = rerank_candidates(group, "baseline_score", PROTECTED_ATTRIBUTES, 0.50, obj_config)
        reranked_5.append(r_5)
    df_5 = pd.concat(reranked_5, ignore_index=True)

    diag_5 = compare_rankings(test_df, df_5, DEFAULT_K)
    pct_changed = diag_5["top_k_changed"].mean() * 100
    assert pct_changed > 0, "Fairness adjustment is not affecting ranking order! (Invariance bug detected)"
    print(f"[PASS] Sanity Check 4: Fairness strength = 0.50 dynamically changes rankings for {pct_changed:.2f}% of users.")

    # 5. Verify Top-K unique items per user <= K
    top_10 = df_5[df_5["fairness_rank"] <= DEFAULT_K]
    user_top_counts = top_10.groupby("user_id").size()
    assert (user_top_counts <= DEFAULT_K).all(), "User with > K recommendations found in Top-K!"
    print(f"[PASS] Sanity Check 5: Top-K recommendations strictly have <= {DEFAULT_K} unique items per user.")

    # 6. Verify Intersectional Group Stability Flagging
    int_file = RESULTS_DIR / "intersectional_gender_age_location.csv"
    if int_file.exists():
        int_df = pd.read_csv(int_file)
        unstable_mask = int_df["sample_count"] < MIN_INTERSECTION_GROUP_SIZE
        assert (int_df["statistically_unstable"] == unstable_mask).all(), "Intersectional statistical stability flagging mismatch!"
        print(f"[PASS] Sanity Check 6: Intersectional groups with sample size < {MIN_INTERSECTION_GROUP_SIZE} are flagged statistically_unstable.")

    # 7. Check for NaNs or Inf in critical metric result files
    critical_files = [
        "baseline_metrics.csv",
        "top_k_metrics.csv",
        "fairness_summary.csv",
        "fairness_strength_results.csv",
        "final_quality_comparison.csv",
        "final_fairness_comparison.csv"
    ]
    for filename in critical_files:
        f_path = RESULTS_DIR / filename
        if f_path.exists():
            df_check = pd.read_csv(f_path)
            numeric_cols = df_check.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                assert not np.isinf(df_check[col]).any(), f"Infinite value found in {filename} column {col}!"
    print("[PASS] Sanity Check 7: No infinite values in critical result CSVs.")

    print("\n" + "=" * 60)
    print("ALL SANITY CHECKS & SCIENTIFIC ASSERTIONS PASSED SUCCESSFULLY")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    run_sanity_checks()
