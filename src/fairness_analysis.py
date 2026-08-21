import joblib
import pandas as pd
import numpy as np
from pathlib import Path

from src.config import (
    RESULTS_DIR,
    PROTECTED_ATTRIBUTES,
    DEFAULT_K
)

def compute_position_exposure(rank):
    """
    Position-weighted exposure: exposure(r) = 1 / log2(r + 1) for r >= 1.
    """
    return 1.0 / np.log2(rank + 1.0)

def compute_group_fairness_metrics(test_df, rank_column="rank", score_column="recommendation_score", k=DEFAULT_K):
    """
    Computes score fairness, recommendation selection fairness, position-weighted
    exposure fairness, and relevance-aware exposure fairness for each protected attribute.
    """
    # Filter to Top-K recommendations
    top_k_df = test_df[test_df[rank_column] <= k].copy()
    top_k_df["position_exposure"] = top_k_df[rank_column].apply(compute_position_exposure)

    detailed_results = {}
    summary_rows = []

    for attr in PROTECTED_ATTRIBUTES:
        # Group metrics across the entire candidate pool
        group_pool_sizes = test_df.groupby(attr).size()
        group_score_means = test_df.groupby(attr)[score_column].mean()
        group_interaction_means = test_df.groupby(attr)["interaction"].mean()

        # Group metrics across Top-K recommendations
        group_top_k_counts = top_k_df.groupby(attr).size().reindex(group_pool_sizes.index, fill_value=0)
        group_top_k_exposures = top_k_df.groupby(attr)["position_exposure"].sum().reindex(group_pool_sizes.index, fill_value=0.0)

        # 1. Selection Rate = Top-K count / candidate count
        selection_rates = group_top_k_counts / group_pool_sizes
        
        # 2. Mean Exposure = Total Position Exposure / candidate count
        mean_exposures = group_top_k_exposures / group_pool_sizes
        
        # Exposure Share = group exposure / total exposure
        total_exp = group_top_k_exposures.sum()
        exposure_shares = group_top_k_exposures / (total_exp if total_exp > 0 else 1.0)

        # 3. Relevance-Aware Exposure = Mean Exposure / Mean Relevance Score
        relevance_ratios = mean_exposures / group_score_means.replace(0, np.nan)

        # Disparate Impact & Statistical Parity Differences
        # A. Score-based
        score_min, score_max = group_score_means.min(), group_score_means.max()
        score_di = score_min / score_max if score_max > 0 else np.nan
        score_spd = score_min - score_max

        # B. Selection Rate
        sel_min, sel_max = selection_rates.min(), selection_rates.max()
        sel_di = sel_min / sel_max if sel_max > 0 else np.nan
        sel_spd = sel_min - sel_max

        # C. Position Exposure
        exp_min, exp_max = mean_exposures.min(), mean_exposures.max()
        exp_di = exp_min / exp_max if exp_max > 0 else np.nan
        exp_spd = exp_min - exp_max

        # D. Relevance-Aware Exposure Disparity
        rel_min, rel_max = relevance_ratios.min(), relevance_ratios.max()
        rel_aware_di = rel_min / rel_max if rel_max > 0 else np.nan

        attr_df = pd.DataFrame({
            "group": group_pool_sizes.index,
            "candidate_count": group_pool_sizes.values,
            "top_k_count": group_top_k_counts.values,
            "selection_rate": selection_rates.values,
            "mean_score": group_score_means.values,
            "mean_interaction_rate": group_interaction_means.values,
            "total_exposure": group_top_k_exposures.values,
            "exposure_share": exposure_shares.values,
            "mean_position_exposure": mean_exposures.values,
            "relevance_aware_ratio": relevance_ratios.values
        })
        detailed_results[attr] = attr_df

        summary_rows.append({
            "protected_attribute": attr,
            "score_based_DI": score_di,
            "score_based_SPD": score_spd,
            "selection_rate_DI": sel_di,
            "selection_rate_SPD": sel_spd,
            "exposure_DI": exp_di,
            "exposure_SPD": exp_spd,
            "relevance_aware_exposure_DI": rel_aware_di
        })

    summary_df = pd.DataFrame(summary_rows)
    return detailed_results, summary_df

def run_fairness_analysis():
    print("=" * 60)
    print("FAIRNESS ANALYSIS (SCORE FAIRNESS VS EXPOSURE FAIRNESS)")
    print("=" * 60)

    test_split_path = RESULTS_DIR / "test_split.csv"
    top_10_path = RESULTS_DIR / "top_10_recommendations.csv"

    if not test_split_path.exists() or not top_10_path.exists():
        raise FileNotFoundError("Prerequisite recommendation files missing. Run top_k_recommender.py first.")

    test_df = pd.read_csv(test_split_path)
    top_10 = pd.read_csv(top_10_path)

    model_path = RESULTS_DIR.parent / "models" / "xgboost_baseline.pkl"
    X_test_path = RESULTS_DIR / "X_test.csv"
    model = joblib.load(model_path)
    X_test = pd.read_csv(X_test_path)
    
    test_df["recommendation_score"] = model.predict_proba(X_test)[:, 1]
    test_df = test_df.sort_values(["user_id", "recommendation_score"], ascending=[True, False]).copy()
    test_df["rank"] = test_df.groupby("user_id").cumcount() + 1

    detailed_results, summary_df = compute_group_fairness_metrics(test_df)

    # Save detailed attribute files
    detailed_results["gender"].to_csv(RESULTS_DIR / "fairness_gender.csv", index=False)
    detailed_results["age_group"].to_csv(RESULTS_DIR / "fairness_age_group.csv", index=False)
    detailed_results["location"].to_csv(RESULTS_DIR / "fairness_location.csv", index=False)

    summary_df.to_csv(RESULTS_DIR / "fairness_summary.csv", index=False)
    summary_df.to_csv(RESULTS_DIR / "top_k_fairness_metrics.csv", index=False)

    # Detailed exposure fairness file
    exposure_rows = []
    for attr, df_attr in detailed_results.items():
        for _, r in df_attr.iterrows():
            exposure_rows.append({
                "attribute": attr,
                "group": r["group"],
                "candidate_count": r["candidate_count"],
                "top_k_count": r["top_k_count"],
                "selection_rate": r["selection_rate"],
                "exposure_share": r["exposure_share"],
                "mean_position_exposure": r["mean_position_exposure"],
                "relevance_aware_ratio": r["relevance_aware_ratio"]
            })
    pd.DataFrame(exposure_rows).to_csv(RESULTS_DIR / "exposure_fairness.csv", index=False)

    print("\nFairness Summary Across Protected Attributes:")
    print(summary_df.to_string(index=False))
    print("\nFairness analysis completed successfully.\n")

if __name__ == "__main__":
    run_fairness_analysis()