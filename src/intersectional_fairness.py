import joblib
import pandas as pd
import numpy as np
from pathlib import Path

from src.config import (
    RESULTS_DIR,
    MODELS_DIR,
    DEFAULT_K,
    MIN_INTERSECTION_GROUP_SIZE
)
from src.fairness_analysis import compute_position_exposure

def compute_intersectional_table(test_df, group_cols, rank_col="rank", score_col="recommendation_score", k=DEFAULT_K):
    """
    Computes detailed metrics for an intersectional subgroup configuration.
    """
    top_k_df = test_df[test_df[rank_col] <= k].copy()
    top_k_df["position_exposure"] = top_k_df[rank_col].apply(compute_position_exposure)

    # Group full test pool
    pool_grouped = test_df.groupby(group_cols, dropna=False)
    pool_counts = pool_grouped.size()
    pool_scores = pool_grouped[score_col].mean()
    pool_interactions = pool_grouped["interaction"].mean()

    # Group top-k recommendations
    top_k_grouped = top_k_df.groupby(group_cols, dropna=False)
    top_k_counts = top_k_grouped.size().reindex(pool_counts.index, fill_value=0)
    top_k_exposures = top_k_grouped["position_exposure"].sum().reindex(pool_counts.index, fill_value=0.0)

    selection_rates = top_k_counts / pool_counts
    mean_exposures = top_k_exposures / pool_counts
    total_exp = top_k_exposures.sum()
    exposure_shares = top_k_exposures / (total_exp if total_exp > 0 else 1.0)
    relevance_ratios = mean_exposures / pool_scores.replace(0, np.nan)

    # Statistical stability flag
    is_unstable = pool_counts < MIN_INTERSECTION_GROUP_SIZE

    # Normal approximation / Wilson 95% CI for selection rate
    z = 1.96
    ci_lowers = []
    ci_uppers = []
    for count, n in zip(top_k_counts, pool_counts):
        p = count / n if n > 0 else 0
        denom = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denom
        margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
        ci_lowers.append(max(0.0, center - margin))
        ci_uppers.append(min(1.0, center + margin))

    max_exp = mean_exposures.max()
    max_sel = selection_rates.max()
    exposure_di = mean_exposures / (max_exp if max_exp > 0 else np.nan)
    exposure_spd = mean_exposures - max_exp
    selection_di = selection_rates / (max_sel if max_sel > 0 else np.nan)
    selection_spd = selection_rates - max_sel

    rows = []
    for idx in pool_counts.index:
        group_label = " x ".join([str(val) for val in (idx if isinstance(idx, tuple) else (idx,))])
        row_dict = {
            "intersection": group_label,
            "sample_count": int(pool_counts[idx]),
            "statistically_unstable": bool(is_unstable[idx]),
            "top_k_count": int(top_k_counts[idx]),
            "selection_rate": float(selection_rates[idx]),
            "selection_rate_ci_lower": float(ci_lowers[len(rows)]),
            "selection_rate_ci_upper": float(ci_uppers[len(rows)]),
            "mean_score": float(pool_scores[idx]),
            "mean_interaction_rate": float(pool_interactions[idx]),
            "total_exposure": float(top_k_exposures[idx]),
            "exposure_share": float(exposure_shares[idx]),
            "mean_position_exposure": float(mean_exposures[idx]),
            "exposure_DI": float(exposure_di[idx]),
            "exposure_SPD": float(exposure_spd[idx]),
            "selection_DI": float(selection_di[idx]),
            "selection_SPD": float(selection_spd[idx]),
            "relevance_aware_ratio": float(relevance_ratios[idx]) if not np.isnan(relevance_ratios[idx]) else 0.0
        }
        # Add individual column names
        if isinstance(idx, tuple):
            for col, val in zip(group_cols, idx):
                row_dict[col] = val
        else:
            row_dict[group_cols[0]] = idx
        rows.append(row_dict)

    return pd.DataFrame(rows)

def run_intersectional_fairness():
    print("=" * 60)
    print("INTERSECTIONAL FAIRNESS EVALUATION")
    print("=" * 60)

    test_split_path = RESULTS_DIR / "test_split.csv"
    X_test_path = RESULTS_DIR / "X_test.csv"
    model_path = MODELS_DIR / "xgboost_baseline.pkl"

    test_df = pd.read_csv(test_split_path)
    X_test = pd.read_csv(X_test_path)
    model = joblib.load(model_path)

    test_df["recommendation_score"] = model.predict_proba(X_test)[:, 1]
    test_df = test_df.sort_values(["user_id", "recommendation_score"], ascending=[True, False]).copy()
    test_df["rank"] = test_df.groupby("user_id").cumcount() + 1

    # 1. Pairwise and 3-way Intersections
    df_gender_age = compute_intersectional_table(test_df, ["gender", "age_group"])
    df_gender_loc = compute_intersectional_table(test_df, ["gender", "location"])
    df_age_loc = compute_intersectional_table(test_df, ["age_group", "location"])
    df_3way = compute_intersectional_table(test_df, ["gender", "age_group", "location"])

    df_gender_age.to_csv(RESULTS_DIR / "intersectional_gender_age.csv", index=False)
    df_gender_loc.to_csv(RESULTS_DIR / "intersectional_gender_location.csv", index=False)
    df_age_loc.to_csv(RESULTS_DIR / "intersectional_age_location.csv", index=False)
    df_3way.to_csv(RESULTS_DIR / "intersectional_gender_age_location.csv", index=False)

    # Summary table across intersections
    intersections = [
        ("Gender x Age Group", df_gender_age),
        ("Gender x Location", df_gender_loc),
        ("Age Group x Location", df_age_loc),
        ("Gender x Age x Location (3-Way)", df_3way)
    ]

    summary_rows = []
    comparison_rows = []

    for name, df_int in intersections:
        stable_subset = df_int[~df_int["statistically_unstable"]]
        if len(stable_subset) > 0:
            worst_stable_exp_di = stable_subset["exposure_DI"].min()
            worst_stable_sel_di = stable_subset["selection_DI"].min()
            max_stable_gap = 1.0 - worst_stable_exp_di
        else:
            worst_stable_exp_di = df_int["exposure_DI"].min()
            worst_stable_sel_di = df_int["selection_DI"].min()
            max_stable_gap = 1.0 - worst_stable_exp_di

        unstable_count = int(df_int["statistically_unstable"].sum())
        total_subgroups = len(df_int)

        summary_rows.append({
            "intersection_type": name,
            "total_subgroups": total_subgroups,
            "statistically_unstable_subgroups (< 30)": unstable_count,
            "worst_exposure_DI (stable)": worst_stable_exp_di,
            "worst_selection_DI (stable)": worst_stable_sel_di,
            "max_fairness_gap": max_stable_gap,
            "mean_exposure_DI": df_int["exposure_DI"].mean()
        })

        # Save comparative records
        for _, r in df_int.iterrows():
            comparison_rows.append({
                "intersection_type": name,
                "intersection": r["intersection"],
                "sample_count": r["sample_count"],
                "statistically_unstable": r["statistically_unstable"],
                "exposure_DI": r["exposure_DI"],
                "selection_DI": r["selection_DI"],
                "fairness_DI": r["exposure_DI"]  # backwards-compatible alias for app.py
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULTS_DIR / "intersectional_fairness_summary.csv", index=False)

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(RESULTS_DIR / "fairness_intersectional_comparison.csv", index=False)

    print("\nIntersectional Fairness Summary:")
    print(summary_df.to_string(index=False))
    print("\nIntersectional fairness evaluation completed.\n")

if __name__ == "__main__":
    run_intersectional_fairness()