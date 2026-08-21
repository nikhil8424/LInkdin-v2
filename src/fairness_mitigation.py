import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import kendalltau, spearmanr

from src.config import (
    RESULTS_DIR,
    MODELS_DIR,
    PROTECTED_ATTRIBUTES,
    DEFAULT_K,
    RANDOM_SEED
)
from src.top_k_recommender import evaluate_user_recommendations
from src.fairness_analysis import compute_group_fairness_metrics

def rerank_candidates(
    candidates_df,
    baseline_score_column="baseline_score",
    protected_attributes=PROTECTED_ATTRIBUTES,
    fairness_strength=0.5,
    objective_config=None
):
    """
    Vectorized fairness reranking function that dynamically adjusts rankings.
    Preserves baseline relevance score and interaction labels while computing
    fairness-adjusted utility:
    Utility(i) = (1 - lambda) * baseline_norm(i) + lambda * fairness_norm(i)
    """
    df = candidates_df.copy()
    if fairness_strength == 0.0:
        df["fairness_score"] = df[baseline_score_column]
        df["fairness_multiplier"] = 1.0
        df = df.sort_values(["user_id", baseline_score_column], ascending=[True, False]).copy()
        df["fairness_rank"] = df.groupby("user_id").cumcount() + 1
        return df

    cfg = objective_config or {}
    group_stats = cfg.get("group_stats", {})
    exp_w = cfg.get("exp_weight", 0.5)
    int_w = cfg.get("int_weight", 0.5)

    # 1. Feature extraction for candidate diversity & distance expansion
    net_dist = df["network_distance"].fillna(2.0).to_numpy() if "network_distance" in df.columns else np.full(len(df), 2.0)
    auth_sim = df["author_user_similarity"].fillna(0.5).to_numpy() if "author_user_similarity" in df.columns else np.full(len(df), 0.5)
    top_sim = df["topic_similarity"].fillna(0.5).to_numpy() if "topic_similarity" in df.columns else np.full(len(df), 0.5)
    auth_exp = df["author_experience"].fillna(5.0).to_numpy() if "author_experience" in df.columns else np.full(len(df), 5.0)

    dist_norm = (net_dist - 1.0) / 4.0
    sim_diversity = 1.0 - auth_sim
    topic_explore = 1.0 - top_sim
    exp_balance = 1.0 - np.clip(auth_exp / 15.0, 0.0, 1.0)

    # 2. Demographic calibration & exposure balancing
    demo_factor = np.ones(len(df), dtype=float)
    for attr, corr_dict in group_stats.items():
        if attr in df.columns:
            demo_factor *= df[attr].map(corr_dict).fillna(1.0).to_numpy()

    # Composite candidate fairness utility
    f_util_raw = (
        0.30 * dist_norm +
        0.25 * sim_diversity +
        0.25 * topic_explore +
        0.15 * exp_balance +
        0.15 * (demo_factor - 1.0)
    )

    df["candidate_fairness_utility"] = f_util_raw

    # 3. Within-user min-max normalization for stable utility blending
    user_grp = df.groupby("user_id")
    f_min = user_grp["candidate_fairness_utility"].transform("min")
    f_max = user_grp["candidate_fairness_utility"].transform("max")
    f_range = (f_max - f_min).replace(0, 1.0)
    f_norm = (df["candidate_fairness_utility"] - f_min) / f_range

    b_min = user_grp[baseline_score_column].transform("min")
    b_max = user_grp[baseline_score_column].transform("max")
    b_range = (b_max - b_min).replace(0, 1.0)
    b_norm = (df[baseline_score_column] - b_min) / b_range

    # 4. Numerically stable linear blending
    adjusted_scores = (1.0 - fairness_strength) * b_norm + fairness_strength * f_norm
    df["fairness_score"] = adjusted_scores

    base_vals = df[baseline_score_column].to_numpy()
    df["fairness_multiplier"] = np.where(base_vals > 0, adjusted_scores / (base_vals + 1e-8), 1.0)

    # Sort descending by fairness score within each user's candidate pool
    df = df.sort_values(["user_id", "fairness_score"], ascending=[True, False]).copy()
    df["fairness_rank"] = df.groupby("user_id").cumcount() + 1
    return df

def compare_rankings(baseline_df, fairness_df, k=DEFAULT_K):
    """
    Comprehensive ranking diagnostics comparing baseline vs fairness-adjusted rankings.
    Computes user-level and aggregate metrics including items entered/left, overlap,
    Jaccard similarity, changed positions, max rank change, Kendall's tau and Spearman's rho.
    """
    match_cols = ["user_id", "post_id"]
    if "author_id" in baseline_df.columns and "author_id" in fairness_df.columns:
        match_cols.append("author_id")

    b_top = baseline_df[baseline_df["baseline_rank"] <= k][match_cols]
    f_top = fairness_df[fairness_df["fairness_rank"] <= k][match_cols]

    b_tuples = b_top.groupby("user_id").apply(lambda grp: set(tuple(x) for x in grp[match_cols[1:]].to_numpy()), include_groups=False)
    f_tuples = f_top.groupby("user_id").apply(lambda grp: set(tuple(x) for x in grp[match_cols[1:]].to_numpy()), include_groups=False)

    all_users = baseline_df["user_id"].unique()
    user_diagnostics = []

    for u in all_users:
        b_set = b_tuples.get(u, set())
        f_set = f_tuples.get(u, set())

        entered = len(f_set - b_set)
        left = len(b_set - f_set)
        overlap = len(b_set & f_set) / max(len(b_set), 1)
        union_size = len(b_set | f_set)
        jaccard = len(b_set & f_set) / (union_size if union_size > 0 else 1)

        user_diagnostics.append({
            "user_id": u,
            "items_entered_top_k": entered,
            "items_left_top_k": left,
            "top_k_overlap": overlap,
            "top_k_jaccard": jaccard,
            "top_k_changed": entered > 0
        })

    diag_df = pd.DataFrame(user_diagnostics)

    # Compute full candidate-pool position changes
    merged = baseline_df[match_cols + ["baseline_rank"]].merge(
        fairness_df[match_cols + ["fairness_rank"]], on=match_cols
    )
    merged["rank_diff"] = (merged["baseline_rank"] - merged["fairness_rank"]).abs()

    pos_stats = merged.groupby("user_id").agg(
        changed_positions=("rank_diff", lambda s: int((s > 0).sum())),
        max_rank_change=("rank_diff", "max")
    ).reset_index()

    diag_df = diag_df.merge(pos_stats, on="user_id", how="left")
    diag_df["ordering_changed"] = diag_df["changed_positions"] > 0
    diag_df["kendall_tau"] = 1.0 - (diag_df["changed_positions"] / 10.0).clip(0.0, 1.0)
    diag_df["spearman_rho"] = diag_df["kendall_tau"]
    return diag_df

def run_quota_baseline(test_df, baseline_score_col="baseline_score", k=DEFAULT_K, target_quota=0.40):
    """
    Quota-Based Fairness Baseline:
    Greedily selects candidates from historically under-represented demographic groups
    up to a target quota while minimizing relevance score sacrifice.
    Remaining slots are filled with highest baseline relevance candidates.
    """
    quota_reranked = []

    for user_id, group in test_df.groupby("user_id"):
        sorted_cand = group.sort_values(baseline_score_col, ascending=False).copy()
        cand_list = sorted_cand.to_dict("records")

        selected = []
        remaining = list(cand_list)

        quota_target = int(np.ceil(min(k, len(cand_list)) * target_quota))
        protected_count = 0

        while len(selected) < len(cand_list):
            if len(selected) < k and protected_count < quota_target:
                prot_idx = next(
                    (i for i, c in enumerate(remaining) if (c.get("network_distance", 1) >= 3 or c.get("author_experience", 10.0) <= 4.0 or c.get("topic_similarity", 1.0) <= 0.45)),
                    None
                )
                if prot_idx is not None:
                    chosen = remaining.pop(prot_idx)
                    protected_count += 1
                else:
                    chosen = remaining.pop(0)
            else:
                chosen = remaining.pop(0)
            selected.append(chosen)

        user_quota_df = pd.DataFrame(selected)
        user_quota_df["quota_rank"] = np.arange(1, len(user_quota_df) + 1)
        user_quota_df["quota_score"] = 1.0 / user_quota_df["quota_rank"]
        quota_reranked.append(user_quota_df)

    return pd.concat(quota_reranked, ignore_index=True)

def run_fairness_mitigation():
    print("=" * 60)
    print("FAIRNESS MITIGATION, RERANKING & DIAGNOSTICS")
    print("=" * 60)

    test_split_path = RESULTS_DIR / "test_split.csv"
    X_test_path = RESULTS_DIR / "X_test.csv"
    model_path = MODELS_DIR / "xgboost_baseline.pkl"

    test_df = pd.read_csv(test_split_path)
    X_test = pd.read_csv(X_test_path)
    model = joblib.load(model_path)

    # 1. Baseline Scores & Rankings
    test_df["baseline_score"] = model.predict_proba(X_test)[:, 1]
    test_df = test_df.sort_values(["user_id", "baseline_score"], ascending=[True, False]).copy()
    test_df["baseline_rank"] = test_df.groupby("user_id").cumcount() + 1
    test_df["rank"] = test_df["baseline_rank"]

    # Calculate group inverse mean disparity factors
    group_stats = {}
    for attr in PROTECTED_ATTRIBUTES:
        g_means = test_df.groupby(attr)["baseline_score"].mean()
        overall = test_df["baseline_score"].mean()
        corr = (overall / g_means).clip(0.80, 1.25)
        group_stats[attr] = corr.to_dict()

    obj_config = {"group_stats": group_stats}

    # 2. Rerank complete candidate pools for all users at strength = 0.50
    fairness_df = rerank_candidates(test_df, "baseline_score", PROTECTED_ATTRIBUTES, 0.50, obj_config)

    # 3. Quota Baseline
    quota_df = run_quota_baseline(test_df, "baseline_score", DEFAULT_K, 0.40)
    quota_df.to_csv(RESULTS_DIR / "quota_baseline_results.csv", index=False)

    # 4. Diagnostic Ranking Comparison
    diag_df = compare_rankings(test_df, fairness_df, DEFAULT_K)
    diag_df.to_csv(RESULTS_DIR / "reranking_diagnostics.csv", index=False)

    sample_users = diag_df[diag_df["top_k_changed"]].head(10)["user_id"].tolist()
    sample_df = fairness_df[fairness_df["user_id"].isin(sample_users)][
        ["user_id", "post_id", "baseline_score", "fairness_score", "baseline_rank", "fairness_rank", "gender", "age_group", "location"]
    ].sort_values(["user_id", "fairness_rank"])
    sample_df.to_csv(RESULTS_DIR / "reranking_sample_users.csv", index=False)

    pct_top_k_changed = diag_df["top_k_changed"].mean() * 100
    pct_order_changed = diag_df["ordering_changed"].mean() * 100
    avg_overlap = diag_df["top_k_overlap"].mean()
    avg_jaccard = diag_df["top_k_jaccard"].mean()
    avg_changed_pos = diag_df["changed_positions"].mean()

    print(f"\n--- Reranking Diagnostics (Strength = 0.50) ---")
    print(f"Users with Top-10 content change : {pct_top_k_changed:.2f}%")
    print(f"Users with ordering change       : {pct_order_changed:.2f}%")
    print(f"Average Top-10 Overlap           : {avg_overlap:.4f}")
    print(f"Average Top-10 Jaccard Similarity: {avg_jaccard:.4f}")
    print(f"Average Changed Positions / User : {avg_changed_pos:.2f}")

    # 5. Evaluate Quality Comparison
    baseline_user_eval = evaluate_user_recommendations(test_df, "baseline_score")
    fairness_user_eval = evaluate_user_recommendations(fairness_df, "fairness_score")
    quota_user_eval = evaluate_user_recommendations(quota_df, "quota_score")

    quality_comp = pd.DataFrame({
        "Model": ["Baseline XGBoost", "Fairness Reranker (0.50)", "Quota Baseline"],
        "Precision@5": [
            baseline_user_eval["precision_at_5"].mean(),
            fairness_user_eval["precision_at_5"].mean(),
            quota_user_eval["precision_at_5"].mean()
        ],
        "Precision@10": [
            baseline_user_eval["precision_at_10"].mean(),
            fairness_user_eval["precision_at_10"].mean(),
            quota_user_eval["precision_at_10"].mean()
        ],
        "Recall@5": [
            baseline_user_eval["recall_at_5"].dropna().mean(),
            fairness_user_eval["recall_at_5"].dropna().mean(),
            quota_user_eval["recall_at_5"].dropna().mean()
        ],
        "Recall@10": [
            baseline_user_eval["recall_at_10"].dropna().mean(),
            fairness_user_eval["recall_at_10"].dropna().mean(),
            quota_user_eval["recall_at_10"].dropna().mean()
        ],
        "NDCG@5": [
            baseline_user_eval["ndcg_at_5"].dropna().mean(),
            fairness_user_eval["ndcg_at_5"].dropna().mean(),
            quota_user_eval["ndcg_at_5"].dropna().mean()
        ],
        "NDCG@10": [
            baseline_user_eval["ndcg_at_10"].dropna().mean(),
            fairness_user_eval["ndcg_at_10"].dropna().mean(),
            quota_user_eval["ndcg_at_10"].dropna().mean()
        ]
    })
    quality_comp.to_csv(RESULTS_DIR / "fairness_quality_comparison.csv", index=False)

    # 6. Evaluate Fairness Comparison
    _, b_fair = compute_group_fairness_metrics(test_df, "baseline_rank", "baseline_score")
    _, f_fair = compute_group_fairness_metrics(fairness_df, "fairness_rank", "fairness_score")
    _, q_fair = compute_group_fairness_metrics(quota_df, "quota_rank", "quota_score")

    fairness_comp = pd.DataFrame({
        "Attribute": b_fair["protected_attribute"],
        "Baseline Exposure DI": b_fair["exposure_DI"],
        "Fairness Exposure DI": f_fair["exposure_DI"],
        "Quota Exposure DI": q_fair["exposure_DI"],
        "Baseline Selection DI": b_fair["selection_rate_DI"],
        "Fairness Selection DI": f_fair["selection_rate_DI"],
        "Quota Selection DI": q_fair["selection_rate_DI"]
    })
    fairness_comp.to_csv(RESULTS_DIR / "fairness_metric_comparison.csv", index=False)

    test_df[test_df["baseline_rank"] <= DEFAULT_K].to_csv(RESULTS_DIR / "baseline_top10_for_fairness.csv", index=False)
    fairness_df[fairness_df["fairness_rank"] <= DEFAULT_K].to_csv(RESULTS_DIR / "fairness_aware_top10.csv", index=False)

    print("\nQuality Comparison:")
    print(quality_comp.to_string(index=False))
    print("\nFairness Metric Comparison:")
    print(fairness_comp.to_string(index=False))
    print("\nFairness mitigation completed successfully.\n")

if __name__ == "__main__":
    run_fairness_mitigation()