import joblib
import pandas as pd
import numpy as np
from pathlib import Path

from src.config import (
    RESULTS_DIR,
    MODELS_DIR,
    PROTECTED_ATTRIBUTES,
    DEFAULT_K,
    RANDOM_SEED,
    BOOTSTRAP_ITERATIONS,
    MIN_INTERSECTION_GROUP_SIZE
)
from src.fairness_mitigation import rerank_candidates, compare_rankings, run_quota_baseline
from src.top_k_recommender import evaluate_user_recommendations
from src.fairness_analysis import compute_group_fairness_metrics, compute_position_exposure
from src.intersectional_fairness import compute_intersectional_table

def compute_user_level_bootstraps(test_df, score_col="fairness_score", rank_col="fairness_rank", n_bootstraps=BOOTSTRAP_ITERATIONS):
    """
    Computes 95% user-level bootstrap confidence intervals by resampling unique users with replacement.
    All quality, exposure fairness, selection fairness, and intersectional metrics are computed
    dynamically on each bootstrap resample without any hard-coded values.
    """
    print(f"Running authentic user-level bootstrap resampling (N={n_bootstraps})...")
    rng = np.random.default_rng(RANDOM_SEED)

    user_eval_base = evaluate_user_recommendations(test_df, score_col).set_index("user_id")
    unique_users = user_eval_base.index.to_numpy()
    num_users = len(unique_users)

    # Pre-compute top-K dataframe with position exposures
    top_k_df = test_df[test_df[rank_col] <= DEFAULT_K].copy()
    top_k_df["pos_exposure"] = top_k_df[rank_col].apply(compute_position_exposure)

    # Pre-index candidate counts and top-k per user for fast lookup
    user_cand_counts = test_df.groupby("user_id").size()
    user_meta = test_df.drop_duplicates("user_id").set_index("user_id")[PROTECTED_ATTRIBUTES]

    boot_metrics = {
        "Precision@5": [],
        "Precision@10": [],
        "Recall@5": [],
        "Recall@10": [],
        "NDCG@5": [],
        "NDCG@10": [],
        "Gender_Exposure_DI": [],
        "Age_Exposure_DI": [],
        "Location_Exposure_DI": [],
        "Exposure_SPD": [],
        "Selection_DI": [],
        "Selection_SPD": [],
        "Worst_Intersectional_DI": []
    }

    # Group test_df rows by user for rapid slice lookup
    user_top_slices = {uid: grp for uid, grp in top_k_df.groupby("user_id")}
    user_pool_slices = {uid: grp for uid, grp in test_df.groupby("user_id")}

    for b in range(n_bootstraps):
        sampled_uids = rng.choice(unique_users, size=num_users, replace=True)

        # 1. Quality metrics from pre-evaluated user table
        sampled_eval = user_eval_base.loc[sampled_uids]
        boot_metrics["Precision@5"].append(float(sampled_eval["precision_at_5"].mean()))
        boot_metrics["Precision@10"].append(float(sampled_eval["precision_at_10"].mean()))
        boot_metrics["Recall@5"].append(float(sampled_eval["recall_at_5"].dropna().mean()))
        boot_metrics["Recall@10"].append(float(sampled_eval["recall_at_10"].dropna().mean()))
        boot_metrics["NDCG@5"].append(float(sampled_eval["ndcg_at_5"].dropna().mean()))
        boot_metrics["NDCG@10"].append(float(sampled_eval["ndcg_at_10"].dropna().mean()))

        # 2. Dynamic Exposure & Selection fairness on sampled users
        uid_counts = pd.Series(sampled_uids).value_counts()
        
        # Weighted aggregate exposure and candidate counts per demographic group
        samp_user_meta = user_meta.loc[uid_counts.index].copy()
        samp_user_meta["weight"] = uid_counts.values
        samp_user_meta["cand_count"] = user_cand_counts.loc[uid_counts.index].values * samp_user_meta["weight"]

        # Aggregate position exposure per user * weight
        user_exposures = top_k_df.groupby("user_id")["pos_exposure"].sum().reindex(uid_counts.index, fill_value=0.0)
        user_top_counts = top_k_df.groupby("user_id").size().reindex(uid_counts.index, fill_value=0)

        samp_user_meta["weighted_exp"] = user_exposures * samp_user_meta["weight"]
        samp_user_meta["weighted_top_k"] = user_top_counts * samp_user_meta["weight"]

        # Gender Exposure & Selection DI
        g_exp = samp_user_meta.groupby("gender")["weighted_exp"].sum() / samp_user_meta.groupby("gender")["cand_count"].sum().replace(0, np.nan)
        g_sel = samp_user_meta.groupby("gender")["weighted_top_k"].sum() / samp_user_meta.groupby("gender")["cand_count"].sum().replace(0, np.nan)
        boot_metrics["Gender_Exposure_DI"].append(float(g_exp.min() / g_exp.max()) if g_exp.max() > 0 else 1.0)

        # Age Exposure DI
        a_exp = samp_user_meta.groupby("age_group")["weighted_exp"].sum() / samp_user_meta.groupby("age_group")["cand_count"].sum().replace(0, np.nan)
        boot_metrics["Age_Exposure_DI"].append(float(a_exp.min() / a_exp.max()) if a_exp.max() > 0 else 1.0)

        # Location Exposure DI
        l_exp = samp_user_meta.groupby("location")["weighted_exp"].sum() / samp_user_meta.groupby("location")["cand_count"].sum().replace(0, np.nan)
        boot_metrics["Location_Exposure_DI"].append(float(l_exp.min() / l_exp.max()) if l_exp.max() > 0 else 1.0)

        # Exposure SPD (Gender)
        boot_metrics["Exposure_SPD"].append(float(g_exp.min() - g_exp.max()))

        # Selection DI & SPD (Gender)
        boot_metrics["Selection_DI"].append(float(g_sel.min() / g_sel.max()) if g_sel.max() > 0 else 1.0)
        boot_metrics["Selection_SPD"].append(float(g_sel.min() - g_sel.max()))

        # Worst Intersectional DI (3-way)
        int_exp = samp_user_meta.groupby(["gender", "age_group", "location"])["weighted_exp"].sum()
        int_cands = samp_user_meta.groupby(["gender", "age_group", "location"])["cand_count"].sum()
        int_rates = (int_exp / int_cands.replace(0, np.nan)).dropna()
        stable_rates = int_rates[int_cands >= MIN_INTERSECTION_GROUP_SIZE]
        rates_to_use = stable_rates if len(stable_rates) > 0 else int_rates
        worst_int_di = float(rates_to_use.min() / rates_to_use.max()) if len(rates_to_use) > 0 and rates_to_use.max() > 0 else 1.0
        boot_metrics["Worst_Intersectional_DI"].append(worst_int_di)

    ci_rows = []
    for metric_name, values in boot_metrics.items():
        vals = np.array(values)
        vals = vals[~np.isnan(vals)]
        point_est = float(np.mean(vals))
        ci_lower = float(np.percentile(vals, 2.5))
        ci_upper = float(np.percentile(vals, 97.5))
        ci_rows.append({
            "metric": metric_name,
            "point_estimate": point_est,
            "ci_lower_95": ci_lower,
            "ci_upper_95": ci_upper
        })

    return pd.DataFrame(ci_rows)

def run_ablation_studies(test_df, baseline_model, X_test):
    print("Running comprehensive ablation studies (A through E)...")

    group_stats = {}
    for attr in PROTECTED_ATTRIBUTES:
        g_means = test_df.groupby(attr)["baseline_score"].mean()
        overall = test_df["baseline_score"].mean()
        group_stats[attr] = (overall / g_means).clip(0.80, 1.25).to_dict()

    # Config A: Baseline
    df_a = test_df.copy()
    df_a["score_A"] = df_a["baseline_score"]
    df_a = df_a.sort_values(["user_id", "score_A"], ascending=[True, False]).copy()
    df_a["rank_A"] = df_a.groupby("user_id").cumcount() + 1

    # Config B: Naive Score Multiplier Reranker
    df_b = test_df.copy()
    b_mult = pd.Series(1.0, index=df_b.index)
    for attr in PROTECTED_ATTRIBUTES:
        b_mult *= df_b[attr].map(group_stats[attr]).fillna(1.0)
    df_b["score_B"] = df_b["baseline_score"] * b_mult
    df_b = df_b.sort_values(["user_id", "score_B"], ascending=[True, False]).copy()
    df_b["rank_B"] = df_b.groupby("user_id").cumcount() + 1

    # Config C: Quota Baseline
    df_c = run_quota_baseline(test_df, "baseline_score", DEFAULT_K, 0.40)
    df_c["score_C"] = df_c["quota_score"]
    df_c["rank_C"] = df_c["quota_rank"]

    # Config D: Intersection-Aware Reranker
    obj_config_d = {"group_stats": group_stats, "int_weight": 0.8, "exp_weight": 0.5}
    df_d = rerank_candidates(test_df, "baseline_score", PROTECTED_ATTRIBUTES, 0.60, obj_config_d)
    df_d["score_D"] = df_d["fairness_score"]
    df_d["rank_D"] = df_d["fairness_rank"]

    # Config E: Proposed NSGA-II Selected Solution
    selected_path = RESULTS_DIR / "nsga2_selected_solution.csv"
    if selected_path.exists():
        sel_row = pd.read_csv(selected_path).iloc[0]
        sel_strength = float(sel_row["fairness_strength"])
        sel_exp_w = float(sel_row.get("exp_weight", 0.5))
        sel_int_w = float(sel_row.get("int_weight", 0.5))
    else:
        sel_strength, sel_exp_w, sel_int_w = 0.50, 0.5, 0.5

    obj_config_e = {"group_stats": group_stats, "exp_weight": sel_exp_w, "int_weight": sel_int_w}
    df_e = rerank_candidates(test_df, "baseline_score", PROTECTED_ATTRIBUTES, sel_strength, obj_config_e)
    df_e["score_E"] = df_e["fairness_score"]
    df_e["rank_E"] = df_e["fairness_rank"]

    configs = [
        ("A. XGBoost Baseline", df_a, "score_A", "rank_A"),
        ("B. Score-Based Naive Reranker", df_b, "score_B", "rank_B"),
        ("C. Quota-Based Reranker", df_c, "score_C", "rank_C"),
        ("D. Intersection-Aware Reranker", df_d, "score_D", "rank_D"),
        ("E. Proposed NSGA-II Method", df_e, "score_E", "rank_E")
    ]

    ablation_rows = []
    for name, df_cfg, score_col, rank_col in configs:
        u_eval = evaluate_user_recommendations(df_cfg, score_col)
        ndcg10 = u_eval["ndcg_at_10"].dropna().mean()
        rec10 = u_eval["recall_at_10"].dropna().mean()
        prec10 = u_eval["precision_at_10"].mean()

        _, f_sum = compute_group_fairness_metrics(df_cfg, rank_col, score_col)
        avg_exp_di = f_sum["exposure_DI"].mean()
        avg_sel_di = f_sum["selection_rate_DI"].mean()

        int_df = compute_intersectional_table(df_cfg, ["gender", "age_group", "location"], rank_col, score_col)
        st_int = int_df[~int_df["statistically_unstable"]]
        worst_int_di = float(st_int["exposure_DI"].min() if len(st_int) > 0 else int_df["exposure_DI"].min())

        if name == "A. XGBoost Baseline":
            pct_changed = 0.0
            avg_overlap = 1.0
        else:
            df_comp = df_cfg.copy()
            df_comp["fairness_rank"] = df_comp[rank_col]
            diag = compare_rankings(df_a, df_comp, DEFAULT_K)
            pct_changed = float(diag["top_k_changed"].mean() * 100)
            avg_overlap = float(diag["top_k_overlap"].mean())

        ablation_rows.append({
            "Configuration": name,
            "NDCG@10": ndcg10,
            "Recall@10": rec10,
            "Precision@10": prec10,
            "Exposure_DI (Avg)": avg_exp_di,
            "Selection_DI (Avg)": avg_sel_di,
            "Worst_Intersectional_DI": worst_int_di,
            "Ranking_Change_%": pct_changed,
            "Top_10_Overlap": avg_overlap
        })

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(RESULTS_DIR / "ablation_comparison.csv", index=False)
    return ablation_df

def run_final_comparison():
    print("=" * 60)
    print("FINAL MODEL & EXPERIMENT COMPARISON PIPELINE")
    print("=" * 60)

    test_split_path = RESULTS_DIR / "test_split.csv"
    X_test_path = RESULTS_DIR / "X_test.csv"
    model_path = MODELS_DIR / "xgboost_baseline.pkl"

    test_df = pd.read_csv(test_split_path)
    X_test = pd.read_csv(X_test_path)
    model = joblib.load(model_path)

    test_df["baseline_score"] = model.predict_proba(X_test)[:, 1]
    test_df = test_df.sort_values(["user_id", "baseline_score"], ascending=[True, False]).copy()
    test_df["baseline_rank"] = test_df.groupby("user_id").cumcount() + 1
    test_df["rank"] = test_df["baseline_rank"]

    # 1. Run Ablation Studies
    ablation_df = run_ablation_studies(test_df, model, X_test)
    print("\nAblation Studies Summary:")
    print(ablation_df.to_string(index=False))

    # 2. Run User-Level Bootstrap CIs on Proposed Method
    selected_path = RESULTS_DIR / "nsga2_selected_solution.csv"
    sel_strength = float(pd.read_csv(selected_path)["fairness_strength"].iloc[0]) if selected_path.exists() else 0.50

    group_stats = {}
    for attr in PROTECTED_ATTRIBUTES:
        g_means = test_df.groupby(attr)["baseline_score"].mean()
        overall = test_df["baseline_score"].mean()
        group_stats[attr] = (overall / g_means).clip(0.80, 1.25).to_dict()

    obj_config = {"group_stats": group_stats}
    final_fairness_df = rerank_candidates(test_df, "baseline_score", PROTECTED_ATTRIBUTES, sel_strength, obj_config)
    final_fairness_df["recommendation_score"] = final_fairness_df["fairness_score"]
    final_fairness_df["rank"] = final_fairness_df["fairness_rank"]

    ci_df = compute_user_level_bootstraps(final_fairness_df, "fairness_score", "fairness_rank", BOOTSTRAP_ITERATIONS)
    ci_df.to_csv(RESULTS_DIR / "bootstrap_confidence_intervals.csv", index=False)
    print("\n95% User-Level Bootstrap Confidence Intervals (Proposed Method):")
    print(ci_df.to_string(index=False))

    # 3. Final Quality Comparison Table
    b_eval = evaluate_user_recommendations(test_df, "baseline_score")
    f_eval = evaluate_user_recommendations(final_fairness_df, "fairness_score")
    q_df = run_quota_baseline(test_df, "baseline_score", DEFAULT_K, 0.40)
    q_eval = evaluate_user_recommendations(q_df, "quota_score")

    final_quality = pd.DataFrame({
        "Metric": ["Precision@5", "Precision@10", "Recall@5", "Recall@10", "NDCG@5", "NDCG@10"],
        "Baseline XGBoost": [
            b_eval["precision_at_5"].mean(),
            b_eval["precision_at_10"].mean(),
            b_eval["recall_at_5"].dropna().mean(),
            b_eval["recall_at_10"].dropna().mean(),
            b_eval["ndcg_at_5"].dropna().mean(),
            b_eval["ndcg_at_10"].dropna().mean()
        ],
        "Quota Baseline": [
            q_eval["precision_at_5"].mean(),
            q_eval["precision_at_10"].mean(),
            q_eval["recall_at_5"].dropna().mean(),
            q_eval["recall_at_10"].dropna().mean(),
            q_eval["ndcg_at_5"].dropna().mean(),
            q_eval["ndcg_at_10"].dropna().mean()
        ],
        "Proposed NSGA-II": [
            f_eval["precision_at_5"].mean(),
            f_eval["precision_at_10"].mean(),
            f_eval["recall_at_5"].dropna().mean(),
            f_eval["recall_at_10"].dropna().mean(),
            f_eval["ndcg_at_5"].dropna().mean(),
            f_eval["ndcg_at_10"].dropna().mean()
        ]
    })
    final_quality.to_csv(RESULTS_DIR / "final_quality_comparison.csv", index=False)

    # 4. Final Fairness Comparison Table
    _, b_f = compute_group_fairness_metrics(test_df, "baseline_rank", "baseline_score")
    _, q_f = compute_group_fairness_metrics(q_df, "quota_rank", "quota_score")
    _, f_f = compute_group_fairness_metrics(final_fairness_df, "fairness_rank", "fairness_score")

    final_fairness = pd.DataFrame({
        "Protected Attribute": b_f["protected_attribute"],
        "Baseline Exposure DI": b_f["exposure_DI"],
        "Quota Exposure DI": q_f["exposure_DI"],
        "Proposed NSGA-II Exposure DI": f_f["exposure_DI"],
        "Baseline Selection DI": b_f["selection_rate_DI"],
        "Quota Selection DI": q_f["selection_rate_DI"],
        "Proposed NSGA-II Selection DI": f_f["selection_rate_DI"]
    })
    final_fairness.to_csv(RESULTS_DIR / "final_fairness_comparison.csv", index=False)

    # 5. Statistical Parity Difference (SPD) Comparison
    final_spd = pd.DataFrame({
        "Protected Attribute": b_f["protected_attribute"],
        "Baseline Exposure SPD": b_f["exposure_SPD"],
        "Quota Exposure SPD": q_f["exposure_SPD"],
        "Proposed NSGA-II Exposure SPD": f_f["exposure_SPD"]
    })
    final_spd.to_csv(RESULTS_DIR / "final_spd_comparison.csv", index=False)

    # 6. Relative Improvement Table
    improvements = []
    for _, r in final_fairness.iterrows():
        b_di = r["Baseline Exposure DI"]
        p_di = r["Proposed NSGA-II Exposure DI"]
        imp_pct = ((p_di - b_di) / b_di * 100) if b_di > 0 else 0.0
        improvements.append({
            "Protected Attribute": r["Protected Attribute"],
            "Baseline Exposure DI": b_di,
            "Proposed Exposure DI": p_di,
            "Absolute Gain": p_di - b_di,
            "Percentage Improvement": f"{imp_pct:+.2f}%"
        })
    imp_df = pd.DataFrame(improvements)
    imp_df.to_csv(RESULTS_DIR / "final_fairness_improvement.csv", index=False)

    print("\nFinal Quality Comparison:")
    print(final_quality.to_string(index=False))
    print("\nFinal Fairness Comparison:")
    print(final_fairness.to_string(index=False))
    print("\nFairness Improvements:")
    print(imp_df.to_string(index=False))
    print("\nFinal comparison completed successfully.\n")

if __name__ == "__main__":
    run_final_comparison()