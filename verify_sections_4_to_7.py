import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import ndcg_score
from scipy.stats import kendalltau, spearmanr

from src.config import RESULTS_DIR, MODELS_DIR, PROTECTED_ATTRIBUTES, DEFAULT_K, MIN_INTERSECTION_GROUP_SIZE
from src.top_k_recommender import precision_at_k, recall_at_k, compute_ndcg_at_k, evaluate_user_recommendations
from src.fairness_analysis import compute_position_exposure, compute_group_fairness_metrics
from src.fairness_mitigation import rerank_candidates, compare_rankings
from src.intersectional_fairness import compute_intersectional_table

print("============================================================")
print("SECTION 4: VERIFY ORIGINAL 'QUALITY NEVER CHANGES' BUG")
print("============================================================")

sweep_df = pd.read_csv(RESULTS_DIR / "fairness_strength_results.csv")
print("Fairness strength sweep shape:", sweep_df.shape)
print("\nMetrics table across strengths:")
print(sweep_df[["fairness_strength", "precision_at_5", "precision_at_10", "recall_at_5", "recall_at_10", "ndcg_at_5", "ndcg_at_10", "percentage_users_changed", "top_k_overlap", "top_k_jaccard"]].to_string(index=False))

# Min, Max, Std across strengths for quality metrics
for col in ["precision_at_5", "precision_at_10", "recall_at_5", "recall_at_10", "ndcg_at_5", "ndcg_at_10"]:
    vals = sweep_df[col].to_numpy()
    diffs = np.diff(vals)
    print(f"\nMetric {col}:")
    print(f"  Min value: {vals.min():.6f}, Max value: {vals.max():.6f}")
    print(f"  Range (Max - Min): {vals.max() - vals.min():.6f}")
    print(f"  Std Dev: {vals.std():.6f}")
    print(f"  Unique values count: {len(np.unique(vals))}")
    print(f"  Min step diff: {np.abs(diffs).min():.6f}, Max step diff: {np.abs(diffs).max():.6f}")

print("\nIndependent Verification of Ranking Dynamics across Strengths:")
test_df = pd.read_csv(RESULTS_DIR / "test_split.csv")
X_test = pd.read_csv(RESULTS_DIR / "X_test.csv")
model = joblib.load(MODELS_DIR / "xgboost_baseline.pkl")
test_df["baseline_score"] = model.predict_proba(X_test)[:, 1]
test_df = test_df.sort_values(["user_id", "baseline_score"], ascending=[True, False]).copy()
test_df["baseline_rank"] = test_df.groupby("user_id").cumcount() + 1
test_df["rank"] = test_df["baseline_rank"]

group_stats = {}
for attr in PROTECTED_ATTRIBUTES:
    g_means = test_df.groupby(attr)["baseline_score"].mean()
    overall = test_df["baseline_score"].mean()
    group_stats[attr] = (overall / g_means).clip(0.80, 1.25).to_dict()
obj_cfg = {"group_stats": group_stats}

dyn_results = []
for s in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    df_s = rerank_candidates(test_df, "baseline_score", PROTECTED_ATTRIBUTES, s, obj_cfg)
    diag = compare_rankings(test_df, df_s, DEFAULT_K)
    
    # rank changes
    merged = test_df[["user_id", "post_id", "baseline_rank"]].merge(
        df_s[["user_id", "post_id", "fairness_rank"]], on=["user_id", "post_id"]
    )
    avg_rank_change = (merged["baseline_rank"] - merged["fairness_rank"]).abs().mean()
    
    dyn_results.append({
        "strength": s,
        "pct_users_changed_top10": diag["top_k_changed"].mean() * 100,
        "pct_users_changed_order": diag["ordering_changed"].mean() * 100,
        "avg_top10_jaccard": diag["top_k_jaccard"].mean(),
        "avg_top10_overlap": diag["top_k_overlap"].mean(),
        "avg_rank_change": avg_rank_change
    })

dyn_df = pd.DataFrame(dyn_results)
print(dyn_df.to_string(index=False))

print("\n============================================================")
print("SECTION 5: SELECT 20+ USERS WITH MULTIPLE PROTECTED GROUPS")
print("============================================================")
# Find users whose candidate pool contains multiple genders, age groups, locations
df_reranked_05 = rerank_candidates(test_df, "baseline_score", PROTECTED_ATTRIBUTES, 0.50, obj_cfg)

user_div = test_df.groupby("user_id").agg(
    n_cands=("post_id", "count"),
    n_genders=("gender", "nunique"),
    n_ages=("age_group", "nunique"),
    n_locs=("location", "nunique")
).reset_index()

multi_group_users = user_div[(user_div["n_genders"] > 1) & (user_div["n_ages"] > 1) & (user_div["n_locs"] > 1) & (user_div["n_cands"] >= 10)]
print(f"Users with diverse candidate pools (>=10 cands, multi-gender, multi-age, multi-loc): {len(multi_group_users)}")

selected_20_users = multi_group_users.head(25)["user_id"].tolist()

rerank_examples = []
for u in selected_20_users:
    b_u = test_df[test_df["user_id"] == u].sort_values("baseline_score", ascending=False)
    f_u = df_reranked_05[df_reranked_05["user_id"] == u].sort_values("fairness_score", ascending=False)
    
    b_top10_ids = b_u.head(10)["post_id"].tolist()
    f_top10_ids = f_u.head(10)["post_id"].tolist()
    
    entering = list(set(f_top10_ids) - set(b_top10_ids))
    leaving = list(set(b_top10_ids) - set(f_top10_ids))
    
    merged_u = b_u[["post_id", "baseline_score", "baseline_rank", "gender", "age_group", "location"]].merge(
        f_u[["post_id", "fairness_score", "fairness_rank"]], on="post_id"
    )
    changed_pos_count = (merged_u["baseline_rank"] != merged_u["fairness_rank"]).sum()
    
    # Check demographic exposure change in top-10
    b_top10_df = b_u.head(10)
    f_top10_df = f_u.head(10)
    
    rerank_examples.append({
        "user_id": u,
        "candidate_count": len(b_u),
        "baseline_top10": ";".join(map(str, b_top10_ids)),
        "fairness_top10": ";".join(map(str, f_top10_ids)),
        "items_entering_top10": ";".join(map(str, entering)) if entering else "None",
        "items_leaving_top10": ";".join(map(str, leaving)) if leaving else "None",
        "num_items_entering": len(entering),
        "changed_positions_count": int(changed_pos_count),
        "baseline_scores_top10_mean": float(b_top10_df["baseline_score"].mean()),
        "fairness_scores_top10_mean": float(f_top10_df["fairness_score"].mean()),
        "baseline_female_top10_count": int((b_top10_df["gender"] == "Female").sum()),
        "fairness_female_top10_count": int((f_top10_df["gender"] == "Female").sum()),
        "baseline_non_tech_loc_count": int((b_top10_df["location"] != "San Francisco").sum()),
        "fairness_non_tech_loc_count": int((f_top10_df["location"] != "San Francisco").sum()),
        "score_changed": True,
        "ranking_changed": changed_pos_count > 0,
        "top_k_membership_changed": len(entering) > 0
    })

rerank_examples_df = pd.DataFrame(rerank_examples)
rerank_examples_df.to_csv(RESULTS_DIR / "reranking_examples.csv", index=False)
print(f"Saved {len(rerank_examples_df)} user reranking examples to results/reranking_examples.csv")
print(rerank_examples_df[["user_id", "candidate_count", "num_items_entering", "changed_positions_count", "baseline_female_top10_count", "fairness_female_top10_count"]].head(10).to_string(index=False))

print("\n============================================================")
print("SECTION 6: INDEPENDENT RECONSTRUCTION OF PRECISION / RECALL / NDCG")
print("============================================================")

np.random.seed(42)
all_test_users = test_df["user_id"].unique()
random_10 = np.random.choice(all_test_users, size=10, replace=False)

# Deliberately selected users with specific candidate counts
user_cand_counts = test_df.groupby("user_id").size()
u_less_5 = user_cand_counts[user_cand_counts < 5].index.tolist()[:2]
u_eq_5 = user_cand_counts[user_cand_counts == 5].index.tolist()[:2]
u_5_to_10 = user_cand_counts[(user_cand_counts > 5) & (user_cand_counts < 10)].index.tolist()[:2]
u_eq_10 = user_cand_counts[user_cand_counts == 10].index.tolist()[:2]
u_gt_10 = user_cand_counts[user_cand_counts > 10].index.tolist()[:2]

deliberate_10 = u_less_5 + u_eq_5 + u_5_to_10 + u_eq_10 + u_gt_10

print(f"Random 10 users: {random_10}")
print(f"Deliberate 10 users: {deliberate_10}")

users_to_verify = list(random_10) + deliberate_10
user_eval_file = pd.read_csv(RESULTS_DIR / "top_k_user_evaluation.csv").set_index("user_id")

mismatches = []
for u in users_to_verify:
    u_data = test_df[test_df["user_id"] == u].sort_values("baseline_score", ascending=False)
    actual_labels = u_data["interaction"].to_numpy()
    pred_scores = u_data["baseline_score"].to_numpy()
    n_cands = len(actual_labels)
    total_rel = actual_labels.sum()
    
    # Manual independent calculation
    # P@5
    manual_p5 = actual_labels[:min(5, n_cands)].sum() / 5.0
    # P@10
    manual_p10 = actual_labels[:min(10, n_cands)].sum() / 10.0
    # R@5
    manual_r5 = (actual_labels[:min(5, n_cands)].sum() / total_rel) if total_rel > 0 else np.nan
    # R@10
    manual_r10 = (actual_labels[:min(10, n_cands)].sum() / total_rel) if total_rel > 0 else np.nan
    # NDCG@5
    if n_cands < 2 or total_rel == 0:
        manual_ndcg5 = np.nan
    else:
        manual_ndcg5 = float(ndcg_score([actual_labels], [pred_scores], k=min(5, n_cands)))
    # NDCG@10
    if n_cands < 2 or total_rel == 0:
        manual_ndcg10 = np.nan
    else:
        manual_ndcg10 = float(ndcg_score([actual_labels], [pred_scores], k=min(10, n_cands)))
        
    recorded = user_eval_file.loc[u]
    
    p5_match = np.isclose(manual_p5, recorded["precision_at_5"], atol=1e-6)
    p10_match = np.isclose(manual_p10, recorded["precision_at_10"], atol=1e-6)
    r5_match = (np.isnan(manual_r5) and np.isnan(recorded["recall_at_5"])) or np.isclose(manual_r5, recorded["recall_at_5"], atol=1e-6)
    r10_match = (np.isnan(manual_r10) and np.isnan(recorded["recall_at_10"])) or np.isclose(manual_r10, recorded["recall_at_10"], atol=1e-6)
    ndcg5_match = (np.isnan(manual_ndcg5) and np.isnan(recorded["ndcg_at_5"])) or np.isclose(manual_ndcg5, recorded["ndcg_at_5"], atol=1e-6)
    ndcg10_match = (np.isnan(manual_ndcg10) and np.isnan(recorded["ndcg_at_10"])) or np.isclose(manual_ndcg10, recorded["ndcg_at_10"], atol=1e-6)
    
    if not (p5_match and p10_match and r5_match and r10_match and ndcg5_match and ndcg10_match):
        mismatches.append(u)
        print(f"Mismatch on user {u}!")

print(f"Total users tested: {len(users_to_verify)}, Mismatches found: {len(mismatches)}")
if len(mismatches) == 0:
    print("✓ All 20 user manual metric reconstructions match top_k_user_evaluation.csv with 100% precision!")

print("\n============================================================")
print("SECTION 7: EXPOSURE FAIRNESS VERIFICATION")
print("============================================================")
# Verify position weights
print("Position weights:")
for r in range(1, 11):
    w = 1.0 / np.log2(r + 1.0)
    print(f"  Rank {r:2d}: {w:.6f}")

# Group-level independent exposure calculation
top10_df = test_df[test_df["baseline_rank"] <= DEFAULT_K].copy()
top10_df["pos_exp"] = 1.0 / np.log2(top10_df["baseline_rank"] + 1.0)

for attr in PROTECTED_ATTRIBUTES:
    pool_counts = test_df.groupby(attr).size()
    top10_counts = top10_df.groupby(attr).size().reindex(pool_counts.index, fill_value=0)
    top10_exps = top10_df.groupby(attr)["pos_exp"].sum().reindex(pool_counts.index, fill_value=0.0)
    
    sel_rates = top10_counts / pool_counts
    mean_exps = top10_exps / pool_counts
    exp_shares = top10_exps / top10_exps.sum()
    
    exp_di = mean_exps.min() / mean_exps.max()
    exp_spd = mean_exps.min() - mean_exps.max()
    sel_di = sel_rates.min() / sel_rates.max()
    sel_spd = sel_rates.min() - sel_rates.max()
    
    print(f"\nAttribute: {attr}")
    print(f"  Exposure DI: {exp_di:.6f}, Exposure SPD: {exp_spd:.6f}")
    print(f"  Selection DI: {sel_di:.6f}, Selection SPD: {sel_spd:.6f}")
    for g in pool_counts.index:
        print(f"    Group '{g}': cand={pool_counts[g]}, top10={top10_counts[g]}, sel_rate={sel_rates[g]:.4f}, exp_share={exp_shares[g]:.4f}, mean_exp={mean_exps[g]:.4f}")

