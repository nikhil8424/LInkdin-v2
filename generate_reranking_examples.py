import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import ndcg_score

from src.config import RESULTS_DIR, MODELS_DIR, PROTECTED_ATTRIBUTES, DEFAULT_K, MIN_INTERSECTION_GROUP_SIZE
from src.top_k_recommender import precision_at_k, recall_at_k, compute_ndcg_at_k, evaluate_user_recommendations
from src.fairness_analysis import compute_position_exposure, compute_group_fairness_metrics
from src.fairness_mitigation import rerank_candidates, compare_rankings
from src.intersectional_fairness import compute_intersectional_table

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

df_reranked_05 = rerank_candidates(test_df, "baseline_score", PROTECTED_ATTRIBUTES, 0.50, obj_cfg)

# Select 25 users spanning diverse demographics and candidate sizes
user_meta = test_df.drop_duplicates("user_id")[["user_id", "gender", "age_group", "location"]]
user_meta["candidate_count"] = test_df.groupby("user_id").size().values

# Pick diverse users
selected_users = []
for g in test_df["gender"].unique():
    for a in test_df["age_group"].unique():
        subset = user_meta[(user_meta["gender"] == g) & (user_meta["age_group"] == a)]
        if len(subset) > 0:
            selected_users.extend(subset.head(3)["user_id"].tolist())

selected_users = list(dict.fromkeys(selected_users))[:25]
print(f"Selected {len(selected_users)} diverse users across demographic groups.")

rerank_examples = []
for u in selected_users:
    b_u = test_df[test_df["user_id"] == u].sort_values("baseline_score", ascending=False)
    f_u = df_reranked_05[df_reranked_05["user_id"] == u].sort_values("fairness_score", ascending=False)
    
    b_top10 = b_u.head(10)
    f_top10 = f_u.head(10)
    
    b_top10_ids = b_top10["post_id"].tolist()
    f_top10_ids = f_top10["post_id"].tolist()
    
    entering = list(set(f_top10_ids) - set(b_top10_ids))
    leaving = list(set(b_top10_ids) - set(f_top10_ids))
    
    merged_u = b_u[["post_id", "baseline_score", "baseline_rank"]].merge(
        f_u[["post_id", "fairness_score", "fairness_rank"]], on="post_id"
    )
    changed_pos_count = int((merged_u["baseline_rank"] != merged_u["fairness_rank"]).sum())
    
    u_gender = b_u["gender"].iloc[0]
    u_age = b_u["age_group"].iloc[0]
    u_loc = b_u["location"].iloc[0]
    
    b_scores_str = ";".join([f"{s:.4f}" for s in b_top10["baseline_score"].tolist()])
    f_scores_str = ";".join([f"{s:.4f}" for s in f_top10["fairness_score"].tolist()])
    
    rerank_examples.append({
        "user_id": u,
        "gender": u_gender,
        "age_group": u_age,
        "location": u_loc,
        "candidate_count": len(b_u),
        "baseline_top10_posts": ";".join(map(str, b_top10_ids)),
        "fairness_top10_posts": ";".join(map(str, f_top10_ids)),
        "items_entering_top10": ";".join(map(str, entering)) if entering else "None",
        "items_leaving_top10": ";".join(map(str, leaving)) if leaving else "None",
        "num_items_entering": len(entering),
        "changed_positions_count": changed_pos_count,
        "baseline_top10_scores": b_scores_str,
        "fairness_top10_scores": f_scores_str,
        "baseline_top10_mean_score": float(b_top10["baseline_score"].mean()),
        "fairness_top10_mean_score": float(f_top10["fairness_score"].mean()),
        "score_changed": True,
        "ranking_changed": changed_pos_count > 0,
        "top_k_membership_changed": len(entering) > 0
    })

rerank_df = pd.DataFrame(rerank_examples)
rerank_df.to_csv(RESULTS_DIR / "reranking_examples.csv", index=False)
print(f"Saved {len(rerank_df)} reranking examples to results/reranking_examples.csv")
print(rerank_df[["user_id", "gender", "age_group", "candidate_count", "num_items_entering", "changed_positions_count", "ranking_changed", "top_k_membership_changed"]].to_string(index=False))
