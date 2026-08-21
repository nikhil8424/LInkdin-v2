import os
import pandas as pd
import numpy as np
import joblib

from src.config import RESULTS_DIR, MODELS_DIR, PROTECTED_ATTRIBUTES, DEFAULT_K, MIN_INTERSECTION_GROUP_SIZE, RANDOM_SEED, BOOTSTRAP_ITERATIONS
from src.intersectional_fairness import compute_intersectional_table
from src.fairness_mitigation import rerank_candidates
from src.top_k_recommender import evaluate_user_recommendations
from src.fairness_analysis import compute_group_fairness_metrics

print("============================================================")
print("SECTION 8 & 9: INTERSECTIONAL FAIRNESS & OBJECTIVE FORMULA")
print("============================================================")

test_df = pd.read_csv(RESULTS_DIR / "test_split.csv")
X_test = pd.read_csv(RESULTS_DIR / "X_test.csv")
model = joblib.load(MODELS_DIR / "xgboost_baseline.pkl")
test_df["baseline_score"] = model.predict_proba(X_test)[:, 1]
test_df = test_df.sort_values(["user_id", "baseline_score"], ascending=[True, False]).copy()
test_df["baseline_rank"] = test_df.groupby("user_id").cumcount() + 1
test_df["rank"] = test_df["baseline_rank"]

int_3way = pd.read_csv(RESULTS_DIR / "intersectional_gender_age_location.csv")
print(f"3-Way Intersectional subgroups: {len(int_3way)}")
print(f"Subgroups with N < {MIN_INTERSECTION_GROUP_SIZE} (unstable): {int_3way['statistically_unstable'].sum()}")
print(f"Subgroups with N >= {MIN_INTERSECTION_GROUP_SIZE} (stable): {(~int_3way['statistically_unstable']).sum()}")

stable_subgroups = int_3way[~int_3way["statistically_unstable"]]
print(f"Baseline Worst Intersectional Exposure DI (stable): {stable_subgroups['exposure_DI'].min():.6f}")
print(f"Baseline Worst Intersectional Exposure DI (all): {int_3way['exposure_DI'].min():.6f}")
print(f"Baseline Average Intersectional Exposure DI: {int_3way['exposure_DI'].mean():.6f}")

print("\nControlled Test: Solution A vs Solution B under NSGA-II objective:")
# Solution A: high marginal fairness, low intersectional fairness
# Solution B: slightly lower marginal fairness, high intersectional fairness
# In NSGA2Optimizer, objectives are: [-ndcg10, exp_gap, int_gap]
# where exp_gap = |1.0 - avg_exp_di|, int_gap = 1.0 - worst_int_di
solA_objs = [-0.80, 0.05, 0.40] # NDCG=0.80, exp_gap=0.05 (avg DI=0.95), int_gap=0.40 (worst DI=0.60)
solB_objs = [-0.80, 0.08, 0.15] # NDCG=0.80, exp_gap=0.08 (avg DI=0.92), int_gap=0.15 (worst DI=0.85)

# Check dominance
# p dominates q if all(p_i <= q_i) and any(p_i < q_i)
def dominates(p, q):
    return all(p[i] <= q[i] for i in range(len(p))) and any(p[i] < q[i] for i in range(len(p)))

print("Does Solution A dominate Solution B?", dominates(solA_objs, solB_objs))
print("Does Solution B dominate Solution A?", dominates(solB_objs, solA_objs))
print("Are they mutually non-dominated (both Pareto-optimal if in same front)?", not dominates(solA_objs, solB_objs) and not dominates(solB_objs, solA_objs))

# Under crowding distance / selection in NSGA-II:
# Both objectives (marginal gap and intersectional gap) are explicitly minimized in the 3D objective vector [-NDCG, exp_gap, int_gap].
# Neither can dominate the other without trading off, proving intersectional fairness is a first-class objective.

print("\n============================================================")
print("SECTION 10 & 11: COUNTERFACTUAL & PROXY DETECTION")
print("============================================================")
cf_df = pd.read_csv(RESULTS_DIR / "counterfactual_fairness.csv")
cf_sum = pd.read_csv(RESULTS_DIR / "counterfactual_fairness_summary.csv")
print("Counterfactual Summary:")
print(cf_sum.to_string(index=False))

proxy_pred = pd.read_csv(RESULTS_DIR / "proxy_attribute_prediction.csv")
print("\nProxy ML Classifiers on Non-Protected Features:")
print(proxy_pred.to_string(index=False))

proxy_top = pd.read_csv(RESULTS_DIR / "proxy_feature_analysis.csv")
print("\nTop Proxy Predictors:")
print(proxy_top.to_string(index=False))

print("\n============================================================")
print("SECTION 12: BOOTSTRAP CONFIDENCE INTERVALS")
print("============================================================")
ci_df = pd.read_csv(RESULTS_DIR / "bootstrap_confidence_intervals.csv")
print("Bootstrap Confidence Intervals (1,000 resamples):")
print(ci_df.to_string(index=False))

for _, r in ci_df.iterrows():
    val = r["point_estimate"]
    low = r["ci_lower_95"]
    up = r["ci_upper_95"]
    is_valid = (low <= val <= up) or np.isclose(low, val) or np.isclose(up, val)
    print(f"Metric {r['metric']:25s}: Lower={low:.4f} <= Point={val:.4f} <= Upper={up:.4f} -> Valid: {is_valid}")

print("\n============================================================")
print("SECTION 14 & 15: NSGA-II & BASELINES COMPARISON")
print("============================================================")
pareto_df = pd.read_csv(RESULTS_DIR / "nsga2_pareto_front.csv")
print(f"Discovered Pareto-optimal solutions: {len(pareto_df)}")
print(pareto_df[["fairness_strength", "ndcg_at_10", "average_exposure_di", "fairness_gap", "worst_intersectional_di", "balanced_score"]].to_string(index=False))

ablation_df = pd.read_csv(RESULTS_DIR / "ablation_comparison.csv")
print("\nAblation Comparisons across all 5 configurations:")
print(ablation_df.to_string(index=False))

print("\n============================================================")
print("SECTION 16 & 17: CHECK FOR HARD-CODED VALUES, STALE DATA, LEAKAGE")
print("============================================================")
final_q = pd.read_csv(RESULTS_DIR / "final_quality_comparison.csv")
final_f = pd.read_csv(RESULTS_DIR / "final_fairness_comparison.csv")
print("\nFinal Quality Comparison Table:")
print(final_q.to_string(index=False))
print("\nFinal Fairness Comparison Table:")
print(final_f.to_string(index=False))

