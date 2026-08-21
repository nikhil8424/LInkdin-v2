import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import ndcg_score

from src.config import (
    RESULTS_DIR,
    MODELS_DIR,
    DEFAULT_K
)

def precision_at_k(actual, k):
    actual = np.asarray(actual)
    if len(actual) == 0 or k == 0:
        return 0.0
    k_eff = min(k, len(actual))
    return float(actual[:k_eff].sum() / k)

def recall_at_k(actual, k):
    actual = np.asarray(actual)
    total_relevant = actual.sum()
    if total_relevant == 0:
        return np.nan
    k_eff = min(k, len(actual))
    return float(actual[:k_eff].sum() / total_relevant)

def compute_ndcg_at_k(actual, scores, k):
    actual = np.asarray(actual)
    scores = np.asarray(scores)
    if len(actual) < 2 or actual.sum() == 0:
        return np.nan
    k_eff = min(k, len(actual))
    return float(ndcg_score([actual], [scores], k=k_eff))

def evaluate_user_recommendations(df, score_column="recommendation_score"):
    """
    Evaluates complete candidate pools per user without premature truncation.
    """
    evaluation_results = []
    
    for user_id, group in df.groupby("user_id"):
        # Sort complete candidate pool by score descending
        sorted_group = group.sort_values(score_column, ascending=False)
        actual = sorted_group["interaction"].to_numpy()
        scores = sorted_group[score_column].to_numpy()
        cand_count = len(actual)
        rel_count = int(actual.sum())

        p5 = precision_at_k(actual, 5)
        p10 = precision_at_k(actual, 10)
        r5 = recall_at_k(actual, 5)
        r10 = recall_at_k(actual, 10)
        ndcg5 = compute_ndcg_at_k(actual, scores, 5)
        ndcg10 = compute_ndcg_at_k(actual, scores, 10)

        evaluation_results.append({
            "user_id": user_id,
            "candidate_count": cand_count,
            "relevant_count": rel_count,
            "precision_at_5": p5,
            "precision_at_10": p10,
            "recall_at_5": r5,
            "recall_at_10": r10,
            "ndcg_at_5": ndcg5,
            "ndcg_at_10": ndcg10
        })

    user_eval_df = pd.DataFrame(evaluation_results)
    return user_eval_df

def run_top_k_recommendation():
    print("=" * 60)
    print("TOP-K RECOMMENDATION & FULL CANDIDATE POOL EVALUATION")
    print("=" * 60)

    model_path = MODELS_DIR / "xgboost_baseline.pkl"
    test_split_path = RESULTS_DIR / "test_split.csv"
    X_test_path = RESULTS_DIR / "X_test.csv"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not test_split_path.exists():
        raise FileNotFoundError(f"Test split not found: {test_split_path}")

    model = joblib.load(model_path)
    test_df = pd.read_csv(test_split_path)
    X_test = pd.read_csv(X_test_path)

    # Generate baseline recommendation scores on held-out test data
    scores = model.predict_proba(X_test)[:, 1]
    test_df["recommendation_score"] = scores

    # Rank complete candidate pool per user
    test_df = test_df.sort_values(["user_id", "recommendation_score"], ascending=[True, False]).copy()
    test_df["rank"] = test_df.groupby("user_id").cumcount() + 1

    # Extract Top-10 recommendations
    top_10 = test_df[test_df["rank"] <= DEFAULT_K].copy()

    # User-level full candidate pool evaluation
    user_eval = evaluate_user_recommendations(test_df, "recommendation_score")

    # Aggregate Metrics (ignoring NaN for recall/NDCG when relevant=0)
    p5 = user_eval["precision_at_5"].mean()
    p10 = user_eval["precision_at_10"].mean()
    r5 = user_eval["recall_at_5"].dropna().mean()
    r10 = user_eval["recall_at_10"].dropna().mean()
    ndcg5 = user_eval["ndcg_at_5"].dropna().mean()
    ndcg10 = user_eval["ndcg_at_10"].dropna().mean()

    # Candidate pool statistics
    cand_counts = user_eval["candidate_count"]
    min_cand = int(cand_counts.min())
    max_cand = int(cand_counts.max())
    mean_cand = float(cand_counts.mean())
    median_cand = float(cand_counts.median())
    pct_fewer_than_k = float((cand_counts < DEFAULT_K).mean() * 100)
    pct_at_least_k = float((cand_counts >= DEFAULT_K).mean() * 100)

    print("\n" + "=" * 60)
    print("TOP-K RECOMMENDATION METRICS (HELD-OUT TEST SET)")
    print("=" * 60)
    print(f"Users evaluated           : {len(user_eval):,}")
    print(f"Total test candidate items: {len(test_df):,}")
    print(f"Candidate count (Min/Max) : {min_cand} / {max_cand}")
    print(f"Candidate count (Mean/Med): {mean_cand:.2f} / {median_cand:.1f}")
    print(f"Users with < 10 candidates: {pct_fewer_than_k:.1f}%")
    print(f"Users with >= 10 cand.    : {pct_at_least_k:.1f}%")
    print(f"Precision@5               : {p5:.4f}")
    print(f"Precision@10              : {p10:.4f}")
    print(f"Recall@5                  : {r5:.4f}")
    print(f"Recall@10                 : {r10:.4f}")
    print(f"NDCG@5                    : {ndcg5:.4f}")
    print(f"NDCG@10                   : {ndcg10:.4f}\n")

    # Save Outputs
    recommendation_cols = [
        "user_id", "post_id", "author_id", "recommendation_score",
        "interaction", "rank", "gender", "age_group", "location"
    ]
    top_10[recommendation_cols].to_csv(RESULTS_DIR / "top_10_recommendations.csv", index=False)
    user_eval.to_csv(RESULTS_DIR / "top_k_user_evaluation.csv", index=False)

    metrics_df = pd.DataFrame({
        "metric": [
            "Precision@5", "Precision@10", "Recall@5", "Recall@10", "NDCG@5", "NDCG@10"
        ],
        "value": [p5, p10, r5, r10, ndcg5, ndcg10]
    })
    metrics_df.to_csv(RESULTS_DIR / "top_k_metrics.csv", index=False)

    notes_df = pd.DataFrame({
        "item": [
            "Number of test users",
            "Total test interaction records",
            "Min candidate count",
            "Max candidate count",
            "Mean candidate count",
            "Median candidate count",
            "Percentage users with < 10 candidates",
            "Percentage users with >= 10 candidates",
            "Evaluation methodology"
        ],
        "value": [
            len(user_eval),
            len(test_df),
            min_cand,
            max_cand,
            f"{mean_cand:.2f}",
            f"{median_cand:.1f}",
            f"{pct_fewer_than_k:.2f}%",
            f"{pct_at_least_k:.2f}%",
            "Full candidate pools evaluated per user on held-out test split without truncation"
        ]
    })
    notes_df.to_csv(RESULTS_DIR / "top_k_evaluation_notes.csv", index=False)
    print("Top-K recommender completed successfully.\n")

if __name__ == "__main__":
    run_top_k_recommendation()