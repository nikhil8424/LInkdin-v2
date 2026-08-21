import joblib
import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.metrics import ndcg_score



BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "synthetic_linkedin_dataset_30000.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "xgboost_baseline.pkl"
)

ENCODER_PATH = (
    BASE_DIR
    / "results"
    / "categorical_encoder.pkl"
)

RESULTS_PATH = BASE_DIR / "results"


# 2. SETTINGS
# ============================================================

K = 10


# 3. REQUIRED COLUMNS
# ============================================================

CATEGORICAL_FEATURES = [
    "professional_field",
    "education",
    "post_topic",
    "content_type"
]

NUMERICAL_FEATURES = [
    "experience_years",
    "network_size",
    "previous_interactions",
    "engagement",
    "author_user_similarity",
    "topic_similarity",
    "post_age_hours",
    "author_experience",
    "author_network_size",
    "network_distance"
]

MODEL_FEATURES = (
    CATEGORICAL_FEATURES
    + NUMERICAL_FEATURES
)


# 4. CHECK REQUIRED FILES
# ============================================================

print("=" * 60)
print("CHECKING REQUIRED FILES")
print("=" * 60)

required_files = [
    DATA_PATH,
    MODEL_PATH,
    ENCODER_PATH
]

for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{file_path}"
        )

    print("✓", file_path)

print()


# 5. LOAD DATASET
# ============================================================

print("=" * 60)
print("LOADING ORIGINAL DATASET")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print()


# 6. LOAD MODEL
# ============================================================

print("=" * 60)
print("XGBOOST MODEL LOADED")
print("=" * 60)

model = joblib.load(MODEL_PATH)

print(MODEL_PATH)
print()


# 7. LOAD ENCODER
# ============================================================

print("=" * 60)
print("ENCODER LOADED")
print("=" * 60)

encoder = joblib.load(ENCODER_PATH)

print(ENCODER_PATH)
print()


# ============================================================
# 8. PREPARE FEATURES
# ============================================================

print("=" * 60)
print("PREPARING FEATURES")
print("=" * 60)

X = df[MODEL_FEATURES].copy()


# Encode categorical features
X_cat = encoder.transform(
    X[CATEGORICAL_FEATURES]
)

encoded_feature_names = (
    encoder.get_feature_names_out(
        CATEGORICAL_FEATURES
    )
)


X_cat = pd.DataFrame(
    X_cat,
    columns=encoded_feature_names,
    index=X.index
)


# Numerical features
X_num = X[
    NUMERICAL_FEATURES
].copy()


# Combine
X_processed = pd.concat(
    [
        X_num.reset_index(drop=True),
        X_cat.reset_index(drop=True)
    ],
    axis=1
)


print(
    "Processed feature shape:",
    X_processed.shape
)

print()


# 9. GENERATE RECOMMENDATION SCORES
# ============================================================

print("=" * 60)
print("GENERATING RECOMMENDATION SCORES")
print("=" * 60)

scores = model.predict_proba(
    X_processed
)[:, 1]


df["recommendation_score"] = scores


print("Scores generated successfully.")

print(
    f"Score range: "
    f"{scores.min():.3f} "
    f"to "
    f"{scores.max():.4f}"
)

print()


# 10. CREATE USER-LEVEL RANKINGS
# ============================================================

print("=" * 60)
print("CREATING USER-LEVEL RANKINGS")
print("=" * 60)


df = df.sort_values(
    [
        "user_id",
        "recommendation_score"
    ],
    ascending=[
        True,
        False
    ]
)


df["rank"] = (
    df.groupby("user_id")
      .cumcount()
      + 1
)


print("User-level rankings created.")
print()


# 11. TOP-K RECOMMENDATIONS
# ============================================================

print("=" * 60)
print("TOP-K RECOMMENDATIONS")
print("=" * 60)

top_k = df[
    df["rank"] <= K
].copy()


print(
    "Number of users:",
    df["user_id"].nunique()
)

print(
    "K:",
    K
)

print(
    "Top-K rows:",
    len(top_k)
)

print()

# 12. SAMPLE USER
# ============================================================

print("=" * 60)
print("SAMPLE USER RECOMMENDATIONS")
print("=" * 60)


sample_user = (
    df["user_id"]
    .value_counts()
    .index[0]
)


sample_recommendations = top_k[
    top_k["user_id"] == sample_user
][
    [
        "user_id",
        "post_id",
        "recommendation_score",
        "interaction",
        "rank"
    ]
]


print(
    "User:",
    sample_user
)

print()

print(
    sample_recommendations.to_string(
        index=False
    )
)

print()


# 13. METRIC FUNCTIONS
# ============================================================

def precision_at_k(
    actual,
    k
):

    actual = np.asarray(
        actual
    )

    k = min(
        k,
        len(actual)
    )

    if k == 0:
        return 0.0

    return (
        actual[:k].sum()
        / k
    )


def recall_at_k(
    actual,
    k
):

    actual = np.asarray(
        actual
    )

    total_relevant = actual.sum()

    if total_relevant == 0:
        return np.nan

    k = min(
        k,
        len(actual)
    )

    return (
        actual[:k].sum()
        / total_relevant
    )


def ndcg_at_k(
    actual,
    scores,
    k
):

    actual = np.asarray(
        actual
    )

    scores = np.asarray(
        scores
    )

    # NDCG requires at least two candidate documents.
    if len(actual) < 2:
        return np.nan

    k = min(
        k,
        len(actual)
    )

    return ndcg_score(
        [actual],
        [scores],
        k=k
    )


# 14. USER-LEVEL EVALUATION
# ============================================================

print("=" * 60)
print("CALCULATING TOP-K METRICS")
print("=" * 60)


evaluation_results = []

ndcg_skipped = 0


for user_id, group in df.groupby(
    "user_id"
):

    # Sort by recommendation score
    group = group.sort_values(
        "recommendation_score",
        ascending=False
    )


    actual = group[
        "interaction"
    ].to_numpy()


    scores_user = group[
        "recommendation_score"
    ].to_numpy()


    precision_5 = precision_at_k(
        actual,
        5
    )


    precision_10 = precision_at_k(
        actual,
        10
    )


    recall_5 = recall_at_k(
        actual,
        5
    )


    # Recall@10 is mathematically 1.0 whenever
    # all candidate documents are included.
    #
    # We therefore DO NOT report Recall@10
    # as a recommendation-quality metric.


    ndcg_5 = ndcg_at_k(
        actual,
        scores_user,
        5
    )


    ndcg_10 = ndcg_at_k(
        actual,
        scores_user,
        10
    )


    if np.isnan(ndcg_5) or np.isnan(ndcg_10):

        ndcg_skipped += 1


    evaluation_results.append({

        "user_id": user_id,

        "candidate_count": len(
            group
        ),

        "relevant_count": int(
            actual.sum()
        ),

        "precision_at_5": precision_5,

        "precision_at_10": precision_10,

        "recall_at_5": recall_5,

        "ndcg_at_5": ndcg_5,

        "ndcg_at_10": ndcg_10
    })


user_evaluation = pd.DataFrame(
    evaluation_results
)


# 15. AGGREGATE METRICS
# ============================================================

precision_5 = (
    user_evaluation[
        "precision_at_5"
    ].mean()
)


precision_10 = (
    user_evaluation[
        "precision_at_10"
    ].mean()
)


recall_5 = (
    user_evaluation[
        "recall_at_5"
    ].mean()
)


ndcg_5 = (
    user_evaluation[
        "ndcg_at_5"
    ].mean()
)


ndcg_10 = (
    user_evaluation[
        "ndcg_at_10"
    ].mean()
)


# 16. FINAL TOP-K METRICS
# ============================================================

print("=" * 60)
print("TOP-K RECOMMENDATION RESULTS")
print("=" * 60)

print(
    f"Precision@5  : {precision_5:.4f}"
)

print(
    f"Precision@10 : {precision_10:.4f}"
)

print(
    f"Recall@5     : {recall_5:.4f}"
)

print(
    f"NDCG@5       : {ndcg_5:.4f}"
)

print(
    f"NDCG@10      : {ndcg_10:.4f}"
)

print()

print(
    "Recall@10 was excluded because "
    "each user has exactly 10 candidate records."
)

print(
    "Therefore Top-10 contains the entire "
    "candidate set and Recall@10 would always be 1.0."
)

print()



# 17. SAVE TOP-K RECOMMENDATIONS
# ============================================================

recommendation_columns = [
    "user_id",
    "post_id",
    "recommendation_score",
    "interaction",
    "rank"
]


top_k[
    recommendation_columns
].to_csv(
    RESULTS_PATH
    / "top_10_recommendations.csv",
    index=False
)



# 18. SAVE USER-LEVEL EVALUATION
# ============================================================

user_evaluation.to_csv(
    RESULTS_PATH
    / "top_k_user_evaluation.csv",
    index=False
)



# 19. SAVE METRICS
# ============================================================

metrics = pd.DataFrame({

    "metric": [

        "Precision@5",

        "Precision@10",

        "Recall@5",

        "NDCG@5",

        "NDCG@10"
    ],

    "value": [

        precision_5,

        precision_10,

        recall_5,

        ndcg_5,

        ndcg_10
    ]
})


metrics.to_csv(
    RESULTS_PATH
    / "top_k_metrics.csv",
    index=False
)



# 20. SAVE EVALUATION NOTES
# ============================================================

evaluation_notes = pd.DataFrame({

    "item": [

        "Number of users",

        "Total interaction records",

        "Candidates per user",

        "Top-K",

        "Recall@10 status"
    ],

    "value": [

        df["user_id"].nunique(),

        len(df),

        df.groupby(
            "user_id"
        ).size().mean(),

        K,

        "Excluded because K equals candidate count"
    ]
})


evaluation_notes.to_csv(
    RESULTS_PATH
    / "top_k_evaluation_notes.csv",
    index=False
)



# 21. COMPLETION
# ============================================================

print("=" * 60)
print("TOP-K RECOMMENDER COMPLETED SUCCESSFULLY")
print("=" * 60)

print()

print("Generated files:")

print(
    "1. top_10_recommendations.csv"
)

print(
    "2. top_k_user_evaluation.csv"
)

print(
    "3. top_k_metrics.csv"
)

print(
    "4. top_k_evaluation_notes.csv"
)

print()

print(
    "Results folder:"
)

print(
    RESULTS_PATH
)