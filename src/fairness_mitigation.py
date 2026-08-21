import joblib
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
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

RESULTS_PATH = (
    BASE_DIR
    / "results"
)


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


PROTECTED_ATTRIBUTES = [
    "gender",
    "age_group",
    "location"
]


K = 10

# Strength of fairness adjustment.
#
# 0.0 = original XGBoost ranking
# 1.0 = full group-rate correction
#
# We start with a moderate value.
FAIRNESS_STRENGTH = 0.50


print("=" * 60)
print("CHECKING REQUIRED FILES")
print("=" * 60)

required_files = [
    DATA_PATH,
    MODEL_PATH,
    ENCODER_PATH
]

for path in required_files:

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    print("✓", path)

print()


print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

df = pd.read_csv(
    DATA_PATH
)

print(
    "Dataset shape:",
    df.shape
)

print()

print("=" * 60)
print("LOADING XGBOOST MODEL")
print("=" * 60)

model = joblib.load(
    MODEL_PATH
)

print(
    "Model loaded successfully."
)

print()


print("=" * 60)
print("LOADING ENCODER")
print("=" * 60)

encoder = joblib.load(
    ENCODER_PATH
)

print(
    "Encoder loaded successfully."
)

print()

print("=" * 60)
print("PREPARING MODEL FEATURES")
print("=" * 60)

X = df[
    MODEL_FEATURES
].copy()


X_categorical = encoder.transform(
    X[CATEGORICAL_FEATURES]
)


encoded_feature_names = (
    encoder.get_feature_names_out(
        CATEGORICAL_FEATURES
    )
)


X_categorical = pd.DataFrame(
    X_categorical,
    columns=encoded_feature_names,
    index=X.index
)


X_numerical = X[
    NUMERICAL_FEATURES
].copy()


X_processed = pd.concat(
    [
        X_numerical,
        X_categorical
    ],
    axis=1
)


print(
    "Processed feature shape:",
    X_processed.shape
)

print()


print("=" * 60)
print("GENERATING BASELINE SCORES")
print("=" * 60)

df["baseline_score"] = (
    model.predict_proba(
        X_processed
    )[:, 1]
)


print(
    "Baseline scores generated."
)

print(
    "Score range:",
    round(
        df["baseline_score"].min(),
        4
    ),
    "to",
    round(
        df["baseline_score"].max(),
        4
    )
)

print()

print("=" * 60)
print("CREATING USER-LEVEL BASELINE RANKINGS")
print("=" * 60)

df = df.sort_values(
    [
        "user_id",
        "baseline_score"
    ],
    ascending=[
        True,
        False
    ]
).copy()


df["baseline_rank"] = (
    df.groupby(
        "user_id"
    ).cumcount() + 1
)


print(
    "Baseline rankings created."
)

print()


print("=" * 60)
print("CALCULATING GROUP STATISTICS")
print("=" * 60)


# We use the mean predicted score of each
# protected group as the baseline group exposure.


group_statistics = {}


for attribute in PROTECTED_ATTRIBUTES:

    statistics = (
        df.groupby(
            attribute
        )[
            "baseline_score"
        ]
        .mean()
    )

    group_statistics[
        attribute
    ] = statistics


    print()
    print(
        f"{attribute}:"
    )

    print(
        statistics
    )


print()


print("=" * 60)
print("CALCULATING FAIRNESS-ADJUSTED SCORES")
print("=" * 60)


df["fairness_multiplier"] = 1.0


for attribute in PROTECTED_ATTRIBUTES:

    group_means = (
        group_statistics[
            attribute
        ]
    )


    overall_mean = (
        df["baseline_score"]
        .mean()
    )


    correction = (
        overall_mean
        / group_means
    )


    # Prevent extreme correction values.

    correction = correction.clip(
        lower=0.80,
        upper=1.20
    )


    # Map correction to every row.

    row_correction = (
        df[attribute]
        .map(correction)
        .fillna(1.0)
    )


    df["fairness_multiplier"] *= (
        1
        + FAIRNESS_STRENGTH
        * (row_correction - 1)
    )


df["fairness_score"] = (
    df["baseline_score"]
    * df["fairness_multiplier"]
)


# Keep score inside valid range.

df["fairness_score"] = (
    df["fairness_score"]
    .clip(
        lower=0.0,
        upper=1.0
    )
)


print(
    "Fairness-adjusted scores generated."
)

print(
    "Fairness score range:",
    round(
        df["fairness_score"].min(),
        4
    ),
    "to",
    round(
        df["fairness_score"].max(),
        4
    )
)

print()


print("=" * 60)
print("CREATING FAIRNESS-AWARE RANKINGS")
print("=" * 60)


df = df.sort_values(
    [
        "user_id",
        "fairness_score"
    ],
    ascending=[
        True,
        False
    ]
).copy()


df["fairness_rank"] = (
    df.groupby(
        "user_id"
    ).cumcount() + 1
)


print(
    "Fairness-aware rankings created."
)

print()


print("=" * 60)
print("CREATING TOP-K RECOMMENDATIONS")
print("=" * 60)


baseline_top_k = (
    df[
        df["baseline_rank"] <= K
    ]
    .copy()
)


fairness_top_k = (
    df[
        df["fairness_rank"] <= K
    ]
    .copy()
)


print(
    "Baseline Top-K rows:",
    len(baseline_top_k)
)

print(
    "Fairness-aware Top-K rows:",
    len(fairness_top_k)
)

print()


def calculate_top_k_metrics(
    data,
    score_column,
    k=10
):

    precisions = []
    recalls = []
    ndcgs = []


    for user_id, user_data in data.groupby(
        "user_id"
    ):

        user_data = user_data.sort_values(
            score_column,
            ascending=False
        )


        actual = (
            user_data[
                "interaction"
            ].to_numpy()
        )


        scores = (
            user_data[
                score_column
            ].to_numpy()
        )


        if len(actual) == 0:
            continue


        top_k_actual = actual[
            :k
        ]


        precision = (
            top_k_actual.sum()
            / len(top_k_actual)
        )


        total_relevant = (
            actual.sum()
        )


        if total_relevant > 0:

            recall = (
                top_k_actual.sum()
                / total_relevant
            )

            recalls.append(
                recall
            )


        precisions.append(
            precision
        )


        if len(actual) >= 2:

            try:

                ndcg = ndcg_score(
                    [actual],
                    [scores],
                    k=min(
                        k,
                        len(actual)
                    )
                )

                ndcgs.append(
                    ndcg
                )

            except ValueError:

                pass


    return {

        "Precision@10":
            np.mean(
                precisions
            ),

        "Recall@10":
            np.mean(
                recalls
            ),

        "NDCG@10":
            np.mean(
                ndcgs
            )
            if ndcgs
            else np.nan
    }


print("=" * 60)
print("RECOMMENDATION QUALITY COMPARISON")
print("=" * 60)


baseline_metrics = (
    calculate_top_k_metrics(
        baseline_top_k,
        "baseline_score",
        K
    )
)


fairness_metrics = (
    calculate_top_k_metrics(
        fairness_top_k,
        "fairness_score",
        K
    )
)


quality_comparison = pd.DataFrame({

    "model": [
        "Baseline XGBoost",
        "Fairness-aware XGBoost"
    ],

    "Precision@10": [
        baseline_metrics[
            "Precision@10"
        ],

        fairness_metrics[
            "Precision@10"
        ]
    ],

    "Recall@10": [
        baseline_metrics[
            "Recall@10"
        ],

        fairness_metrics[
            "Recall@10"
        ]
    ],

    "NDCG@10": [
        baseline_metrics[
            "NDCG@10"
        ],

        fairness_metrics[
            "NDCG@10"
        ]
    ]
})


print(
    quality_comparison.to_string(
        index=False
    )
)

print()


def calculate_group_fairness(
    data,
    score_column,
    attribute
):

    group_scores = (
        data.groupby(
            attribute
        )[
            score_column
        ]
        .mean()
    )


    max_score = (
        group_scores.max()
    )

    min_score = (
        group_scores.min()
    )


    spd = (
        min_score
        - max_score
    )


    if max_score == 0:

        di = np.nan

    else:

        di = (
            min_score
            / max_score
        )


    return {

        "attribute":
            attribute,

        "minimum_group_score":
            min_score,

        "maximum_group_score":
            max_score,

        "SPD":
            spd,

        "DI":
            di
    }


print("=" * 60)
print("FAIRNESS COMPARISON")
print("=" * 60)


fairness_results = []


for attribute in PROTECTED_ATTRIBUTES:

    baseline_result = (
        calculate_group_fairness(
            baseline_top_k,
            "baseline_score",
            attribute
        )
    )


    fair_result = (
        calculate_group_fairness(
            fairness_top_k,
            "fairness_score",
            attribute
        )
    )


    fairness_results.append({

        "attribute":
            attribute,

        "baseline_SPD":
            baseline_result[
                "SPD"
            ],

        "fairness_SPD":
            fair_result[
                "SPD"
            ],

        "baseline_DI":
            baseline_result[
                "DI"
            ],

        "fairness_DI":
            fair_result[
                "DI"
            ]
    })


fairness_comparison = pd.DataFrame(
    fairness_results
)


print(
    fairness_comparison.to_string(
        index=False
    )
)

print()


print("=" * 60)
print("INTERSECTIONAL FAIRNESS COMPARISON")
print("=" * 60)


INTERSECTIONAL_GROUPS = [

    [
        "gender",
        "age_group"
    ],

    [
        "gender",
        "location"
    ],

    [
        "age_group",
        "location"
    ]
]


intersectional_results = []


for group_columns in INTERSECTIONAL_GROUPS:

    group_name = (
        " × ".join(
            group_columns
        )
    )


    baseline_group_scores = (
        baseline_top_k
        .groupby(
            group_columns
        )[
            "baseline_score"
        ]
        .mean()
    )


    fair_group_scores = (
        fairness_top_k
        .groupby(
            group_columns
        )[
            "fairness_score"
        ]
        .mean()
    )


    if (
        len(
            baseline_group_scores
        ) > 1
    ):

        baseline_min = (
            baseline_group_scores.min()
        )

        baseline_max = (
            baseline_group_scores.max()
        )

        baseline_di = (
            baseline_min
            / baseline_max
            if baseline_max != 0
            else np.nan
        )

        baseline_spd = (
            baseline_min
            - baseline_max
        )

    else:

        baseline_di = np.nan
        baseline_spd = np.nan


    if (
        len(
            fair_group_scores
        ) > 1
    ):

        fair_min = (
            fair_group_scores.min()
        )

        fair_max = (
            fair_group_scores.max()
        )

        fair_di = (
            fair_min
            / fair_max
            if fair_max != 0
            else np.nan
        )

        fair_spd = (
            fair_min
            - fair_max
        )

    else:

        fair_di = np.nan
        fair_spd = np.nan


    intersectional_results.append({

        "intersection":
            group_name,

        "baseline_SPD":
            baseline_spd,

        "fairness_SPD":
            fair_spd,

        "baseline_DI":
            baseline_di,

        "fairness_DI":
            fair_di
    })


intersectional_comparison = (
    pd.DataFrame(
        intersectional_results
    )
)


print(
    intersectional_comparison.to_string(
        index=False
    )
)

print()


print("=" * 60)
print("SAMPLE BASELINE RECOMMENDATIONS")
print("=" * 60)


sample_user = (
    df["user_id"]
    .iloc[0]
)


sample_baseline = (
    baseline_top_k[
        baseline_top_k[
            "user_id"
        ] == sample_user
    ]
    [
        [
            "user_id",
            "post_id",
            "baseline_score",
            "interaction"
        ]
    ]
    .sort_values(
        "baseline_score",
        ascending=False
    )
    .head(K)
)


print(
    sample_baseline.to_string(
        index=False
    )
)

print()


print("=" * 60)
print("SAMPLE FAIRNESS-AWARE RECOMMENDATIONS")
print("=" * 60)


sample_fair = (
    fairness_top_k[
        fairness_top_k[
            "user_id"
        ] == sample_user
    ]
    [
        [
            "user_id",
            "post_id",
            "fairness_score",
            "interaction"
        ]
    ]
    .sort_values(
        "fairness_score",
        ascending=False
    )
    .head(K)
)


print(
    sample_fair.to_string(
        index=False
    )
)

print()

print("=" * 60)
print("SAVING RESULTS")
print("=" * 60)


baseline_path = (
    RESULTS_PATH
    / "baseline_top10_for_fairness.csv"
)


fairness_path = (
    RESULTS_PATH
    / "fairness_aware_top10.csv"
)


quality_path = (
    RESULTS_PATH
    / "fairness_quality_comparison.csv"
)


fairness_metrics_path = (
    RESULTS_PATH
    / "fairness_metric_comparison.csv"
)


intersectional_path = (
    RESULTS_PATH
    / "fairness_intersectional_comparison.csv"
)


baseline_top_k.to_csv(
    baseline_path,
    index=False
)


fairness_top_k.to_csv(
    fairness_path,
    index=False
)


quality_comparison.to_csv(
    quality_path,
    index=False
)


fairness_comparison.to_csv(
    fairness_metrics_path,
    index=False
)


intersectional_comparison.to_csv(
    intersectional_path,
    index=False
)


print()
print(
    "1. baseline_top10_for_fairness.csv"
)

print(
    "2. fairness_aware_top10.csv"
)

print(
    "3. fairness_quality_comparison.csv"
)

print(
    "4. fairness_metric_comparison.csv"
)

print(
    "5. fairness_intersectional_comparison.csv"
)

print()


print("=" * 60)
print("FAIRNESS MITIGATION COMPLETED")
print("=" * 60)

print()

print(
    "Fairness strength:",
    FAIRNESS_STRENGTH
)

print(
    "Top-K:",
    K
)

print()

print(
    "Baseline and fairness-aware"
)

print(
    "recommendation results have been saved."
)

print()

print(
    "Results folder:"
)

print(
    RESULTS_PATH
)

print()