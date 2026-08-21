import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

RESULTS_PATH = (
    BASE_DIR
    / "results"
)



K = 10

FAIRNESS_STRENGTHS = [
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0
]

PROTECTED_ATTRIBUTES = [
    "gender",
    "age_group",
    "location"
]

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



print("=" * 60)
print("CHECKING REQUIRED FILES")
print("=" * 60)

for path in [
    DATA_PATH,
    MODEL_PATH,
    ENCODER_PATH
]:

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

print()



def apply_fairness_adjustment(
    data,
    strength
):

    result = data.copy()

    result["fairness_multiplier"] = 1.0


  
    overall_mean = (
        result["baseline_score"]
        .mean()
    )



    for attribute in PROTECTED_ATTRIBUTES:

        group_means = (
            result.groupby(
                attribute
            )[
                "baseline_score"
            ]
            .mean()
        )


        correction = (
            overall_mean
            / group_means
        )


        # Prevent extreme corrections

        correction = correction.clip(
            lower=0.80,
            upper=1.20
        )


        row_correction = (
            result[attribute]
            .map(correction)
            .fillna(1.0)
        )


        result["fairness_multiplier"] *= (

            1
            + strength
            * (
                row_correction - 1
            )

        )


  
    result["fairness_score"] = (

        result["baseline_score"]
        * result["fairness_multiplier"]

    )


    result["fairness_score"] = (
        result["fairness_score"]
        .clip(
            lower=0.0,
            upper=1.0
        )
    )


    return result


def calculate_quality_metrics(
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

        user_data = (
            user_data
            .sort_values(
                score_column,
                ascending=False
            )
        )


        actual = (
            user_data[
                "interaction"
            ]
            .to_numpy()
        )


        scores = (
            user_data[
                score_column
            ]
            .to_numpy()
        )


        if len(actual) == 0:
            continue


        top_k = actual[:k]

        precisions.append(
            top_k.mean()
        )


     
        total_relevant = (
            actual.sum()
        )


        if total_relevant > 0:

            recalls.append(
                top_k.sum()
                / total_relevant
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

        "precision_at_10":
            np.mean(
                precisions
            ),

        "recall_at_10":
            np.mean(
                recalls
            ),

        "ndcg_at_10":
            np.mean(
                ndcgs
            )
            if ndcgs
            else np.nan
    }



def calculate_fairness_metrics(
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


    if max_score != 0:

        di = (
            min_score
            / max_score
        )

    else:

        di = np.nan


    return spd, di



print("=" * 60)
print("RUNNING FAIRNESS STRENGTH EXPERIMENT")
print("=" * 60)

print()

all_results = []


for strength in FAIRNESS_STRENGTHS:

    print(
        f"Testing fairness strength: "
        f"{strength:.1f}"
    )


    adjusted_df = (
        apply_fairness_adjustment(
            df,
            strength
        )
    )


 
    adjusted_df = (
        adjusted_df
        .sort_values(
            [
                "user_id",
                "fairness_score"
            ],
            ascending=[
                True,
                False
            ]
        )
    )


    adjusted_df["fairness_rank"] = (
        adjusted_df
        .groupby(
            "user_id"
        )
        .cumcount()
        + 1
    )


    top_k_df = (
        adjusted_df[
            adjusted_df[
                "fairness_rank"
            ] <= K
        ]
        .copy()
    )



    quality = (
        calculate_quality_metrics(
            top_k_df,
            "fairness_score",
            K
        )
    )



    gender_spd, gender_di = (
        calculate_fairness_metrics(
            top_k_df,
            "fairness_score",
            "gender"
        )
    )


    age_spd, age_di = (
        calculate_fairness_metrics(
            top_k_df,
            "fairness_score",
            "age_group"
        )
    )


    location_spd, location_di = (
        calculate_fairness_metrics(
            top_k_df,
            "fairness_score",
            "location"
        )
    )


  
    all_results.append({

        "fairness_strength":
            strength,

        "precision_at_10":
            quality[
                "precision_at_10"
            ],

        "recall_at_10":
            quality[
                "recall_at_10"
            ],

        "ndcg_at_10":
            quality[
                "ndcg_at_10"
            ],

        "gender_spd":
            gender_spd,

        "gender_di":
            gender_di,

        "age_group_spd":
            age_spd,

        "age_group_di":
            age_di,

        "location_spd":
            location_spd,

        "location_di":
            location_di
    })


    print(
        f"  NDCG@10       : "
        f"{quality['ndcg_at_10']:.4f}"
    )

    print(
        f"  Gender DI     : "
        f"{gender_di:.4f}"
    )

    print(
        f"  Age-group DI  : "
        f"{age_di:.4f}"
    )

    print(
        f"  Location DI   : "
        f"{location_di:.4f}"
    )

    print()


results_df = pd.DataFrame(
    all_results
)



print("=" * 60)
print("FAIRNESS STRENGTH RESULTS")
print("=" * 60)

print()

print(
    results_df.to_string(
        index=False
    )
)

print()



print("=" * 60)
print("BEST BALANCED CONFIGURATION")
print("=" * 60)



results_df["fairness_gap"] = (

    abs(
        1
        - results_df[
            "gender_di"
        ]
    )

    +

    abs(
        1
        - results_df[
            "age_group_di"
        ]
    )

    +

    abs(
        1
        - results_df[
            "location_di"
        ]
    )
)



max_ndcg = (
    results_df[
        "ndcg_at_10"
    ].max()
)


min_ndcg = (
    results_df[
        "ndcg_at_10"
    ].min()
)


if max_ndcg != min_ndcg:

    results_df["ndcg_normalized"] = (

        (
            results_df[
                "ndcg_at_10"
            ]
            - min_ndcg
        )

        /

        (
            max_ndcg
            - min_ndcg
        )
    )

else:

    results_df[
        "ndcg_normalized"
    ] = 1.0


results_df["balanced_score"] = (

    results_df[
        "ndcg_normalized"
    ]

    -

    results_df[
        "fairness_gap"
    ]
)


best_row = (
    results_df
    .sort_values(
        "balanced_score",
        ascending=False
    )
    .iloc[0]
)


print(
    "Best fairness strength:",
    best_row[
        "fairness_strength"
    ]
)

print(
    "NDCG@10:",
    round(
        best_row[
            "ndcg_at_10"
        ],
        4
    )
)

print(
    "Gender DI:",
    round(
        best_row[
            "gender_di"
        ],
        4
    )
)

print(
    "Age-group DI:",
    round(
        best_row[
            "age_group_di"
        ],
        4
    )
)

print(
    "Location DI:",
    round(
        best_row[
            "location_di"
        ],
        4
    )
)

print(
    "Balanced score:",
    round(
        best_row[
            "balanced_score"
        ],
        4
    )
)

print()



print("=" * 60)
print("SAVING EXPERIMENT RESULTS")
print("=" * 60)


results_path = (
    RESULTS_PATH
    / "fairness_strength_results.csv"
)


results_df.to_csv(
    results_path,
    index=False
)


print(
    "Saved:"
)

print(
    results_path
)

print()


print("=" * 60)
print("CREATING FAIRNESS VS QUALITY PLOT")
print("=" * 60)


plt.figure(
    figsize=(10, 6)
)


plt.plot(
    results_df[
        "fairness_strength"
    ],
    results_df[
        "ndcg_at_10"
    ],
    marker="o",
    label="NDCG@10"
)


plt.xlabel(
    "Fairness Strength"
)

plt.ylabel(
    "NDCG@10"
)

plt.title(
    "Recommendation Quality vs Fairness Strength"
)

plt.grid(
    True,
    alpha=0.3
)

plt.legend()

plt.tight_layout()


quality_plot_path = (
    RESULTS_PATH
    / "fairness_strength_ndcg.png"
)


plt.savefig(
    quality_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "Saved:"
)

print(
    quality_plot_path
)

print()


plt.figure(
    figsize=(10, 6)
)


plt.plot(
    results_df[
        "fairness_strength"
    ],
    results_df[
        "gender_di"
    ],
    marker="o",
    label="Gender DI"
)


plt.plot(
    results_df[
        "fairness_strength"
    ],
    results_df[
        "age_group_di"
    ],
    marker="o",
    label="Age-group DI"
)


plt.plot(
    results_df[
        "fairness_strength"
    ],
    results_df[
        "location_di"
    ],
    marker="o",
    label="Location DI"
)


plt.axhline(
    y=1.0,
    linestyle="--",
    linewidth=1
)


plt.xlabel(
    "Fairness Strength"
)

plt.ylabel(
    "Disparate Impact"
)

plt.title(
    "Fairness Metrics vs Fairness Strength"
)

plt.grid(
    True,
    alpha=0.3
)

plt.legend()

plt.tight_layout()


fairness_plot_path = (
    RESULTS_PATH
    / "fairness_strength_di.png"
)


plt.savefig(
    fairness_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "Saved:"
)

print(
    fairness_plot_path
)

print()



print("=" * 60)
print("FAIRNESS STRENGTH EXPERIMENT COMPLETED")
print("=" * 60)

print()

print(
    "Generated files:"
)

print(
    "1. fairness_strength_results.csv"
)

print(
    "2. fairness_strength_ndcg.png"
)

print(
    "3. fairness_strength_di.png"
)

print()

print(
    "The experiment tested:"
)

print(
    FAIRNESS_STRENGTHS
)

print()

print(
    "Results folder:"
)

print(
    RESULTS_PATH
)