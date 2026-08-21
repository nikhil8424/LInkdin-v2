import joblib
import pandas as pd

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)



BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "synthetic_linkedin_dataset_30000.csv"
)

RESULTS_PATH = BASE_DIR / "results"

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "xgboost_baseline.pkl"
)



print("=" * 60)
print("LOADING DATA")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print()


if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Baseline model not found:\n{MODEL_PATH}\n\n"
        "Run recommendation.py first."
    )


model = joblib.load(MODEL_PATH)

print("=" * 60)
print("BASELINE MODEL LOADED")
print("=" * 60)

print(MODEL_PATH)
print()


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


ENCODER_PATH = (
    RESULTS_PATH
    / "categorical_encoder.pkl"
)

if not ENCODER_PATH.exists():

    raise FileNotFoundError(
        f"Encoder not found:\n{ENCODER_PATH}\n\n"
        "Run data_preprocessing.py first."
    )


encoder = joblib.load(
    ENCODER_PATH
)

print("=" * 60)
print("ENCODER LOADED")
print("=" * 60)

print(ENCODER_PATH)
print()


X = df[
    MODEL_FEATURES
].copy()


# Encode categorical variables

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
    columns=encoded_feature_names
)


# Numerical features

X_num = (
    X[NUMERICAL_FEATURES]
    .reset_index(drop=True)
)


# Combine

X_processed = pd.concat(
    [
        X_num,
        X_cat
    ],
    axis=1
)


print("=" * 60)
print("FEATURES PREPARED")
print("=" * 60)

print(
    "Processed feature shape:",
    X_processed.shape
)

print()

print("=" * 60)
print("GENERATING BASELINE PREDICTIONS")
print("=" * 60)


df["prediction_probability"] = (
    model.predict_proba(
        X_processed
    )[:, 1]
)


df["predicted_interaction"] = (
    model.predict(
        X_processed
    )
)


print("Predictions generated.")

print()


accuracy = accuracy_score(
    df["interaction"],
    df["predicted_interaction"]
)

precision = precision_score(
    df["interaction"],
    df["predicted_interaction"],
    zero_division=0
)

recall = recall_score(
    df["interaction"],
    df["predicted_interaction"],
    zero_division=0
)

f1 = f1_score(
    df["interaction"],
    df["predicted_interaction"],
    zero_division=0
)


print("=" * 60)
print("OVERALL BASELINE PERFORMANCE")
print("=" * 60)

print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)

print()


def calculate_group_metrics(
    data,
    protected_column
):

    results = []


    for group, group_data in data.groupby(
        protected_column
    ):

        actual = group_data[
            "interaction"
        ]

        predicted = group_data[
            "predicted_interaction"
        ]

        probability = group_data[
            "prediction_probability"
        ]


        actual_positive_rate = (
            actual.mean()
        )


        predicted_positive_rate = (
            predicted.mean()
        )


        average_score = (
            probability.mean()
        )


        group_accuracy = accuracy_score(
            actual,
            predicted
        )


        group_precision = precision_score(
            actual,
            predicted,
            zero_division=0
        )


        group_recall = recall_score(
            actual,
            predicted,
            zero_division=0
        )


        group_f1 = f1_score(
            actual,
            predicted,
            zero_division=0
        )


        results.append({

            protected_column: group,

            "sample_count": len(group_data),

            "actual_interaction_rate":
                actual_positive_rate,

            "predicted_positive_rate":
                predicted_positive_rate,

            "average_prediction_score":
                average_score,

            "accuracy":
                group_accuracy,

            "precision":
                group_precision,

            "recall":
                group_recall,

            "f1":
                group_f1
        })


    return pd.DataFrame(results)


print("=" * 60)
print("GENDER FAIRNESS")
print("=" * 60)


gender_metrics = calculate_group_metrics(
    df,
    "gender"
)


print(
    gender_metrics.to_string(
        index=False
    )
)

print()


print("=" * 60)
print("AGE-GROUP FAIRNESS")
print("=" * 60)


age_metrics = calculate_group_metrics(
    df,
    "age_group"
)


print(
    age_metrics.to_string(
        index=False
    )
)

print()


print("=" * 60)
print("LOCATION FAIRNESS")
print("=" * 60)


location_metrics = calculate_group_metrics(
    df,
    "location"
)


print(
    location_metrics.to_string(
        index=False
    )
)

print()


def statistical_parity_difference(
    data,
    protected_column
):

    group_rates = (
        data
        .groupby(
            protected_column
        )["predicted_interaction"]
        .mean()
    )


    if len(group_rates) < 2:

        return None


    max_rate = group_rates.max()

    min_rate = group_rates.min()


    return min_rate - max_rate


def disparate_impact(
    data,
    protected_column
):

    group_rates = (
        data
        .groupby(
            protected_column
        )["predicted_interaction"]
        .mean()
    )


    if len(group_rates) < 2:

        return None


    max_rate = group_rates.max()

    min_rate = group_rates.min()


    if max_rate == 0:

        return None


    return min_rate / max_rate

gender_spd = (
    statistical_parity_difference(
        df,
        "gender"
    )
)

gender_di = (
    disparate_impact(
        df,
        "gender"
    )
)


age_spd = (
    statistical_parity_difference(
        df,
        "age_group"
    )
)

age_di = (
    disparate_impact(
        df,
        "age_group"
    )
)


location_spd = (
    statistical_parity_difference(
        df,
        "location"
    )
)

location_di = (
    disparate_impact(
        df,
        "location"
    )
)


print("=" * 60)
print("FAIRNESS SUMMARY")
print("=" * 60)

print(
    f"Gender SPD       : {gender_spd:.4f}"
)

print(
    f"Gender DI        : {gender_di:.4f}"
)

print()

print(
    f"Age-group SPD    : {age_spd:.4f}"
)

print(
    f"Age-group DI     : {age_di:.4f}"
)

print()

print(
    f"Location SPD     : {location_spd:.4f}"
)

print(
    f"Location DI      : {location_di:.4f}"
)

print()


gender_path = (
    RESULTS_PATH
    / "fairness_gender.csv"
)

age_path = (
    RESULTS_PATH
    / "fairness_age_group.csv"
)

location_path = (
    RESULTS_PATH
    / "fairness_location.csv"
)


gender_metrics.to_csv(
    gender_path,
    index=False
)

age_metrics.to_csv(
    age_path,
    index=False
)

location_metrics.to_csv(
    location_path,
    index=False
)


fairness_summary = pd.DataFrame({

    "protected_attribute": [

        "gender",

        "age_group",

        "location"
    ],

    "statistical_parity_difference": [

        gender_spd,

        age_spd,

        location_spd
    ],

    "disparate_impact": [

        gender_di,

        age_di,

        location_di
    ]
})


summary_path = (
    RESULTS_PATH
    / "fairness_summary.csv"
)


fairness_summary.to_csv(
    summary_path,
    index=False
)


prediction_path = (
    RESULTS_PATH
    / "fairness_predictions.csv"
)


df[
    [
        "user_id",
        "post_id",
        "gender",
        "age_group",
        "location",
        "interaction",
        "prediction_probability",
        "predicted_interaction"
    ]
].to_csv(
    prediction_path,
    index=False
)


print("=" * 60)
print("FAIRNESS ANALYSIS COMPLETED")
print("=" * 60)

print()

print("Saved files:")

print(
    "1. fairness_gender.csv"
)

print(
    "2. fairness_age_group.csv"
)

print(
    "3. fairness_location.csv"
)

print(
    "4. fairness_summary.csv"
)

print(
    "5. fairness_predictions.csv"
)

print()

print("Results folder:")

print(RESULTS_PATH)