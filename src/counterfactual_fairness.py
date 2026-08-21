import joblib
import pandas as pd
import numpy as np

from pathlib import Path


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


required_files = {
    "Dataset": DATA_PATH,
    "XGBoost model": MODEL_PATH,
    "Categorical encoder": ENCODER_PATH
}


for name, path in required_files.items():

    if not path.exists():

        raise FileNotFoundError(
            f"{name} not found:\n{path}"
        )

    print(f"✓ {path}")


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


required_columns = (
    MODEL_FEATURES
    + PROTECTED_ATTRIBUTES
    + ["interaction"]
)


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    raise ValueError(
        "Missing columns:\n"
        + str(missing_columns)
    )


print("=" * 60)
print("COLUMN VALIDATION")
print("=" * 60)

print("All required columns are present.")

print()



print("=" * 60)
print("LOADING XGBOOST MODEL")
print("=" * 60)


model = joblib.load(
    MODEL_PATH
)


print("Model loaded successfully.")

print()


print("=" * 60)
print("LOADING CATEGORICAL ENCODER")
print("=" * 60)


encoder = joblib.load(
    ENCODER_PATH
)


print("Encoder loaded successfully.")

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
print("GENERATING ORIGINAL PREDICTIONS")
print("=" * 60)


original_scores = model.predict_proba(
    X_processed
)[:, 1]


df["original_prediction"] = (
    original_scores
)


print(
    "Original predictions generated."
)

print(
    "Score range:",
    round(original_scores.min(), 4),
    "to",
    round(original_scores.max(), 4)
)

print()


def perform_counterfactual_analysis(
    data,
    attribute,
    model,
    encoder
):

    print("=" * 60)

    print(
        f"COUNTERFACTUAL ANALYSIS: {attribute.upper()}"
    )

    print("=" * 60)


    attribute_values = (
        data[attribute]
        .dropna()
        .unique()
        .tolist()
    )


    print(
        "Protected attribute values:",
        attribute_values
    )

    print()


    results = []


    for counterfactual_value in attribute_values:

     
        counterfactual_df = data.copy()


       
        counterfactual_df[
            attribute
        ] = counterfactual_value


        X_cf = counterfactual_df[
            MODEL_FEATURES
        ].copy()


   
        X_cf_categorical = encoder.transform(
            X_cf[
                CATEGORICAL_FEATURES
            ]
        )


        X_cf_categorical = pd.DataFrame(
            X_cf_categorical,
            columns=encoded_feature_names,
            index=X_cf.index
        )


        X_cf_numerical = X_cf[
            NUMERICAL_FEATURES
        ].copy()


        X_cf_processed = pd.concat(
            [
                X_cf_numerical,
                X_cf_categorical
            ],
            axis=1
        )


   
        counterfactual_scores = (
            model.predict_proba(
                X_cf_processed
            )[:, 1]
        )


        score_difference = (
            counterfactual_scores
            - data["original_prediction"].to_numpy()
        )


        absolute_difference = np.abs(
            score_difference
        )


        mean_difference = (
            score_difference.mean()
        )


        mean_absolute_difference = (
            absolute_difference.mean()
        )


        max_absolute_difference = (
            absolute_difference.max()
        )


        changed_predictions = (
            np.sum(
                absolute_difference > 1e-10
            )
        )


        percentage_changed = (
            changed_predictions
            / len(data)
            * 100
        )


        results.append({

            "protected_attribute":
                attribute,

            "counterfactual_value":
                counterfactual_value,

            "sample_count":
                len(data),

            "mean_original_score":
                data[
                    "original_prediction"
                ].mean(),

            "mean_counterfactual_score":
                counterfactual_scores.mean(),

            "mean_score_difference":
                mean_difference,

            "mean_absolute_score_difference":
                mean_absolute_difference,

            "maximum_absolute_score_difference":
                max_absolute_difference,

            "changed_predictions":
                changed_predictions,

            "percentage_predictions_changed":
                percentage_changed
        })


        print(
            f"Counterfactual value: "
            f"{counterfactual_value}"
        )

        print(
            f"Mean score: "
            f"{counterfactual_scores.mean():.6f}"
        )

        print(
            f"Mean difference: "
            f"{mean_difference:.10f}"
        )

        print(
            f"Predictions changed: "
            f"{changed_predictions}"
        )

        print()


    return pd.DataFrame(
        results
    )


gender_results = (
    perform_counterfactual_analysis(
        df,
        "gender",
        model,
        encoder
    )
)



age_results = (
    perform_counterfactual_analysis(
        df,
        "age_group",
        model,
        encoder
    )
)


location_results = (
    perform_counterfactual_analysis(
        df,
        "location",
        model,
        encoder
    )
)


print("=" * 60)
print("COMBINING COUNTERFACTUAL RESULTS")
print("=" * 60)


counterfactual_results = pd.concat(
    [
        gender_results,
        age_results,
        location_results
    ],
    ignore_index=True
)


print(
    counterfactual_results.to_string(
        index=False
    )
)

print()


print("=" * 60)
print("COUNTERFACTUAL FAIRNESS SUMMARY")
print("=" * 60)


summary = []


for attribute in PROTECTED_ATTRIBUTES:

    attribute_data = (
        counterfactual_results[
            counterfactual_results[
                "protected_attribute"
            ] == attribute
        ]
    )


    summary.append({

        "protected_attribute":
            attribute,

        "maximum_mean_absolute_difference":
            attribute_data[
                "mean_absolute_score_difference"
            ].max(),

        "maximum_prediction_difference":
            attribute_data[
                "maximum_absolute_score_difference"
            ].max(),

        "total_changed_predictions":
            attribute_data[
                "changed_predictions"
            ].max()
    })


counterfactual_summary = pd.DataFrame(
    summary
)


print(
    counterfactual_summary.to_string(
        index=False
    )
)

print()


print("=" * 60)
print("COUNTERFACTUAL INTERPRETATION")
print("=" * 60)


print()
print(
    "The protected attributes gender, age_group,"
)

print(
    "and location are not included directly"
)

print(
    "in the XGBoost model features."
)

print()

print(
    "Therefore, changing only a protected"
)

print(
    "attribute while keeping model features"
)

print(
    "unchanged should not change the prediction."
)

print()

print(
    "A zero or near-zero counterfactual"
)

print(
    "difference indicates no direct dependence"
)

print(
    "on that protected attribute."
)

print()

print(
    "This does NOT prove complete fairness."
)

print(
    "Indirect or proxy-based disparities can"
)

print(
    "still exist through other model features."
)

print()



print("=" * 60)
print("SAVING COUNTERFACTUAL RESULTS")
print("=" * 60)


all_results_path = (
    RESULTS_PATH
    / "counterfactual_fairness.csv"
)


summary_path = (
    RESULTS_PATH
    / "counterfactual_fairness_summary.csv"
)


gender_path = (
    RESULTS_PATH
    / "counterfactual_gender.csv"
)


age_path = (
    RESULTS_PATH
    / "counterfactual_age_group.csv"
)


location_path = (
    RESULTS_PATH
    / "counterfactual_location.csv"
)


counterfactual_results.to_csv(
    all_results_path,
    index=False
)


counterfactual_summary.to_csv(
    summary_path,
    index=False
)


gender_results.to_csv(
    gender_path,
    index=False
)


age_results.to_csv(
    age_path,
    index=False
)


location_results.to_csv(
    location_path,
    index=False
)


print()
print(
    "1. counterfactual_fairness.csv"
)

print(
    "2. counterfactual_fairness_summary.csv"
)

print(
    "3. counterfactual_gender.csv"
)

print(
    "4. counterfactual_age_group.csv"
)

print(
    "5. counterfactual_location.csv"
)

print()


print("=" * 60)
print("COUNTERFACTUAL FAIRNESS ANALYSIS COMPLETED")
print("=" * 60)

print()

print(
    "Counterfactual analysis has been completed"
)

print(
    "for gender, age_group, and location."
)

print()

print(
    "Results saved in:"
)

print(
    RESULTS_PATH
)