import pandas as pd

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)



BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_PATH = BASE_DIR / "results"

PREDICTIONS_PATH = (
    RESULTS_PATH
    / "fairness_predictions.csv"
)



print("=" * 60)
print("CHECKING REQUIRED FILE")
print("=" * 60)

if not PREDICTIONS_PATH.exists():

    raise FileNotFoundError(
        f"Fairness prediction file not found:\n"
        f"{PREDICTIONS_PATH}\n\n"
        "Run fairness_analysis.py first."
    )

print("Prediction file found:")
print(PREDICTIONS_PATH)

print()


print("=" * 60)
print("LOADING FAIRNESS PREDICTIONS")
print("=" * 60)

df = pd.read_csv(
    PREDICTIONS_PATH
)

print(
    "Dataset shape:",
    df.shape
)

print()

print("Columns:")

print(
    df.columns.tolist()
)

print()



REQUIRED_COLUMNS = [

    "user_id",

    "post_id",

    "gender",

    "age_group",

    "location",

    "interaction",

    "prediction_probability",

    "predicted_interaction"
]


missing_columns = [

    column

    for column in REQUIRED_COLUMNS

    if column not in df.columns
]


if missing_columns:

    raise ValueError(
        "Missing required columns: "
        + str(missing_columns)
    )


print("=" * 60)
print("COLUMN VALIDATION")
print("=" * 60)

print("All required columns are present.")

print()

def calculate_intersectional_metrics(
    data,
    group_columns
):

    results = []


    grouped = data.groupby(
        group_columns,
        dropna=False
    )


    for group_values, group_data in grouped:


        if not isinstance(
            group_values,
            tuple
        ):

            group_values = (
                group_values,
            )


        group_info = {}


        for column, value in zip(
            group_columns,
            group_values
        ):

            group_info[column] = value


     
        actual = group_data[
            "interaction"
        ]

        predicted = group_data[
            "predicted_interaction"
        ]

        probability = group_data[
            "prediction_probability"
        ]


        actual_interaction_rate = (
            actual.mean()
        )


        predicted_positive_rate = (
            predicted.mean()
        )


       
        average_prediction_score = (
            probability.mean()
        )


        accuracy = accuracy_score(
            actual,
            predicted
        )


        precision = precision_score(
            actual,
            predicted,
            zero_division=0
        )


        recall = recall_score(
            actual,
            predicted,
            zero_division=0
        )


        f1 = f1_score(
            actual,
            predicted,
            zero_division=0
        )


        group_info.update({

            "sample_count":
                len(group_data),

            "actual_interaction_rate":
                actual_interaction_rate,

            "predicted_positive_rate":
                predicted_positive_rate,

            "average_prediction_score":
                average_prediction_score,

            "accuracy":
                accuracy,

            "precision":
                precision,

            "recall":
                recall,

            "f1":
                f1
        })


        results.append(
            group_info
        )


    return pd.DataFrame(
        results
    )



def calculate_fairness_summary(
    metrics_df,
    group_name
):

    predicted_rates = (
        metrics_df[
            "predicted_positive_rate"
        ]
    )


    max_rate = (
        predicted_rates.max()
    )

    min_rate = (
        predicted_rates.min()
    )


   
    spd = (
        min_rate
        - max_rate
    )


   
    if max_rate == 0:

        disparate_impact = None

    else:

        disparate_impact = (
            min_rate
            / max_rate
        )


    
    rate_difference = (
        max_rate
        - min_rate
    )


    max_group = metrics_df.loc[
        metrics_df[
            "predicted_positive_rate"
        ].idxmax()
    ]


    min_group = metrics_df.loc[
        metrics_df[
            "predicted_positive_rate"
        ].idxmin()
    ]


    return {

        "intersection":
            group_name,

        "number_of_groups":
            len(metrics_df),

        "minimum_predicted_rate":
            min_rate,

        "maximum_predicted_rate":
            max_rate,

        "rate_difference":
            rate_difference,

        "statistical_parity_difference":
            spd,

        "disparate_impact":
            disparate_impact,

        "highest_rate_group":
            str(
                max_group.to_dict()
            ),

        "lowest_rate_group":
            str(
                min_group.to_dict()
            )
    }

print("=" * 60)
print("INTERSECTION 1: GENDER × AGE GROUP")
print("=" * 60)

gender_age_metrics = (
    calculate_intersectional_metrics(
        df,
        [
            "gender",
            "age_group"
        ]
    )
)


print(
    gender_age_metrics.to_string(
        index=False
    )
)

print()


print("=" * 60)
print("INTERSECTION 2: GENDER × LOCATION")
print("=" * 60)

gender_location_metrics = (
    calculate_intersectional_metrics(
        df,
        [
            "gender",
            "location"
        ]
    )
)


print(
    gender_location_metrics.to_string(
        index=False
    )
)

print()



print("=" * 60)
print("INTERSECTION 3: AGE GROUP × LOCATION")
print("=" * 60)

age_location_metrics = (
    calculate_intersectional_metrics(
        df,
        [
            "age_group",
            "location"
        ]
    )
)


print(
    age_location_metrics.to_string(
        index=False
    )
)

print()


# 10. GENDER × AGE GROUP × LOCATION
# ============================================================

print("=" * 60)
print(
    "INTERSECTION 4: GENDER × AGE GROUP × LOCATION"
)
print("=" * 60)

gender_age_location_metrics = (
    calculate_intersectional_metrics(
        df,
        [
            "gender",
            "age_group",
            "location"
        ]
    )
)


# Do not print the entire three-way table
# because it can become large.

print(
    "Number of intersectional groups:",
    len(
        gender_age_location_metrics
    )
)

print()

print(
    "Smallest groups:"
)

print(
    gender_age_location_metrics
    .sort_values(
        "sample_count"
    )
    .head(10)
    .to_string(index=False)
)

print()

print(
    "Largest groups:"
)

print(
    gender_age_location_metrics
    .sort_values(
        "sample_count",
        ascending=False
    )
    .head(10)
    .to_string(index=False)
)

print()


# 11. FAIRNESS SUMMARIES
# ============================================================

print("=" * 60)
print("INTERSECTIONAL FAIRNESS SUMMARY")
print("=" * 60)


summary_results = []


summary_results.append(
    calculate_fairness_summary(
        gender_age_metrics,
        "gender × age_group"
    )
)


summary_results.append(
    calculate_fairness_summary(
        gender_location_metrics,
        "gender × location"
    )
)


summary_results.append(
    calculate_fairness_summary(
        age_location_metrics,
        "age_group × location"
    )
)


summary_results.append(
    calculate_fairness_summary(
        gender_age_location_metrics,
        "gender × age_group × location"
    )
)


intersectional_summary = pd.DataFrame(
    summary_results
)


print(
    intersectional_summary[
        [
            "intersection",
            "number_of_groups",
            "minimum_predicted_rate",
            "maximum_predicted_rate",
            "rate_difference",
            "statistical_parity_difference",
            "disparate_impact"
        ]
    ].to_string(
        index=False
    )
)

print()


# 12. FIND MOST DISADVANTAGED INTERSECTION
# ============================================================

print("=" * 60)
print("MOST DISADVANTAGED INTERSECTIONAL GROUPS")
print("=" * 60)


def show_lowest_groups(
    metrics_df,
    group_columns,
    n=10
):

    columns_to_show = (
        group_columns
        + [
            "sample_count",
            "actual_interaction_rate",
            "predicted_positive_rate",
            "average_prediction_score",
            "accuracy",
            "precision",
            "recall",
            "f1"
        ]
    )


    return (
        metrics_df
        .sort_values(
            "predicted_positive_rate"
        )
        [columns_to_show]
        .head(n)
    )


print()
print("Gender × Age Group:")
print()

print(
    show_lowest_groups(
        gender_age_metrics,
        [
            "gender",
            "age_group"
        ]
    ).to_string(
        index=False
    )
)

print()


print("Gender × Location:")
print()

print(
    show_lowest_groups(
        gender_location_metrics,
        [
            "gender",
            "location"
        ]
    ).to_string(
        index=False
    )
)

print()


print("Age Group × Location:")
print()

print(
    show_lowest_groups(
        age_location_metrics,
        [
            "age_group",
            "location"
        ]
    ).to_string(
        index=False
    )
)

print()



# 13. FIND MOST FAVORED INTERSECTION

print("=" * 60)
print("HIGHEST PREDICTED-RATE INTERSECTIONAL GROUPS")
print("=" * 60)


def show_highest_groups(
    metrics_df,
    group_columns,
    n=10
):

    columns_to_show = (
        group_columns
        + [
            "sample_count",
            "actual_interaction_rate",
            "predicted_positive_rate",
            "average_prediction_score",
            "accuracy",
            "precision",
            "recall",
            "f1"
        ]
    )


    return (
        metrics_df
        .sort_values(
            "predicted_positive_rate",
            ascending=False
        )
        [columns_to_show]
        .head(n)
    )


print()
print("Gender × Age Group:")
print()

print(
    show_highest_groups(
        gender_age_metrics,
        [
            "gender",
            "age_group"
        ]
    ).to_string(
        index=False
    )
)

print()


print("Gender × Location:")
print()

print(
    show_highest_groups(
        gender_location_metrics,
        [
            "gender",
            "location"
        ]
    ).to_string(
        index=False
    )
)

print()


print("Age Group × Location:")
print()

print(
    show_highest_groups(
        age_location_metrics,
        [
            "age_group",
            "location"
        ]
    ).to_string(
        index=False
    )
)

print()


print("=" * 60)
print("SAVING INTERSECTIONAL RESULTS")
print("=" * 60)


gender_age_path = (
    RESULTS_PATH
    / "intersectional_gender_age.csv"
)

gender_location_path = (
    RESULTS_PATH
    / "intersectional_gender_location.csv"
)

age_location_path = (
    RESULTS_PATH
    / "intersectional_age_location.csv"
)

gender_age_location_path = (
    RESULTS_PATH
    / "intersectional_gender_age_location.csv"
)

summary_path = (
    RESULTS_PATH
    / "intersectional_fairness_summary.csv"
)


gender_age_metrics.to_csv(
    gender_age_path,
    index=False
)


gender_location_metrics.to_csv(
    gender_location_path,
    index=False
)


age_location_metrics.to_csv(
    age_location_path,
    index=False
)


gender_age_location_metrics.to_csv(
    gender_age_location_path,
    index=False
)


intersectional_summary.to_csv(
    summary_path,
    index=False
)


print()
print("Saved files:")

print(
    "1. intersectional_gender_age.csv"
)

print(
    "2. intersectional_gender_location.csv"
)

print(
    "3. intersectional_age_location.csv"
)

print(
    "4. intersectional_gender_age_location.csv"
)

print(
    "5. intersectional_fairness_summary.csv"
)

print()



print("=" * 60)
print("INTERSECTIONAL FAIRNESS ANALYSIS COMPLETED")
print("=" * 60)

print()

print(
    "The baseline recommender has now been"
)

print(
    "evaluated across individual and"
)

print(
    "intersectional protected groups."
)

print()