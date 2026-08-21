import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder



BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "synthetic_linkedin_dataset_30000.csv"
)

RESULTS_PATH = BASE_DIR / "results"


print("=" * 60)
print("PROJECT PATHS")
print("=" * 60)

print("Project folder:")
print(BASE_DIR)

print()

print("Dataset:")
print(DATA_PATH)

print()

print("Results folder:")
print(RESULTS_PATH)

print()



if not DATA_PATH.exists():

    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}"
    )


RESULTS_PATH.mkdir(
    parents=True,
    exist_ok=True
)



df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("DATASET LOADED")
print("=" * 60)

print("Shape:", df.shape)

print()



print("=" * 60)
print("DATA VALIDATION")
print("=" * 60)

missing_values = df.isnull().sum().sum()

duplicate_rows = df.duplicated().sum()

print("Missing values:", missing_values)

print("Duplicate rows:", duplicate_rows)


if missing_values > 0:

    raise ValueError(
        "Dataset contains missing values."
    )


if duplicate_rows > 0:

    raise ValueError(
        "Dataset contains duplicate rows."
    )


print("Validation passed.")

print()



# ID columns
ID_COLUMNS = [
    "user_id",
    "post_id",
    "author_id"
]


# Protected / audit attributes
PROTECTED_COLUMNS = [
    "gender",
    "age_group",
    "location"
]


# Categorical recommendation features
CATEGORICAL_FEATURES = [
    "professional_field",
    "education",
    "post_topic",
    "content_type"
]


# Numerical recommendation features
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


# Target
TARGET = "interaction"


# Potential target-derived feature
LEAKAGE_COLUMN = "interaction_probability"


required_columns = (
    ID_COLUMNS
    + PROTECTED_COLUMNS
    + CATEGORICAL_FEATURES
    + NUMERICAL_FEATURES
    + [
        TARGET,
        LEAKAGE_COLUMN
    ]
)


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    raise ValueError(
        f"Missing columns in dataset: {missing_columns}"
    )


print("=" * 60)
print("COLUMN CHECK")
print("=" * 60)

print("All required columns are present.")

print()



y = df[TARGET].copy()


print("=" * 60)
print("TARGET")
print("=" * 60)

print(
    y.value_counts()
)

print()



MODEL_FEATURES = (
    CATEGORICAL_FEATURES
    + NUMERICAL_FEATURES
)


X = df[
    MODEL_FEATURES
].copy()


print("=" * 60)
print("MODEL FEATURES")
print("=" * 60)

print(
    "Number of features before encoding:",
    X.shape[1]
)

print()

print("Categorical features:")

print(
    CATEGORICAL_FEATURES
)

print()

print("Numerical features:")

print(
    NUMERICAL_FEATURES
)

print()


protected_data = df[
    PROTECTED_COLUMNS
].copy()


print("=" * 60)
print("PROTECTED ATTRIBUTES")
print("=" * 60)

print(
    protected_data.head()
)

print()



(
    X_train,
    X_test,
    y_train,
    y_test,
    protected_train,
    protected_test
) = train_test_split(

    X,
    y,
    protected_data,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print("=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

print(
    "X_train:",
    X_train.shape
)

print(
    "X_test :",
    X_test.shape
)

print(
    "y_train:",
    y_train.shape
)

print(
    "y_test :",
    y_test.shape
)

print(
    "protected_train:",
    protected_train.shape
)

print(
    "protected_test :",
    protected_test.shape
)

print()



encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)


# Fit ONLY on training data
X_train_cat = encoder.fit_transform(
    X_train[
        CATEGORICAL_FEATURES
    ]
)


# Transform test data
X_test_cat = encoder.transform(
    X_test[
        CATEGORICAL_FEATURES
    ]
)



X_train_num = (
    X_train[
        NUMERICAL_FEATURES
    ]
    .reset_index(drop=True)
)


X_test_num = (
    X_test[
        NUMERICAL_FEATURES
    ]
    .reset_index(drop=True)
)


encoded_feature_names = (
    encoder.get_feature_names_out(
        CATEGORICAL_FEATURES
    )
)


X_train_cat = pd.DataFrame(
    X_train_cat,

    columns=encoded_feature_names
)


X_test_cat = pd.DataFrame(
    X_test_cat,

    columns=encoded_feature_names
)



X_train_processed = pd.concat(

    [
        X_train_num,

        X_train_cat.reset_index(
            drop=True
        )
    ],

    axis=1
)


X_test_processed = pd.concat(

    [
        X_test_num,

        X_test_cat.reset_index(
            drop=True
        )
    ],

    axis=1
)


y_train = (
    y_train
    .reset_index(drop=True)
)


y_test = (
    y_test
    .reset_index(drop=True)
)


protected_train = (
    protected_train
    .reset_index(drop=True)
)


protected_test = (
    protected_test
    .reset_index(drop=True)
)



print("=" * 60)
print("PROCESSED DATA")
print("=" * 60)

print(
    "X_train processed:",
    X_train_processed.shape
)

print(
    "X_test processed :",
    X_test_processed.shape
)

print()

print(
    "First five processed training records:"
)

print(
    X_train_processed.head()
)

print()

print(
    "Target training distribution:"
)

print(
    y_train.value_counts()
)

print()

print(
    "Target testing distribution:"
)

print(
    y_test.value_counts()
)

print()



print("=" * 60)
print("SAVING PROCESSED DATA")
print("=" * 60)

print(
    "Saving to:"
)

print(
    RESULTS_PATH
)

print()


# Training features
X_train_processed.to_csv(

    RESULTS_PATH
    / "X_train.csv",

    index=False
)


# Testing features
X_test_processed.to_csv(

    RESULTS_PATH
    / "X_test.csv",

    index=False
)


# Training target
y_train.to_csv(

    RESULTS_PATH
    / "y_train.csv",

    index=False
)


# Testing target
y_test.to_csv(

    RESULTS_PATH
    / "y_test.csv",

    index=False
)


# Training protected attributes
protected_train.to_csv(

    RESULTS_PATH
    / "protected_train.csv",

    index=False
)


# Testing protected attributes
protected_test.to_csv(

    RESULTS_PATH
    / "protected_test.csv",

    index=False
)


joblib.dump(

    encoder,

    RESULTS_PATH
    / "categorical_encoder.pkl"
)


print()

print("=" * 60)
print("VERIFYING SAVED FILES")
print("=" * 60)


expected_files = [

    "X_train.csv",

    "X_test.csv",

    "y_train.csv",

    "y_test.csv",

    "protected_train.csv",

    "protected_test.csv",

    "categorical_encoder.pkl"
]


all_files_created = True


for filename in expected_files:

    file_path = (
        RESULTS_PATH
        / filename
    )


    if file_path.exists():

        size_kb = (
            file_path.stat().st_size
            / 1024
        )

        print(
            f"✓ {filename:<30}"
            f"{size_kb:.2f} KB"
        )

    else:

        print(
            f"✗ {filename:<30}"
            "NOT FOUND"
        )

        all_files_created = False


print()

print("=" * 60)

if all_files_created:

    print(
        "PREPROCESSING COMPLETED SUCCESSFULLY"
    )

else:

    print(
        "PREPROCESSING FINISHED WITH ERRORS"
    )

print("=" * 60)

print()

print(
    "Results folder:"
)

print(
    RESULTS_PATH
)