import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from src.config import (
    DATA_PATH,
    RESULTS_DIR,
    RANDOM_SEED,
    ID_COLUMNS,
    PROTECTED_ATTRIBUTES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    MODEL_FEATURES,
    TARGET
)

def preprocess_and_split():
    print("=" * 60)
    print("DATA PREPROCESSING & USER-STRATIFIED 70/15/15 SPLITTING")
    print("=" * 60)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset: {df.shape[0]:,} rows, {df.shape[1]} columns, {df['user_id'].nunique():,} unique users")

    missing = df.isnull().sum().sum()
    duplicates = df.duplicated().sum()
    print(f"Missing values: {missing}, Duplicates: {duplicates}")

    if missing > 0 or duplicates > 0:
        raise ValueError("Dataset contains missing or duplicate records.")

    # 1. User-Level Stratified Splitting: 70% Train Users, 15% Val Users, 15% Test Users
    # This guarantees zero evaluation leakage and preserves complete candidate pools for each user.
    user_metadata = df.drop_duplicates("user_id")[["user_id", "gender", "age_group", "location"]]
    
    train_users, temp_users = train_test_split(
        user_metadata,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=user_metadata["gender"]
    )
    val_users, test_users = train_test_split(
        temp_users,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temp_users["gender"]
    )

    train_df = df[df["user_id"].isin(train_users["user_id"])].reset_index(drop=True)
    val_df = df[df["user_id"].isin(val_users["user_id"])].reset_index(drop=True)
    test_df = df[df["user_id"].isin(test_users["user_id"])].reset_index(drop=True)

    print(f"\nSplit Distribution:")
    print(f"Train split : {train_df.shape[0]:,} rows ({len(train_users):,} users, {train_df.shape[0]/len(df):.1%})")
    print(f"Val split   : {val_df.shape[0]:,} rows ({len(val_users):,} users, {val_df.shape[0]/len(df):.1%})")
    print(f"Test split  : {test_df.shape[0]:,} rows ({len(test_users):,} users, {test_df.shape[0]/len(df):.1%})")

    # Save full splits with IDs and metadata
    train_df.to_csv(RESULTS_DIR / "train_split.csv", index=False)
    val_df.to_csv(RESULTS_DIR / "val_split.csv", index=False)
    test_df.to_csv(RESULTS_DIR / "test_split.csv", index=False)

    # 2. Categorical Encoding (fit ONLY on training split)
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    train_cat = encoder.fit_transform(train_df[CATEGORICAL_FEATURES])
    val_cat = encoder.transform(val_df[CATEGORICAL_FEATURES])
    test_cat = encoder.transform(test_df[CATEGORICAL_FEATURES])

    encoded_feature_names = encoder.get_feature_names_out(CATEGORICAL_FEATURES)
    train_cat_df = pd.DataFrame(train_cat, columns=encoded_feature_names)
    val_cat_df = pd.DataFrame(val_cat, columns=encoded_feature_names)
    test_cat_df = pd.DataFrame(test_cat, columns=encoded_feature_names)

    # 3. Numerical features
    train_num_df = train_df[NUMERICAL_FEATURES].reset_index(drop=True)
    val_num_df = val_df[NUMERICAL_FEATURES].reset_index(drop=True)
    test_num_df = test_df[NUMERICAL_FEATURES].reset_index(drop=True)

    # Combined matrices
    X_train = pd.concat([train_num_df, train_cat_df], axis=1)
    X_val = pd.concat([val_num_df, val_cat_df], axis=1)
    X_test = pd.concat([test_num_df, test_cat_df], axis=1)

    y_train = train_df[TARGET]
    y_val = val_df[TARGET]
    y_test = test_df[TARGET]

    protected_train = train_df[PROTECTED_ATTRIBUTES]
    protected_val = val_df[PROTECTED_ATTRIBUTES]
    protected_test = test_df[PROTECTED_ATTRIBUTES]

    X_train.to_csv(RESULTS_DIR / "X_train.csv", index=False)
    X_val.to_csv(RESULTS_DIR / "X_val.csv", index=False)
    X_test.to_csv(RESULTS_DIR / "X_test.csv", index=False)

    y_train.to_csv(RESULTS_DIR / "y_train.csv", index=False)
    y_val.to_csv(RESULTS_DIR / "y_val.csv", index=False)
    y_test.to_csv(RESULTS_DIR / "y_test.csv", index=False)

    protected_train.to_csv(RESULTS_DIR / "protected_train.csv", index=False)
    protected_val.to_csv(RESULTS_DIR / "protected_val.csv", index=False)
    protected_test.to_csv(RESULTS_DIR / "protected_test.csv", index=False)

    joblib.dump(encoder, RESULTS_DIR / "categorical_encoder.pkl")

    print("\nSaved split artifacts to:", RESULTS_DIR)
    print("User-stratified pre-processing completed successfully.\n")

if __name__ == "__main__":
    preprocess_and_split()