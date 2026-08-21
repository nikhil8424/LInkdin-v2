import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import ndcg_score

print("=== 1. DATA SPLIT VERIFICATION ===")
raw_df = pd.read_csv('data/synthetic_linkedin_dataset_30000.csv')
train_df = pd.read_csv('results/train_split.csv')
val_df = pd.read_csv('results/val_split.csv')
test_df = pd.read_csv('results/test_split.csv')

print(f"Total raw observations: {len(raw_df):,}")
print(f"Train split: {len(train_df):,} rows ({len(train_df)/len(raw_df):.2%}), {train_df['user_id'].nunique():,} unique users")
print(f"Val split:   {len(val_df):,} rows ({len(val_df)/len(raw_df):.2%}), {val_df['user_id'].nunique():,} unique users")
print(f"Test split:  {len(test_df):,} rows ({len(test_df)/len(raw_df):.2%}), {test_df['user_id'].nunique():,} unique users")

train_users = set(train_df['user_id'])
val_users = set(val_df['user_id'])
test_users = set(test_df['user_id'])

print("User overlap train & val:", len(train_users & val_users))
print("User overlap train & test:", len(train_users & test_users))
print("User overlap val & test:", len(val_users & test_users))

# Verify features in X_train, X_val, X_test
X_train = pd.read_csv('results/X_train.csv')
X_val = pd.read_csv('results/X_val.csv')
X_test = pd.read_csv('results/X_test.csv')
print(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}, X_test shape: {X_test.shape}")

# Check if protected attributes in X
for attr in ['gender', 'age_group', 'location']:
    print(f"Attribute '{attr}' in X_train: {attr in X_train.columns}, in X_test: {attr in X_test.columns}")

print("\n=== 2. CANDIDATE POOL PIPELINE VERIFICATION ===")
user_cand_counts = test_df.groupby("user_id").size()
print(f"Test users: {len(user_cand_counts)}")
print(f"Min candidates: {user_cand_counts.min()}, Max: {user_cand_counts.max()}, Mean: {user_cand_counts.mean():.2f}, Median: {user_cand_counts.median():.1f}")
print(f"Users with < 5 candidates: {(user_cand_counts < 5).sum()} ({(user_cand_counts < 5).mean():.2%})")
print(f"Users with == 5 candidates: {(user_cand_counts == 5).sum()}")
print(f"Users with 5 < cand < 10: {((user_cand_counts > 5) & (user_cand_counts < 10)).sum()}")
print(f"Users with == 10 candidates: {(user_cand_counts == 10).sum()}")
print(f"Users with > 10 candidates: {(user_cand_counts > 10).sum()} ({(user_cand_counts > 10).mean():.2%})")
