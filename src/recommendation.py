import joblib
import pandas as pd

from pathlib import Path

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from sklearn.metrics import ndcg_score



BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_PATH = BASE_DIR / "results"
MODELS_PATH = BASE_DIR / "models"


X_train = pd.read_csv(
    RESULTS_PATH / "X_train.csv"
)

X_test = pd.read_csv(
    RESULTS_PATH / "X_test.csv"
)

y_train = pd.read_csv(
    RESULTS_PATH / "y_train.csv"
).squeeze()

y_test = pd.read_csv(
    RESULTS_PATH / "y_test.csv"
).squeeze()


print("=" * 60)
print("PROCESSED DATA LOADED")
print("=" * 60)

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

print()


model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)


print("=" * 60)
print("TRAINING XGBOOST")
print("=" * 60)

model.fit(
    X_train,
    y_train
)

print("XGBoost training completed.")
print()



# 4. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)

y_probability = model.predict_proba(
    X_test
)[:, 1]


# 5. CLASSIFICATION METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


# 6. NDCG
# ============================================================


ndcg_5 = ndcg_score(
    [y_test.to_numpy()],
    [y_probability],
    k=5
)

ndcg_10 = ndcg_score(
    [y_test.to_numpy()],
    [y_probability],
    k=10
)


# 7. DISPLAY RESULTS
# ============================================================

print("=" * 60)
print("BASELINE MODEL RESULTS")
print("=" * 60)

print(f"Accuracy   : {accuracy:.4f}")
print(f"Precision  : {precision:.4f}")
print(f"Recall     : {recall:.4f}")
print(f"F1 Score   : {f1:.4f}")
print(f"ROC-AUC    : {roc_auc:.4f}")
print(f"NDCG@5     : {ndcg_5:.4f}")
print(f"NDCG@10    : {ndcg_10:.4f}")

print()


# 8. CLASSIFICATION REPORT
# ============================================================

print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_pred
    )
)


# 9. CONFUSION MATRIX
# ============================================================

print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)

print()


# 10. FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)

print("=" * 60)
print("TOP 15 FEATURES")
print("=" * 60)

print(
    feature_importance.head(15).to_string(
        index=False
    )
)

print()


# 11. SAVE MODEL
# ============================================================

MODELS_PATH.mkdir(
    parents=True,
    exist_ok=True
)

model_path = MODELS_PATH / "xgboost_baseline.pkl"

joblib.dump(
    model,
    model_path
)

print("=" * 60)
print("MODEL SAVED")
print("=" * 60)

print(f"Saved to: {model_path}")

print()


# 12. SAVE METRICS
# ============================================================

metrics = pd.DataFrame({
    "metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC",
        "NDCG@5",
        "NDCG@10"
    ],
    "value": [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc,
        ndcg_5,
        ndcg_10
    ]
})

metrics_path = RESULTS_PATH / "baseline_metrics.csv"

metrics.to_csv(
    metrics_path,
    index=False
)

print("Metrics saved to:")
print(metrics_path)