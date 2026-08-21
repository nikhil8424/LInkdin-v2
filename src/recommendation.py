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

from src.config import (
    RESULTS_DIR,
    MODELS_DIR,
    RANDOM_SEED
)

def train_baseline_model():
    print("=" * 60)
    print("TRAINING XGBOOST RECOMMENDATION MODEL")
    print("=" * 60)

    X_train = pd.read_csv(RESULTS_DIR / "X_train.csv")
    X_val = pd.read_csv(RESULTS_DIR / "X_val.csv")
    X_test = pd.read_csv(RESULTS_DIR / "X_test.csv")

    y_train = pd.read_csv(RESULTS_DIR / "y_train.csv").squeeze()
    y_val = pd.read_csv(RESULTS_DIR / "y_val.csv").squeeze()
    y_test = pd.read_csv(RESULTS_DIR / "y_test.csv").squeeze()

    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val  : {X_val.shape}, y_val  : {y_val.shape}")
    print(f"X_test : {X_test.shape}, y_test : {y_test.shape}")

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    print("XGBoost model training completed.")

    # Evaluate on held-out test set
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)

    print("\n" + "=" * 60)
    print("TEST SET CLASSIFICATION METRICS")
    print("=" * 60)
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {auc:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    model_path = MODELS_DIR / "xgboost_baseline.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")

    metrics_df = pd.DataFrame({
        "metric": ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
        "value": [acc, prec, rec, f1, auc]
    })
    metrics_path = RESULTS_DIR / "baseline_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to: {metrics_path}")

    # Feature Importance
    feat_imp = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)
    feat_imp.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)

if __name__ == "__main__":
    train_baseline_model()