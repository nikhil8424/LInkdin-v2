import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import RESULTS_DIR, MODELS_DIR

def run_shap_analysis():
    print("=" * 60)
    print("SHAP EXPLAINABILITY ANALYSIS")
    print("=" * 60)

    model_path = MODELS_DIR / "xgboost_baseline.pkl"
    X_test_path = RESULTS_DIR / "X_test.csv"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not X_test_path.exists():
        raise FileNotFoundError(f"X_test not found: {X_test_path}")

    model = joblib.load(model_path)
    X_test = pd.read_csv(X_test_path)
    print(f"Loaded model and X_test with shape: {X_test.shape}")

    # TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    print("SHAP values calculated successfully.")

    # Global feature importance
    mean_abs_shap = abs(shap_values).mean(axis=0)
    shap_importance = pd.DataFrame({
        "feature": X_test.columns,
        "mean_absolute_shap": mean_abs_shap
    }).sort_values("mean_absolute_shap", ascending=False).reset_index(drop=True)

    shap_importance_path = RESULTS_DIR / "shap_feature_importance.csv"
    shap_importance.to_csv(shap_importance_path, index=False)
    print(f"SHAP feature importance saved to: {shap_importance_path}")

    # Top features summary
    print("\nTop 10 Features by SHAP Importance:")
    print(shap_importance.head(10).to_string(index=False))

    # Summary Bar Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title("SHAP Global Feature Importance (Test Set)", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "shap_summary_bar.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Beeswarm Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Feature Impact Distribution (Beeswarm)", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "shap_summary_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Sample local explanation
    sample_idx = 0
    sample_row = X_test.iloc[[sample_idx]]
    sample_shap = shap_values[sample_idx]
    
    local_df = pd.DataFrame({
        "feature": X_test.columns,
        "feature_value": sample_row.values[0],
        "shap_value": sample_shap
    }).sort_values(by="shap_value", key=abs, ascending=False).reset_index(drop=True)
    local_df.to_csv(RESULTS_DIR / "shap_local_explanation.csv", index=False)

    # Waterfall plot
    plt.figure(figsize=(10, 6))
    explanation = shap.Explanation(
        values=sample_shap,
        base_values=explainer.expected_value,
        data=sample_row.values[0],
        feature_names=X_test.columns.tolist()
    )
    shap.plots.waterfall(explanation, max_display=10, show=False)
    plt.title(f"SHAP Waterfall Explanation (Sample Record {sample_idx})", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "shap_waterfall_sample.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\nSHAP analysis and visualization completed.\n")

if __name__ == "__main__":
    run_shap_analysis()