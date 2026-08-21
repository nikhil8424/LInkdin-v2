import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

from pathlib import Path



BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_PATH = BASE_DIR / "results"
MODELS_PATH = BASE_DIR / "models"

MODEL_PATH = (
    MODELS_PATH
    / "xgboost_baseline.pkl"
)

X_TEST_PATH = (
    RESULTS_PATH
    / "X_test.csv"
)


# 2. CHECK REQUIRED FILES
# ============================================================

print("=" * 60)
print("CHECKING REQUIRED FILES")
print("=" * 60)

required_files = [
    MODEL_PATH,
    X_TEST_PATH
]

for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{file_path}"
        )

    print("✓", file_path)

print()


# 3. LOAD MODEL
# ============================================================

print("=" * 60)
print("LOADING XGBOOST MODEL")
print("=" * 60)

model = joblib.load(
    MODEL_PATH
)

print("Model loaded successfully.")

print()


# 4. LOAD TEST DATA
# ============================================================

print("=" * 60)
print("LOADING TEST DATA")
print("=" * 60)

X_test = pd.read_csv(
    X_TEST_PATH
)

print(
    "X_test shape:",
    X_test.shape
)

print()


# 5. CREATE SHAP EXPLAINER
# ============================================================

print("=" * 60)
print("CREATING SHAP EXPLAINER")
print("=" * 60)

explainer = shap.TreeExplainer(
    model
)

print("SHAP TreeExplainer created.")

print()


# 6. CALCULATE SHAP VALUES
# ============================================================

print("=" * 60)
print("CALCULATING SHAP VALUES")
print("=" * 60)

shap_values = explainer.shap_values(
    X_test
)

print(
    "SHAP values calculated."
)

print()


# 7. CHECK SHAP OUTPUT
# ============================================================

print("=" * 60)
print("SHAP OUTPUT")
print("=" * 60)

print(
    "SHAP values shape:",
    shap_values.shape
)

print(
    "Test data shape:",
    X_test.shape
)

print()


# 8. GLOBAL FEATURE IMPORTANCE
# ============================================================

print("=" * 60)
print("GLOBAL SHAP FEATURE IMPORTANCE")
print("=" * 60)

mean_abs_shap = (
    abs(shap_values)
    .mean(axis=0)
)


shap_importance = pd.DataFrame({

    "feature":
        X_test.columns,

    "mean_absolute_shap":
        mean_abs_shap

})


shap_importance = (
    shap_importance
    .sort_values(
        "mean_absolute_shap",
        ascending=False
    )
    .reset_index(drop=True)
)


print(
    shap_importance
    .head(20)
    .to_string(index=False)
)

print()


# 9. SAVE SHAP FEATURE IMPORTANCE
# ============================================================

shap_importance_path = (
    RESULTS_PATH
    / "shap_feature_importance.csv"
)

shap_importance.to_csv(
    shap_importance_path,
    index=False
)

print(
    "SHAP feature importance saved:"
)

print(
    shap_importance_path
)

print()


# 10. GLOBAL SHAP SUMMARY BAR PLOT
# ============================================================

print("=" * 60)
print("CREATING SHAP SUMMARY PLOT")
print("=" * 60)

plt.figure()

shap.summary_plot(
    shap_values,
    X_test,
    plot_type="bar",
    show=False
)

plt.tight_layout()

summary_bar_path = (
    RESULTS_PATH
    / "shap_summary_bar.png"
)

plt.savefig(
    summary_bar_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved:"
)

print(
    summary_bar_path
)

print()


# 11. SHAP BEESWARM PLOT
# ============================================================

print("=" * 60)
print("CREATING SHAP BEESWARM PLOT")
print("=" * 60)

plt.figure()

shap.summary_plot(
    shap_values,
    X_test,
    show=False
)

plt.tight_layout()

beeswarm_path = (
    RESULTS_PATH
    / "shap_summary_beeswarm.png"
)

plt.savefig(
    beeswarm_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved:"
)

print(
    beeswarm_path
)

print()


# 12. LOCAL EXPLANATION
# ============================================================

print("=" * 60)
print("CREATING LOCAL EXPLANATION")
print("=" * 60)

sample_index = 0

sample_row = X_test.iloc[
    sample_index
]

sample_shap_values = (
    shap_values[sample_index]
)


local_explanation = pd.DataFrame({

    "feature":
        X_test.columns,

    "feature_value":
        sample_row.values,

    "shap_value":
        sample_shap_values

})


local_explanation[
    "absolute_shap"
] = (
    local_explanation[
        "shap_value"
    ].abs()
)


local_explanation = (
    local_explanation
    .sort_values(
        "absolute_shap",
        ascending=False
    )
    .reset_index(drop=True)
)


print(
    "Sample test record:",
    sample_index
)

print()

print(
    local_explanation
    .head(15)
    .to_string(index=False)
)

print()


# 13. SAVE LOCAL EXPLANATION
# ============================================================

local_path = (
    RESULTS_PATH
    / "shap_local_explanation.csv"
)

local_explanation.to_csv(
    local_path,
    index=False
)

print(
    "Local explanation saved:"
)

print(
    local_path
)

print()


# 14. LOCAL WATERFALL PLOT
# ============================================================

print("=" * 60)
print("CREATING SHAP WATERFALL PLOT")
print("=" * 60)

explanation = shap.Explanation(

    values=sample_shap_values,

    base_values=explainer.expected_value,

    data=sample_row.values,

    feature_names=X_test.columns.tolist()
)


plt.figure()

shap.plots.waterfall(
    explanation,
    max_display=15,
    show=False
)

plt.tight_layout()

waterfall_path = (
    RESULTS_PATH
    / "shap_waterfall_sample.png"
)

plt.savefig(
    waterfall_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved:"
)

print(
    waterfall_path
)

print()


# 15. TOP FEATURES
# ============================================================

print("=" * 60)
print("TOP 10 SHAP FEATURES")
print("=" * 60)

print(
    shap_importance
    .head(10)
    .to_string(index=False)
)

print()



# 16. FINAL OUTPUT
# ============================================================

print("=" * 60)
print("SHAP ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 60)

print()

print("Generated files:")

print(
    "1. shap_feature_importance.csv"
)

print(
    "2. shap_summary_bar.png"
)

print(
    "3. shap_summary_beeswarm.png"
)

print(
    "4. shap_local_explanation.csv"
)

print(
    "5. shap_waterfall_sample.png"
)

print()

print(
    "Results folder:"
)

print(
    RESULTS_PATH
)