import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import label_binarize

from src.config import (
    RESULTS_DIR,
    MODELS_DIR,
    PROTECTED_ATTRIBUTES,
    MODEL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    RANDOM_SEED
)

def perform_direct_invariance_analysis(test_df, model, encoder):
    """
    Direct Sensitive Attribute Invariance Test:
    Tests whether directly mutating protected attributes alters model predictions.
    Since protected attributes are excluded from MODEL_FEATURES, direct variation
    yields 0.0 prediction change, confirming direct attribute invariance.
    """
    print("=" * 60)
    print("DIRECT SENSITIVE ATTRIBUTE INVARIANCE TEST")
    print("=" * 60)

    # Prepare baseline features
    X = test_df[MODEL_FEATURES].copy()
    X_cat = encoder.transform(X[CATEGORICAL_FEATURES])
    encoded_feature_names = encoder.get_feature_names_out(CATEGORICAL_FEATURES)
    X_cat_df = pd.DataFrame(X_cat, columns=encoded_feature_names)
    X_num_df = X[NUMERICAL_FEATURES].reset_index(drop=True)
    X_processed = pd.concat([X_num_df, X_cat_df], axis=1)

    original_scores = model.predict_proba(X_processed)[:, 1]
    test_df["original_prediction"] = original_scores

    all_invariance_results = []
    summary_rows = []

    for attr in PROTECTED_ATTRIBUTES:
        attr_values = test_df[attr].dropna().unique().tolist()
        attr_results = []

        for cf_val in attr_values:
            cf_df = test_df.copy()
            cf_df[attr] = cf_val

            # Features are derived from MODEL_FEATURES
            X_cf = cf_df[MODEL_FEATURES].copy()
            X_cf_cat = encoder.transform(X_cf[CATEGORICAL_FEATURES])
            X_cf_cat_df = pd.DataFrame(X_cf_cat, columns=encoded_feature_names)
            X_cf_num_df = X_cf[NUMERICAL_FEATURES].reset_index(drop=True)
            X_cf_processed = pd.concat([X_cf_num_df, X_cf_cat_df], axis=1)

            cf_scores = model.predict_proba(X_cf_processed)[:, 1]
            diff = cf_scores - original_scores
            abs_diff = np.abs(diff)
            changed_count = int(np.sum(abs_diff > 1e-9))

            res = {
                "protected_attribute": attr,
                "counterfactual_value": cf_val,
                "sample_count": len(test_df),
                "mean_original_score": float(original_scores.mean()),
                "mean_counterfactual_score": float(cf_scores.mean()),
                "mean_score_difference": float(diff.mean()),
                "mean_absolute_difference": float(abs_diff.mean()),
                "maximum_absolute_difference": float(abs_diff.max()),
                "changed_predictions": changed_count,
                "percentage_predictions_changed": float(changed_count / len(test_df) * 100)
            }
            attr_results.append(res)
            all_invariance_results.append(res)

        attr_df = pd.DataFrame(attr_results)
        attr_df.to_csv(RESULTS_DIR / f"counterfactual_{attr}.csv", index=False)

        summary_rows.append({
            "protected_attribute": attr,
            "distinct_values_tested": len(attr_values),
            "max_absolute_score_change": float(attr_df["maximum_absolute_difference"].max()),
            "changed_predictions_count": int(attr_df["changed_predictions"].sum()),
            "direct_invariance_status": "Passed (Strictly Invariant)" if attr_df["maximum_absolute_difference"].max() < 1e-9 else "Failed"
        })

    invariance_df = pd.DataFrame(all_invariance_results)
    invariance_df.to_csv(RESULTS_DIR / "counterfactual_fairness.csv", index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULTS_DIR / "counterfactual_fairness_summary.csv", index=False)
    print("Direct invariance summary:")
    print(summary_df.to_string(index=False))

def run_proxy_detection():
    """
    Proxy Detection:
    Trains classifiers predicting each protected attribute from non-protected model features.
    Quantifies potential indirect/proxy leakage in features like professional_field,
    experience_years, education, network_size, etc.
    """
    print("\n" + "=" * 60)
    print("PROXY DETECTION & FEATURE PREDICTIVE POWER ANALYSIS")
    print("=" * 60)

    X_train = pd.read_csv(RESULTS_DIR / "X_train.csv")
    X_val = pd.read_csv(RESULTS_DIR / "X_val.csv")
    X_test = pd.read_csv(RESULTS_DIR / "X_test.csv")

    prot_train = pd.read_csv(RESULTS_DIR / "protected_train.csv")
    prot_val = pd.read_csv(RESULTS_DIR / "protected_val.csv")
    prot_test = pd.read_csv(RESULTS_DIR / "protected_test.csv")

    prediction_results = []
    feature_importances_all = []
    proxy_feature_analysis = []

    for attr in PROTECTED_ATTRIBUTES:
        y_tr = prot_train[attr]
        y_v = prot_val[attr]
        y_te = prot_test[attr]

        # Train a proxy detection classifier
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
        clf.fit(X_train, y_tr)

        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)

        classes = clf.classes_
        bal_acc = balanced_accuracy_score(y_te, y_pred)
        f1_macro = f1_score(y_te, y_pred, average="macro", zero_division=0)

        # Multi-class / binary ROC-AUC
        if len(classes) == 2:
            auc = roc_auc_score(y_te, y_proba[:, 1])
        else:
            y_te_bin = label_binarize(y_te, classes=classes)
            auc = roc_auc_score(y_te_bin, y_proba, multi_class="ovr", average="macro")

        # Proxy strength assessment
        if auc >= 0.70:
            proxy_risk = "High Proxy Signal"
        elif auc >= 0.58:
            proxy_risk = "Moderate Proxy Signal"
        else:
            proxy_risk = "Low / Noise Proxy Signal"

        prediction_results.append({
            "protected_attribute": attr,
            "num_classes": len(classes),
            "ROC_AUC": auc,
            "balanced_accuracy": bal_acc,
            "macro_F1": f1_macro,
            "proxy_risk_level": proxy_risk
        })

        # Feature importance for this protected attribute
        for feat, imp in zip(X_train.columns, clf.feature_importances_):
            feature_importances_all.append({
                "protected_attribute": attr,
                "feature": feat,
                "importance": imp
            })

        # Top 3 most predictive features for this attribute
        feat_df = pd.DataFrame({"feature": X_train.columns, "importance": clf.feature_importances_})
        top_feats = feat_df.sort_values("importance", ascending=False).head(3)
        for rank, (_, row) in enumerate(top_feats.iterrows(), 1):
            proxy_feature_analysis.append({
                "protected_attribute": attr,
                "predictive_rank": rank,
                "feature": row["feature"],
                "importance_score": row["importance"],
                "attribute_ROC_AUC": auc,
                "proxy_risk_assessment": proxy_risk
            })

    pred_df = pd.DataFrame(prediction_results)
    pred_df.to_csv(RESULTS_DIR / "proxy_attribute_prediction.csv", index=False)

    imp_df = pd.DataFrame(feature_importances_all).sort_values(
        ["protected_attribute", "importance"], ascending=[True, False]
    )
    imp_df.to_csv(RESULTS_DIR / "proxy_feature_importance.csv", index=False)

    analysis_df = pd.DataFrame(proxy_feature_analysis)
    analysis_df.to_csv(RESULTS_DIR / "proxy_feature_analysis.csv", index=False)

    print("Proxy Attribute Prediction Summary:")
    print(pred_df.to_string(index=False))
    print("\nTop Proxy Features by Protected Attribute:")
    print(analysis_df.to_string(index=False))

def run_counterfactual_fairness():
    test_split_path = RESULTS_DIR / "test_split.csv"
    encoder_path = RESULTS_DIR / "categorical_encoder.pkl"
    model_path = MODELS_DIR / "xgboost_baseline.pkl"

    test_df = pd.read_csv(test_split_path)
    encoder = joblib.load(encoder_path)
    model = joblib.load(model_path)

    perform_direct_invariance_analysis(test_df, model, encoder)
    run_proxy_detection()
    print("\nCounterfactual & proxy diagnostics completed successfully.\n")

if __name__ == "__main__":
    run_counterfactual_fairness()