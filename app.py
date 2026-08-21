import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Fair LinkedIn Recommendation System",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, modern look
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #3B82F6;
    }
    .badge-pass {
        background-color: #DCFCE7;
        color: #166534;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# PROJECT PATHS & HELPERS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
RESULTS = BASE_DIR / "results"
DATA = BASE_DIR / "data"
MODELS = BASE_DIR / "models"

def load_csv(filename):
    path = RESULTS / filename
    if path.exists():
        return pd.read_csv(path)
    return None

def show_missing_file(filename):
    st.warning(f"Result artifact not found: `{filename}`. Please execute the pipeline to generate it.")

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("⚖️ FairRec Navigator")
st.sidebar.caption("XGBoost • SHAP • Fairness • NSGA-II")

page = st.sidebar.radio(
    "Select Stage",
    [
        "Overview & Methodology",
        "Dataset & Splitting",
        "Baseline Model (XGBoost)",
        "Top-K Recommendations",
        "SHAP Explainability",
        "Score vs. Exposure Fairness",
        "Intersectional Fairness",
        "Counterfactual & Proxy Analysis",
        "Fairness Mitigation & Reranking",
        "Fairness Strength Experiment",
        "NSGA-II Multi-Objective Optimization",
        "Ablation Studies & Final Comparison"
    ]
)

st.sidebar.divider()
st.sidebar.markdown("**Reproducibility Parameters:**")
st.sidebar.caption("• Random Seed: `42`")
st.sidebar.caption("• Split: 70% Train / 15% Val / 15% Test")
st.sidebar.caption("• Bootstrap Resamples: `1,000`")
st.sidebar.caption("• Min Intersection Size: `30`")

# ============================================================
# 1. OVERVIEW & METHODOLOGY
# ============================================================

if page == "Overview & Methodology":
    st.markdown('<div class="main-header">⚖️ Fair LinkedIn Recommendation System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Methodologically Repaired Recommendation, Explainability, Exposure Fairness, and NSGA-II Pipeline</div>', unsafe_allow_html=True)

    st.markdown("""
    This project implements an end-to-end fairness-aware recommendation pipeline for professional LinkedIn-style interactions.
    The system addresses key trade-offs between **recommendation utility (NDCG@10, Precision@10, Recall@10)** and **multi-group demographic exposure parity**.
    """)

    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Dataset", "30,000 Rows", "Interactions")
    col2.metric("Split Scheme", "70 / 15 / 15", "Train / Val / Test")
    col3.metric("Candidate Evaluation", "Full Pools", "No Truncation")
    col4.metric("Optimization", "NSGA-II", "Pareto-Optimal")

    st.divider()

    st.subheader("Pipeline Architecture")
    st.markdown("""
    ```mermaid
    graph TD
        A[30,000 Synthetic Interactions] --> B[Stratified 70/15/15 Splitting]
        B --> C[Train Split: 21,000 rows]
        B --> D[Val Split: 4,500 rows]
        B --> E[Test Split: 4,500 rows]
        C --> F[XGBoost Classifier Training]
        D --> F
        F --> G[Held-Out Test Scoring]
        E --> G
        G --> H[Full Candidate Pool Ranking]
        H --> I[SHAP Feature Explainability]
        H --> J[Position-Weighted Exposure Fairness]
        H --> K[Intersectional 3-Way Fairness]
        H --> L[Proxy Attribute ML Classifiers]
        H --> M[Fairness Reranker & Quota Baseline]
        M --> N[Fairness Strength Sweep: 0.0 to 1.0]
        N --> O[Genuine NSGA-II Genetic Optimization]
        O --> P[Pareto Front & Multi-Criteria Decision]
        P --> Q[1000 User-Level Bootstrap CIs & Ablations]
    ```
    """)

    st.info(r"""
    **Methodological Audit Highlights:**
    - **No Evaluation Leakage:** Recommendation quality and fairness metrics are evaluated strictly on the 4,500-row held-out test split.
    - **Dynamic Ranking Changes:** Reranking alters candidate positions dynamically, providing genuine trade-off curves.
    - **True Exposure vs. Score Fairness:** Clearly isolates model score distributions from position-weighted exposure $1/\log_2(r+1)$.
    - **Proxy Detection:** Exposing subtle correlations between model features and sensitive attributes.
    """)

# ============================================================
# 2. DATASET & SPLITTING
# ============================================================

elif page == "Dataset & Splitting":
    st.header("Dataset & Clean Train/Val/Test Splitting")
    
    test_split = load_csv("test_split.csv")
    train_split = load_csv("train_split.csv")
    val_split = load_csv("val_split.csv")

    if test_split is not None and train_split is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Training Records", f"{len(train_split):,}", "70.0%")
        c2.metric("Validation Records", f"{len(val_split):,}", "15.0%")
        c3.metric("Testing Records", f"{len(test_split):,}", "15.0%")
        c4.metric("Test Users", f"{test_split['user_id'].nunique():,}", "Unique")

        st.subheader("Held-Out Test Split Preview")
        st.dataframe(test_split.head(15))

        st.subheader("Demographic Distributions (Test Set)")
        t1, t2, t3 = st.tabs(["Gender", "Age Group", "Location"])
        with t1:
            st.bar_chart(test_split["gender"].value_counts())
        with t2:
            st.bar_chart(test_split["age_group"].value_counts())
        with t3:
            st.bar_chart(test_split["location"].value_counts())
    else:
        show_missing_file("test_split.csv")

# ============================================================
# 3. BASELINE MODEL
# ============================================================

elif page == "Baseline Model (XGBoost)":
    st.header("Baseline XGBoost Recommendation Model")
    
    metrics = load_csv("baseline_metrics.csv")
    feat_imp = load_csv("feature_importance.csv")

    if metrics is not None:
        st.subheader("Held-Out Test Classification Performance")
        m_dict = dict(zip(metrics["metric"], metrics["value"]))
        cols = st.columns(len(m_dict))
        for col, (k, v) in zip(cols, m_dict.items()):
            col.metric(k, f"{v:.4f}")

        st.dataframe(metrics)

    if feat_imp is not None:
        st.subheader("Model Feature Importance (Gini Gain)")
        st.bar_chart(feat_imp.set_index("feature")["importance"].head(12))

# ============================================================
# 4. TOP-K RECOMMENDATIONS
# ============================================================

elif page == "Top-K Recommendations":
    st.header("Top-K Recommendation Performance (Full-Pool Evaluation)")

    metrics = load_csv("top_k_metrics.csv")
    notes = load_csv("top_k_evaluation_notes.csv")
    recs = load_csv("top_10_recommendations.csv")
    user_eval = load_csv("top_k_user_evaluation.csv")

    if metrics is not None:
        st.subheader("Candidate Ranking Metrics (Test Set)")
        cols = st.columns(len(metrics))
        for col, row in zip(cols, metrics.itertuples()):
            col.metric(row.metric, f"{row.value:.4f}")

    if notes is not None:
        st.subheader("Candidate Pool Distribution Diagnostics")
        st.dataframe(notes)

    if user_eval is not None:
        st.subheader("User Candidate Pool Distribution")
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.hist(user_eval["candidate_count"], bins=range(1, 25), color="#3B82F6", edgecolor="black", alpha=0.7)
        ax.set_xlabel("Candidate Count per User")
        ax.set_ylabel("User Count")
        ax.set_title("Distribution of Candidate Pool Sizes in Test Set")
        st.pyplot(fig)
        plt.close()

    if recs is not None:
        st.subheader("Explore User Top-10 Recommendations")
        users = recs["user_id"].unique()
        sel_user = st.selectbox("Select User ID", users)
        user_recs = recs[recs["user_id"] == sel_user]
        st.dataframe(user_recs)

# ============================================================
# 5. SHAP EXPLAINABILITY
# ============================================================

elif page == "SHAP Explainability":
    st.header("SHAP Explainability Analysis")

    importance = load_csv("shap_feature_importance.csv")
    if importance is not None:
        st.subheader("Global Mean Absolute SHAP Importance")
        st.dataframe(importance.head(15))

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Global Summary Bar Plot")
        bar_p = RESULTS / "shap_summary_bar.png"
        if bar_p.exists():
            st.image(str(bar_p))
    with c2:
        st.subheader("SHAP Beeswarm Feature Impact")
        bee_p = RESULTS / "shap_summary_beeswarm.png"
        if bee_p.exists():
            st.image(str(bee_p))

    st.subheader("Local Explanation (Waterfall)")
    water_p = RESULTS / "shap_waterfall_sample.png"
    if water_p.exists():
        st.image(str(water_p))

# ============================================================
# 6. SCORE VS EXPOSURE FAIRNESS
# ============================================================

elif page == "Score vs. Exposure Fairness":
    st.header("Score Fairness vs. Recommendation Exposure Fairness")

    st.markdown(r"""
    **Crucial Distinction:**
    - **Score-Based DI / SPD:** Averages model predicted probabilities across groups.
    - **Recommendation Selection DI / SPD:** Measures actual entry rate into Top-K lists.
    - **Position-Weighted Exposure DI / SPD:** Applies logarithmic discount $w(r) = 1/\log_2(r+1)$ to ranked positions.
    - **Relevance-Aware Exposure:** Compares actual exposure to expected utility.
    """)

    summary = load_csv("fairness_summary.csv")
    exposure = load_csv("exposure_fairness.csv")

    if summary is not None:
        st.subheader("Comprehensive Fairness Metric Summary")
        st.dataframe(summary)

    if exposure is not None:
        st.subheader("Group-Level Exposure Disparities")
        st.dataframe(exposure)

        t1, t2, t3 = st.tabs(["Gender Exposure", "Age Exposure", "Location Exposure"])
        with t1:
            g_exp = exposure[exposure["attribute"] == "gender"]
            st.bar_chart(g_exp.set_index("group")["mean_position_exposure"])
        with t2:
            a_exp = exposure[exposure["attribute"] == "age_group"]
            st.bar_chart(a_exp.set_index("group")["mean_position_exposure"])
        with t3:
            l_exp = exposure[exposure["attribute"] == "location"]
            st.bar_chart(l_exp.set_index("group")["mean_position_exposure"])

# ============================================================
# 7. INTERSECTIONAL FAIRNESS
# ============================================================

elif page == "Intersectional Fairness":
    st.header("Intersectional Multi-Attribute Fairness Analysis")

    st.markdown(r"""
    Single-attribute metrics can mask severe disparities in compounded subgroups.
    We evaluate pairwise and 3-way intersections, flagging groups with $N < 30$ as `statistically_unstable`.
    """)

    summary = load_csv("intersectional_fairness_summary.csv")
    int_3way = load_csv("intersectional_gender_age_location.csv")

    if summary is not None:
        st.subheader("Worst-Case Intersectional Gaps")
        st.dataframe(summary)

    if int_3way is not None:
        st.subheader("3-Way Subgroups (Gender x Age Group x Location)")
        st.dataframe(int_3way)

        st.subheader("Subgroup Exposure DI Distribution")
        st.bar_chart(int_3way.set_index("intersection")["exposure_DI"])

# ============================================================
# 8. COUNTERFACTUAL & PROXY ANALYSIS
# ============================================================

elif page == "Counterfactual & Proxy Analysis":
    st.header("Direct Attribute Invariance & Proxy Feature Analysis")

    st.subheader("1. Direct Sensitive Attribute Invariance")
    st.markdown("""
    Because protected attributes (`gender`, `age_group`, `location`) are excluded from `MODEL_FEATURES`,
    directly mutating them on the input feature vector produces **0.0 score difference**.
    """)
    cf_sum = load_csv("counterfactual_fairness_summary.csv")
    if cf_sum is not None:
        st.dataframe(cf_sum)

    st.divider()

    st.subheader("2. Machine Learning Proxy Detection")
    st.markdown("""
    Although protected attributes are not direct features, other features may act as **statistical proxies**.
    Classifiers are trained to predict protected attributes using only non-protected features.
    """)

    proxy_pred = load_csv("proxy_attribute_prediction.csv")
    proxy_feats = load_csv("proxy_feature_analysis.csv")

    if proxy_pred is not None:
        st.dataframe(proxy_pred)

    p_plot = RESULTS / "proxy_attribute_auc.png"
    if p_plot.exists():
        st.image(str(p_plot))

    if proxy_feats is not None:
        st.subheader("Top Proxy Signals by Attribute")
        st.dataframe(proxy_feats)

# ============================================================
# 9. FAIRNESS MITIGATION & RERANKING
# ============================================================

elif page == "Fairness Mitigation & Reranking":
    st.header("Fairness-Aware Reranking & Ranking Diagnostics")

    q_comp = load_csv("fairness_quality_comparison.csv")
    f_comp = load_csv("fairness_metric_comparison.csv")
    diag = load_csv("reranking_diagnostics.csv")
    samples = load_csv("reranking_sample_users.csv")

    if q_comp is not None and f_comp is not None:
        st.subheader("Quality & Fairness Trade-Off Comparison")
        st.dataframe(q_comp)
        st.dataframe(f_comp)

    if diag is not None:
        st.subheader("Ranking Change Diagnostics (Baseline vs. Mitigation)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Users with Top-10 Changed", f"{diag['top_k_changed'].mean()*100:.2f}%")
        c2.metric("Users with Order Changed", f"{diag['ordering_changed'].mean()*100:.2f}%")
        c3.metric("Average Top-10 Overlap", f"{diag['top_k_overlap'].mean():.4f}")
        c4.metric("Average Jaccard Similarity", f"{diag['top_k_jaccard'].mean():.4f}")

    if samples is not None:
        st.subheader("Sample Users with Reranked Items")
        st.dataframe(samples)

# ============================================================
# 10. FAIRNESS STRENGTH EXPERIMENT
# ============================================================

elif page == "Fairness Strength Experiment":
    st.header(r"Fairness Strength Sweep ($\lambda \in [0.0, 1.0]$)")

    sweep_df = load_csv("fairness_strength_results.csv")
    if sweep_df is not None:
        st.dataframe(sweep_df)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("NDCG@10 vs. Fairness Strength")
            ndcg_p = RESULTS / "fairness_strength_ndcg.png"
            if ndcg_p.exists():
                st.image(str(ndcg_p))
        with c2:
            st.subheader("Exposure DI vs. Fairness Strength")
            di_p = RESULTS / "fairness_strength_exposure_di.png"
            if di_p.exists():
                st.image(str(di_p))

        c3, c4 = st.columns(2)
        with c3:
            st.subheader("Reranking Dynamics (% Changed & Overlap)")
            dyn_p = RESULTS / "fairness_strength_ranking_changes.png"
            if dyn_p.exists():
                st.image(str(dyn_p))
        with c4:
            st.subheader("Worst-Case Intersectional DI vs. Strength")
            int_p = RESULTS / "fairness_strength_worst_intersectional_di.png"
            if int_p.exists():
                st.image(str(int_p))

# ============================================================
# 11. NSGA-II MULTI-OBJECTIVE OPTIMIZATION
# ============================================================

elif page == "NSGA-II Multi-Objective Optimization":
    st.header("Genuine NSGA-II Multi-Objective Optimization")

    pareto_df = load_csv("nsga2_pareto_front.csv")
    selected_df = load_csv("nsga2_selected_solution.csv")
    all_df = load_csv("nsga2_all_solutions.csv")

    if selected_df is not None:
        st.subheader("Selected Multi-Criteria Pareto Solution")
        st.dataframe(selected_df)

    if pareto_df is not None:
        st.subheader("Discovered Pareto-Optimal Front")
        st.dataframe(pareto_df)

    p_plot = RESULTS / "nsga2_pareto_front.png"
    if p_plot.exists():
        st.image(str(p_plot))

# ============================================================
# 12. ABLATION STUDIES & FINAL COMPARISON
# ============================================================

elif page == "Ablation Studies & Final Comparison":
    st.header("🏆 Final Model Comparison, Ablations & Confidence Intervals")

    ablation = load_csv("ablation_comparison.csv")
    ci_df = load_csv("bootstrap_confidence_intervals.csv")
    quality = load_csv("final_quality_comparison.csv")
    fairness = load_csv("final_fairness_comparison.csv")
    imp = load_csv("final_fairness_improvement.csv")

    if ablation is not None:
        st.subheader("1. Comprehensive 5-Way Ablation Study")
        st.dataframe(ablation)

    if ci_df is not None:
        st.subheader("2. 95% User-Level Bootstrap Confidence Intervals (1,000 Resamples)")
        st.dataframe(ci_df)

    if quality is not None and fairness is not None:
        st.subheader("3. Final Method Comparison (Baseline vs. Quota vs. NSGA-II)")
        st.dataframe(quality)
        st.dataframe(fairness)

    if imp is not None:
        st.subheader("4. Measured Fairness Gains")
        st.dataframe(imp)

    st.success("""
    **Conclusion:**
    The pipeline demonstrates statistically verified, reproducible fairness gains in exposure and selection parity
    while accurately quantifying the empirical trade-off cost on recommendation ranking metrics.
    """)