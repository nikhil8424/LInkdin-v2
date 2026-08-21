import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Fair LinkedIn Recommendation System",
    page_icon="⚖️",
    layout="wide"
)


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

RESULTS = BASE_DIR / "results"
DATA = BASE_DIR / "data"
MODELS = BASE_DIR / "models"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_csv(filename):

    path = RESULTS / filename

    if path.exists():
        return pd.read_csv(path)

    return None


def show_missing_file(filename):

    st.warning(
        f"Result file not found: `{filename}`"
    )


# ============================================================
# TITLE
# ============================================================

st.title(
    "⚖️ Fair LinkedIn Recommendation System"
)

st.markdown(
    """
    ### Bias-Aware Recommendation using XGBoost,
    Explainable AI, Fairness Analysis and Multi-Objective Optimization
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Overview",
        "Dataset",
        "Baseline Model",
        "Top-K Recommendations",
        "SHAP Explainability",
        "Group Fairness",
        "Intersectional Fairness",
        "Counterfactual Fairness",
        "Fairness Mitigation",
        "Fairness Strength",
        "NSGA-II Optimization",
        "Final Comparison"
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.header("Project Overview")

    st.markdown(
        """
        This system develops a recommendation model for
        LinkedIn-style content while explicitly evaluating
        recommendation quality and fairness.

        The complete pipeline consists of:

        **Dataset → Preprocessing → XGBoost → Top-K Recommendation
        → SHAP → Fairness Analysis → Counterfactual Analysis
        → Fairness Mitigation → Fairness Strength Experiment
        → Multi-Objective Optimization**
        """
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Dataset Records",
        "30,000"
    )

    col2.metric(
        "Users",
        "3,000"
    )

    col3.metric(
        "Features",
        "22"
    )

    col4.metric(
        "Top-K",
        "10"
    )

    st.divider()

    st.subheader("Project Pipeline")

    pipeline = [
        "1. Dataset Validation",
        "2. Data Preprocessing",
        "3. XGBoost Baseline",
        "4. Top-K Recommendation",
        "5. SHAP Explainability",
        "6. Group Fairness",
        "7. Intersectional Fairness",
        "8. Counterfactual Fairness",
        "9. Fairness Mitigation",
        "10. Fairness Strength Experiment",
        "11. NSGA-II / Pareto Optimization",
        "12. Final Comparison"
    ]

    for item in pipeline:
        st.write("✓", item)


# ============================================================
# DATASET
# ============================================================

elif page == "Dataset":

    st.header("Dataset Analysis")

    dataset_path = (
        DATA /
        "synthetic_linkedin_dataset_30000.csv"
    )

    if dataset_path.exists():

        df = pd.read_csv(
            dataset_path
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Rows",
            f"{df.shape[0]:,}"
        )

        col2.metric(
            "Columns",
            df.shape[1]
        )

        col3.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )

        col4.metric(
            "Duplicate Rows",
            int(df.duplicated().sum())
        )

        st.subheader(
            "Dataset Preview"
        )

        st.dataframe(
            df.head(20),
            use_container_width=True
        )

        st.subheader(
            "Target Distribution"
        )

        target_counts = (
            df["interaction"]
            .value_counts()
            .sort_index()
        )

        st.bar_chart(
            target_counts
        )

        st.subheader(
            "Gender Distribution"
        )

        gender_counts = (
            df["gender"]
            .value_counts()
        )

        st.bar_chart(
            gender_counts
        )

        st.subheader(
            "Age Group Distribution"
        )

        age_counts = (
            df["age_group"]
            .value_counts()
        )

        st.bar_chart(
            age_counts
        )

    else:

        st.error(
            "Dataset not found."
        )


# ============================================================
# BASELINE MODEL
# ============================================================

elif page == "Baseline Model":

    st.header(
        "Baseline XGBoost Model"
    )

    metrics = load_csv(
        "baseline_metrics.csv"
    )

    if metrics is not None:

        st.subheader(
            "Classification Performance"
        )

        metric_dict = dict(
            zip(
                metrics["metric"],
                metrics["value"]
            )
        )

        cols = st.columns(
            len(metric_dict)
        )

        for col, (name, value) in zip(
            cols,
            metric_dict.items()
        ):

            col.metric(
                name,
                f"{value:.4f}"
            )

        st.dataframe(
            metrics,
            use_container_width=True
        )

    else:

        show_missing_file(
            "baseline_metrics.csv"
        )


# ============================================================
# TOP-K RECOMMENDATIONS
# ============================================================

elif page == "Top-K Recommendations":

    st.header(
        "Top-K Recommendation Performance"
    )

    metrics = load_csv(
        "top_k_metrics.csv"
    )

    recommendations = load_csv(
        "top_10_recommendations.csv"
    )

    if metrics is not None:

        st.subheader(
            "Recommendation Metrics"
        )

        cols = st.columns(
            len(metrics)
        )

        for col, row in zip(
            cols,
            metrics.itertuples()
        ):

            col.metric(
                row.metric,
                f"{row.value:.4f}"
            )

        st.dataframe(
            metrics,
            use_container_width=True
        )

    if recommendations is not None:

        st.subheader(
            "Sample Recommendations"
        )

        users = (
            recommendations["user_id"]
            .unique()
        )

        selected_user = st.selectbox(
            "Select User",
            users
        )

        user_recommendations = (
            recommendations[
                recommendations["user_id"]
                == selected_user
            ]
        )

        st.dataframe(
            user_recommendations,
            use_container_width=True
        )

    st.info(
        """
        Recall@10 is not used because the dataset
        contains exactly 10 candidate interaction records
        per user. Selecting Top-10 therefore includes the
        entire candidate set.
        """
    )


# ============================================================
# SHAP
# ============================================================

elif page == "SHAP Explainability":

    st.header(
        "SHAP Explainability"
    )

    importance = load_csv(
        "shap_feature_importance.csv"
    )

    if importance is not None:

        st.subheader(
            "Global Feature Importance"
        )

        top_features = (
            importance
            .sort_values(
                "mean_absolute_shap",
                ascending=True
            )
            .tail(15)
        )

        st.bar_chart(
            top_features.set_index(
                "feature"
            )[
                "mean_absolute_shap"
            ]
        )

        st.dataframe(
            importance.head(20),
            use_container_width=True
        )

    st.subheader(
        "SHAP Summary Plot"
    )

    shap_summary = (
        RESULTS /
        "shap_summary_bar.png"
    )

    if shap_summary.exists():

        st.image(
            str(shap_summary),
            use_container_width=True
        )

    st.subheader(
        "SHAP Beeswarm Plot"
    )

    shap_beeswarm = (
        RESULTS /
        "shap_summary_beeswarm.png"
    )

    if shap_beeswarm.exists():

        st.image(
            str(shap_beeswarm),
            use_container_width=True
        )

    st.subheader(
        "Local Explanation"
    )

    waterfall = (
        RESULTS /
        "shap_waterfall_sample.png"
    )

    if waterfall.exists():

        st.image(
            str(waterfall),
            use_container_width=True
        )


# ============================================================
# GROUP FAIRNESS
# ============================================================

elif page == "Group Fairness":

    st.header(
        "Group Fairness Analysis"
    )

    tabs = st.tabs(
        [
            "Gender",
            "Age Group",
            "Location",
            "Summary"
        ]
    )

    files = [
        "fairness_gender.csv",
        "fairness_age_group.csv",
        "fairness_location.csv"
    ]

    for tab, filename in zip(
        tabs[:3],
        files
    ):

        with tab:

            data = load_csv(
                filename
            )

            if data is not None:

                st.dataframe(
                    data,
                    use_container_width=True
                )

    with tabs[3]:

        summary = load_csv(
            "fairness_summary.csv"
        )

        if summary is not None:

            st.dataframe(
                summary,
                use_container_width=True
            )


# ============================================================
# INTERSECTIONAL FAIRNESS
# ============================================================

elif page == "Intersectional Fairness":

    st.header(
        "Intersectional Fairness"
    )

    data = load_csv(
        "fairness_intersectional_comparison.csv"
    )

    if data is not None:

        st.dataframe(
            data,
            use_container_width=True
        )

        st.subheader(
            "Intersectional DI"
        )

        if "fairness_DI" in data.columns:

            chart = data.set_index(
                "intersection"
            )["fairness_DI"]

            st.bar_chart(
                chart
            )


# ============================================================
# COUNTERFACTUAL FAIRNESS
# ============================================================

elif page == "Counterfactual Fairness":

    st.header(
        "Counterfactual Fairness"
    )

    summary = load_csv(
        "counterfactual_fairness_summary.csv"
    )

    if summary is not None:

        st.dataframe(
            summary,
            use_container_width=True
        )

    st.info(
        """
        The current model does not directly use gender,
        age_group or location as model features. Therefore,
        changing these protected attributes while keeping
        model features unchanged produces zero direct
        prediction change.

        This does not prove complete fairness because
        indirect proxy effects may still exist.
        """
    )


# ============================================================
# FAIRNESS MITIGATION
# ============================================================

elif page == "Fairness Mitigation":

    st.header(
        "Fairness-Aware Recommendation"
    )

    quality = load_csv(
        "fairness_quality_comparison.csv"
    )

    fairness = load_csv(
        "fairness_metric_comparison.csv"
    )

    if quality is not None:

        st.subheader(
            "Recommendation Quality"
        )

        st.dataframe(
            quality,
            use_container_width=True
        )

    if fairness is not None:

        st.subheader(
            "Fairness Comparison"
        )

        st.dataframe(
            fairness,
            use_container_width=True
        )

    baseline = load_csv(
        "baseline_top10_for_fairness.csv"
    )

    fair = load_csv(
        "fairness_aware_top10.csv"
    )

    if baseline is not None and fair is not None:

        st.subheader(
            "Recommendation Comparison"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.write(
                "Baseline Recommendations"
            )

            st.dataframe(
                baseline.head(10),
                use_container_width=True
            )

        with col2:

            st.write(
                "Fairness-Aware Recommendations"
            )

            st.dataframe(
                fair.head(10),
                use_container_width=True
            )


# ============================================================
# FAIRNESS STRENGTH
# ============================================================

elif page == "Fairness Strength":

    st.header(
        "Fairness Strength Experiment"
    )

    data = load_csv(
        "fairness_strength_results.csv"
    )

    if data is not None:

        st.dataframe(
            data,
            use_container_width=True
        )

        st.subheader(
            "NDCG vs Fairness Strength"
        )

        chart_data = data.set_index(
            "fairness_strength"
        )[[
            "ndcg_at_10"
        ]]

        st.line_chart(
            chart_data
        )

        st.subheader(
            "Fairness DI vs Fairness Strength"
        )

        di_data = data.set_index(
            "fairness_strength"
        )[[
            "gender_di",
            "age_group_di",
            "location_di"
        ]]

        st.line_chart(
            di_data
        )

        st.subheader(
            "Generated Plots"
        )

        ndcg_plot = (
            RESULTS /
            "fairness_strength_ndcg.png"
        )

        di_plot = (
            RESULTS /
            "fairness_strength_di.png"
        )

        if ndcg_plot.exists():

            st.image(
                str(ndcg_plot),
                use_container_width=True
            )

        if di_plot.exists():

            st.image(
                str(di_plot),
                use_container_width=True
            )


# ============================================================
# NSGA-II
# ============================================================

elif page == "NSGA-II Optimization":

    st.header(
        "Multi-Objective / Pareto Optimization"
    )

    all_solutions = load_csv(
        "nsga2_all_solutions.csv"
    )

    pareto = load_csv(
        "nsga2_pareto_front.csv"
    )

    selected = load_csv(
        "nsga2_selected_solution.csv"
    )

    if selected is not None:

        st.subheader(
            "Selected Configuration"
        )

        st.dataframe(
            selected,
            use_container_width=True
        )

    if pareto is not None:

        st.subheader(
            "Pareto Front"
        )

        st.dataframe(
            pareto,
            use_container_width=True
        )

    plot = (
        RESULTS /
        "nsga2_pareto_front.png"
    )

    if plot.exists():

        st.image(
            str(plot),
            use_container_width=True
        )


# ============================================================
# FINAL COMPARISON
# ============================================================

elif page == "Final Comparison":

    st.header(
        "🏆 Final Model Comparison"
    )

    quality = load_csv(
        "final_quality_comparison.csv"
    )

    fairness = load_csv(
        "final_fairness_comparison.csv"
    )

    spd = load_csv(
        "final_spd_comparison.csv"
    )

    improvement = load_csv(
        "final_fairness_improvement.csv"
    )

    if quality is not None:

        st.subheader(
            "Recommendation Quality"
        )

        st.dataframe(
            quality,
            use_container_width=True
        )

        st.bar_chart(
            quality.set_index(
                "Metric"
            )
        )

    if fairness is not None:

        st.subheader(
            "Fairness Comparison — DI"
        )

        st.dataframe(
            fairness,
            use_container_width=True
        )

        st.bar_chart(
            fairness.set_index(
                "Protected Attribute"
            )
        )

    if spd is not None:

        st.subheader(
            "Statistical Parity Difference"
        )

        st.dataframe(
            spd,
            use_container_width=True
        )

    if improvement is not None:

        st.subheader(
            "Fairness Improvement"
        )

        st.dataframe(
            improvement,
            use_container_width=True
        )

    st.success(
        """
        Overall, the fairness-aware optimization
        improves demographic fairness metrics while
        preserving the measured recommendation quality
        in the current experimental setup.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "Fair LinkedIn Recommendation System"
)

st.sidebar.caption(
    "XGBoost • SHAP • Fairness • Multi-Objective Optimization"
)