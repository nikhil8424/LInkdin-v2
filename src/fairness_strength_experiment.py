import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import (
    RESULTS_DIR,
    MODELS_DIR,
    PROTECTED_ATTRIBUTES,
    FAIRNESS_STRENGTHS,
    DEFAULT_K
)
from src.fairness_mitigation import rerank_candidates, compare_rankings
from src.top_k_recommender import evaluate_user_recommendations
from src.fairness_analysis import compute_group_fairness_metrics
from src.intersectional_fairness import compute_intersectional_table

def run_fairness_strength_experiment():
    print("=" * 60)
    print("FAIRNESS STRENGTH EXPERIMENTAL SWEEP [0.0 TO 1.0]")
    print("=" * 60)

    test_split_path = RESULTS_DIR / "test_split.csv"
    X_test_path = RESULTS_DIR / "X_test.csv"
    model_path = MODELS_DIR / "xgboost_baseline.pkl"

    test_df = pd.read_csv(test_split_path)
    X_test = pd.read_csv(X_test_path)
    model = joblib.load(model_path)

    # 1. Baseline Scores
    test_df["baseline_score"] = model.predict_proba(X_test)[:, 1]
    test_df = test_df.sort_values(["user_id", "baseline_score"], ascending=[True, False]).copy()
    test_df["baseline_rank"] = test_df.groupby("user_id").cumcount() + 1
    test_df["rank"] = test_df["baseline_rank"]

    # Precompute group inverse mean calibration stats
    group_stats = {}
    for attr in PROTECTED_ATTRIBUTES:
        g_means = test_df.groupby(attr)["baseline_score"].mean()
        overall = test_df["baseline_score"].mean()
        corr = (overall / g_means).clip(0.80, 1.25)
        group_stats[attr] = corr.to_dict()

    obj_config = {"group_stats": group_stats}

    all_sweep_results = []
    final_fairness_top10 = None

    for strength in FAIRNESS_STRENGTHS:
        print(f"Testing Fairness Strength: {strength:.1f} ...")

        # Vectorized reranking
        strength_df = rerank_candidates(test_df, "baseline_score", PROTECTED_ATTRIBUTES, strength, obj_config)

        if strength == 1.0:
            final_fairness_top10 = strength_df[strength_df["fairness_rank"] <= DEFAULT_K].copy()

        # Quality Evaluation
        user_eval = evaluate_user_recommendations(strength_df, "fairness_score")
        p5 = user_eval["precision_at_5"].mean()
        p10 = user_eval["precision_at_10"].mean()
        r5 = user_eval["recall_at_5"].dropna().mean()
        r10 = user_eval["recall_at_10"].dropna().mean()
        ndcg5 = user_eval["ndcg_at_5"].dropna().mean()
        ndcg10 = user_eval["ndcg_at_10"].dropna().mean()

        # Ranking Change Diagnostics
        diag_df = compare_rankings(test_df, strength_df, DEFAULT_K)
        pct_users_changed = float(diag_df["top_k_changed"].mean() * 100)
        pct_items_moved = float((diag_df["items_entered_top_k"] > 0).mean() * 100)
        top_k_overlap = float(diag_df["top_k_overlap"].mean())
        top_k_jaccard = float(diag_df["top_k_jaccard"].mean())

        # Multiplier stats
        multipliers = strength_df["fairness_multiplier"]
        mean_mult = float(multipliers.mean())
        min_mult = float(multipliers.min())
        max_mult = float(multipliers.max())

        # Marginal Fairness Evaluation
        _, fair_summary = compute_group_fairness_metrics(strength_df, "fairness_rank", "fairness_score")
        fair_dict = fair_summary.set_index("protected_attribute").to_dict(orient="index")

        gender_exp_di = fair_dict["gender"]["exposure_DI"]
        age_exp_di = fair_dict["age_group"]["exposure_DI"]
        loc_exp_di = fair_dict["location"]["exposure_DI"]

        gender_sel_di = fair_dict["gender"]["selection_rate_DI"]
        age_sel_di = fair_dict["age_group"]["selection_rate_DI"]
        loc_sel_di = fair_dict["location"]["selection_rate_DI"]

        gender_spd = fair_dict["gender"]["exposure_SPD"]
        age_spd = fair_dict["age_group"]["exposure_SPD"]
        loc_spd = fair_dict["location"]["exposure_SPD"]

        # Intersectional Evaluation (3-way)
        df_3way = compute_intersectional_table(strength_df, ["gender", "age_group", "location"], "fairness_rank", "fairness_score")
        stable_3way = df_3way[~df_3way["statistically_unstable"]]
        worst_int_di = float(stable_3way["exposure_DI"].min() if len(stable_3way) > 0 else df_3way["exposure_DI"].min())
        max_int_gap = float(1.0 - worst_int_di)

        avg_exp_di = (gender_exp_di + age_exp_di + loc_exp_di) / 3.0
        fairness_gap = abs(1.0 - avg_exp_di)

        all_sweep_results.append({
            "fairness_strength": strength,
            "precision_at_5": p5,
            "precision_at_10": p10,
            "recall_at_5": r5,
            "recall_at_10": r10,
            "ndcg_at_5": ndcg5,
            "ndcg_at_10": ndcg10,
            "gender_exposure_di": gender_exp_di,
            "age_exposure_di": age_exp_di,
            "location_exposure_di": loc_exp_di,
            "gender_selection_di": gender_sel_di,
            "age_selection_di": age_sel_di,
            "location_selection_di": loc_sel_di,
            "gender_spd": gender_spd,
            "age_group_spd": age_spd,
            "location_spd": loc_spd,
            "average_exposure_di": avg_exp_di,
            "fairness_gap": fairness_gap,
            "worst_intersectional_di": worst_int_di,
            "max_intersectional_gap": max_int_gap,
            "top_k_overlap": top_k_overlap,
            "top_k_jaccard": top_k_jaccard,
            "percentage_users_changed": pct_users_changed,
            "percentage_items_moved": pct_items_moved,
            "mean_fairness_multiplier": mean_mult,
            "min_fairness_multiplier": min_mult,
            "max_fairness_multiplier": max_mult
        })

    sweep_df = pd.DataFrame(all_sweep_results)
    
    # Normalizations
    min_ndcg, max_ndcg = sweep_df["ndcg_at_10"].min(), sweep_df["ndcg_at_10"].max()
    sweep_df["ndcg_normalized"] = (
        (sweep_df["ndcg_at_10"] - min_ndcg) / (max_ndcg - min_ndcg)
        if max_ndcg > min_ndcg else 1.0
    )
    sweep_df["balanced_score"] = 0.5 * sweep_df["ndcg_normalized"] + 0.5 * (1.0 - sweep_df["fairness_gap"])

    # Backwards compatibility columns for app.py
    sweep_df["gender_di"] = sweep_df["gender_exposure_di"]
    sweep_df["age_group_di"] = sweep_df["age_exposure_di"]
    sweep_df["location_di"] = sweep_df["location_exposure_di"]

    results_csv_path = RESULTS_DIR / "fairness_strength_results.csv"
    sweep_df.to_csv(results_csv_path, index=False)
    print(f"\nSaved fairness sweep results to: {results_csv_path}")

    # Generate Diagnostic Plots
    generate_diagnostic_plots(sweep_df, test_df, final_fairness_top10)

def generate_diagnostic_plots(sweep_df, test_df, final_fairness_top10):
    print("Generating comprehensive diagnostic plots...")

    # Plot 1: NDCG@10 vs Fairness Strength
    plt.figure(figsize=(8, 5))
    plt.plot(sweep_df["fairness_strength"], sweep_df["ndcg_at_10"], marker="o", color="#2563EB", linewidth=2.5)
    plt.title("NDCG@10 vs. Fairness Strength", fontsize=14, pad=12)
    plt.xlabel(r"Fairness Strength ($\lambda$)", fontsize=12)
    plt.ylabel("NDCG@10", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "fairness_strength_ndcg.png", dpi=300)
    plt.close()

    # Plot 2: Recall@10 vs Fairness Strength
    plt.figure(figsize=(8, 5))
    plt.plot(sweep_df["fairness_strength"], sweep_df["recall_at_10"], marker="s", color="#10B981", linewidth=2.5)
    plt.title("Recall@10 vs. Fairness Strength", fontsize=14, pad=12)
    plt.xlabel(r"Fairness Strength ($\lambda$)", fontsize=12)
    plt.ylabel("Recall@10", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "fairness_strength_recall.png", dpi=300)
    plt.close()

    # Plot 3: Exposure DI vs Fairness Strength
    plt.figure(figsize=(8, 5))
    plt.plot(sweep_df["fairness_strength"], sweep_df["gender_exposure_di"], marker="o", label="Gender Exposure DI", linewidth=2)
    plt.plot(sweep_df["fairness_strength"], sweep_df["age_exposure_di"], marker="s", label="Age Exposure DI", linewidth=2)
    plt.plot(sweep_df["fairness_strength"], sweep_df["location_exposure_di"], marker="^", label="Location Exposure DI", linewidth=2)
    plt.axhline(1.0, color="gray", linestyle="--", alpha=0.7, label="Ideal Parity (1.0)")
    plt.title("Exposure Disparate Impact vs. Fairness Strength", fontsize=14, pad=12)
    plt.xlabel(r"Fairness Strength ($\lambda$)", fontsize=12)
    plt.ylabel("Exposure DI (Min/Max Ratio)", fontsize=12)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "fairness_strength_di.png", dpi=300)
    plt.savefig(RESULTS_DIR / "fairness_strength_exposure_di.png", dpi=300)
    plt.close()

    # Plot 4: Selection DI vs Fairness Strength
    plt.figure(figsize=(8, 5))
    plt.plot(sweep_df["fairness_strength"], sweep_df["gender_selection_di"], marker="o", label="Gender Selection DI", linewidth=2)
    plt.plot(sweep_df["fairness_strength"], sweep_df["age_selection_di"], marker="s", label="Age Selection DI", linewidth=2)
    plt.plot(sweep_df["fairness_strength"], sweep_df["location_selection_di"], marker="^", label="Location Selection DI", linewidth=2)
    plt.axhline(1.0, color="gray", linestyle="--", alpha=0.7, label="Ideal Parity (1.0)")
    plt.title("Selection Rate DI vs. Fairness Strength", fontsize=14, pad=12)
    plt.xlabel(r"Fairness Strength ($\lambda$)", fontsize=12)
    plt.ylabel("Selection Rate DI", fontsize=12)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "fairness_strength_selection_di.png", dpi=300)
    plt.close()

    # Plot 5: Worst Intersectional DI vs Fairness Strength
    plt.figure(figsize=(8, 5))
    plt.plot(sweep_df["fairness_strength"], sweep_df["worst_intersectional_di"], marker="d", color="#8B5CF6", linewidth=2.5)
    plt.axhline(1.0, color="gray", linestyle="--", alpha=0.7, label="Ideal Parity (1.0)")
    plt.title("Worst-Case Intersectional DI vs. Fairness Strength", fontsize=14, pad=12)
    plt.xlabel(r"Fairness Strength ($\lambda$)", fontsize=12)
    plt.ylabel("Worst Intersectional DI (3-Way)", fontsize=12)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "fairness_strength_worst_intersectional_di.png", dpi=300)
    plt.close()

    # Plot 6: Percentage Users Changed & Top-K Overlap
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(sweep_df["fairness_strength"], sweep_df["percentage_users_changed"], color="#EF4444", marker="o", linewidth=2.5, label="% Users Changed")
    ax1.set_xlabel(r"Fairness Strength ($\lambda$)", fontsize=12)
    ax1.set_ylabel("% Users with Changed Top-10", color="#EF4444", fontsize=12)
    ax1.tick_params(axis="y", labelcolor="#EF4444")
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2 = ax1.twinx()
    ax2.plot(sweep_df["fairness_strength"], sweep_df["top_k_overlap"], color="#3B82F6", marker="s", linewidth=2.5, linestyle="--", label="Top-10 Overlap")
    ax2.set_ylabel("Average Top-10 Overlap", color="#3B82F6", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="#3B82F6")

    plt.title("Reranking Dynamics Across Fairness Strengths", fontsize=14, pad=12)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "fairness_strength_ranking_changes.png", dpi=300)
    plt.close()

    # Plot 7: Proxy Classifier AUCs
    proxy_path = RESULTS_DIR / "proxy_attribute_prediction.csv"
    if proxy_path.exists():
        proxy_df = pd.read_csv(proxy_path)
        plt.figure(figsize=(7, 4.5))
        bars = plt.bar(proxy_df["protected_attribute"], proxy_df["ROC_AUC"], color=["#3B82F6", "#10B981", "#F59E0B"], width=0.5)
        plt.axhline(0.5, color="gray", linestyle="--", label="Random Chance (0.50)")
        plt.title("Proxy Prediction ROC-AUC by Sensitive Attribute", fontsize=13, pad=12)
        plt.ylabel("ROC-AUC Score", fontsize=11)
        plt.ylim(0.0, 1.0)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.3f}", ha="center", va="bottom", fontweight="bold")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "proxy_attribute_auc.png", dpi=300)
        plt.close()

    print("All diagnostic plots generated and saved successfully.\n")

if __name__ == "__main__":
    run_fairness_strength_experiment()