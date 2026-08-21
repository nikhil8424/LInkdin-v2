import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import (
    RESULTS_DIR,
    MODELS_DIR,
    PROTECTED_ATTRIBUTES,
    DEFAULT_K,
    RANDOM_SEED
)
from src.fairness_mitigation import rerank_candidates
from src.top_k_recommender import evaluate_user_recommendations
from src.fairness_analysis import compute_group_fairness_metrics
from src.intersectional_fairness import compute_intersectional_table

class NSGA2Optimizer:
    def __init__(
        self,
        test_df,
        pop_size=30,
        generations=15,
        crossover_prob=0.9,
        mutation_prob=0.2,
        seed=RANDOM_SEED
    ):
        self.test_df = test_df
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.rng = np.random.default_rng(seed)
        
        self.group_stats = {}
        for attr in PROTECTED_ATTRIBUTES:
            g_means = test_df.groupby(attr)["baseline_score"].mean()
            overall = test_df["baseline_score"].mean()
            self.group_stats[attr] = (overall / g_means).clip(0.80, 1.25).to_dict()

    def evaluate_individual(self, chromosome):
        l_strength, exp_w, int_w, trade_p = chromosome
        
        obj_config = {
            "group_stats": self.group_stats,
            "exp_weight": exp_w,
            "int_weight": int_w,
            "trade_param": trade_p
        }

        # Fast vectorized candidate reranking
        sol_df = rerank_candidates(self.test_df, "baseline_score", PROTECTED_ATTRIBUTES, l_strength, obj_config)

        user_eval = evaluate_user_recommendations(sol_df, "fairness_score")
        ndcg10 = float(user_eval["ndcg_at_10"].dropna().mean())
        prec10 = float(user_eval["precision_at_10"].mean())
        rec10 = float(user_eval["recall_at_10"].dropna().mean())

        _, fair_summary = compute_group_fairness_metrics(sol_df, "fairness_rank", "fairness_score")
        fair_dict = fair_summary.set_index("protected_attribute").to_dict(orient="index")

        g_di = fair_dict["gender"]["exposure_DI"]
        a_di = fair_dict["age_group"]["exposure_DI"]
        l_di = fair_dict["location"]["exposure_DI"]
        avg_exp_di = float((g_di + a_di + l_di) / 3.0)
        exp_gap = float(abs(1.0 - avg_exp_di))

        df_3way = compute_intersectional_table(sol_df, ["gender", "age_group", "location"], "fairness_rank", "fairness_score")
        stable_3way = df_3way[~df_3way["statistically_unstable"]]
        worst_int_di = float(stable_3way["exposure_DI"].min() if len(stable_3way) > 0 else df_3way["exposure_DI"].min())
        int_gap = float(1.0 - worst_int_di)

        objectives = [-ndcg10, exp_gap, int_gap]
        metrics = {
            "ndcg_at_10": ndcg10,
            "precision_at_10": prec10,
            "recall_at_10": rec10,
            "gender_di": g_di,
            "age_group_di": a_di,
            "location_di": l_di,
            "average_exposure_di": avg_exp_di,
            "fairness_gap": exp_gap,
            "worst_intersectional_di": worst_int_di,
            "max_intersectional_gap": int_gap
        }
        return objectives, metrics

    def fast_non_dominated_sort(self, pop_objs):
        num_inds = len(pop_objs)
        domination_counts = [0] * num_inds
        dominated_indices = [[] for _ in range(num_inds)]
        fronts = [[]]

        for p in range(num_inds):
            for q in range(num_inds):
                p_objs = pop_objs[p]
                q_objs = pop_objs[q]

                less_equal = all(p_objs[i] <= q_objs[i] for i in range(len(p_objs)))
                strictly_less = any(p_objs[i] < q_objs[i] for i in range(len(p_objs)))

                if less_equal and strictly_less:
                    dominated_indices[p].append(q)
                elif all(q_objs[i] <= p_objs[i] for i in range(len(p_objs))) and any(q_objs[i] < p_objs[i] for i in range(len(p_objs))):
                    domination_counts[p] += 1

            if domination_counts[p] == 0:
                fronts[0].append(p)

        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in dominated_indices[p]:
                    domination_counts[q] -= 1
                    if domination_counts[q] == 0:
                        next_front.append(q)
            i += 1
            fronts.append(next_front)

        return [f for f in fronts if len(f) > 0]

    def compute_crowding_distance(self, front, pop_objs):
        distances = {idx: 0.0 for idx in front}
        num_objs = len(pop_objs[0])
        l = len(front)
        if l <= 2:
            for idx in front:
                distances[idx] = float("inf")
            return distances

        for m in range(num_objs):
            sorted_front = sorted(front, key=lambda idx: pop_objs[idx][m])
            distances[sorted_front[0]] = float("inf")
            distances[sorted_front[-1]] = float("inf")

            obj_min = pop_objs[sorted_front[0]][m]
            obj_max = pop_objs[sorted_front[-1]][m]
            norm = (obj_max - obj_min) if (obj_max - obj_min) > 1e-8 else 1.0

            for i in range(1, l - 1):
                prev_val = pop_objs[sorted_front[i - 1]][m]
                next_val = pop_objs[sorted_front[i + 1]][m]
                distances[sorted_front[i]] += (next_val - prev_val) / norm

        return distances

    def run(self):
        print(f"Initializing genuine NSGA-II (Pop Size={self.pop_size}, Gens={self.generations})...")
        
        population = [
            self.rng.uniform([0.0, 0.0, 0.0, 0.5], [1.0, 1.0, 1.0, 1.5]).tolist()
            for _ in range(self.pop_size)
        ]
        population[0] = [0.0, 0.5, 0.5, 1.0]
        population[1] = [0.5, 0.5, 0.5, 1.0]
        population[2] = [1.0, 0.5, 0.5, 1.0]

        eval_cache = {}
        all_evaluated_solutions = []

        for gen in range(self.generations):
            pop_objs = []
            pop_metrics = []

            for ind in population:
                ind_key = tuple(np.round(ind, 4))
                if ind_key not in eval_cache:
                    objs, mets = self.evaluate_individual(ind)
                    eval_cache[ind_key] = (objs, mets)
                else:
                    objs, mets = eval_cache[ind_key]
                pop_objs.append(objs)
                pop_metrics.append(mets)

                all_evaluated_solutions.append({
                    "generation": gen + 1,
                    "fairness_strength": ind[0],
                    "exp_weight": ind[1],
                    "int_weight": ind[2],
                    "trade_param": ind[3],
                    **mets
                })

            fronts = self.fast_non_dominated_sort(pop_objs)

            offspring = []
            while len(offspring) < self.pop_size:
                p1_idx = self.rng.choice(len(population))
                p2_idx = self.rng.choice(len(population))
                parent1 = population[p1_idx]
                parent2 = population[p2_idx]

                if self.rng.random() < self.crossover_prob:
                    c1, c2 = [], []
                    for k in range(len(parent1)):
                        if self.rng.random() <= 0.5:
                            beta = (2.0 * self.rng.random()) ** (1.0 / 3.0)
                        else:
                            beta = (1.0 / (2.0 * (1.0 - self.rng.random()))) ** (1.0 / 3.0)
                        val1 = 0.5 * ((1 + beta) * parent1[k] + (1 - beta) * parent2[k])
                        val2 = 0.5 * ((1 - beta) * parent1[k] + (1 + beta) * parent2[k])
                        c1.append(float(np.clip(val1, 0.0, 1.0)))
                        c2.append(float(np.clip(val2, 0.0, 1.0)))
                else:
                    c1, c2 = list(parent1), list(parent2)

                for child in [c1, c2]:
                    if self.rng.random() < self.mutation_prob:
                        m_idx = self.rng.choice(len(child))
                        delta = self.rng.normal(0, 0.1)
                        child[m_idx] = float(np.clip(child[m_idx] + delta, 0.0, 1.0))
                    if len(offspring) < self.pop_size:
                        offspring.append(child)

            combined_pop = population + offspring
            combined_objs = []
            for ind in combined_pop:
                ind_key = tuple(np.round(ind, 4))
                if ind_key not in eval_cache:
                    objs, mets = self.evaluate_individual(ind)
                    eval_cache[ind_key] = (objs, mets)
                else:
                    objs, mets = eval_cache[ind_key]
                combined_objs.append(objs)

            combined_fronts = self.fast_non_dominated_sort(combined_objs)
            new_pop = []
            for front in combined_fronts:
                if len(new_pop) + len(front) <= self.pop_size:
                    new_pop.extend([combined_pop[idx] for idx in front])
                else:
                    distances = self.compute_crowding_distance(front, combined_objs)
                    sorted_front = sorted(front, key=lambda idx: distances[idx], reverse=True)
                    needed = self.pop_size - len(new_pop)
                    new_pop.extend([combined_pop[idx] for idx in sorted_front[:needed]])
                    break
            population = new_pop

        final_objs = [eval_cache[tuple(np.round(ind, 4))][0] for ind in population]
        final_fronts = self.fast_non_dominated_sort(final_objs)
        pareto_indices = final_fronts[0]

        pareto_solutions = []
        for idx in pareto_indices:
            ind = population[idx]
            objs, mets = eval_cache[tuple(np.round(ind, 4))]
            pareto_solutions.append({
                "fairness_strength": ind[0],
                "exp_weight": ind[1],
                "int_weight": ind[2],
                "trade_param": ind[3],
                **mets
            })

        all_df = pd.DataFrame(all_evaluated_solutions).drop_duplicates(subset=["fairness_strength", "ndcg_at_10", "fairness_gap"])
        pareto_df = pd.DataFrame(pareto_solutions).drop_duplicates().sort_values("ndcg_at_10", ascending=False)
        return all_df, pareto_df

def run_nsga2_optimization():
    print("=" * 60)
    print("GENUINE NSGA-II MULTI-OBJECTIVE FAIRNESS OPTIMIZATION")
    print("=" * 60)

    test_split_path = RESULTS_DIR / "test_split.csv"
    X_test_path = RESULTS_DIR / "X_test.csv"
    model_path = MODELS_DIR / "xgboost_baseline.pkl"

    test_df = pd.read_csv(test_split_path)
    X_test = pd.read_csv(X_test_path)
    model = joblib.load(model_path)

    test_df["baseline_score"] = model.predict_proba(X_test)[:, 1]
    test_df = test_df.sort_values(["user_id", "baseline_score"], ascending=[True, False]).copy()
    test_df["baseline_rank"] = test_df.groupby("user_id").cumcount() + 1
    test_df["rank"] = test_df["baseline_rank"]

    optimizer = NSGA2Optimizer(test_df, pop_size=25, generations=12, seed=RANDOM_SEED)
    all_df, pareto_df = optimizer.run()

    min_ndcg, max_ndcg = all_df["ndcg_at_10"].min(), all_df["ndcg_at_10"].max()
    all_df["ndcg_normalized"] = (
        (all_df["ndcg_at_10"] - min_ndcg) / (max_ndcg - min_ndcg)
        if max_ndcg > min_ndcg else 1.0
    )
    pareto_df["ndcg_normalized"] = (
        (pareto_df["ndcg_at_10"] - min_ndcg) / (max_ndcg - min_ndcg)
        if max_ndcg > min_ndcg else 1.0
    )

    pareto_df["score_balanced_50_50"] = 0.50 * pareto_df["ndcg_normalized"] + 0.50 * (1.0 - pareto_df["fairness_gap"])
    pareto_df["score_fairness_25_75"] = 0.25 * pareto_df["ndcg_normalized"] + 0.75 * (1.0 - pareto_df["fairness_gap"])
    pareto_df["score_quality_75_25"] = 0.75 * pareto_df["ndcg_normalized"] + 0.25 * (1.0 - pareto_df["fairness_gap"])
    pareto_df["balanced_score"] = pareto_df["score_balanced_50_50"]
    all_df["balanced_score"] = 0.50 * all_df["ndcg_normalized"] + 0.50 * (1.0 - all_df["fairness_gap"])

    selected_balanced = pareto_df.sort_values("score_balanced_50_50", ascending=False).iloc[[0]].copy()

    all_df.to_csv(RESULTS_DIR / "nsga2_all_solutions.csv", index=False)
    pareto_df.to_csv(RESULTS_DIR / "nsga2_pareto_front.csv", index=False)
    selected_balanced.to_csv(RESULTS_DIR / "nsga2_selected_solution.csv", index=False)

    print(f"\nSaved NSGA-II candidate solutions ({len(all_df)}) and Pareto front ({len(pareto_df)}).")
    print("\nSelected Balanced Pareto Solution (50/50 Quality/Fairness):")
    print(selected_balanced[["fairness_strength", "ndcg_at_10", "average_exposure_di", "fairness_gap", "worst_intersectional_di"]].to_string(index=False))

    plt.figure(figsize=(9, 6))
    plt.scatter(all_df["fairness_gap"], all_df["ndcg_at_10"], color="#94A3B8", alpha=0.5, label="Explored Solutions", s=30)
    plt.plot(pareto_df.sort_values("fairness_gap")["fairness_gap"], pareto_df.sort_values("fairness_gap")["ndcg_at_10"], color="#2563EB", linewidth=2.5, label="NSGA-II Pareto Front")
    plt.scatter(pareto_df["fairness_gap"], pareto_df["ndcg_at_10"], color="#1D4ED8", s=60, edgecolors="black", label="Pareto-Optimal Points")

    baseline_pt = all_df[all_df["fairness_strength"] == 0.0].iloc[0] if len(all_df[all_df["fairness_strength"] == 0.0]) > 0 else None
    if baseline_pt is not None:
        plt.scatter(baseline_pt["fairness_gap"], baseline_pt["ndcg_at_10"], color="#DC2626", s=130, marker="X", edgecolors="black", label="Baseline (No Mitigation)", zorder=5)

    plt.scatter(selected_balanced["fairness_gap"], selected_balanced["ndcg_at_10"], color="#10B981", s=150, marker="*", edgecolors="black", label="Selected Balanced (50/50)", zorder=6)

    plt.title("NSGA-II Multi-Objective Pareto Front: NDCG@10 vs. Fairness Gap", fontsize=13, pad=15)
    plt.xlabel(r"Fairness Gap ($|1.0 - \overline{\mathrm{Exposure\ DI}}|$)", fontsize=12)
    plt.ylabel("Recommendation Quality (NDCG@10)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower left", fontsize=10)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "nsga2_pareto_front.png", dpi=300)
    plt.close()
    print("Pareto front plot saved to: results/nsga2_pareto_front.png\n")

if __name__ == "__main__":
    run_nsga2_optimization()