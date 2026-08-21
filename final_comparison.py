import pandas as pd
from pathlib import Path


# ============================================================
# FINAL MODEL COMPARISON
# ============================================================

RESULTS = Path("results")


# ============================================================
# 1. CHECK REQUIRED FILES
# ============================================================

required_files = [
    "fairness_strength_results.csv",
    "nsga2_selected_solution.csv",
    "fairness_intersectional_comparison.csv"
]

print("=" * 70)
print("CHECKING RESULT FILES")
print("=" * 70)

for filename in required_files:

    path = RESULTS / filename

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{path}\n\n"
            "Make sure all previous project stages have been run."
        )

    print("✓", path)

print()


# ============================================================
# 2. LOAD EXISTING RESULTS
# ============================================================

strength = pd.read_csv(
    RESULTS / "fairness_strength_results.csv"
)

nsga = pd.read_csv(
    RESULTS / "nsga2_selected_solution.csv"
)

intersectional = pd.read_csv(
    RESULTS / "fairness_intersectional_comparison.csv"
)


print("=" * 70)
print("RESULT FILES LOADED")
print("=" * 70)

print(
    "Fairness-strength results:",
    strength.shape
)

print(
    "NSGA-II selected solution:",
    nsga.shape
)

print(
    "Intersectional results:",
    intersectional.shape
)

print()


# ============================================================
# 3. SELECT THREE CONFIGURATIONS
# ============================================================

# Baseline:
# fairness strength = 0.0

baseline = strength[
    strength["fairness_strength"] == 0.0
].iloc[0]


# Fairness-aware model:
# fairness strength = 0.5

fairness_aware = strength[
    strength["fairness_strength"] == 0.5
].iloc[0]


# Pareto-selected configuration:
# selected by NSGA-II/Pareto analysis

nsga_selected = nsga.iloc[0]


# ============================================================
# 4. RECOMMENDATION QUALITY COMPARISON
# ============================================================

quality_comparison = pd.DataFrame({

    "Metric": [
        "Precision@10",
        "Recall@10",
        "NDCG@10"
    ],

    "Baseline XGBoost": [

        baseline[
            "precision_at_10"
        ],

        baseline[
            "recall_at_10"
        ],

        baseline[
            "ndcg_at_10"
        ]
    ],

    "Fairness-aware (0.5)": [

        fairness_aware[
            "precision_at_10"
        ],

        fairness_aware[
            "recall_at_10"
        ],

        fairness_aware[
            "ndcg_at_10"
        ]
    ],

    "Pareto-selected (1.0)": [

        nsga_selected[
            "precision_at_10"
        ],

        nsga_selected[
            "recall_at_10"
        ],

        nsga_selected[
            "ndcg_at_10"
        ]
    ]
})


print("=" * 70)
print("FINAL RECOMMENDATION QUALITY COMPARISON")
print("=" * 70)

print(
    quality_comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print()


# ============================================================
# 5. GROUP FAIRNESS COMPARISON
# ============================================================

fairness_comparison = pd.DataFrame({

    "Protected Attribute": [
        "Gender",
        "Age Group",
        "Location"
    ],

    "Baseline DI": [

        baseline[
            "gender_di"
        ],

        baseline[
            "age_group_di"
        ],

        baseline[
            "location_di"
        ]
    ],

    "Fairness-aware DI": [

        fairness_aware[
            "gender_di"
        ],

        fairness_aware[
            "age_group_di"
        ],

        fairness_aware[
            "location_di"
        ]
    ],

    "Pareto-selected DI": [

        nsga_selected[
            "gender_di"
        ],

        nsga_selected[
            "age_group_di"
        ],

        nsga_selected[
            "location_di"
        ]
    ]
})


print("=" * 70)
print("FINAL GROUP FAIRNESS COMPARISON")
print("=" * 70)

print(
    fairness_comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print()


# ============================================================
# 6. SPD COMPARISON
# ============================================================

spd_comparison = pd.DataFrame({

    "Protected Attribute": [
        "Gender",
        "Age Group",
        "Location"
    ],

    "Baseline SPD": [

        baseline[
            "gender_spd"
        ],

        baseline[
            "age_group_spd"
        ],

        baseline[
            "location_spd"
        ]
    ],

    "Fairness-aware SPD": [

        fairness_aware[
            "gender_spd"
        ],

        fairness_aware[
            "age_group_spd"
        ],

        fairness_aware[
            "location_spd"
        ]
    ],

    "Pareto-selected SPD": [

        nsga_selected[
            "gender_spd"
        ],

        nsga_selected[
            "age_group_spd"
        ],

        nsga_selected[
            "location_spd"
        ]
    ]
})


print("=" * 70)
print("FINAL SPD COMPARISON")
print("=" * 70)

print(
    spd_comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print()


# ============================================================
# 7. INTERSECTIONAL FAIRNESS
# ============================================================

print("=" * 70)
print("INTERSECTIONAL FAIRNESS COMPARISON")
print("=" * 70)


intersectional_columns = [
    "intersection",
    "baseline_SPD",
    "fairness_SPD",
    "baseline_DI",
    "fairness_DI"
]


print(
    intersectional[
        intersectional_columns
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print()


# ============================================================
# 8. QUALITY IMPROVEMENT
# ============================================================

quality_improvement = pd.DataFrame({

    "Metric": [
        "Precision@10",
        "Recall@10",
        "NDCG@10"
    ],

    "Baseline_to_Fairness": [

        fairness_aware[
            "precision_at_10"
        ]
        -
        baseline[
            "precision_at_10"
        ],

        fairness_aware[
            "recall_at_10"
        ]
        -
        baseline[
            "recall_at_10"
        ],

        fairness_aware[
            "ndcg_at_10"
        ]
        -
        baseline[
            "ndcg_at_10"
        ]
    ],

    "Baseline_to_Pareto": [

        nsga_selected[
            "precision_at_10"
        ]
        -
        baseline[
            "precision_at_10"
        ],

        nsga_selected[
            "recall_at_10"
        ]
        -
        baseline[
            "recall_at_10"
        ],

        nsga_selected[
            "ndcg_at_10"
        ]
        -
        baseline[
            "ndcg_at_10"
        ]
    ]
})


print("=" * 70)
print("QUALITY IMPROVEMENT")
print("=" * 70)

print(
    quality_improvement.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print()


# ============================================================
# 9. FAIRNESS IMPROVEMENT
# ============================================================

fairness_improvement = pd.DataFrame({

    "Protected Attribute": [
        "Gender",
        "Age Group",
        "Location"
    ],

    "DI Improvement: Fairness-aware": [

        fairness_aware[
            "gender_di"
        ]
        -
        baseline[
            "gender_di"
        ],

        fairness_aware[
            "age_group_di"
        ]
        -
        baseline[
            "age_group_di"
        ],

        fairness_aware[
            "location_di"
        ]
        -
        baseline[
            "location_di"
        ]
    ],

    "DI Improvement: Pareto-selected": [

        nsga_selected[
            "gender_di"
        ]
        -
        baseline[
            "gender_di"
        ],

        nsga_selected[
            "age_group_di"
        ]
        -
        baseline[
            "age_group_di"
        ],

        nsga_selected[
            "location_di"
        ]
        -
        baseline[
            "location_di"
        ]
    ]
})


print("=" * 70)
print("FAIRNESS IMPROVEMENT")
print("=" * 70)

print(
    fairness_improvement.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print()


# ============================================================
# 10. SAVE FINAL COMPARISON FILES
# ============================================================

quality_comparison.to_csv(
    RESULTS / "final_quality_comparison.csv",
    index=False
)


fairness_comparison.to_csv(
    RESULTS / "final_fairness_comparison.csv",
    index=False
)


spd_comparison.to_csv(
    RESULTS / "final_spd_comparison.csv",
    index=False
)


quality_improvement.to_csv(
    RESULTS / "final_quality_improvement.csv",
    index=False
)


fairness_improvement.to_csv(
    RESULTS / "final_fairness_improvement.csv",
    index=False
)


# ============================================================
# 11. FINAL SUMMARY
# ============================================================

print("=" * 70)
print("FINAL COMPARISON COMPLETED")
print("=" * 70)

print()

print("Models/configurations compared:")

print(
    "1. Baseline XGBoost"
)

print(
    "2. Fairness-aware configuration (strength = 0.5)"
)

print(
    "3. Pareto-selected configuration (strength = 1.0)"
)

print()

print("Generated files:")

print(
    "1. results/final_quality_comparison.csv"
)

print(
    "2. results/final_fairness_comparison.csv"
)

print(
    "3. results/final_spd_comparison.csv"
)

print(
    "4. results/final_quality_improvement.csv"
)

print(
    "5. results/final_fairness_improvement.csv"
)

print()

print("=" * 70)
print("ALL FINAL COMPARISONS SAVED SUCCESSFULLY")
print("=" * 70)