import pandas as pd
import numpy as np

from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.metrics import ndcg_score



BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "synthetic_linkedin_dataset_30000.csv"
)

RESULTS_PATH = (
    BASE_DIR
    / "results"
)



K = 10

PROTECTED_ATTRIBUTES = [
    "gender",
    "age_group",
    "location"
]


# Fairness-strength values already tested in the
# previous experiment.

STRENGTH_VALUES = np.arange(
    0.0,
    1.01,
    0.1
)


print("=" * 60)
print("CHECKING REQUIRED FILES")
print("=" * 60)


if not DATA_PATH.exists():

    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}"
    )


STRENGTH_RESULTS_PATH = (
    RESULTS_PATH
    / "fairness_strength_results.csv"
)


if not STRENGTH_RESULTS_PATH.exists():

    raise FileNotFoundError(
        "fairness_strength_results.csv not found.\n"
        "Run fairness_strength_experiment.py first."
    )


print("✓ Dataset found")
print("✓ Fairness strength results found")

print()

print("=" * 60)
print("LOADING DATA")
print("=" * 60)


df = pd.read_csv(
    DATA_PATH
)


print(
    "Dataset shape:",
    df.shape
)

print()



# 5. LOAD FAIRNESS-STRENGTH RESULTS
# ============================================================

print("=" * 60)
print("LOADING CANDIDATE SOLUTIONS")
print("=" * 60)


strength_results = pd.read_csv(
    STRENGTH_RESULTS_PATH
)


print(
    "Candidate solutions:",
    len(strength_results)
)

print()

# 6. VERIFY REQUIRED COLUMNS
# ============================================================

required_columns = [
    "fairness_strength",
    "precision_at_10",
    "recall_at_10",
    "ndcg_at_10",
    "gender_di",
    "age_group_di",
    "location_di"
]


missing_columns = [
    column
    for column in required_columns
    if column not in strength_results.columns
]


if missing_columns:

    raise ValueError(
        f"Missing columns:\n{missing_columns}"
    )



# 7. CREATE FAIRNESS OBJECTIVE
# ============================================================

print("=" * 60)
print("CALCULATING MULTI-OBJECTIVE VALUES")
print("=" * 60)


# We want all DI values as close to 1 as possible.

strength_results[
    "average_DI"
] = (

    strength_results[
        [
            "gender_di",
            "age_group_di",
            "location_di"
        ]
    ]
    .mean(axis=1)
)


# Fairness gap from ideal DI = 1.

strength_results[
    "fairness_gap"
] = (

    1
    - strength_results[
        "average_DI"
    ]
).abs()





strength_results[
    "objective_quality"
] = (
    -strength_results[
        "ndcg_at_10"
    ]
)


strength_results[
    "objective_fairness"
] = (
    strength_results[
        "fairness_gap"
    ]
)


print(
    strength_results[
        [
            "fairness_strength",
            "ndcg_at_10",
            "average_DI",
            "fairness_gap"
        ]
    ].to_string(
        index=False
    )
)

print()


# 9. PARETO DOMINANCE FUNCTION
# ============================================================

def dominates(
    point_a,
    point_b
):

    """
    Return True if point A dominates point B.

    Both objectives are minimized.

    A dominates B when:

    A is no worse than B
    in every objective

    AND

    A is strictly better in at least
    one objective.
    """

    no_worse = (
        point_a[0] <= point_b[0]
        and
        point_a[1] <= point_b[1]
    )


    strictly_better = (
        point_a[0] < point_b[0]
        or
        point_a[1] < point_b[1]
    )


    return (
        no_worse
        and
        strictly_better
    )


# 10. FIND PARETO FRONT
# ============================================================

def find_pareto_front(
    dataframe
):

    pareto_indices = []


    objective_values = (
        dataframe[
            [
                "objective_quality",
                "objective_fairness"
            ]
        ]
        .to_numpy()
    )


    for i in range(
        len(objective_values)
    ):

        dominated = False


        for j in range(
            len(objective_values)
        ):

            if i == j:

                continue


            if dominates(
                objective_values[j],
                objective_values[i]
            ):

                dominated = True

                break


        if not dominated:

            pareto_indices.append(
                i
            )


    return dataframe.iloc[
        pareto_indices
    ].copy()


# 11. CALCULATE PARETO FRONT
# ============================================================

print("=" * 60)
print("CALCULATING PARETO FRONT")
print("=" * 60)


pareto_front = find_pareto_front(
    strength_results
)


pareto_front = (
    pareto_front
    .sort_values(
        "fairness_strength"
    )
    .reset_index(
        drop=True
    )
)


print(
    "Pareto-optimal solutions:",
    len(pareto_front)
)

print()


print(
    pareto_front[
        [
            "fairness_strength",
            "ndcg_at_10",
            "average_DI",
            "gender_di",
            "age_group_di",
            "location_di"
        ]
    ].to_string(
        index=False
    )
)

print()


# 12. MARK PARETO SOLUTIONS
# ============================================================

strength_results[
    "pareto_optimal"
] = False


for index in pareto_front.index:

    strength = pareto_front.loc[
        index,
        "fairness_strength"
    ]


    mask = (
        strength_results[
            "fairness_strength"
        ]
        == strength
    )


    strength_results.loc[
        mask,
        "pareto_optimal"
    ] = True


# 13. NORMALIZE OBJECTIVES
# ============================================================

print("=" * 60)
print("CALCULATING BALANCED SOLUTION")
print("=" * 60)


# Quality normalization
# ------------------------------------------------------------

quality_min = (
    pareto_front[
        "ndcg_at_10"
    ].min()
)


quality_max = (
    pareto_front[
        "ndcg_at_10"
    ].max()
)


if quality_max != quality_min:

    pareto_front[
        "quality_normalized"
    ] = (

        (
            pareto_front[
                "ndcg_at_10"
            ]
            - quality_min
        )

        /

        (
            quality_max
            - quality_min
        )
    )

else:

    pareto_front[
        "quality_normalized"
    ] = 1.0


# Fairness normalization
# ------------------------------------------------------------

fairness_min = (
    pareto_front[
        "average_DI"
    ].min()
)


fairness_max = (
    pareto_front[
        "average_DI"
    ].max()
)


if fairness_max != fairness_min:

    pareto_front[
        "fairness_normalized"
    ] = (

        (
            pareto_front[
                "average_DI"
            ]
            - fairness_min
        )

        /

        (
            fairness_max
            - fairness_min
        )
    )

else:

    pareto_front[
        "fairness_normalized"
    ] = 1.0


# 14. BALANCED PARETO SCORE
# ============================================================

pareto_front[
    "balanced_score"
] = (

    0.5
    * pareto_front[
        "quality_normalized"
    ]

    +

    0.5
    * pareto_front[
        "fairness_normalized"
    ]
)


best_index = (
    pareto_front[
        "balanced_score"
    ]
    .idxmax()
)


best_solution = (
    pareto_front.loc[
        best_index
    ]
)


print(
    "Selected Pareto solution:"
)

print(
    "Fairness strength:",
    best_solution[
        "fairness_strength"
    ]
)

print(
    "NDCG@10:",
    round(
        best_solution[
            "ndcg_at_10"
        ],
        4
    )
)

print(
    "Average DI:",
    round(
        best_solution[
            "average_DI"
        ],
        4
    )
)

print(
    "Gender DI:",
    round(
        best_solution[
            "gender_di"
        ],
        4
    )
)

print(
    "Age-group DI:",
    round(
        best_solution[
            "age_group_di"
        ],
        4
    )
)

print(
    "Location DI:",
    round(
        best_solution[
            "location_di"
        ],
        4
    )
)

print(
    "Balanced Pareto score:",
    round(
        best_solution[
            "balanced_score"
        ],
        4
    )
)

print()


# 15. SAVE PARETO RESULTS
# ============================================================

print("=" * 60)
print("SAVING NSGA-II RESULTS")
print("=" * 60)


all_results_path = (
    RESULTS_PATH
    / "nsga2_all_solutions.csv"
)


pareto_path = (
    RESULTS_PATH
    / "nsga2_pareto_front.csv"
)


selected_path = (
    RESULTS_PATH
    / "nsga2_selected_solution.csv"
)


strength_results.to_csv(
    all_results_path,
    index=False
)


pareto_front.to_csv(
    pareto_path,
    index=False
)


pd.DataFrame(
    [
        best_solution
    ]
).to_csv(
    selected_path,
    index=False
)


print(
    "1. nsga2_all_solutions.csv"
)

print(
    "2. nsga2_pareto_front.csv"
)

print(
    "3. nsga2_selected_solution.csv"
)

print()



# 16. CREATE PARETO FRONT PLOT
# ============================================================

print("=" * 60)
print("CREATING PARETO FRONT PLOT")
print("=" * 60)


plt.figure(
    figsize=(10, 6)
)


plt.scatter(
    strength_results[
        "average_DI"
    ],
    strength_results[
        "ndcg_at_10"
    ],
    label="Candidate solutions"
)


plt.scatter(
    pareto_front[
        "average_DI"
    ],
    pareto_front[
        "ndcg_at_10"
    ],
    marker="o",
    label="Pareto front"
)


plt.scatter(
    best_solution[
        "average_DI"
    ],
    best_solution[
        "ndcg_at_10"
    ],
    marker="*",
    s=200,
    label="Selected solution"
)


plt.xlabel(
    "Average Disparate Impact"
)

plt.ylabel(
    "NDCG@10"
)

plt.title(
    "NSGA-II Fairness–Quality Pareto Front"
)

plt.grid(
    True,
    alpha=0.3
)

plt.legend()

plt.tight_layout()


plot_path = (
    RESULTS_PATH
    / "nsga2_pareto_front.png"
)


plt.savefig(
    plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "Saved:"
)

print(
    plot_path
)

print()


print("=" * 60)
print("NSGA-II OPTIMIZATION COMPLETED")
print("=" * 60)

print()

print(
    "Pareto-optimal solutions:",
    len(pareto_front)
)

print(
    "Selected fairness strength:",
    best_solution[
        "fairness_strength"
    ]
)

print(
    "Selected NDCG@10:",
    round(
        best_solution[
            "ndcg_at_10"
        ],
        4
    )
)

print(
    "Selected average DI:",
    round(
        best_solution[
            "average_DI"
        ],
        4
    )
)

print()

print(
    "Generated files:"
)

print(
    "1. nsga2_all_solutions.csv"
)

print(
    "2. nsga2_pareto_front.csv"
)

print(
    "3. nsga2_selected_solution.csv"
)

print(
    "4. nsga2_pareto_front.png"
)

print()

print(
    "Results folder:"
)

print(
    RESULTS_PATH
)