# Fairness-Aware LinkedIn Recommendation System

A fairness-aware LinkedIn-style recommendation system that combines machine learning recommendations with explainability and multi-objective fairness optimization.

The project is designed to recommend relevant professional connections while measuring and reducing potential demographic bias in the recommendation results.

## Key Features

- Synthetic LinkedIn-style professional dataset
- XGBoost-based recommendation model
- Top-K connection recommendations
- SHAP-based recommendation explanations
- Gender-based fairness analysis
- Intersectional fairness analysis
- Fairness-aware recommendation comparison
- NSGA-II multi-objective optimization
- Pareto-optimal solution selection
- Comparison of recommendation quality and fairness
- CSV-based experiment results and evaluation

## Project Objective

Traditional recommendation systems mainly optimize relevance or prediction accuracy. However, a recommendation system can perform well while still producing systematically different outcomes for different demographic groups.

This project addresses that problem by treating recommendation quality and fairness as related but competing objectives.

The system therefore evaluates:

1. Recommendation quality
2. Fairness across gender groups
3. Intersectional fairness
4. Explainability of recommendations
5. Trade-offs between recommendation performance and fairness

## Methodology

The overall workflow is:

```text
LinkedIn-Style Dataset
        |
        v
Data Preprocessing
        |
        v
Feature Engineering
        |
        v
XGBoost Recommendation Model
        |
        +--------------------+
        |                    |
        v                    v
   Top-K Results       SHAP Explanations
        |
        v
Fairness Evaluation
        |
        +----------------------+
        |                      |
        v                      v
 Gender Fairness      Intersectional Fairness
        |                      |
        +----------+-----------+
                   |
                   v
          NSGA-II Optimization
                   |
                   v
        Pareto-Optimal Solutions
                   |
                   v
       Quality vs Fairness Analysis
```

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core implementation |
| Pandas | Data processing |
| NumPy | Numerical computation |
| Scikit-learn | Machine learning utilities and evaluation |
| XGBoost | Recommendation/prediction model |
| SHAP | Model explainability |
| NSGA-II | Multi-objective fairness optimization |
| Matplotlib | Visualization |
| Seaborn | Statistical visualization |
| CSV | Experiment result storage |

## Dataset

The project uses a synthetic LinkedIn-style dataset.

The dataset contains professional and recommendation-related attributes designed to represent a realistic professional networking environment without relying on private LinkedIn user data.

Typical information may include:

- User/profile attributes
- Professional features
- Skills
- Experience-related information
- Connection-related features
- Demographic attributes used for fairness evaluation
- Recommendation labels or outcomes

Because the dataset is synthetic, the project can be reproduced without exposing real users' personal information.

## Recommendation Model

XGBoost is used as the primary machine learning model.

The model learns the relationship between user/profile features and recommendation outcomes.

The prediction score can then be used to rank potential recommendations.

### Top-K Recommendation

For every user:

1. Candidate profiles are generated.
2. Features are passed to the trained XGBoost model.
3. The model produces a recommendation score.
4. Candidates are sorted by score.
5. The highest-ranked K candidates are returned.

This produces the final Top-K recommendation list.

## Explainability with SHAP

SHAP (SHapley Additive exPlanations) is used to understand why the recommendation model produces its predictions.

SHAP can identify:

- Features that increase recommendation scores
- Features that decrease recommendation scores
- Relative importance of individual features
- Feature-level contribution to model predictions

This improves transparency and makes the recommendation system easier to analyze.

## Fairness Analysis

The project evaluates fairness at multiple levels.

### 1. Gender Fairness

Recommendation outcomes are compared across gender groups.

The analysis can examine differences in metrics such as:

- Recommendation exposure
- Selection/recommendation rate
- Average recommendation score
- True-positive or related performance measures
- Fairness gaps

A smaller difference between groups generally indicates better demographic parity for the evaluated metric.

### 2. Intersectional Fairness

Intersectional fairness evaluates combinations of demographic attributes rather than analyzing each attribute independently.

For example:

```text
Gender + another demographic/group attribute
```

This is important because a model may appear fair when each attribute is examined separately but still produce unequal outcomes for specific intersectional groups.

## NSGA-II Optimization

NSGA-II (Non-dominated Sorting Genetic Algorithm II) is used as a multi-objective optimization method.

Instead of reducing fairness and recommendation quality to one arbitrary combined score, NSGA-II searches for solutions that balance multiple objectives.

The optimization considers objectives such as:

```text
Objective 1: Maximize recommendation quality
Objective 2: Minimize fairness disparity
```

The output is a set of Pareto-optimal solutions.

### Why NSGA-II?

NSGA-II is appropriate because recommendation quality and fairness can conflict with each other.

For example:

```text
Higher Recommendation Quality
          |
          |        *
          |      *
          |    *
          |  *
          | *
          +----------------------> Better Fairness
```

Rather than selecting a single solution too early, NSGA-II provides multiple non-dominated alternatives.

## Pareto Front

A solution is Pareto-optimal when no other solution can improve one objective without worsening at least one other objective.

The resulting Pareto front represents the available trade-offs between:

- Recommendation performance
- Fairness

This allows the final solution to be selected according to the project's fairness-performance requirements.

## Results

The `results/` directory contains the generated experimental outputs.

Important result files include:

### `fairness_strength_results.csv`

Contains fairness-strength experiment results and measurements used to analyze how fairness changes under different optimization settings.

### `nsga2_selected_solution.csv`

Contains the selected NSGA-II solution from the multi-objective optimization experiment.

### `fairness_intersectional_comparison.csv`

Contains comparisons of fairness across intersectional demographic groups.

These files can be used for further analysis, visualization, reporting, and comparison of different recommendation strategies.

## Project Structure

```text
linkedin-fair-recommendation/
│
├── data/
│   └── synthetic LinkedIn-style dataset files
│
├── results/
│   ├── fairness_strength_results.csv
│   ├── nsga2_selected_solution.csv
│   └── fairness_intersectional_comparison.csv
│
├── src/
│   ├── data preprocessing
│   ├── recommendation model
│   ├── fairness analysis
│   ├── SHAP analysis
│   └── NSGA-II optimization
│
├── notebooks/
│   └── experimental notebooks
│
├── requirements.txt
├── README.md
└── project scripts
```

> Update the folder names above if your local repository uses a different structure.

## Installation

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd linkedin-fair-recommendation
```

### 2. Create a Virtual Environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If a `requirements.txt` file is not available, install the main dependencies with:

```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn
```

Install any additional optimization dependency required by the NSGA-II implementation used in the project.

## How to Run the Project

Follow these steps to run the complete project from start to finish.

### Step 1: Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd linkedin-fair-recommendation
```

### Step 2: Create a Virtual Environment

On Windows PowerShell:

```powershell
python -m venv venv
```

Activate the environment:

```powershell
venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, you can activate it using:

```powershell
venv\Scripts\activate
```

For macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

Install the project dependencies:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not present, install the main dependencies:

```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn
```

Install any additional package required by the NSGA-II implementation used in the project.

### Step 4: Check the Dataset

Make sure the synthetic LinkedIn-style dataset is placed in the expected `data/` directory.

The general structure should be:

```text
linkedin-fair-recommendation/
│
├── data/
│   └── <dataset-file>
│
├── results/
├── final_comparison.py
├── requirements.txt
└── ...
```

If your dataset uses a different filename or folder, update the dataset path in the corresponding Python script.

### Step 5: Run Data Preprocessing

Run the preprocessing script used by the project.

Example:

```bash
python <preprocessing_script>.py
```

This step prepares the data and generates the features required by the recommendation model.

### Step 6: Train the Recommendation Model

Run the XGBoost recommendation model:

```bash
python <recommendation_model_script>.py
```

The model learns from the processed LinkedIn-style data and produces recommendation scores.

### Step 7: Generate Top-K Recommendations

Run the recommendation-generation script:

```bash
python <top_k_script>.py
```

This ranks candidate profiles and generates the Top-K recommendations for users.

### Step 8: Generate SHAP Explanations

Run the SHAP analysis script:

```bash
python <shap_script>.py
```

This generates feature-level explanations showing which features influence the recommendation predictions.

### Step 9: Run Gender Fairness Analysis

Run the gender fairness experiment:

```bash
python <gender_fairness_script>.py
```

This evaluates recommendation outcomes across gender groups and calculates the fairness-related metrics used by the project.

### Step 10: Run Intersectional Fairness Analysis

Run the intersectional fairness analysis:

```bash
python <intersectional_fairness_script>.py
```

This evaluates fairness across combinations of demographic groups.

### Step 11: Run NSGA-II Optimization

Run the NSGA-II optimization script:

```bash
python <nsga2_script>.py
```

NSGA-II searches for Pareto-optimal solutions that balance recommendation quality and fairness.

The resulting selected solution is saved in:

```text
results/nsga2_selected_solution.csv
```

### Step 12: Run the Final Comparison

After all experiments have completed, run:

```bash
python final_comparison.py
```

This checks the generated result files and performs the final fairness comparison.

A successful run should show:

```text
======================================================================
CHECKING RESULT FILES
======================================================================
✓ results\\fairness_strength_results.csv
✓ results\\nsga2_selected_solution.csv
✓ results\\fairness_intersectional_comparison.csv
```

The script then compares the generated fairness results and displays the final experimental comparison.

### Complete Execution Order

Run the project in this general order:

```text
1. Create virtual environment
        ↓
2. Install dependencies
        ↓
3. Prepare dataset
        ↓
4. Run preprocessing
        ↓
5. Train XGBoost model
        ↓
6. Generate Top-K recommendations
        ↓
7. Generate SHAP explanations
        ↓
8. Run gender fairness analysis
        ↓
9. Run intersectional fairness analysis
        ↓
10. Run NSGA-II optimization
        ↓
11. Run final_comparison.py
        ↓
12. Inspect results/
```

> **Important:** Replace the placeholder script names such as `<preprocessing_script>.py`, `<recommendation_model_script>.py`, and `<nsga2_script>.py` with the actual filenames present in your repository.

### Quick Run

If all preprocessing, model, fairness, and optimization scripts are already configured in your repository, the final verification can be performed with:

```bash
venv\Scripts\activate
python final_comparison.py
```

This verifies that the required result files have been generated successfully.

## Result Verification

A successful verification should confirm the presence of:

```text
✓ results\fairness_strength_results.csv
✓ results\nsga2_selected_solution.csv
✓ results\fairness_intersectional_comparison.csv
```

## Evaluation Framework

The project evaluates the recommendation system from two complementary perspectives.

### Recommendation Performance

Measures how effectively the system identifies relevant recommendations.

Possible metrics include:

- Accuracy
- Precision
- Recall
- F1-score
- Ranking-based metrics
- Top-K performance

### Fairness Performance

Measures whether recommendation outcomes differ substantially between protected or demographic groups.

Possible measures include:

- Group selection-rate difference
- Statistical parity difference
- Exposure disparity
- Recommendation score disparity
- Intersectional fairness gap

The exact metrics used should be interpreted according to the implementation and experiment configuration in the repository.

## Fairness vs Recommendation Quality

The central idea of the project is that fairness should not simply be treated as an afterthought.

A recommendation model can achieve high predictive performance while producing unequal recommendation exposure.

Therefore, the project compares:

```text
Baseline Model
      vs
Fairness-Aware Model
      vs
NSGA-II Selected Solution
```

This makes it possible to study whether fairness improvements can be achieved while maintaining acceptable recommendation quality.

## Reproducibility

To reproduce the experiments:

1. Set up the Python environment.
2. Install all dependencies.
3. Ensure the dataset is available in the expected location.
4. Run preprocessing.
5. Train the recommendation model.
6. Generate Top-K recommendations.
7. Generate SHAP explanations.
8. Run gender fairness analysis.
9. Run intersectional fairness analysis.
10. Run the NSGA-II optimization.
11. Generate the result CSV files.
12. Run the final comparison script.

## Limitations

- The dataset is synthetic and may not capture every characteristic of a real professional networking platform.
- Fairness metrics depend on the selected demographic groups and evaluation methodology.
- Fairness improvements may reduce recommendation performance in some cases.
- SHAP explains model behavior but does not by itself guarantee that the model is fair.
- NSGA-II provides Pareto-optimal trade-offs rather than one universally optimal solution.
- Results should not be interpreted as evidence about the fairness of the real LinkedIn platform.

## Future Scope

Possible extensions include:

- Testing on larger and more diverse datasets
- Adding additional protected attributes
- Evaluating temporal fairness
- Incorporating fairness constraints directly into model training
- Comparing additional recommendation algorithms
- Adding graph-based recommendation methods
- Using Graph Neural Networks for professional-network modeling
- Adding online/real-time recommendation evaluation
- Developing an interactive fairness dashboard
- Adding more ranking-specific fairness metrics
- Evaluating fairness across multiple intersectional groups
- Comparing NSGA-II with other multi-objective optimization algorithms

## Research Contribution

The main contribution of this project is the integration of:

```text
Machine Learning Recommendation
            +
Explainable AI
            +
Fairness Evaluation
            +
Intersectional Analysis
            +
Multi-Objective Optimization
```

This creates a framework for studying recommendation systems not only in terms of predictive performance but also in terms of fairness, transparency, and demographic impact.

## Conclusion

This project demonstrates how a LinkedIn-style recommendation system can be extended beyond conventional relevance-based recommendation.

By combining XGBoost, SHAP, fairness analysis, intersectional evaluation, and NSGA-II optimization, the system provides a framework for investigating the trade-off between recommendation quality and fairness.

The resulting Pareto-optimal solutions allow different fairness-performance trade-offs to be examined rather than assuming that a single objective is sufficient.

## Author

**Payal Salve**

This project was developed as an academic/research project focused on Machine Learning, Explainable AI, Fairness-Aware Recommendation Systems, and Multi-Objective Optimization.

## License

This project is intended for academic and research purposes.

Add an appropriate open-source license such as MIT if you want to make the repository publicly reusable.
