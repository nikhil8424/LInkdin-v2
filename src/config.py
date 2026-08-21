import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"

# Ensure directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Raw Data Path
DATA_PATH = DATA_DIR / "synthetic_linkedin_dataset_30000.csv"
POSTS_PATH = DATA_DIR / "synthetic_linkedin_posts_5000.csv"
INTERACTIONS_PATH = DATA_DIR / "synthetic_linkedin_interactions_30000.csv"

# Global Constants for Reproducibility
RANDOM_SEED = 42
K_VALUES = [5, 10]
DEFAULT_K = 10
BOOTSTRAP_ITERATIONS = 1000
MIN_INTERSECTION_GROUP_SIZE = 30

# Feature Definitions
ID_COLUMNS = [
    "user_id",
    "post_id",
    "author_id"
]

PROTECTED_ATTRIBUTES = [
    "gender",
    "age_group",
    "location"
]

CATEGORICAL_FEATURES = [
    "professional_field",
    "education",
    "post_topic",
    "content_type"
]

NUMERICAL_FEATURES = [
    "experience_years",
    "network_size",
    "previous_interactions",
    "engagement",
    "author_user_similarity",
    "topic_similarity",
    "post_age_hours",
    "author_experience",
    "author_network_size",
    "network_distance"
]

MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
TARGET = "interaction"
LEAKAGE_COLUMN = "interaction_probability"

# Fairness Experiment Parameter Grid
FAIRNESS_STRENGTHS = [
    0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
]
