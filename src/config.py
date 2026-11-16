"""
Configuration module for Weed Species Classification using ResNet152V2.
Contains project paths, dataset parameters, hyperparameter defaults, and environment setups.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Dataset Paths
DATASET_DIR = BASE_DIR / "dataset"
IMAGES_DIR = DATASET_DIR
LABELS_CSV = DATASET_DIR / "labels.csv"
SPLIT_DIR = DATASET_DIR / "split_files"
TRAIN_CSV = SPLIT_DIR / "train.csv"
VAL_CSV = SPLIT_DIR / "val.csv"
TEST_CSV = SPLIT_DIR / "test.csv"
CV_SPLIT_DIR = SPLIT_DIR / "cv_folds"

# Model Paths
MODELS_DIR = BASE_DIR / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
BEST_MODEL_PATH = MODELS_DIR / "weed_resnet152v2_best.keras"
FINAL_MODEL_PATH = MODELS_DIR / "weed_resnet152v2_final.keras"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
HISTORY_PATH = MODELS_DIR / "training_history.json"
EVALUATION_METRICS_PATH = MODELS_DIR / "evaluation_metrics.json"

# Results & Output Paths
RESULTS_DIR = BASE_DIR / "results"
EDA_DIR = RESULTS_DIR / "eda"
GRAPHS_DIR = RESULTS_DIR / "graphs"
CONFUSION_MATRIX_DIR = RESULTS_DIR / "confusion_matrix"
CLASSIFICATION_REPORT_DIR = RESULTS_DIR / "classification_report"
GRADCAM_DIR = RESULTS_DIR / "gradcam"
TRAINING_PLOTS_DIR = RESULTS_DIR / "training_plots"
EVAL_METRICS_DIR = RESULTS_DIR / "evaluation_metrics"
LOGS_DIR = BASE_DIR / "logs"

# Class order is derived from the current folder-labeled dataset.
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
CLASS_NAMES = sorted(
    (
        folder.name
        for folder in DATASET_DIR.iterdir()
        if folder.is_dir()
        and any(
            image.is_file() and image.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            for image in folder.iterdir()
        )
    ),
    key=str.casefold
) if DATASET_DIR.exists() else []

LABEL_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}
CLASS_TO_LABEL = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# Image & Model Specifications
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_CHANNELS = 3
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
NUM_CLASSES = len(CLASS_NAMES)

# Reproducibility Seed
RANDOM_SEED = 42

# Training Hyperparameters
BATCH_SIZE = 32
STAGE1_EPOCHS = 15
STAGE2_EPOCHS = 15
STAGE1_LEARNING_RATE = 1e-3
STAGE2_LEARNING_RATE = 1e-5
EARLY_STOPPING_PATIENCE = 7
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.2
MIN_LEARNING_RATE = 1e-7

# Regularization Architecture Parameters
DROPOUT_RATE_1 = 0.45
DROPOUT_RATE_2 = 0.30
DENSE_UNITS = 512
FINE_TUNE_LAYERS = 30  # Top upper layers of ResNet152V2 to unfreeze in Stage 2

def init_directories():
    """Create all required directories in the project hierarchy."""
    directories = [
        DATASET_DIR,
        IMAGES_DIR,
        SPLIT_DIR,
        CV_SPLIT_DIR,
        MODELS_DIR,
        CHECKPOINTS_DIR,
        RESULTS_DIR,
        EDA_DIR,
        GRAPHS_DIR,
        CONFUSION_MATRIX_DIR,
        CLASSIFICATION_REPORT_DIR,
        GRADCAM_DIR,
        TRAINING_PLOTS_DIR,
        EVAL_METRICS_DIR,
        LOGS_DIR
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    return True

if __name__ == "__main__":
    init_directories()
    print("Project directory structure initialized successfully.")
    print(f"Base Directory: {BASE_DIR}")
    print(f"Total Classes: {NUM_CLASSES}")
    for idx, name in LABEL_TO_CLASS.items():
        print(f"  {idx}: {name}")
