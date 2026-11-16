"""
Utility functions for reproducibility, hardware acceleration detection,
logging, JSON serialization, and metrics tracking.
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import random
import logging
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from src.config import RANDOM_SEED, LOGS_DIR

def set_reproducible_seeds(seed=RANDOM_SEED):
    """Set random seeds across python, numpy, and tensorflow for reproducible runs."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # Ensure deterministic TensorFlow operations where supported
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def setup_hardware():
    """
    Detects hardware accelerator (GPU/CPU).
    Configures memory growth on GPUs if detected, otherwise sets up CPU execution.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[Hardware Setup] GPU detected and configured: {len(gpus)} device(s) - {[gpu.name for gpu in gpus]}")
            return "GPU", len(gpus)
        except RuntimeError as e:
            print(f"[Hardware Setup] Error configuring GPU memory growth: {e}")
            return "GPU (Configuration Error)", len(gpus)
    else:
        print("[Hardware Setup] No GPU detected by TensorFlow. Running on CPU with multi-threading.")
        return "CPU", 0

def setup_logger(name: str, log_filename: str = None) -> logging.Logger:
    """Configures a standardized logger writing to both console and a log file."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers if called multiple times
    if logger.handlers:
        return logger
        
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_filename:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOGS_DIR / log_filename
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger

def save_json(data: dict, filepath: Path) -> None:
    """Save dictionary to a nicely formatted JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Custom serializer for non-standard types like numpy types
    def convert_types(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Path):
            return str(obj)
        return str(obj)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, default=convert_types)

def load_json(filepath: Path) -> dict:
    """Load a dictionary from a JSON file."""
    if not filepath.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_class_weights(labels_array, num_classes: int = None) -> dict:
    """
    Compute balanced class weights for the labels present in the training split.
    """
    unique_classes = np.unique(labels_array)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=labels_array
    )
    class_weight_dict = {int(cls): float(weight) for cls, weight in zip(unique_classes, weights)}
    return class_weight_dict

if __name__ == "__main__":
    set_reproducible_seeds()
    device, count = setup_hardware()
    logger = setup_logger("UtilsTest", "utils_test.log")
    logger.info(f"Utils verification passed. Device: {device}, Count: {count}")
