"""
Data Preprocessing Module for Weed Species Classification with ResNet152V2.
Implements:
- Standard ResNet152V2 image preprocessing (scaled to [-1, 1] via tf.keras.applications.resnet_v2.preprocess_input)
- Biological identity-preserving training-only data augmentation
- Preprocessing configuration export to models/model_metadata.json
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_v2_preprocess_input

from src.config import (
    IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, INPUT_SHAPE,
    NUM_CLASSES, CLASS_NAMES, LABEL_TO_CLASS, METADATA_PATH,
    RANDOM_SEED, init_directories
)
from src.utils import setup_logger, save_json

logger = setup_logger("DataPreprocessing", "data_preprocessing.log")

def get_data_augmentation_pipeline() -> tf.keras.Sequential:
    """
    Builds training-only augmentation pipeline.
    Preserves botanical integrity of weed foliage, leaf venation, and floral morphology.
    - Random horizontal flip (invariance to field capture orientation)
    - Small rotation (+/- 8% ~ 15-20 deg)
    - Small zoom (+/- 8%)
    - Small translation (+/- 5%)
    - Moderate contrast variation (simulating changing solar irradiance in rangeland pastures)
    """
    augmentation_layers = [
        layers.RandomFlip("horizontal", seed=RANDOM_SEED, name="aug_random_flip"),
        layers.RandomRotation(factor=0.08, fill_mode="reflect", seed=RANDOM_SEED, name="aug_random_rotation"),
        layers.RandomZoom(height_factor=(-0.08, 0.08), fill_mode="reflect", seed=RANDOM_SEED, name="aug_random_zoom"),
        layers.RandomTranslation(height_factor=0.05, width_factor=0.05, fill_mode="reflect", seed=RANDOM_SEED, name="aug_random_translation"),
        layers.RandomContrast(factor=0.15, seed=RANDOM_SEED, name="aug_random_contrast")
    ]
    return tf.keras.Sequential(augmentation_layers, name="weed_data_augmentation")

def preprocess_image_tensor(image_tensor: tf.Tensor) -> tf.Tensor:
    """
    Applies standard ResNet152V2 preprocessing to an image tensor.
    Expects pixel values in [0, 255]. ResNetV2 scales to [-1.0, 1.0].
    NOTE: Never divide by 255 before or after this function.
    """
    image_tensor = tf.cast(image_tensor, tf.float32)
    return resnet_v2_preprocess_input(image_tensor)

def save_preprocessing_metadata():
    """Save preprocessing parameters to model_metadata.json."""
    init_directories()
    metadata = {
        "model_architecture": "ResNet152V2 Transfer Learning",
        "input_resolution": f"{IMAGE_HEIGHT}x{IMAGE_WIDTH}x{IMAGE_CHANNELS}",
        "target_size": [IMAGE_HEIGHT, IMAGE_WIDTH],
        "color_mode": "RGB",
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "label_mapping": LABEL_TO_CLASS,
        "preprocessing_method": "tf.keras.applications.resnet_v2.preprocess_input",
        "scaling_range": "[-1.0, 1.0]",
        "training_augmentation": {
            "random_flip": "horizontal",
            "random_rotation_factor": 0.08,
            "random_zoom_range": [-0.08, 0.08],
            "random_translation": [0.05, 0.05],
            "random_contrast_factor": 0.15,
            "fill_mode": "reflect"
        },
        "tensorflow_version": tf.__version__,
        "dataset": "Folder-labeled plant species dataset"
    }
    save_json(metadata, METADATA_PATH)
    logger.info(f"Saved preprocessing metadata to {METADATA_PATH}")

if __name__ == "__main__":
    logger.info("Initializing and validating data preprocessing module...")
    aug_model = get_data_augmentation_pipeline()
    save_preprocessing_metadata()
    logger.info("Augmentation pipeline structure:")
    aug_model.summary(print_fn=logger.info)
    logger.info("Data preprocessing module verification completed.")
