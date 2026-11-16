"""
Model Architecture Module for Weed Species Classification using ResNet152V2.
Implements:
- Transfer learning with ImageNet pre-trained ResNet152V2 backbone
- Regularized classification head (GlobalAveragePooling2D, Dual Dense, Dual Dropout, BatchNormalization)
- Multi-metric compilation (Accuracy, Precision, Recall, AUC-ROC)
- Controlled fine-tuning unfreezing with BatchNormalization protection
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import ResNet152V2

from src.config import (
    INPUT_SHAPE, NUM_CLASSES, DROPOUT_RATE_1, DROPOUT_RATE_2,
    DENSE_UNITS, STAGE1_LEARNING_RATE, STAGE2_LEARNING_RATE,
    MODELS_DIR, CLASS_NAMES, init_directories
)
from src.utils import setup_logger

logger = setup_logger("ModelArchitecture", "model.log")

def build_resnet152v2_model(
    input_shape: tuple = INPUT_SHAPE,
    num_classes: int = NUM_CLASSES,
    dense_units: int = DENSE_UNITS,
    dropout_rate_1: float = DROPOUT_RATE_1,
    dropout_rate_2: float = DROPOUT_RATE_2,
    learning_rate: float = STAGE1_LEARNING_RATE,
    label_smoothing: float = 0.05
) -> tf.keras.Model:
    """
    Constructs the ResNet152V2 Transfer Learning Architecture.
    Backbone is initially frozen for Stage 1 Feature Extraction.
    """
    logger.info("Instantiating ResNet152V2 pre-trained on ImageNet (include_top=False)...")
    base_model = ResNet152V2(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    
    # Stage 1: Freeze all backbone weights to protect pre-trained features
    base_model.trainable = False
    
    inputs = base_model.input
    x = base_model.output
    
    # 1. Global Average Pooling (aggregates 2D spatial feature maps into 2048-dim embedding)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    
    # 2. First Regularized Dense Block
    x = layers.BatchNormalization(name="head_bn_1")(x)
    x = layers.Dense(
        dense_units,
        activation="relu",
        kernel_initializer="he_normal",
        name="dense_feature_1"
    )(x)
    x = layers.Dropout(dropout_rate_1, name="dropout_1")(x)
    
    # 3. Second Dense Bottleneck Block
    x = layers.BatchNormalization(name="head_bn_2")(x)
    x = layers.Dense(
        dense_units // 2,
        activation="relu",
        kernel_initializer="he_normal",
        name="dense_feature_2"
    )(x)
    x = layers.Dropout(dropout_rate_2, name="dropout_2")(x)
    
    # 4. Final softmax classification head
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",
        name="weed_predictions"
    )(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="WeedClassifier_ResNet152V2")
    
    # Compile with Adam optimizer and multi-class metrics
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
    metrics = [
        tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="roc_auc")
    ]
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    total_params = model.count_params()
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_params = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
    
    logger.info(f"Model compiled. Total params: {total_params:,} | Trainable: {trainable_params:,} | Non-trainable: {non_trainable_params:,}")
    return model

def configure_model_for_finetuning(
    model: tf.keras.Model,
    unfreeze_layers: int = 30,
    learning_rate: float = STAGE2_LEARNING_RATE
) -> tf.keras.Model:
    """
    Configures model for Stage 2 Fine-Tuning:
    - Finds the ResNet152V2 backbone layers
    - Unfreezes the top `unfreeze_layers` convolutional layers
    - Keeps all BatchNormalization layers frozen to avoid corrupting moving statistics
    - Recompiles with a reduced learning rate (1e-5)
    """
    logger.info(f"Preparing Stage 2 Fine-Tuning: unfreezing top {unfreeze_layers} upper layers...")
    
    # Unfreeze the base model
    # We inspect all layers up to the classification head
    # Head starts at 'global_avg_pool'
    head_start_idx = None
    for i, layer in enumerate(model.layers):
        if layer.name == "global_avg_pool":
            head_start_idx = i
            break
            
    if head_start_idx is None:
        head_start_idx = len(model.layers) - 7
        
    backbone_layers = model.layers[:head_start_idx]
    cutoff_index = max(0, len(backbone_layers) - unfreeze_layers)
    
    for i, layer in enumerate(backbone_layers):
        if i >= cutoff_index:
            # Keep BatchNormalization frozen during fine-tuning
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True
        else:
            layer.trainable = False
            
    # Always keep newly added classification head trainable
    for layer in model.layers[head_start_idx:]:
        layer.trainable = True
        
    # Recompile with small learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
    metrics = [
        tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="roc_auc")
    ]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    logger.info(f"Stage 2 fine-tuning configuration complete. Trainable parameters: {trainable_params:,}")
    return model

if __name__ == "__main__":
    init_directories()
    logger.info("Verifying model construction...")
    model = build_resnet152v2_model()
    logger.info("Testing fine-tuning reconfiguration...")
    model = configure_model_for_finetuning(model, unfreeze_layers=30)
    logger.info("Model verification successful!")
