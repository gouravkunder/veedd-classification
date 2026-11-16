"""
Training entry point for two-stage ResNet152V2 transfer learning.
Stage 1 trains the classification head while the backbone is frozen.
Stage 2 fine-tunes the upper backbone layers and exports the best model.
Both stages incorporate:
- EarlyStopping (val_loss monitoring, patience 7, restore_best_weights)
- ReduceLROnPlateau (factor 0.2, patience 3, min_lr 1e-7)
- ModelCheckpoint (saves best validation loss model in .keras format)
- CSVLogger and TensorBoard logging
- Balanced class weight handling for severe weed imbalance
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger, TensorBoard
)

from src.config import (
    BATCH_SIZE, STAGE1_EPOCHS, STAGE1_LEARNING_RATE,
    STAGE2_EPOCHS, STAGE2_LEARNING_RATE, FINE_TUNE_LAYERS,
    EARLY_STOPPING_PATIENCE, REDUCE_LR_PATIENCE, REDUCE_LR_FACTOR, MIN_LEARNING_RATE,
    CHECKPOINTS_DIR, LOGS_DIR, MODELS_DIR, CLASS_NAMES_PATH, CLASS_NAMES,
    init_directories
)
from src.utils import set_reproducible_seeds, setup_hardware, setup_logger, save_json
from src.data_loader import get_data_loaders
from src.model import build_resnet152v2_model
from src.data_preprocessing import save_preprocessing_metadata

logger = setup_logger("Stage1Train", "stage1_train.log")

def train_stage1(
    epochs: int = STAGE1_EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = STAGE1_LEARNING_RATE,
    sample_fraction: float = None
) -> tuple:
    """
    Executes Stage 1 training (Feature Extraction).
    Returns (trained_model, history_dict, best_checkpoint_path).
    """
    init_directories()
    set_reproducible_seeds()
    device, count = setup_hardware()
    
    logger.info("=" * 60)
    logger.info("STARTING STAGE 1 TRAINING: FEATURE EXTRACTION")
    logger.info(f"Hardware: {device} ({count} devices) | Epochs: {epochs} | Batch Size: {batch_size}")
    logger.info("=" * 60)
    
    # Save official class names JSON in models/ directory for teammates
    save_json(CLASS_NAMES, CLASS_NAMES_PATH)
    save_preprocessing_metadata()
    logger.info(f"Exported class names order to {CLASS_NAMES_PATH}")
    
    # Load dataset pipelines
    train_ds, val_ds, test_ds, class_weights = get_data_loaders(
        batch_size=batch_size,
        augment_train=True,
        sample_fraction=sample_fraction
    )
    
    # Build model with frozen backbone
    model = build_resnet152v2_model(learning_rate=learning_rate)
    
    # Configure Callbacks
    stage1_checkpoint_path = CHECKPOINTS_DIR / "stage1_best.keras"
    csv_log_path = LOGS_DIR / "stage1_training.csv"
    tb_log_dir = LOGS_DIR / "tensorboard_stage1"
    
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            mode="min",
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=MIN_LEARNING_RATE,
            mode="min",
            verbose=1
        ),
        ModelCheckpoint(
            filepath=str(stage1_checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1
        ),
        CSVLogger(
            filename=str(csv_log_path),
            separator=",",
            append=False
        ),
        TensorBoard(
            log_dir=str(tb_log_dir),
            histogram_freq=1
        )
    ]
    
    logger.info("Commencing Stage 1 model fit...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    history_dict = history.history
    # Convert any float32 to python float for JSON serialization
    serialized_history = {k: [float(v) for v in vals] for k, vals in history_dict.items()}
    save_json(serialized_history, CHECKPOINTS_DIR / "stage1_history.json")
    
    best_val_loss = min(serialized_history.get("val_loss", [float("inf")]))
    best_val_acc = max(serialized_history.get("val_accuracy", [0.0]))
    logger.info(f"Stage 1 Training Complete. Best Val Loss: {best_val_loss:.4f} | Best Val Accuracy: {best_val_acc:.4f}")
    logger.info(f"Stage 1 Best Checkpoint saved to: {stage1_checkpoint_path}")
    
    return model, serialized_history, stage1_checkpoint_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and export the complete two-stage weed classifier")
    parser.add_argument("--epochs", type=int, default=STAGE1_EPOCHS, help="Maximum Stage 1 epochs")
    parser.add_argument("--fine-tune-epochs", type=int, default=STAGE2_EPOCHS, help="Maximum Stage 2 fine-tuning epochs")
    parser.add_argument("--fine-tune-lr", type=float, default=STAGE2_LEARNING_RATE, help="Stage 2 learning rate")
    parser.add_argument("--unfreeze-layers", type=int, default=FINE_TUNE_LAYERS, help="Number of upper backbone layers to unfreeze")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=STAGE1_LEARNING_RATE, help="Initial learning rate")
    parser.add_argument("--sample-fraction", type=float, default=None, help="Optional stratified fraction for experiments; omit for final training")
    args = parser.parse_args()
    
    train_stage1(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        sample_fraction=args.sample_fraction
    )

    from src.fine_tuning import run_fine_tuning

    run_fine_tuning(
        epochs=args.fine_tune_epochs,
        batch_size=args.batch_size,
        learning_rate=args.fine_tune_lr,
        unfreeze_layers=args.unfreeze_layers,
        sample_fraction=args.sample_fraction
    )
