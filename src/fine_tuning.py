"""
Stage 2 Fine-Tuning Module for ResNet152V2 Weed Classification.
Loads the best Stage 1 model, unfreezes top convolutional layers (with BatchNormalization frozen),
fine-tunes with a conservative learning rate (1e-5), compares Stage 1 vs Stage 2 checkpoints,
and securely exports the superior model to models/weed_resnet152v2_best.keras and final.keras.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import shutil
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger, TensorBoard
)

from src.config import (
    BATCH_SIZE, STAGE2_EPOCHS, STAGE2_LEARNING_RATE, FINE_TUNE_LAYERS,
    EARLY_STOPPING_PATIENCE, REDUCE_LR_PATIENCE, REDUCE_LR_FACTOR, MIN_LEARNING_RATE,
    CHECKPOINTS_DIR, LOGS_DIR, BEST_MODEL_PATH, FINAL_MODEL_PATH,
    HISTORY_PATH, METADATA_PATH, CLASS_NAMES_PATH, CLASS_NAMES, init_directories
)
from src.utils import set_reproducible_seeds, setup_hardware, setup_logger, save_json, load_json
from src.data_loader import get_data_loaders
from src.model import configure_model_for_finetuning
from src.train import train_stage1
from src.data_preprocessing import save_preprocessing_metadata

logger = setup_logger("FineTuning", "fine_tuning.log")

def run_fine_tuning(
    epochs: int = STAGE2_EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = STAGE2_LEARNING_RATE,
    unfreeze_layers: int = FINE_TUNE_LAYERS,
    sample_fraction: float = None
) -> tf.keras.Model:
    """
    Executes Stage 2 fine-tuning and checkpoint comparison.
    """
    init_directories()
    set_reproducible_seeds()
    device, count = setup_hardware()
    save_preprocessing_metadata()
    
    logger.info("=" * 60)
    logger.info("STARTING STAGE 2 FINE-TUNING")
    logger.info(f"Hardware: {device} | Epochs: {epochs} | LR: {learning_rate} | Unfreeze Layers: {unfreeze_layers}")
    logger.info("=" * 60)
    
    stage1_checkpoint_path = CHECKPOINTS_DIR / "stage1_best.keras"
    stage1_history_path = CHECKPOINTS_DIR / "stage1_history.json"
    
    if not stage1_checkpoint_path.exists():
        logger.info("No Stage 1 checkpoint found. Running Stage 1 Feature Extraction first...")
        _, s1_history, _ = train_stage1(
            epochs=10,
            batch_size=batch_size,
            sample_fraction=sample_fraction
        )
    else:
        logger.info(f"Loading existing Stage 1 checkpoint: {stage1_checkpoint_path}")
        if stage1_history_path.exists():
            s1_history = load_json(stage1_history_path)
        else:
            s1_history = {}
            
    # Load Stage 1 model
    logger.info(f"Loading Stage 1 model weights from {stage1_checkpoint_path}...")
    stage1_model = tf.keras.models.load_model(stage1_checkpoint_path)
    if stage1_model.output_shape[-1] != len(CLASS_NAMES):
        raise ValueError(
            f"Stage 1 checkpoint has {stage1_model.output_shape[-1]} outputs, "
            f"but the current dataset has {len(CLASS_NAMES)} classes. "
            "Run Stage 1 training with the current dataset before fine-tuning."
        )
    
    # Configure for Stage 2
    model = configure_model_for_finetuning(
        model=stage1_model,
        unfreeze_layers=unfreeze_layers,
        learning_rate=learning_rate
    )
    
    # Load data
    train_ds, val_ds, test_ds, class_weights = get_data_loaders(
        batch_size=batch_size,
        augment_train=True,
        sample_fraction=sample_fraction
    )
    
    stage2_checkpoint_path = CHECKPOINTS_DIR / "stage2_best.keras"
    csv_log_path = LOGS_DIR / "stage2_training.csv"
    tb_log_dir = LOGS_DIR / "tensorboard_stage2"
    
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
            filepath=str(stage2_checkpoint_path),
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
    
    logger.info("Commencing Stage 2 fine-tuning fit...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    s2_history = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    save_json(s2_history, CHECKPOINTS_DIR / "stage2_history.json")
    
    # Consolidated History
    consolidated_history = {
        "stage1": s1_history,
        "stage2": s2_history
    }
    save_json(consolidated_history, HISTORY_PATH)
    logger.info(f"Consolidated training history saved to {HISTORY_PATH}")
    
    # Checkpoint Comparison Protocol
    logger.info("=" * 60)
    logger.info("CHECKPOINT SELECTION PROTOCOL: STAGE 1 VS STAGE 2")
    logger.info("=" * 60)
    
    s1_val_loss = min(s1_history.get("val_loss", [float("inf")]))
    s1_val_acc = max(s1_history.get("val_accuracy", [0.0]))
    s2_val_loss = min(s2_history.get("val_loss", [float("inf")]))
    s2_val_acc = max(s2_history.get("val_accuracy", [0.0]))
    
    logger.info(f"Stage 1 Best Checkpoint -> Val Loss: {s1_val_loss:.4f} | Val Accuracy: {s1_val_acc:.4f}")
    logger.info(f"Stage 2 Best Checkpoint -> Val Loss: {s2_val_loss:.4f} | Val Accuracy: {s2_val_acc:.4f}")
    
    if s2_val_loss < s1_val_loss:
        logger.info("Stage 2 Fine-Tuning demonstrated superior generalization (lower val_loss).")
        selected_checkpoint = stage2_checkpoint_path
        best_stage = "Stage 2 (Fine-Tuning)"
        best_val_loss = s2_val_loss
        best_val_acc = s2_val_acc
    else:
        logger.info("Stage 1 Feature Extraction demonstrated equal/superior validation stability. Retaining Stage 1 checkpoint.")
        selected_checkpoint = stage1_checkpoint_path
        best_stage = "Stage 1 (Feature Extraction)"
        best_val_loss = s1_val_loss
        best_val_acc = s1_val_acc
        
    # Copy selected best model to canonical paths
    shutil.copy2(selected_checkpoint, BEST_MODEL_PATH)
    shutil.copy2(selected_checkpoint, FINAL_MODEL_PATH)
    logger.info(f"Successfully saved canonical best model to:  {BEST_MODEL_PATH}")
    logger.info(f"Successfully saved canonical final model to: {FINAL_MODEL_PATH}")
    
    # Update Metadata
    if METADATA_PATH.exists():
        metadata = load_json(METADATA_PATH)
    else:
        metadata = {}
        
    metadata.update({
        "selected_best_stage": best_stage,
        "best_val_loss": float(best_val_loss),
        "best_val_accuracy": float(best_val_acc),
        "stage1_metrics": {"best_val_loss": float(s1_val_loss), "best_val_accuracy": float(s1_val_acc)},
        "stage2_metrics": {"best_val_loss": float(s2_val_loss), "best_val_accuracy": float(s2_val_acc)},
        "final_model_path": str(FINAL_MODEL_PATH.name),
        "best_model_path": str(BEST_MODEL_PATH.name),
        "class_names_path": str(CLASS_NAMES_PATH.name),
        "sample_fraction": sample_fraction,
        "training_complete": True
    })
    save_json(metadata, METADATA_PATH)
    logger.info(f"Updated metadata saved to {METADATA_PATH}")
    
    # Save class names JSON
    save_json(CLASS_NAMES, CLASS_NAMES_PATH)
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2 Fine-Tuning")
    parser.add_argument("--epochs", type=int, default=STAGE2_EPOCHS, help="Number of epochs for Stage 2")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=STAGE2_LEARNING_RATE, help="Fine-tuning learning rate")
    parser.add_argument("--unfreeze-layers", type=int, default=FINE_TUNE_LAYERS, help="Number of upper layers to unfreeze")
    parser.add_argument("--sample-fraction", type=float, default=None, help="Fraction of dataset to sample (0.0 to 1.0) for testing")
    args = parser.parse_args()
    
    run_fine_tuning(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        unfreeze_layers=args.unfreeze_layers,
        sample_fraction=args.sample_fraction
    )
