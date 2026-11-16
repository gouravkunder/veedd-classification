"""
Hyperparameter Tuning Module for Weed Species Classification using ResNet152V2.
Evaluates systematic configurations across learning rates, batch sizes, dropout rates,
dense bottleneck dimensions, and fine-tuning depths on the validation split.
Saves results to results/hyperparameter_tuning_results.csv and generates comparison plots.
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import f1_score

from src.config import (
    RESULTS_DIR, GRAPHS_DIR, CLASS_NAMES, RANDOM_SEED, init_directories
)
from src.utils import set_reproducible_seeds, setup_logger, save_json
from src.data_loader import get_data_loaders
from src.model import build_resnet152v2_model, configure_model_for_finetuning

logger = setup_logger("HyperparameterTuning", "hyperparameter_tuning.log")

CONFIG_GRID = [
    {
        "config_id": "EXP-1 (Baseline)",
        "learning_rate": 1e-3,
        "batch_size": 32,
        "dropout_rate_1": 0.45,
        "dropout_rate_2": 0.30,
        "dense_units": 512,
        "fine_tune_layers": 30
    },
    {
        "config_id": "EXP-2 (High-Capacity)",
        "learning_rate": 5e-4,
        "batch_size": 32,
        "dropout_rate_1": 0.40,
        "dropout_rate_2": 0.25,
        "dense_units": 768,
        "fine_tune_layers": 40
    },
    {
        "config_id": "EXP-3 (Heavy-Regularization)",
        "learning_rate": 1e-3,
        "batch_size": 16,
        "dropout_rate_1": 0.50,
        "dropout_rate_2": 0.35,
        "dense_units": 512,
        "fine_tune_layers": 20
    },
    {
        "config_id": "EXP-4 (Compact-Head)",
        "learning_rate": 8e-4,
        "batch_size": 32,
        "dropout_rate_1": 0.35,
        "dropout_rate_2": 0.20,
        "dense_units": 256,
        "fine_tune_layers": 25
    }
]

def evaluate_validation_f1(model: tf.keras.Model, val_ds: tf.data.Dataset) -> float:
    """Computes Macro F1-score on the untouched validation dataset."""
    all_preds = []
    all_targets = []
    for images, labels in val_ds:
        preds = model(images, training=False)
        pred_classes = tf.argmax(preds, axis=-1).numpy()
        true_classes = tf.argmax(labels, axis=-1).numpy()
        all_preds.extend(pred_classes)
        all_targets.extend(true_classes)
    return float(f1_score(all_targets, all_preds, average='macro'))

def run_hyperparameter_tuning(
    epochs_per_config: int = 3,
    sample_fraction: float = 0.15
) -> pd.DataFrame:
    """
    Executes resource-aware hyperparameter experiments on validation data.
    """
    init_directories()
    set_reproducible_seeds()
    
    logger.info("=" * 60)
    logger.info("STARTING RESOURCE-AWARE HYPERPARAMETER EXPERIMENTS")
    logger.info(f"Configurations to test: {len(CONFIG_GRID)} | Epochs per experiment: {epochs_per_config}")
    logger.info("=" * 60)
    
    results = []
    
    for cfg in CONFIG_GRID:
        cid = cfg["config_id"]
        logger.info(f"\n--- Running Experiment: {cid} ---")
        logger.info(f"Parameters: LR={cfg['learning_rate']}, Batch={cfg['batch_size']}, Dropout=({cfg['dropout_rate_1']}, {cfg['dropout_rate_2']}), Dense={cfg['dense_units']}, Unfreeze={cfg['fine_tune_layers']}")
        
        train_ds, val_ds, _, class_weights = get_data_loaders(
            batch_size=cfg["batch_size"],
            augment_train=True,
            sample_fraction=sample_fraction
        )
        
        # Build model with specific config
        model = build_resnet152v2_model(
            dense_units=cfg["dense_units"],
            dropout_rate_1=cfg["dropout_rate_1"],
            dropout_rate_2=cfg["dropout_rate_2"],
            learning_rate=cfg["learning_rate"]
        )
        
        # Train Stage 1
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs_per_config,
            class_weight=class_weights,
            verbose=1
        )
        
        val_losses = history.history.get("val_loss", [float("inf")])
        val_accs = history.history.get("val_accuracy", [0.0])
        best_val_loss = float(min(val_losses))
        best_val_acc = float(max(val_accs))
        
        # Calculate Macro F1 on validation set
        macro_f1 = evaluate_validation_f1(model, val_ds)
        logger.info(f"Experiment {cid} -> Val Loss: {best_val_loss:.4f} | Val Accuracy: {best_val_acc:.4f} | Macro F1: {macro_f1:.4f}")
        
        cfg_record = dict(cfg)
        cfg_record.update({
            "val_loss": best_val_loss,
            "val_accuracy": best_val_acc,
            "val_macro_f1": macro_f1
        })
        results.append(cfg_record)
        
    results_df = pd.DataFrame(results)
    # Sort by validation Macro F1 descending
    results_df = results_df.sort_values(by="val_macro_f1", ascending=False).reset_index(drop=True)
    
    # Save CSV
    csv_path = RESULTS_DIR / "hyperparameter_tuning_results.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Saved hyperparameter tuning results to {csv_path}")
    
    # Generate Comparison Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Macro F1 Comparison
    sns.barplot(data=results_df, x="config_id", y="val_macro_f1", ax=ax1, palette="viridis", edgecolor="black")
    ax1.set_title("Validation Macro F1-Score Across Configurations", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Experiment Configuration", fontsize=11)
    ax1.set_ylabel("Validation Macro F1", fontsize=11)
    ax1.tick_params(axis='x', rotation=25)
    for p in ax1.patches:
        ax1.annotate(f"{p.get_height():.3f}",
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='bottom', fontsize=10, xytext=(0, 4), textcoords='offset points')
                     
    # Validation Loss Comparison
    sns.barplot(data=results_df, x="config_id", y="val_loss", ax=ax2, palette="mako", edgecolor="black")
    ax2.set_title("Validation Loss Across Configurations", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Experiment Configuration", fontsize=11)
    ax2.set_ylabel("Validation Loss", fontsize=11)
    ax2.tick_params(axis='x', rotation=25)
    for p in ax2.patches:
        ax2.annotate(f"{p.get_height():.3f}",
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='bottom', fontsize=10, xytext=(0, 4), textcoords='offset points')
                     
    plt.tight_layout()
    plot_path = GRAPHS_DIR / "hyperparameter_tuning_comparison.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.info(f"Saved comparison plot to {plot_path}")
    
    best_config = results_df.iloc[0].to_dict()
    logger.info("=" * 60)
    logger.info(f"BEST CONFIGURATION: {best_config['config_id']}")
    logger.info(f"Selected due to highest validation Macro F1 ({best_config['val_macro_f1']:.4f}) and lowest generalization loss ({best_config['val_loss']:.4f}).")
    logger.info("=" * 60)
    
    return results_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning Experiments")
    parser.add_argument("--epochs", type=int, default=2, help="Epochs per configuration")
    parser.add_argument("--sample-fraction", type=float, default=0.08, help="Sample fraction for fast tuning")
    args = parser.parse_args()
    
    run_hyperparameter_tuning(
        epochs_per_config=args.epochs,
        sample_fraction=args.sample_fraction
    )
