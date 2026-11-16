"""
Training History Visualization Module.
Plots:
1. Training Loss vs Validation Loss (Stage 1 and Stage 2)
2. Training Accuracy vs Validation Accuracy (Stage 1 and Stage 2)
3. Learning Rate Progression (incorporating ReduceLROnPlateau drops)
4. Multi-metric Stage 1 vs Stage 2 Comprehensive Comparison Panel
Saves publication-quality figures to results/training_plots/.
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    TRAINING_PLOTS_DIR, HISTORY_PATH, CHECKPOINTS_DIR, init_directories
)
from src.utils import setup_logger, load_json

logger = setup_logger("TrainingPlots", "training_plots.log")

def plot_training_history(history_data: dict = None):
    """Generates all training visual plots from history dictionary."""
    init_directories()
    logger.info("Generating training trajectory plots...")
    
    if history_data is None:
        if HISTORY_PATH.exists():
            history_data = load_json(HISTORY_PATH)
        elif (CHECKPOINTS_DIR / "stage1_history.json").exists():
            s1 = load_json(CHECKPOINTS_DIR / "stage1_history.json")
            s2 = load_json(CHECKPOINTS_DIR / "stage2_history.json") if (CHECKPOINTS_DIR / "stage2_history.json").exists() else {}
            history_data = {"stage1": s1, "stage2": s2}
        else:
            logger.warning(f"No history file found at {HISTORY_PATH}. Cannot render plots.")
            return
            
    s1 = history_data.get("stage1", {})
    s2 = history_data.get("stage2", {})
    
    # 1. Stage 1 Accuracy & Loss Plot
    if s1 and "loss" in s1:
        epochs_s1 = range(1, len(s1["loss"]) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.plot(epochs_s1, s1.get("accuracy", []), 'o-', label="Training Accuracy", color="#2ca02c", linewidth=2)
        ax1.plot(epochs_s1, s1.get("val_accuracy", []), 's--', label="Validation Accuracy", color="#1f77b4", linewidth=2)
        ax1.set_title("Stage 1 (Feature Extraction): Accuracy", fontsize=12, fontweight='bold')
        ax1.set_xlabel("Epoch", fontsize=11)
        ax1.set_ylabel("Accuracy", fontsize=11)
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(epochs_s1, s1.get("loss", []), 'o-', label="Training Loss", color="#d62728", linewidth=2)
        ax2.plot(epochs_s1, s1.get("val_loss", []), 's--', label="Validation Loss", color="#ff7f0e", linewidth=2)
        ax2.set_title("Stage 1 (Feature Extraction): Loss", fontsize=12, fontweight='bold')
        ax2.set_xlabel("Epoch", fontsize=11)
        ax2.set_ylabel("Loss", fontsize=11)
        ax2.legend(loc="upper right")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        s1_path = TRAINING_PLOTS_DIR / "stage1_performance.png"
        plt.savefig(s1_path, dpi=300)
        plt.close()
        logger.info(f"Saved: {s1_path}")
        
    # 2. Stage 2 Accuracy & Loss Plot
    if s2 and "loss" in s2:
        epochs_s2 = range(1, len(s2["loss"]) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.plot(epochs_s2, s2.get("accuracy", []), 'o-', label="Training Accuracy", color="#2ca02c", linewidth=2)
        ax1.plot(epochs_s2, s2.get("val_accuracy", []), 's--', label="Validation Accuracy", color="#1f77b4", linewidth=2)
        ax1.set_title("Stage 2 (Fine-Tuning): Accuracy", fontsize=12, fontweight='bold')
        ax1.set_xlabel("Epoch", fontsize=11)
        ax1.set_ylabel("Accuracy", fontsize=11)
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(epochs_s2, s2.get("loss", []), 'o-', label="Training Loss", color="#d62728", linewidth=2)
        ax2.plot(epochs_s2, s2.get("val_loss", []), 's--', label="Validation Loss", color="#ff7f0e", linewidth=2)
        ax2.set_title("Stage 2 (Fine-Tuning): Loss", fontsize=12, fontweight='bold')
        ax2.set_xlabel("Epoch", fontsize=11)
        ax2.set_ylabel("Loss", fontsize=11)
        ax2.legend(loc="upper right")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        s2_path = TRAINING_PLOTS_DIR / "stage2_performance.png"
        plt.savefig(s2_path, dpi=300)
        plt.close()
        logger.info(f"Saved: {s2_path}")
        
    # 3. Overall Combined Trajectory (Loss & Accuracy across all epochs)
    comb_train_acc = s1.get("accuracy", []) + s2.get("accuracy", [])
    comb_val_acc = s1.get("val_accuracy", []) + s2.get("val_accuracy", [])
    comb_train_loss = s1.get("loss", []) + s2.get("loss", [])
    comb_val_loss = s1.get("val_loss", []) + s2.get("val_loss", [])
    
    if comb_train_acc:
        total_epochs = range(1, len(comb_train_acc) + 1)
        split_point = len(s1.get("accuracy", []))
        
        # Accuracy Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(total_epochs, comb_train_acc, 'b-', label='Training Accuracy', linewidth=2)
        ax.plot(total_epochs, comb_val_acc, 'g--', label='Validation Accuracy', linewidth=2)
        if split_point > 0 and len(s2.get("accuracy", [])) > 0:
            ax.axvline(x=split_point + 0.5, color='gray', linestyle=':', label='Fine-Tuning Transition')
        ax.set_title("Training vs Validation Accuracy (Two-Stage Transfer Learning)", fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.legend(loc="lower right", fontsize=10)
        plt.tight_layout()
        acc_path = TRAINING_PLOTS_DIR / "accuracy_vs_epoch.png"
        plt.savefig(acc_path, dpi=300)
        plt.close()
        logger.info(f"Saved: {acc_path}")
        
        # Loss Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(total_epochs, comb_train_loss, 'r-', label='Training Loss', linewidth=2)
        ax.plot(total_epochs, comb_val_loss, 'orange', linestyle='--', label='Validation Loss', linewidth=2)
        if split_point > 0 and len(s2.get("loss", [])) > 0:
            ax.axvline(x=split_point + 0.5, color='gray', linestyle=':', label='Fine-Tuning Transition')
        ax.set_title("Training vs Validation Loss (Two-Stage Transfer Learning)", fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Categorical Crossentropy Loss", fontsize=11)
        ax.legend(loc="upper right", fontsize=10)
        plt.tight_layout()
        loss_path = TRAINING_PLOTS_DIR / "loss_vs_epoch.png"
        plt.savefig(loss_path, dpi=300)
        plt.close()
        logger.info(f"Saved: {loss_path}")
        
    # 4. Learning Rate Schedule Progression
    lrs = s1.get("learning_rate", s1.get("lr", [])) + s2.get("learning_rate", s2.get("lr", []))
    if lrs:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(1, len(lrs) + 1), lrs, 'm-o', linewidth=2)
        ax.set_yscale('log')
        ax.set_title("Learning Rate Schedule Progression (ReduceLROnPlateau)", fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Learning Rate (Log Scale)", fontsize=11)
        plt.tight_layout()
        lr_path = TRAINING_PLOTS_DIR / "learning_rate_schedule.png"
        plt.savefig(lr_path, dpi=300)
        plt.close()
        logger.info(f"Saved: {lr_path}")

if __name__ == "__main__":
    plot_training_history()
