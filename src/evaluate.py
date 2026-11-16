"""
Comprehensive Model Evaluation Module on Untouched Test Set.
Computes:
- Test Accuracy, Precision, Recall, F1-Score (Macro and Weighted)
- Per-class classification report and confusion matrix
- Multi-class One-vs-Rest ROC curves and Precision-Recall curves
- Sample galleries of correctly classified and misclassified instances
Exports all results to results/ and models/evaluation_metrics.json.
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
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import tensorflow as tf
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)

from src.config import (
    FINAL_MODEL_PATH, BEST_MODEL_PATH, CLASS_NAMES_PATH,
    TEST_CSV, IMAGES_DIR, RESULTS_DIR, CONFUSION_MATRIX_DIR,
    CLASSIFICATION_REPORT_DIR, EVAL_METRICS_DIR, GRAPHS_DIR,
    EVALUATION_METRICS_PATH, CLASS_NAMES, NUM_CLASSES, RANDOM_SEED,
    IMAGE_HEIGHT, IMAGE_WIDTH, init_directories
)
from src.utils import setup_logger, save_json, load_json
from src.data_loader import get_data_loaders, create_tf_dataset
from src.data_preprocessing import preprocess_image_tensor

logger = setup_logger("ModelEvaluation", "model_evaluation.log")

def evaluate_test_set(
    model_path: Path = None,
    batch_size: int = 32,
    sample_fraction: float = None
) -> dict:
    """
    Evaluates the final model against the untouched test split.
    """
    init_directories()
    logger.info("=" * 60)
    logger.info("STARTING UNTOUCHED TEST SET COMPREHENSIVE EVALUATION")
    logger.info("=" * 60)
    
    if model_path is None or not Path(model_path).exists():
        if FINAL_MODEL_PATH.exists():
            model_path = FINAL_MODEL_PATH
        elif BEST_MODEL_PATH.exists():
            model_path = BEST_MODEL_PATH
        else:
            raise FileNotFoundError(
                f"No trained model found at {FINAL_MODEL_PATH} or {BEST_MODEL_PATH}. "
                "Please run train.py or fine_tuning.py first."
            )
            
    logger.info(f"Loading trained model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # Load class names
    if CLASS_NAMES_PATH.exists():
        class_names = load_json(CLASS_NAMES_PATH)
    else:
        class_names = CLASS_NAMES
        
    # Load test split
    if not TEST_CSV.exists():
        raise FileNotFoundError(f"Test split CSV not found at {TEST_CSV}")
        
    test_df = pd.read_csv(TEST_CSV)
    if sample_fraction is not None and 0.0 < sample_fraction < 1.0:
        logger.info(f"Evaluating on stratified sample fraction: {sample_fraction:.2f}")
        test_df = test_df.groupby("Label", group_keys=False).apply(
            lambda x: x.sample(frac=sample_fraction, random_state=RANDOM_SEED)
        ).reset_index(drop=True)
        
    logger.info(f"Loaded untouched test set with {len(test_df)} samples.")
    
    test_ds = create_tf_dataset(test_df, is_training=False, batch_size=batch_size, shuffle=False, augment=False)
    
    # Predict probabilities across test set
    logger.info("Generating predictions across all test images...")
    y_prob_list = []
    y_true_list = []
    
    for images, labels in test_ds:
        preds = model(images, training=False).numpy()
        y_prob_list.append(preds)
        y_true_list.append(labels.numpy())
        
    y_probs = np.vstack(y_prob_list)
    y_true_onehot = np.vstack(y_true_list)
    y_true = np.argmax(y_true_onehot, axis=1)
    y_pred = np.argmax(y_probs, axis=1)
    
    # 1. Overall Metrics
    test_acc = float(accuracy_score(y_true, y_pred))
    macro_prec = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
    weighted_prec = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
    weighted_rec = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    
    # 2. Multi-class ROC-AUC (One-vs-Rest)
    try:
        roc_auc_macro = float(roc_auc_score(y_true_onehot, y_probs, multi_class='ovr', average='macro'))
        roc_auc_weighted = float(roc_auc_score(y_true_onehot, y_probs, multi_class='ovr', average='weighted'))
    except Exception as e:
        logger.warning(f"Could not compute ROC-AUC: {e}")
        roc_auc_macro = 0.0
        roc_auc_weighted = 0.0
        
    logger.info(f"Test Accuracy:       {test_acc * 100:.2f}%")
    logger.info(f"Macro Precision:     {macro_prec:.4f} | Weighted Precision: {weighted_prec:.4f}")
    logger.info(f"Macro Recall:        {macro_rec:.4f}  | Weighted Recall:    {weighted_rec:.4f}")
    logger.info(f"Macro F1-Score:      {macro_f1:.4f}   | Weighted F1-Score:  {weighted_f1:.4f}")
    logger.info(f"Macro ROC-AUC (OvR): {roc_auc_macro:.4f}")
    
    # 3. Per-Class Report
    report_dict = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    report_text = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
    
    # Save classification report text and CSV
    with open(CLASSIFICATION_REPORT_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(CLASSIFICATION_REPORT_DIR / "classification_report.csv")
    logger.info(f"Saved classification report to {CLASSIFICATION_REPORT_DIR}")
    
    # 4. Confusion Matrix (Raw & Normalized Heatmaps)
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title("Confusion Matrix (Absolute Counts)", fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlabel("Predicted Label", fontsize=11)
    ax1.set_ylabel("True Label", fontsize=11)
    ax1.tick_params(axis='x', rotation=30)
    
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens', xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_title("Normalized Confusion Matrix (Recall %)", fontsize=13, fontweight='bold', pad=10)
    ax2.set_xlabel("Predicted Label", fontsize=11)
    ax2.set_ylabel("True Label", fontsize=11)
    ax2.tick_params(axis='x', rotation=30)
    
    plt.tight_layout()
    cm_path = CONFUSION_MATRIX_DIR / "confusion_matrix_heatmap.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrix heatmap to {cm_path}")
    
    # 5. Per-Class F1-Score Bar Chart
    per_class_f1 = [report_dict[c]['f1-score'] for c in class_names]
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(class_names, per_class_f1, color=sns.color_palette("tab10", n_colors=9), edgecolor='black', alpha=0.85)
    ax.set_title("Per-Class F1-Score on Untouched Test Set", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Weed Species", fontsize=12)
    ax.set_ylabel("F1-Score", fontsize=12)
    ax.set_ylim(0, 1.15)
    plt.xticks(rotation=30, ha='right', fontsize=11)
    for bar in bars:
        ax.annotate(f"{bar.get_height():.3f}",
                    (bar.get_x() + bar.get_width() / 2., bar.get_height()),
                    ha='center', va='bottom', fontsize=10, xytext=(0, 4), textcoords='offset points', fontweight='bold')
    plt.tight_layout()
    f1_plot_path = GRAPHS_DIR / "per_class_f1_score.png"
    plt.savefig(f1_plot_path, dpi=300)
    plt.close()
    logger.info(f"Saved per-class F1 chart to {f1_plot_path}")
    
    # 6. ROC Curves (One-vs-Rest)
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, cname in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_probs[:, i])
        c_auc = roc_auc_score(y_true_onehot[:, i], y_probs[:, i])
        ax.plot(fpr, tpr, label=f"{cname} (AUC = {c_auc:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.500)')
    ax.set_title("Multi-Class One-vs-Rest ROC Curves", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    roc_plot_path = GRAPHS_DIR / "roc_curves.png"
    plt.savefig(roc_plot_path, dpi=300)
    plt.close()
    logger.info(f"Saved ROC curves to {roc_plot_path}")
    
    # 7. Precision-Recall Curves
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, cname in enumerate(class_names):
        prec_c, rec_c, _ = precision_recall_curve(y_true_onehot[:, i], y_probs[:, i])
        ap = average_precision_score(y_true_onehot[:, i], y_probs[:, i])
        ax.plot(rec_c, prec_c, label=f"{cname} (AP = {ap:.3f})", linewidth=2)
    ax.set_title("Multi-Class Precision-Recall Curves", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    pr_plot_path = GRAPHS_DIR / "precision_recall_curves.png"
    plt.savefig(pr_plot_path, dpi=300)
    plt.close()
    logger.info(f"Saved Precision-Recall curves to {pr_plot_path}")
    
    # 8. Correctly Classified vs Misclassified Sample Galleries
    correct_indices = np.where(y_true == y_pred)[0]
    misclassified_indices = np.where(y_true != y_pred)[0]
    
    logger.info(f"Test samples breakdown: Correct={len(correct_indices)}, Misclassified={len(misclassified_indices)}")
    
    # Gallery helper
    def plot_sample_gallery(indices, title, output_path, max_samples=12):
        if len(indices) == 0:
            return
        selected = np.random.choice(indices, size=min(max_samples, len(indices)), replace=False)
        cols = 4
        rows = int(np.ceil(len(selected) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        axes = np.array(axes).reshape(-1)
        
        for idx_pos, sample_idx in enumerate(selected):
            ax = axes[idx_pos]
            fname = test_df.iloc[sample_idx]["Filename"]
            true_label = class_names[y_true[sample_idx]]
            pred_label = class_names[y_pred[sample_idx]]
            conf = y_probs[sample_idx, y_pred[sample_idx]] * 100
            
            img_path = IMAGES_DIR / fname
            try:
                img = Image.open(img_path)
                ax.imshow(img)
            except Exception as e:
                ax.text(0.5, 0.5, "Image Error", ha='center')
                
            ax.set_xticks([])
            ax.set_yticks([])
            color = "green" if true_label == pred_label else "red"
            ax.set_title(f"True: {true_label}\nPred: {pred_label} ({conf:.1f}%)",
                         fontsize=10, fontweight='bold', color=color)
                         
        for unused_ax in axes[len(selected):]:
            unused_ax.axis('off')
            
        plt.suptitle(title, fontsize=15, fontweight='bold', y=0.99)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        logger.info(f"Saved sample gallery to {output_path}")

    plot_sample_gallery(
        correct_indices,
        "Representative Correctly Classified Test Samples",
        EVAL_METRICS_DIR / "correctly_classified_samples.png"
    )
    plot_sample_gallery(
        misclassified_indices,
        "Representative Misclassified Test Samples (Challenging Rangeland Cases)",
        EVAL_METRICS_DIR / "misclassified_samples.png"
    )
    
    # Save Consolidated Evaluation Metrics JSON
    metrics_summary = {
        "model_path": str(model_path.name),
        "total_test_samples": len(test_df),
        "test_accuracy": test_acc,
        "macro_precision": macro_prec,
        "weighted_precision": weighted_prec,
        "macro_recall": macro_rec,
        "weighted_recall": weighted_rec,
        "macro_f1_score": macro_f1,
        "weighted_f1_score": weighted_f1,
        "roc_auc_macro_ovr": roc_auc_macro,
        "roc_auc_weighted_ovr": roc_auc_weighted,
        "per_class_metrics": {
            c: {
                "precision": report_dict[c]["precision"],
                "recall": report_dict[c]["recall"],
                "f1_score": report_dict[c]["f1-score"],
                "support": int(report_dict[c]["support"])
            } for c in class_names
        }
    }
    
    save_json(metrics_summary, EVAL_METRICS_DIR / "evaluation_metrics.json")
    save_json(metrics_summary, EVALUATION_METRICS_PATH)
    logger.info(f"Saved evaluation metrics to {EVALUATION_METRICS_PATH}")
    logger.info("Evaluation pipeline execution complete.")
    return metrics_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ResNet152V2 Weed Classifier")
    parser.add_argument("--model-path", type=str, default=None, help="Path to .keras model")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--sample-fraction", type=float, default=None, help="Fraction of test set (0.0 - 1.0) for evaluation")
    args = parser.parse_args()
    
    evaluate_test_set(
        model_path=Path(args.model_path) if args.model_path else None,
        batch_size=args.batch_size,
        sample_fraction=args.sample_fraction
    )
