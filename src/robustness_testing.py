"""
Robustness Testing Module for Weed Species Classification using ResNet152V2.
Tests performance stability under real-world agricultural conditions:
- Solar illumination shifts (Low Brightness 0.7x, High Brightness 1.3x)
- Dynamic contrast shifts (Low Contrast 0.7x, High Contrast 1.3x)
- Camera focus / motion blur (Gaussian blur sigma 1.0, 2.0)
- Environmental sensor noise (Gaussian noise std 0.05, 0.10)
- Resolution variation (Downscale 192x192 upscaled to 224x224)
Exports results to results/evaluation_metrics/ and comparison plots to results/graphs/.
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
from PIL import Image, ImageFilter, ImageEnhance
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score

from src.config import (
    FINAL_MODEL_PATH, BEST_MODEL_PATH, CLASS_NAMES_PATH,
    TEST_CSV, IMAGES_DIR, RESULTS_DIR, EVAL_METRICS_DIR, GRAPHS_DIR,
    CLASS_NAMES, IMAGE_HEIGHT, IMAGE_WIDTH, RANDOM_SEED, init_directories
)
from src.utils import setup_logger, save_json, load_json
from src.data_preprocessing import preprocess_image_tensor

logger = setup_logger("RobustnessTesting", "robustness_testing.log")

def apply_perturbation(img_pil: Image.Image, perturbation_type: str) -> Image.Image:
    """Applies specific synthetic environmental perturbation to a PIL image."""
    if perturbation_type == "baseline":
        return img_pil
    elif perturbation_type == "low_brightness":
        enhancer = ImageEnhance.Brightness(img_pil)
        return enhancer.enhance(0.7)
    elif perturbation_type == "high_brightness":
        enhancer = ImageEnhance.Brightness(img_pil)
        return enhancer.enhance(1.3)
    elif perturbation_type == "low_contrast":
        enhancer = ImageEnhance.Contrast(img_pil)
        return enhancer.enhance(0.7)
    elif perturbation_type == "high_contrast":
        enhancer = ImageEnhance.Contrast(img_pil)
        return enhancer.enhance(1.3)
    elif perturbation_type == "mild_blur":
        return img_pil.filter(ImageFilter.GaussianBlur(radius=1.0))
    elif perturbation_type == "moderate_blur":
        return img_pil.filter(ImageFilter.GaussianBlur(radius=2.0))
    elif perturbation_type == "mild_noise":
        arr = np.array(img_pil, dtype=np.float32)
        noise = np.random.normal(0, 0.05 * 255, arr.shape)
        noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)
    elif perturbation_type == "moderate_noise":
        arr = np.array(img_pil, dtype=np.float32)
        noise = np.random.normal(0, 0.10 * 255, arr.shape)
        noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)
    elif perturbation_type == "resizing_variation":
        # Downscale to 192x192 then upscale back to test interpolation resilience
        downscaled = img_pil.resize((192, 192), Image.Resampling.BILINEAR)
        return downscaled.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.Resampling.BILINEAR)
    else:
        return img_pil

def run_robustness_evaluation(
    model_path: Path = None,
    sample_size: int = 500
) -> pd.DataFrame:
    """
    Evaluates model resilience under 10 environmental perturbation regimes.
    """
    init_directories()
    logger.info("=" * 60)
    logger.info("STARTING AGRICULTURAL ROBUSTNESS & STABILITY TESTING")
    logger.info(f"Evaluation sample size: {sample_size} images per perturbation")
    logger.info("=" * 60)
    
    if model_path is None:
        if FINAL_MODEL_PATH.exists():
            model_path = FINAL_MODEL_PATH
        elif BEST_MODEL_PATH.exists():
            model_path = BEST_MODEL_PATH
        else:
            raise FileNotFoundError("Trained model not found for robustness testing.")
            
    model = tf.keras.models.load_model(model_path)
    test_df = pd.read_csv(TEST_CSV)
    eval_df = test_df.sample(min(sample_size, len(test_df)), random_state=RANDOM_SEED).reset_index(drop=True)
    
    perturbations = [
        ("baseline", "Baseline (Untouched)"),
        ("low_brightness", "Low Brightness (0.7x)"),
        ("high_brightness", "High Brightness (1.3x)"),
        ("low_contrast", "Low Contrast (0.7x)"),
        ("high_contrast", "High Contrast (1.3x)"),
        ("mild_blur", "Mild Blur (Sigma=1.0)"),
        ("moderate_blur", "Moderate Blur (Sigma=2.0)"),
        ("mild_noise", "Mild Noise (Std=0.05)"),
        ("moderate_noise", "Moderate Noise (Std=0.10)"),
        ("resizing_variation", "Resizing Shift (192->224)")
    ]
    
    results = []
    baseline_acc = None
    baseline_f1 = None
    
    for p_key, p_label in perturbations:
        logger.info(f"Evaluating Perturbation: {p_label}...")
        y_true = []
        y_pred = []
        
        # Batch process samples
        batch_tensors = []
        batch_labels = []
        
        for _, row in eval_df.iterrows():
            img_path = IMAGES_DIR / row["Filename"]
            try:
                with Image.open(img_path) as img:
                    img_rgb = img.convert("RGB")
                    perturbed = apply_perturbation(img_rgb, p_key)
                    resized = perturbed.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.Resampling.BILINEAR)
                    arr = np.array(resized, dtype=np.float32)
                    batch_tensors.append(arr)
                    batch_labels.append(row["Label"])
            except Exception as e:
                continue
                
            if len(batch_tensors) >= 32:
                t = preprocess_image_tensor(tf.convert_to_tensor(np.array(batch_tensors)))
                preds = model(t, training=False).numpy()
                y_pred.extend(np.argmax(preds, axis=1))
                y_true.extend(batch_labels)
                batch_tensors = []
                batch_labels = []
                
        if batch_tensors:
            t = preprocess_image_tensor(tf.convert_to_tensor(np.array(batch_tensors)))
            preds = model(t, training=False).numpy()
            y_pred.extend(np.argmax(preds, axis=1))
            y_true.extend(batch_labels)
            
        acc = float(accuracy_score(y_true, y_pred))
        f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        
        if p_key == "baseline":
            baseline_acc = acc
            baseline_f1 = f1
            rel_drop = 0.0
        else:
            rel_drop = float(((baseline_acc - acc) / baseline_acc) * 100) if baseline_acc else 0.0
            
        logger.info(f"  -> Accuracy: {acc*100:.2f}% | Macro F1: {f1:.4f} | Accuracy Drop: {rel_drop:.2f}%")
        results.append({
            "perturbation_key": p_key,
            "perturbation_label": p_label,
            "accuracy": acc,
            "macro_f1": f1,
            "relative_acc_degradation_pct": rel_drop
        })
        
    results_df = pd.DataFrame(results)
    
    # Save CSV and JSON
    csv_out = EVAL_METRICS_DIR / "robustness_testing_report.csv"
    json_out = EVAL_METRICS_DIR / "robustness_testing_report.json"
    results_df.to_csv(csv_out, index=False)
    save_json(results_df.to_dict(orient="records"), json_out)
    logger.info(f"Saved robustness report to {csv_out}")
    
    # Generate Comparison Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Accuracy Plot
    sns.barplot(data=results_df, x="perturbation_label", y="accuracy", ax=ax1, palette="crest", edgecolor="black")
    ax1.set_title("Accuracy Under Image Perturbations", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Perturbation Regime", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.tick_params(axis='x', rotation=40)
    for p in ax1.patches:
        ax1.annotate(f"{p.get_height()*100:.1f}%",
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points', fontweight='bold')
                     
    # Degradation Plot
    non_baseline_df = results_df[results_df["perturbation_key"] != "baseline"]
    sns.barplot(data=non_baseline_df, x="perturbation_label", y="relative_acc_degradation_pct", ax=ax2, palette="flare", edgecolor="black")
    ax2.set_title("Relative Accuracy Degradation (%) vs Baseline", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Perturbation Regime", fontsize=11)
    ax2.set_ylabel("Accuracy Drop (%)", fontsize=11)
    ax2.tick_params(axis='x', rotation=40)
    for p in ax2.patches:
        ax2.annotate(f"{p.get_height():.1f}%",
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points', fontweight='bold')
                     
    plt.tight_layout()
    plot_path = GRAPHS_DIR / "robustness_evaluation_comparison.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.info(f"Saved robustness comparison plot to {plot_path}")
    logger.info("Robustness testing complete.")
    return results_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robustness Testing")
    parser.add_argument("--samples", type=int, default=300, help="Number of test samples per perturbation")
    args = parser.parse_args()
    
    run_robustness_evaluation(sample_size=args.samples)
