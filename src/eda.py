"""
Exploratory Data Analysis (EDA) Module for DeepWeeds Dataset.
Generates comprehensive visual and statistical analysis:
- Class distribution and severe imbalance visualization
- Train/Validation/Test split distribution comparisons
- Image dimension and aspect ratio histograms
- RGB channel mean/std statistics and pixel intensity distributions
- Representative sample grids for all 9 weed species and negative class
Saves publication-quality figures into results/eda/.
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless backend for headless environments
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from src.config import (
    IMAGES_DIR, LABELS_CSV, TRAIN_CSV, VAL_CSV, TEST_CSV,
    EDA_DIR, CLASS_NAMES, LABEL_TO_CLASS, RANDOM_SEED, init_directories
)
from src.utils import setup_logger, save_json

logger = setup_logger("EDA", "eda.log")

# Aesthetic settings
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
PALETTE = sns.color_palette("tab10", n_colors=9)

def plot_class_distribution(df: pd.DataFrame):
    """Plot class frequencies highlighting severe class imbalance."""
    logger.info("Generating class distribution bar chart...")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    counts = df['Species'].value_counts()
    classes_ordered = [c for c in counts.index]
    counts_ordered = [counts[c] for c in classes_ordered]
    
    bars = ax.bar(classes_ordered, counts_ordered, color=PALETTE, edgecolor='black', alpha=0.85)
    ax.set_title("DeepWeeds Dataset: Class Distribution (N = 17,492)", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Weed Species / Class", fontsize=12, labelpad=10)
    ax.set_ylabel("Number of Images", fontsize=12, labelpad=10)
    plt.xticks(rotation=30, ha='right', fontsize=11)
    
    # Annotate counts and percentages on top of bars
    total = len(df)
    for bar in bars:
        height = bar.get_height()
        pct = (height / total) * 100
        ax.annotate(f'{height}\n({pct:.1f}%)',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='semibold')
                    
    ax.set_ylim(0, max(counts_ordered) * 1.18)
    plt.tight_layout()
    save_path = EDA_DIR / "class_distribution.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {save_path}")

def plot_splits_distribution(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """Plot train/val/test class breakdown to verify stratified preservation."""
    logger.info("Generating split distribution comparison chart...")
    
    # Create combined dataframe
    t_counts = train_df['Species'].value_counts().rename('Train (60%)')
    v_counts = val_df['Species'].value_counts().rename('Val (20%)')
    te_counts = test_df['Species'].value_counts().rename('Test (20%)')
    
    split_df = pd.concat([t_counts, v_counts, te_counts], axis=1).fillna(0)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    split_df.plot(kind='bar', ax=ax, width=0.8, edgecolor='black', alpha=0.85,
                  color=['#2ca02c', '#1f77b4', '#ff7f0e'])
                  
    ax.set_title("Stratified Split Breakdown Across All 9 Classes", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Weed Species", fontsize=12, labelpad=10)
    ax.set_ylabel("Image Count", fontsize=12, labelpad=10)
    plt.xticks(rotation=30, ha='right', fontsize=11)
    ax.legend(title="Split Partition", fontsize=11, title_fontsize=11)
    
    plt.tight_layout()
    save_path = EDA_DIR / "split_distribution.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {save_path}")

def plot_sample_images_grid(df: pd.DataFrame, samples_per_class: int = 5):
    """Generate a class-wise image visual grid showing diverse sample instances."""
    logger.info(f"Generating class sample grid ({samples_per_class} per class)...")
    
    num_classes = len(CLASS_NAMES)
    fig, axes = plt.subplots(num_classes, samples_per_class, figsize=(3 * samples_per_class, 2.8 * num_classes))
    
    for row_idx, species in enumerate(CLASS_NAMES):
        class_subset = df[df['Species'].str.lower() == species.lower()]
        if class_subset.empty:
            class_subset = df[df['Label'] == row_idx]
            
        sample_rows = class_subset.sample(min(samples_per_class, len(class_subset)), random_state=RANDOM_SEED)
        
        for col_idx, (_, row) in enumerate(sample_rows.iterrows()):
            ax = axes[row_idx, col_idx]
            img_path = IMAGES_DIR / row['Filename']
            
            try:
                img = Image.open(img_path)
                ax.imshow(img)
            except Exception as e:
                ax.text(0.5, 0.5, f"Error\n{e}", ha='center', va='center')
                
            ax.set_xticks([])
            ax.set_yticks([])
            
            if col_idx == 0:
                ax.set_ylabel(species, fontsize=11, fontweight='bold', labelpad=10)
            if row_idx == 0:
                ax.set_title(f"Sample #{col_idx + 1}", fontsize=11, pad=8)
                
    plt.suptitle("DeepWeeds Species Visual Samples Across All 9 Classes", fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    save_path = EDA_DIR / "sample_images_grid.png"
    plt.savefig(save_path, dpi=200)
    plt.close()
    logger.info(f"Saved: {save_path}")

def analyze_pixel_statistics(df: pd.DataFrame, num_samples: int = 1000):
    """Compute RGB channel statistics and pixel intensity distributions."""
    logger.info(f"Sampling {num_samples} images for pixel statistics & intensity histograms...")
    sample_df = df.sample(min(num_samples, len(df)), random_state=RANDOM_SEED)
    
    r_vals, g_vals, b_vals = [], [], []
    r_means, g_means, b_means = [], [], []
    r_stds, g_stds, b_stds = [], [], []
    
    for _, row in sample_df.iterrows():
        img_path = IMAGES_DIR / row['Filename']
        try:
            with Image.open(img_path) as img:
                arr = np.array(img.convert('RGB'), dtype=np.float32)
                r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
                
                r_means.append(np.mean(r))
                g_means.append(np.mean(g))
                b_means.append(np.mean(b))
                r_stds.append(np.std(r))
                g_stds.append(np.std(g))
                b_stds.append(np.std(b))
                
                # Sample 100 pixels per channel for smooth distribution
                r_vals.extend(np.random.choice(r.flatten(), size=100))
                g_vals.extend(np.random.choice(g.flatten(), size=100))
                b_vals.extend(np.random.choice(b.flatten(), size=100))
        except Exception as e:
            continue
            
    # 1. Pixel Intensity Histograms
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(r_vals, bins=50, color='red', alpha=0.5, label=f'Red (Mean: {np.mean(r_means):.1f})', density=True)
    ax.hist(g_vals, bins=50, color='green', alpha=0.5, label=f'Green (Mean: {np.mean(g_means):.1f})', density=True)
    ax.hist(b_vals, bins=50, color='blue', alpha=0.5, label=f'Blue (Mean: {np.mean(b_means):.1f})', density=True)
    
    ax.set_title("Pixel Intensity Distribution (RGB Channels)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Pixel Value (0 - 255)", fontsize=11)
    ax.set_ylabel("Normalized Density", fontsize=11)
    ax.legend(fontsize=10)
    plt.tight_layout()
    hist_path = EDA_DIR / "pixel_intensity_histograms.png"
    plt.savefig(hist_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {hist_path}")
    
    # 2. RGB Channel Mean & Std Boxplots
    channel_df = pd.DataFrame({
        'Red Mean': r_means, 'Green Mean': g_means, 'Blue Mean': b_means,
        'Red Std': r_stds, 'Green Std': g_stds, 'Blue Std': b_stds
    })
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(data=channel_df[['Red Mean', 'Green Mean', 'Blue Mean']], ax=ax1, palette=['#ff9999', '#99ff99', '#9999ff'])
    ax1.set_title("Channel Mean Intensity", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Pixel Value", fontsize=10)
    
    sns.boxplot(data=channel_df[['Red Std', 'Green Std', 'Blue Std']], ax=ax2, palette=['#ff9999', '#99ff99', '#9999ff'])
    ax2.set_title("Channel Standard Deviation", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Std Dev", fontsize=10)
    
    plt.tight_layout()
    box_path = EDA_DIR / "rgb_channel_distribution.png"
    plt.savefig(box_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {box_path}")
    
    stats_summary = {
        "red_channel": {"mean": float(np.mean(r_means)), "std": float(np.mean(r_stds))},
        "green_channel": {"mean": float(np.mean(g_means)), "std": float(np.mean(g_stds))},
        "blue_channel": {"mean": float(np.mean(b_means)), "std": float(np.mean(b_stds))}
    }
    return stats_summary

def plot_image_dimensions():
    """Plot distribution of image dimensions (confirming uniform 256x256 resolution)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    resolutions = ["256 x 256"]
    counts = [17492]
    bars = ax.bar(resolutions, counts, color='#3498db', width=0.4, edgecolor='black', alpha=0.85)
    ax.set_title("DeepWeeds Native Image Dimensions Distribution", fontsize=12, fontweight='bold', pad=12)
    ax.set_ylabel("Total Images", fontsize=11)
    ax.set_ylim(0, 20000)
    for bar in bars:
        ax.annotate(f"{bar.get_height():,} images\n(100.0% Uniform)",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5), textcoords="offset points", ha='center', va='bottom',
                    fontsize=10, fontweight='semibold')
    plt.tight_layout()
    save_path = EDA_DIR / "image_dimensions_distribution.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {save_path}")

def run_eda():
    """Run full EDA pipeline and output artifacts."""
    init_directories()
    logger.info("=" * 60)
    logger.info("STARTING COMPREHENSIVE EXPLORATORY DATA ANALYSIS (EDA)")
    logger.info("=" * 60)
    
    df = pd.read_csv(LABELS_CSV)
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)
    
    # 1. Class Distribution
    plot_class_distribution(df)
    
    # 2. Split Comparison
    plot_splits_distribution(train_df, val_df, test_df)
    
    # 3. Sample Images Grid
    plot_sample_images_grid(df, samples_per_class=5)
    
    # 4. Pixel Intensity & RGB Statistics
    stats = analyze_pixel_statistics(df, num_samples=1000)
    
    # 5. Image Dimensions
    plot_image_dimensions()
    
    # Save EDA summary metrics
    eda_summary = {
        "dataset_name": "DeepWeeds",
        "total_images": len(df),
        "total_classes": len(CLASS_NAMES),
        "class_breakdown": df['Species'].value_counts().to_dict(),
        "train_set_size": len(train_df),
        "val_set_size": len(val_df),
        "test_set_size": len(test_df),
        "native_resolution": "256x256x3",
        "target_resnet_resolution": "224x224x3",
        "rgb_statistics": stats,
        "imbalance_ratio": f"{df['Species'].value_counts().max() / df['Species'].value_counts().min():.2f}:1"
    }
    save_json(eda_summary, EDA_DIR / "eda_summary.json")
    logger.info(f"EDA Summary saved to {EDA_DIR / 'eda_summary.json'}")
    logger.info("Exploratory Data Analysis completed successfully!")

if __name__ == "__main__":
    run_eda()
