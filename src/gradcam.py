"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Module for ResNet152V2.
Explains model decisions by localizing discriminative botanical regions
(foliage, leaves, spines, flowerheads) that influence predictions.
Generates side-by-side visualizations (Original, Heatmap, Superimposed Overlay)
and exports to results/gradcam/.
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import tensorflow as tf

from src.config import (
    FINAL_MODEL_PATH, BEST_MODEL_PATH, CLASS_NAMES_PATH,
    TEST_CSV, IMAGES_DIR, GRADCAM_DIR, CLASS_NAMES,
    IMAGE_HEIGHT, IMAGE_WIDTH, init_directories
)
from src.utils import setup_logger, load_json
from src.data_preprocessing import preprocess_image_tensor

logger = setup_logger("GradCAM", "gradcam.log")

def get_gradcam_model(model: tf.keras.Model, last_conv_layer_name: str = "post_relu") -> tf.keras.Model:
    """
    Constructs sub-model mapping inputs to the target convolutional layer and final predictions.
    In ResNet152V2, 'post_relu' is the terminal feature map prior to GlobalAveragePooling.
    """
    try:
        target_layer = model.get_layer(last_conv_layer_name)
    except ValueError:
        # Fallback if backbone layer name differs
        conv_layers = [l.name for l in model.layers if "conv" in l.name or "relu" in l.name]
        last_conv_layer_name = conv_layers[-1] if conv_layers else "post_relu"
        target_layer = model.get_layer(last_conv_layer_name)
        
    logger.info(f"Using convolutional layer '{last_conv_layer_name}' for Grad-CAM gradients.")
    return tf.keras.Model(
        inputs=model.inputs,
        outputs=[target_layer.output, model.output]
    )

def make_gradcam_heatmap(
    img_tensor: tf.Tensor,
    grad_model: tf.keras.Model,
    pred_index: int = None
) -> np.ndarray:
    """
    Computes normalized Grad-CAM activation heatmap for a target class.
    """
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor, training=False)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Compute gradients of top class score with respect to feature maps
    grads = tape.gradient(class_channel, conv_outputs)
    
    # Global average pooling of gradients gives feature map importance weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weighted sum of feature map channels
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Apply ReLU (only features that positively influence the target class)
    heatmap = tf.maximum(heatmap, 0.0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    return heatmap.numpy()

def overlay_gradcam(
    original_img_pil: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: str = "jet"
) -> np.ndarray:
    """
    Overlays colored activation heatmap onto original RGB image.
    """
    # Resize heatmap to match image dimensions
    heatmap_uint8 = np.uint8(255 * heatmap)
    
    # Colormap
    cmap = matplotlib.colormaps[colormap]
    cmap_colors = cmap(np.arange(256))[:, :3]  # Discard alpha
    colored_heatmap = cmap_colors[heatmap_uint8]
    
    # Resize colored heatmap to match original image size
    colored_heatmap = Image.fromarray(np.uint8(colored_heatmap * 255))
    colored_heatmap = colored_heatmap.resize((original_img_pil.width, original_img_pil.height), Image.Resampling.BILINEAR)
    colored_heatmap = np.array(colored_heatmap) / 255.0
    
    # Normalize original image to [0, 1]
    orig_arr = np.array(original_img_pil.convert("RGB"), dtype=np.float32) / 255.0
    
    # Superimpose
    superimposed = (1.0 - alpha) * orig_arr + alpha * colored_heatmap
    superimposed = np.clip(superimposed, 0.0, 1.0)
    return superimposed

def generate_sample_gradcam_visualizations(
    model: tf.keras.Model = None,
    num_samples: int = 8,
    save_dir: Path = GRADCAM_DIR
):
    """
    Generates representative Grad-CAM panels for test samples across diverse weed species.
    """
    init_directories()
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if model is None:
        if FINAL_MODEL_PATH.exists():
            model = tf.keras.models.load_model(FINAL_MODEL_PATH)
        elif BEST_MODEL_PATH.exists():
            model = tf.keras.models.load_model(BEST_MODEL_PATH)
        else:
            raise FileNotFoundError("No trained model found for Grad-CAM.")
            
    class_names = load_json(CLASS_NAMES_PATH) if CLASS_NAMES_PATH.exists() else CLASS_NAMES
    grad_model = get_gradcam_model(model, "post_relu")
    
    test_df = pd.read_csv(TEST_CSV)
    # Pick a stratified selection of samples across different classes
    sampled_indices = []
    for cls_id in range(len(class_names)):
        cls_subset = test_df[test_df["Label"] == cls_id]
        if not cls_subset.empty:
            sampled_indices.append(cls_subset.index[0])
            
    logger.info(f"Generating Grad-CAM figures for {len(sampled_indices)} weed classes...")
    
    for idx_pos, df_idx in enumerate(sampled_indices[:num_samples]):
        row = test_df.iloc[df_idx]
        fname = row["Filename"]
        true_label = class_names[row["Label"]]
        img_path = IMAGES_DIR / fname
        
        orig_img = Image.open(img_path).convert("RGB")
        resized_img = orig_img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.Resampling.BILINEAR)
        img_arr = np.array(resized_img, dtype=np.float32)
        
        # Preprocess
        input_tensor = preprocess_image_tensor(tf.expand_dims(img_arr, 0))
        
        # Predict
        preds = model(input_tensor, training=False).numpy()[0]
        pred_idx = int(np.argmax(preds))
        pred_label = class_names[pred_idx]
        confidence = float(preds[pred_idx]) * 100.0
        
        # Make heatmap
        heatmap = make_gradcam_heatmap(input_tensor, grad_model, pred_index=pred_idx)
        superimposed = overlay_gradcam(resized_img, heatmap, alpha=0.5)
        
        # Plot 3-panel figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Panel 1: Original
        axes[0].imshow(resized_img)
        axes[0].set_title(f"Input Image: {fname}\nGround Truth: {true_label}", fontsize=11, fontweight='bold')
        axes[0].axis('off')
        
        # Panel 2: Heatmap
        im = axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title(f"Grad-CAM Class Activation Map\n(Layer: 'post_relu')", fontsize=11, fontweight='bold')
        axes[1].axis('off')
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Panel 3: Superimposed
        color = "green" if pred_label == true_label else "red"
        axes[2].imshow(superimposed)
        axes[2].set_title(f"Botanical Attention Overlay\nPred: {pred_label} ({confidence:.2f}%)",
                          fontsize=11, fontweight='bold', color=color)
        axes[2].axis('off')
        
        plt.tight_layout()
        output_filename = f"gradcam_sample_{idx_pos + 1}_{true_label.replace(' ', '_').lower()}.png"
        fig_save_path = save_dir / output_filename
        plt.savefig(fig_save_path, dpi=250)
        plt.close()
        logger.info(f"Saved: {fig_save_path}")
        
    logger.info("Grad-CAM visualization generation complete.")

if __name__ == "__main__":
    generate_sample_gradcam_visualizations()
