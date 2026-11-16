"""
Inference & Prediction Module for Plant Species Classification using ResNet152V2.
Designed for independent teammate usage: loads the pre-trained .keras model directly,
loads class_names.json, performs exact ResNet152V2 preprocessing, and reports
the predicted plant species, top-3 confidences, and visualization.
"""

import os
import sys
import argparse
import json
from pathlib import Path
import keras
import numpy as np
from PIL import Image

# Suppress unnecessary TensorFlow debug information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

# Default Paths (relative to script or project root)
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent if CURRENT_DIR.name == "src" else CURRENT_DIR

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "weed_resnet152v2_final.keras"
FALLBACK_MODEL_PATH = PROJECT_ROOT / "models" / "weed_resnet152v2_best.keras"
DEFAULT_CLASSES_PATH = PROJECT_ROOT / "models" / "class_names.json"

def load_inference_model(model_path: Path = None):
    """Loads the trained .keras model."""
    if model_path is None or not Path(model_path).exists():
        if DEFAULT_MODEL_PATH.exists():
            model_path = DEFAULT_MODEL_PATH
        elif FALLBACK_MODEL_PATH.exists():
            model_path = FALLBACK_MODEL_PATH
        else:
            raise FileNotFoundError(
                f"Trained model not found at {DEFAULT_MODEL_PATH} or {FALLBACK_MODEL_PATH}. "
                "Ensure weed_resnet152v2_final.keras is present in models/."
            )
            
    # Load complete model
    model = keras.models.load_model(model_path)
    return model

def load_class_mapping(classes_path: Path = None) -> list:
    """Loads class names in exact trained order."""
    if classes_path is None:
        classes_path = DEFAULT_CLASSES_PATH
        
    if Path(classes_path).exists():
        with open(classes_path, "r", encoding="utf-8") as f:
            class_names = json.load(f)
    else:
        raise FileNotFoundError(f"Class names file not found at: {classes_path}")

    if not isinstance(class_names, list) or not all(isinstance(name, str) for name in class_names):
        raise ValueError(f"Class names file must contain a JSON list of strings: {classes_path}")
    return class_names

def preprocess_image_for_resnet(img_pil: Image.Image, target_size=(224, 224)) -> tf.Tensor:
    """
    Applies exact ResNet152V2 preprocessing:
    1. Converts image to RGB
    2. Resizes to 224 x 224
    3. Normalizes pixel values [0, 255] -> [-1.0, 1.0]
    """
    img_rgb = img_pil.convert("RGB")
    img_resized = img_rgb.resize(target_size, Image.Resampling.BILINEAR)
    img_array = np.array(img_resized, dtype=np.float32)
    # ResNetV2 formula: (x / 127.5) - 1.0
    img_tensor = tf.convert_to_tensor(img_array / 127.5 - 1.0, dtype=tf.float32)
    # Add batch dimension (1, 224, 224, 3)
    return tf.expand_dims(img_tensor, axis=0)


def predict_image(img_pil: Image.Image, model, class_names: list) -> dict:
    """Predict a class and confidence scores for an already-open image."""
    input_tensor = preprocess_image_for_resnet(img_pil)
    probs = model(input_tensor, training=False).numpy()[0]
    if len(class_names) != len(probs):
        raise ValueError(
            f"Model outputs {len(probs)} classes, but the class names file has "
            f"{len(class_names)} entries."
        )

    top_indices = np.argsort(probs)[::-1][:min(3, len(probs))]
    top_predictions = [
        {
            "rank": rank + 1,
            "class": class_names[int(idx)],
            "confidence": float(probs[idx]) * 100.0,
        }
        for rank, idx in enumerate(top_indices)
    ]
    predicted = top_predictions[0]
    return {
        "predicted_species": predicted["class"],
        "confidence_percentage": predicted["confidence"],
        "top_predictions": top_predictions,
    }


def predict_weed(
    image_path: str,
    model=None,
    class_names: list = None,
    save_viz: bool = True,
    output_viz_dir: str = "results"
) -> dict:
    """
    Performs inference on a single image and returns prediction details.
    """
    img_file = Path(image_path)
    if not img_file.exists():
        raise FileNotFoundError(f"Target image not found at: {image_path}")
        
    if model is None:
        model = load_inference_model()
    if class_names is None:
        class_names = load_class_mapping()
        
    with Image.open(img_file) as image:
        orig_img = image.convert("RGB")
    prediction = predict_image(orig_img, model, class_names)
    pred_class = prediction["predicted_species"]
    confidence_pct = prediction["confidence_percentage"]
    top3_results = prediction["top_predictions"]
    
    # Formatted terminal display
    print("\n" + "=" * 50)
    print(f"IMAGE: {img_file.name}")
    print("=" * 50)
    print(f"Predicted Species: {pred_class}")
    print(f"Confidence:     {confidence_pct:.2f}%")
    print("\nTop Predictions:")
    for item in top3_results:
        print(f"{item['rank']}. {item['class']} - {item['confidence']:.2f}%")
    print("=" * 50)
    
    # Optionally save visualization
    if save_viz:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
        ax1.imshow(orig_img.convert("RGB"))
        ax1.set_title(f"Target Image: {img_file.name}", fontsize=11, fontweight='bold')
        ax1.axis('off')
        
        # Horizontal bar chart of top 3
        top_classes = [item['class'] for item in top3_results][::-1]
        top_confs = [item['confidence'] for item in top3_results][::-1]
        bars = ax2.barh(top_classes, top_confs, color=['#ff7f0e', '#1f77b4', '#2ca02c'], edgecolor='black')
        ax2.set_xlim(0, 105)
        ax2.set_xlabel("Confidence (%)", fontsize=11)
        ax2.set_title(f"Prediction: {pred_class} ({confidence_pct:.1f}%)", fontsize=12, fontweight='bold', color='green')
        for bar in bars:
            ax2.annotate(f"{bar.get_width():.2f}%",
                         (bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2),
                         va='center', fontsize=10, fontweight='bold')
                         
        plt.tight_layout()
        out_dir = Path(output_viz_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        viz_path = out_dir / f"prediction_{img_file.stem}.png"
        plt.savefig(viz_path, dpi=200)
        plt.close()
        print(f"[Visualization saved to {viz_path}]")
        
    return {
        "image": str(img_file),
        "predicted_weed": pred_class,
        "predicted_species": pred_class,
        "confidence_percentage": confidence_pct,
        "top_predictions": top3_results
    }

def main(argv=None):
    parser = argparse.ArgumentParser(description="Predict Plant Species from Image using ResNet152V2")
    parser.add_argument("--image", type=str, required=True, help="Path to input image (JPG/PNG)")
    parser.add_argument("--model", type=str, default=None, help="Path to .keras model")
    parser.add_argument("--classes", type=str, default=None, help="Path to class_names.json")
    parser.add_argument("--no-viz", action="store_true", help="Do not save visualization plot")
    args = parser.parse_args(argv)
    
    predict_weed(
        image_path=args.image,
        model=load_inference_model(Path(args.model) if args.model else None),
        class_names=load_class_mapping(Path(args.classes) if args.classes else None),
        save_viz=not args.no_viz
    )


if __name__ == "__main__":
    main()
