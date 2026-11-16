"""
High-Performance TensorFlow tf.data Pipeline Module for DeepWeeds.
Implements:
- Asynchronous image loading, decoding, resizing, and one-hot encoding
- Dynamic training augmentation
- ResNet152V2 preprocessing [-1.0, 1.0]
- Prefetched and batched data loaders with balanced class weight computation
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import tensorflow as tf

from src.config import (
    DATASET_DIR, LABELS_CSV, TRAIN_CSV, VAL_CSV, TEST_CSV,
    IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CLASSES,
    BATCH_SIZE, RANDOM_SEED, init_directories
)
from src.data_preprocessing import get_data_augmentation_pipeline, preprocess_image_tensor
from src.utils import setup_logger, get_class_weights

logger = setup_logger("DataLoader", "data_loader.log")
AUTOTUNE = tf.data.AUTOTUNE

# Global augmentation pipeline instance for tf.data graph integration
_AUGMENTATION_MODEL = None

def get_shared_augmentation_pipeline():
    global _AUGMENTATION_MODEL
    if _AUGMENTATION_MODEL is None:
        _AUGMENTATION_MODEL = get_data_augmentation_pipeline()
    return _AUGMENTATION_MODEL

def parse_image_path_and_label(filename_tensor, label_tensor, is_training: bool = False, augment: bool = False):
    """
    Parses an individual image path and label into a preprocessed image tensor
    and one-hot encoded label tensor.
    """
    # Construct absolute path string
    file_path = tf.strings.join([str(DATASET_DIR), os.sep, filename_tensor])
    img_bytes = tf.io.read_file(file_path)
    image = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    # Ensure float32 casting
    image = tf.image.resize(image, [IMAGE_HEIGHT, IMAGE_WIDTH], method=tf.image.ResizeMethod.BILINEAR)
    
    # Apply training-only data augmentation if requested
    if is_training and augment:
        aug_pipeline = get_shared_augmentation_pipeline()
        # Add temporary batch dimension for keras layer then squeeze back
        image = tf.expand_dims(image, 0)
        image = aug_pipeline(image, training=True)
        image = tf.squeeze(image, 0)
        
    # Standard ResNetV2 Preprocessing: scales [0, 255] -> [-1.0, 1.0]
    image = preprocess_image_tensor(image)
    
    # One-hot encode label for multiclass categorical crossentropy
    one_hot_label = tf.one_hot(label_tensor, depth=NUM_CLASSES)
    return image, one_hot_label

def create_tf_dataset(
    df: pd.DataFrame,
    is_training: bool = False,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = True,
    augment: bool = False,
    max_samples: int = None
) -> tf.data.Dataset:
    """
    Builds a scalable, prefetched tf.data.Dataset from a pandas DataFrame.
    """
    if max_samples is not None and max_samples < len(df):
        df = df.sample(n=max_samples, random_state=RANDOM_SEED).reset_index(drop=True)
        
    filenames = df["Filename"].values.astype(str)
    labels = df["Label"].values.astype(np.int32)
    
    ds = tf.data.Dataset.from_tensor_slices((filenames, labels))
    
    if shuffle and is_training:
        ds = ds.shuffle(buffer_size=min(len(df), 2048), seed=RANDOM_SEED, reshuffle_each_iteration=True)
        
    # Map parsing function in parallel
    ds = ds.map(
        lambda fname, lbl: parse_image_path_and_label(fname, lbl, is_training=is_training, augment=augment),
        num_parallel_calls=AUTOTUNE
    )
    
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

def get_data_loaders(
    batch_size: int = BATCH_SIZE,
    augment_train: bool = True,
    sample_fraction: float = None
) -> tuple:
    """
    Loads train, validation, and test datasets and calculates balanced class weights.
    Returns:
        (train_dataset, val_dataset, test_dataset, class_weights_dict)
    """
    init_directories()
    from src.dataset_inspection import build_dataset_manifest, create_dataset_splits

    manifest = build_dataset_manifest()
    manifest.to_csv(LABELS_CSV, index=False)
    split_paths = (TRAIN_CSV, VAL_CSV, TEST_CSV)
    splits_match = all(path.exists() for path in split_paths)
    if splits_match:
        try:
            existing_splits = pd.concat(
                [pd.read_csv(path) for path in split_paths],
                ignore_index=True
            )
            expected = manifest.set_index("Filename").sort_index()
            actual = existing_splits.set_index("Filename").sort_index()
            splits_match = (
                existing_splits["Filename"].is_unique
                and expected.index.equals(actual.index)
                and expected["Label"].equals(actual["Label"])
                and expected["Species"].equals(actual["Species"])
            )
        except (KeyError, pd.errors.EmptyDataError, pd.errors.ParserError):
            splits_match = False

    if not splits_match:
        logger.warning("Existing splits do not match current class folders; regenerating stratified splits.")
        create_dataset_splits(manifest)

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)
    
    if sample_fraction is not None and 0.0 < sample_fraction < 1.0:
        logger.info(f"Using stratified sample fraction: {sample_fraction:.2f} for fast experimentation.")
        train_df = train_df.groupby("Label", group_keys=False).apply(
            lambda x: x.sample(frac=sample_fraction, random_state=RANDOM_SEED)
        ).reset_index(drop=True)
        val_df = val_df.groupby("Label", group_keys=False).apply(
            lambda x: x.sample(frac=sample_fraction, random_state=RANDOM_SEED)
        ).reset_index(drop=True)
        test_df = test_df.groupby("Label", group_keys=False).apply(
            lambda x: x.sample(frac=sample_fraction, random_state=RANDOM_SEED)
        ).reset_index(drop=True)
        
    logger.info(f"Building data loaders: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}, Batch Size={batch_size}")
    
    # Calculate balanced class weights from training distribution
    class_weights = get_class_weights(train_df["Label"].values, num_classes=NUM_CLASSES)
    logger.info(f"Calculated balanced class weights for training: {class_weights}")
    
    train_ds = create_tf_dataset(train_df, is_training=True, batch_size=batch_size, shuffle=True, augment=augment_train)
    val_ds = create_tf_dataset(val_df, is_training=False, batch_size=batch_size, shuffle=False, augment=False)
    test_ds = create_tf_dataset(test_df, is_training=False, batch_size=batch_size, shuffle=False, augment=False)
    
    return train_ds, val_ds, test_ds, class_weights

if __name__ == "__main__":
    logger.info("Testing data loader pipeline...")
    train_ds, val_ds, test_ds, weights = get_data_loaders(batch_size=16)
    
    # Fetch a sample batch to verify tensors
    for images, labels in train_ds.take(1):
        logger.info(f"Batch Images Tensor Shape: {images.shape}, dtype: {images.dtype}")
        logger.info(f"Batch Labels Tensor Shape: {labels.shape}, dtype: {labels.dtype}")
        logger.info(f"Pixel values min: {tf.reduce_min(images):.4f}, max: {tf.reduce_max(images):.4f}")
        assert images.shape == (16, 224, 224, 3), f"Unexpected image shape: {images.shape}"
        assert labels.shape == (16, NUM_CLASSES), f"Unexpected label shape: {labels.shape}"
        # ResNetV2 normalized range is [-1.0, 1.0]
        assert -1.1 <= tf.reduce_min(images) <= 0.0, "Preprocessing min out of range"
        assert 0.0 <= tf.reduce_max(images) <= 1.1, "Preprocessing max out of range"
        logger.info("Data loader pipeline verified successfully with verified tensor dimensions and ResNetV2 scaling.")
