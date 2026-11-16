"""
Inspect a folder-labeled image dataset and generate stratified train/validation/test splits.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hashlib

import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.config import (
    CLASS_NAMES,
    DATASET_DIR,
    LABELS_CSV,
    LABEL_TO_CLASS,
    RANDOM_SEED,
    RESULTS_DIR,
    SPLIT_DIR,
    CV_SPLIT_DIR,
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    SUPPORTED_IMAGE_EXTENSIONS,
    init_directories,
)
from src.utils import save_json, setup_logger

logger = setup_logger("DatasetInspection", "dataset_inspection.log")


def calculate_file_hash(filepath: Path, chunk_size: int = 65536) -> str:
    """Calculate SHA-256 for exact duplicate detection."""
    sha256 = hashlib.sha256()
    with filepath.open("rb") as image_file:
        while chunk := image_file.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()


def _build_raw_dataset_manifest() -> pd.DataFrame:
    """Build labels from supported images directly inside each class folder."""
    if not CLASS_NAMES:
        raise ValueError(
            f"No class folders with images were found directly under {DATASET_DIR}."
        )

    rows = []
    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = DATASET_DIR / class_name
        image_paths = sorted(
            (
                path for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            ),
            key=lambda path: path.name.casefold(),
        )
        if not image_paths:
            raise ValueError(f"Class folder contains no supported images: {class_dir}")

        rows.extend(
            {
                "Filename": path.relative_to(DATASET_DIR).as_posix(),
                "Label": label,
                "Species": class_name,
            }
            for path in image_paths
        )

    manifest = pd.DataFrame(rows, columns=["Filename", "Label", "Species"])
    if manifest["Filename"].duplicated().any():
        duplicates = manifest.loc[manifest["Filename"].duplicated(), "Filename"].tolist()
        raise ValueError(f"Duplicate image paths found in the dataset: {duplicates[:10]}")
    return manifest


def _remove_exact_duplicates(df: pd.DataFrame) -> tuple:
    """Return one row per image content and details for duplicate records."""
    seen_hashes = {}
    unique_rows = []
    duplicates = []
    for row in df.itertuples(index=False):
        digest = calculate_file_hash(DATASET_DIR / Path(row.Filename))
        if digest in seen_hashes:
            original = seen_hashes[digest]
            if original["Label"] != row.Label:
                raise ValueError(
                    "Identical image content has conflicting class labels: "
                    f"{original['Filename']} and {row.Filename}."
                )
            duplicates.append(
                {
                    "original": original["Filename"],
                    "duplicate": row.Filename,
                    "sha256": digest,
                }
            )
            continue
        seen_hashes[digest] = {"Filename": row.Filename, "Label": row.Label}
        unique_rows.append(
            {"Filename": row.Filename, "Label": row.Label, "Species": row.Species}
        )
    return pd.DataFrame(unique_rows, columns=df.columns), duplicates


def build_dataset_manifest() -> pd.DataFrame:
    """Build a deduplicated manifest from the current class folders."""
    raw_manifest = _build_raw_dataset_manifest()
    manifest, _ = _remove_exact_duplicates(raw_manifest)
    return manifest


def create_dataset_splits(df: pd.DataFrame) -> dict:
    """Create stratified 60/20/20 splits and five stratified CV folds."""
    if df.empty:
        raise ValueError("Cannot create dataset splits from an empty manifest.")

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    CV_SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(LABELS_CSV, index=False)

    train_df, temp_df = train_test_split(
        df,
        test_size=0.40,
        random_state=RANDOM_SEED,
        stratify=df["Label"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temp_df["Label"],
    )

    train_df = train_df.sort_values("Filename").reset_index(drop=True)
    val_df = val_df.sort_values("Filename").reset_index(drop=True)
    test_df = test_df.sort_values("Filename").reset_index(drop=True)
    train_df.to_csv(TRAIN_CSV, index=False)
    val_df.to_csv(VAL_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    fold_summaries = {}
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["Label"])):
        fold_train = df.iloc[train_idx].reset_index(drop=True)
        fold_val = df.iloc[val_idx].reset_index(drop=True)
        fold_train.to_csv(CV_SPLIT_DIR / f"fold_{fold}_train.csv", index=False)
        fold_val.to_csv(CV_SPLIT_DIR / f"fold_{fold}_val.csv", index=False)
        fold_summaries[f"fold_{fold}"] = {
            "train_count": len(fold_train),
            "val_count": len(fold_val),
        }

    return {
        "train_count": len(train_df),
        "val_count": len(val_df),
        "test_count": len(test_df),
        "cv_folds": fold_summaries,
    }


def inspect_dataset() -> dict:
    """Validate class-folder images, refresh the manifest, and regenerate all splits."""
    init_directories()
    logger.info("Starting folder-labeled dataset inspection.")

    raw_df = _build_raw_dataset_manifest()
    df, duplicates = _remove_exact_duplicates(raw_df)
    missing_files = []
    corrupted_files = []
    dimensions = []
    formats = {}
    channels = {"RGB": 0, "Grayscale": 0, "RGBA": 0, "Other": 0}

    for idx, row in df.iterrows():
        image_path = DATASET_DIR / Path(row["Filename"])
        if not image_path.is_file():
            missing_files.append(row["Filename"])
            continue

        try:
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                dimensions.append(image.size)
                image_format = image.format or image_path.suffix.lstrip(".").upper()
                formats[image_format] = formats.get(image_format, 0) + 1
                if image.mode == "RGB":
                    channels["RGB"] += 1
                elif image.mode in ("L", "1"):
                    channels["Grayscale"] += 1
                elif image.mode == "RGBA":
                    channels["RGBA"] += 1
                else:
                    channels["Other"] += 1
        except (OSError, ValueError) as error:
            corrupted_files.append({"filename": row["Filename"], "error": str(error)})
            continue

        if (idx + 1) % 1000 == 0 or idx + 1 == len(df):
            logger.info("Checked %s/%s images.", idx + 1, len(df))

    split_stats = create_dataset_splits(df)
    common_size = max(set(dimensions), key=dimensions.count) if dimensions else (0, 0)
    report = {
        "dataset_name": "Folder-labeled plant species dataset",
        "total_records_in_csv": len(df),
        "total_images_in_class_folders": len(raw_df),
        "unique_images_used": len(df),
        "class_count": len(CLASS_NAMES),
        "class_distribution": df["Species"].value_counts().to_dict(),
        "label_distribution": {
            LABEL_TO_CLASS[label]: int(count)
            for label, count in df["Label"].value_counts().items()
        },
        "missing_files_count": len(missing_files),
        "missing_files_list": missing_files,
        "corrupted_images_count": len(corrupted_files),
        "corrupted_images_list": corrupted_files,
        "exact_duplicates_count": len(duplicates),
        "exact_duplicates_list": duplicates,
        "channel_distribution": channels,
        "format_distribution": formats,
        "dimensions": {
            "min_width": min((width for width, _ in dimensions), default=0),
            "max_width": max((width for width, _ in dimensions), default=0),
            "min_height": min((height for _, height in dimensions), default=0),
            "max_height": max((height for _, height in dimensions), default=0),
            "most_common_dimension": f"{common_size[0]}x{common_size[1]}",
        },
        "splits": split_stats,
    }
    save_json(report, RESULTS_DIR / "dataset_inspection_report.json")

    summary_path = RESULTS_DIR / "dataset_inspection_summary.md"
    with summary_path.open("w", encoding="utf-8") as summary:
        summary.write("# Plant Species Dataset Inspection\n\n")
        summary.write(f"- **Classes:** {report['class_count']}\n")
        summary.write(f"- **Images:** {report['total_records_in_csv']}\n")
        summary.write(f"- **Missing files:** {report['missing_files_count']}\n")
        summary.write(f"- **Corrupted images:** {report['corrupted_images_count']}\n")
        summary.write(f"- **Exact duplicates:** {report['exact_duplicates_count']}\n")
        summary.write(
            f"- **Most common image dimensions:** "
            f"{report['dimensions']['most_common_dimension']}\n\n"
        )
        summary.write("## Class distribution\n\n")
        summary.write("| Class | Images |\n|---|---:|\n")
        for class_name in CLASS_NAMES:
            summary.write(
                f"| {class_name} | {report['class_distribution'].get(class_name, 0)} |\n"
            )
        summary.write("\n## Dataset splits\n\n")
        for split_name in ("train_count", "val_count", "test_count"):
            summary.write(f"- **{split_name.replace('_count', '').title()}:** {split_stats[split_name]}\n")

    logger.info(
        "Inspection complete: %s images across %s classes; splits: %s",
        len(df),
        len(CLASS_NAMES),
        split_stats,
    )
    return report


if __name__ == "__main__":
    report = inspect_dataset()
    print("\n--- DATASET INSPECTION SUMMARY ---")
    print(f"Classes: {report['class_count']}")
    print(f"Images: {report['total_records_in_csv']}")
    print(f"Corrupted images: {report['corrupted_images_count']}")
    print(f"Missing files: {report['missing_files_count']}")
