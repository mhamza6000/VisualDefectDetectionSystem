"""Extract, preview, and split the casting image dataset."""

import hashlib
import shutil
import warnings
import zipfile
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoImageProcessor
from preprocessing import IMAGE_SIZE, build_image_transform, preprocess_image
from visualization import display_dataset_samples


BASE_DIR = Path(__file__).resolve().parent
ZIP_FILE = BASE_DIR / "archive (1).zip"
DATASET_DIR = BASE_DIR / "casting_extracted" 
OUTPUT_DIR = BASE_DIR / "casting_imagefolder"
ARCHIVE_PREFIX = "casting_512x512"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LABELS = {"def_front": "Defective", "ok_front": "Normal"}
SOURCE_CLASSES = {label: folder for folder, label in LABELS.items()}
CLASS_NAMES = ["Normal", "Defective"]
ID2LABEL = dict(enumerate(CLASS_NAMES))
LABEL2ID = {label: index for index, label in ID2LABEL.items()}
SPLIT_NAMES = ("train", "validation", "test")
SPLIT_RATIOS = (0.70, 0.15, 0.15)
RANDOM_SEED = 42
SAMPLES_PER_CLASS = 2


class CastingImageDataset(Dataset):
    """Load and transform flat split-folder images from the manifest."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_root: Path,
        processor: AutoImageProcessor,
        augment: bool = False,
    ) -> None:
        self.samples = list(
            zip(dataframe["image_name"], dataframe["label"], dataframe["split"])
        )
        self.image_root = image_root
        self.transform = build_image_transform(processor, augment=augment)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_name, label, split = self.samples[index]
        image_path = self.image_root / split / image_name
        if not image_path.is_file():
            source_path = DATASET_DIR / SOURCE_CLASSES[label] / image_name
            if not source_path.is_file():
                raise FileNotFoundError(
                    f"Image {image_name!r} is missing from split {split!r} "
                    f"({image_path}) and source class {label!r} ({source_path}). "
                    "Run prepare_dataset() to rebuild the dataset splits."
                )
            warnings.warn(
                f"Split copy is missing for {image_name!r} in {split!r}; "
                f"loading the original image from {source_path}. "
                "Run prepare_dataset() to rebuild split folders.",
                RuntimeWarning,
                stacklevel=2,
            )
            image_path = source_path
        with Image.open(image_path) as image:
            pixels = self.transform(image)
        return pixels, LABEL2ID[label]


def extract_dataset() -> Path:
    """Extract the casting_512x512 images from the zip archive."""
    if not ZIP_FILE.is_file():
        raise FileNotFoundError(f"Dataset archive not found: {ZIP_FILE}")

    if DATASET_DIR.exists() and all(
        any(
            image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS
            for image in (DATASET_DIR / class_folder).glob("*")
        )
        for class_folder in SOURCE_CLASSES.values()
    ):
        return DATASET_DIR

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_FILE) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue

            parts = Path(member.filename.replace("\\", "/")).parts
            if ".." in parts or ARCHIVE_PREFIX not in parts:
                continue

            prefix_index = parts.index(ARCHIVE_PREFIX)
            dataset_parts = parts[prefix_index + 1 :]
            if not dataset_parts or Path(dataset_parts[-1]).suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            class_folder = next(
                (part for part in dataset_parts if part in SOURCE_CLASSES.values()),
                None,
            )
            if class_folder is None:
                continue

            destination = DATASET_DIR / class_folder / dataset_parts[-1]
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)

    missing_classes = [
        class_folder
        for class_folder in SOURCE_CLASSES.values()
        if not any(
            image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS
            for image in (DATASET_DIR / class_folder).glob("*")
        )
    ]
    if missing_classes:
        raise FileNotFoundError(
            f"No images found for class folder(s) {missing_classes} "
            f"in {ZIP_FILE} under {ARCHIVE_PREFIX!r}."
        )
    return DATASET_DIR


def create_image_dataframe(dataset_dir: Path) -> pd.DataFrame:
    """List unique extracted images with labels and SHA-256 content hashes."""
    rows = []
    for class_folder in SOURCE_CLASSES.values():
        class_dir = dataset_dir / class_folder
        if not class_dir.is_dir():
            continue
        for image in sorted(class_dir.iterdir()):
            if not image.is_file() or image.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            rows.append(
                {
                    "image_name": image.name,
                    "label": LABELS[class_folder],
                    "sha256": digest,
                }
            )

    dataframe = pd.DataFrame(rows)
    if dataframe.empty:
        raise FileNotFoundError(
            f"No images found directly inside expected class folders "
            f"{list(SOURCE_CLASSES.values())} under {dataset_dir}. "
            "Run extract_dataset() to extract the archive into the expected layout."
        )
    duplicate_hashes = dataframe[dataframe["sha256"].duplicated(keep=False)]
    conflicting_labels = duplicate_hashes.groupby("sha256")["label"].nunique()
    conflicting_labels = conflicting_labels[conflicting_labels > 1]
    if not conflicting_labels.empty:
        raise ValueError(
            "Identical image content appears under different labels; "
            "resolve these conflicting source labels before splitting."
        )

    duplicate_count = int(dataframe["sha256"].duplicated().sum())
    if duplicate_count:
        dataframe = dataframe.drop_duplicates("sha256", keep="first").reset_index(drop=True)
        print(f"Removed {duplicate_count} duplicate image file(s) by SHA-256.")

    if dataframe["image_name"].duplicated().any():
        raise ValueError("Image names must be unique for flat split folders.")
    return dataframe


def split_images(
    dataframe: pd.DataFrame,
    dataset_dir: Path,
    output_dir: Path = OUTPUT_DIR,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Create stratified train/validation/test splits and copy their images."""
    dataframe = dataframe.reset_index(drop=True).copy()
    if "sha256" not in dataframe:
        dataframe["sha256"] = [
            hashlib.sha256(
                (dataset_dir / SOURCE_CLASSES[row.label] / row.image_name).read_bytes()
            ).hexdigest()
            for row in dataframe.itertuples(index=False)
        ]
    if dataframe["sha256"].duplicated().any():
        raise ValueError("Duplicate image content must be removed before splitting.")

    train_data, validation_test = train_test_split(
        dataframe,
        test_size=SPLIT_RATIOS[1] + SPLIT_RATIOS[2],
        stratify=dataframe["label"],
        random_state=seed,
    )
    validation_data, test_data = train_test_split(
        validation_test,
        test_size=SPLIT_RATIOS[2] / (SPLIT_RATIOS[1] + SPLIT_RATIOS[2]),
        stratify=validation_test["label"],
        random_state=seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLIT_NAMES:
        split_dir = output_dir / split
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir()

    split_data = pd.concat(
        [
            train_data.assign(split="train"),
            validation_data.assign(split="validation"),
            test_data.assign(split="test"),
        ]
    ).sort_index()
    for row in split_data.itertuples(index=False):
        shutil.copyfile(
            dataset_dir / SOURCE_CLASSES[row.label] / row.image_name,
            output_dir / row.split / row.image_name,
        )

    dataframe = split_data
    dataframe.to_csv(output_dir / "image_labels.csv", index=False)
    for split in SPLIT_NAMES:
        print(f"{split}: {(dataframe['split'] == split).sum()} images")
    return dataframe


def prepare_dataset(show_samples: bool = True) -> pd.DataFrame:
    """Extract images, build the label manifest, preview, and split train/test."""
    dataset_dir = extract_dataset()
    dataframe = create_image_dataframe(dataset_dir)
    if show_samples:
        display_dataset_samples(
            dataframe, dataset_dir, SOURCE_CLASSES, SAMPLES_PER_CLASS
        )
    dataframe = split_images(dataframe, dataset_dir, seed=RANDOM_SEED)
    print(f"Labels and splits saved in: {OUTPUT_DIR / 'image_labels.csv'}")
    return dataframe
