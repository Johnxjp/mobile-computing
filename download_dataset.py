"""Download the Fashion Product Images (Small) dataset from Hugging Face."""

import json
from pathlib import Path

from datasets import load_dataset


DATA_DIR = Path("data")
IMAGES_DIR = DATA_DIR / "images"
METADATA_FILE = DATA_DIR / "products.jsonl"

DATASET_ID = "ashraq/fashion-product-images-small"
METADATA_FIELDS = [
    "id",
    "gender",
    "masterCategory",
    "subCategory",
    "articleType",
    "baseColour",
    "season",
    "year",
    "usage",
    "productDisplayName",
]


def main():
    print(f"Loading dataset: {DATASET_ID}")
    ds = load_dataset(DATASET_ID, split="train")
    print(f"Dataset loaded: {len(ds)} products")

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Saving images to {IMAGES_DIR}/ and metadata to {METADATA_FILE}")

    with open(METADATA_FILE, "w") as f:
        for i, row in enumerate(ds):
            product_id = row["id"]

            # Save image
            image = row["image"]
            image_path = IMAGES_DIR / f"{product_id}.jpg"
            image.save(image_path)

            # Collect metadata
            metadata = {field: row[field] for field in METADATA_FIELDS}
            metadata["image_path"] = str(image_path)
            f.write(json.dumps(metadata) + "\n")

            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{len(ds)} products")

    print(f"Done! Saved {len(ds)} products.")
    print(f"  Images: {IMAGES_DIR}/")
    print(f"  Metadata: {METADATA_FILE}")


if __name__ == "__main__":
    main()
