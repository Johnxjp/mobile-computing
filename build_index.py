"""Pre-compute CLIP image embeddings and build a product metadata index."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
IMAGES_DIR = DATA_DIR / "images"
METADATA_FILE = DATA_DIR / "products.jsonl"
EMBEDDINGS_FILE = DATA_DIR / "clip_embeddings.npz"
PARQUET_FILE = DATA_DIR / "products.parquet"

MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"
BATCH_SIZE = 128
EMBEDDING_DIM = 512


def load_metadata() -> pd.DataFrame:
    """Load product metadata from JSONL into a DataFrame."""
    records = []
    with open(METADATA_FILE) as f:
        for line in f:
            records.append(json.loads(line))
    df = pd.DataFrame(records)
    print(f"Loaded metadata for {len(df)} products")
    return df


def build_embeddings(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Compute CLIP image embeddings for all products.

    Returns (product_ids, embeddings) where embeddings is (N, D) float32.
    """
    import open_clip
    import torch
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED, device=device
    )
    model.eval()

    product_ids = df["id"].values
    all_embeddings = []
    valid_ids = []
    skipped = 0

    for start in range(0, len(product_ids), BATCH_SIZE):
        batch_ids = product_ids[start : start + BATCH_SIZE]
        images = []
        batch_valid_ids = []

        for pid in batch_ids:
            image_path = IMAGES_DIR / f"{pid}.jpg"
            if not image_path.exists():
                skipped += 1
                continue
            try:
                img = Image.open(image_path).convert("RGB")
                images.append(preprocess(img))
                batch_valid_ids.append(pid)
            except Exception as e:
                print(f"  Warning: failed to load {image_path}: {e}")
                skipped += 1
                continue

        if not images:
            continue

        batch_tensor = torch.stack(images).to(device)
        with torch.no_grad():
            embeddings = model.encode_image(batch_tensor)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            all_embeddings.append(embeddings.cpu().numpy())

        valid_ids.extend(batch_valid_ids)

        processed = min(start + BATCH_SIZE, len(product_ids))
        if processed % 1000 < BATCH_SIZE or processed == len(product_ids):
            print(f"  Embedded {processed}/{len(product_ids)} products")

    if skipped:
        print(f"  Skipped {skipped} products (missing/corrupt images)")

    embeddings_array = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    ids_array = np.array(valid_ids)

    return ids_array, embeddings_array


def build_mock_embeddings(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Generate random embeddings for testing without network access."""
    product_ids = df["id"].values
    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((len(product_ids), EMBEDDING_DIM)).astype(np.float32)
    # L2 normalize to match real CLIP output
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    return product_ids, embeddings


def main():
    parser = argparse.ArgumentParser(description="Build CLIP embedding index")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Generate random embeddings (for testing without network access)",
    )
    args = parser.parse_args()

    df = load_metadata()

    if args.mock:
        print("\nGenerating mock embeddings (random, for testing only)...")
        product_ids, embeddings = build_mock_embeddings(df)
    else:
        print(f"\nLoading CLIP model ({MODEL_NAME}/{PRETRAINED})...")
        product_ids, embeddings = build_embeddings(df)
    print(f"Embeddings shape: {embeddings.shape}")

    np.savez(EMBEDDINGS_FILE, ids=product_ids, embeddings=embeddings)
    print(f"Saved embeddings to {EMBEDDINGS_FILE}")

    # Save metadata as parquet
    df.to_parquet(PARQUET_FILE, index=False)
    print(f"Saved metadata to {PARQUET_FILE}")

    # Print summary
    print(f"\nIndex built:")
    print(f"  Products: {len(product_ids)}")
    print(f"  Embedding dim: {embeddings.shape[1]}")
    print(
        f"  Embeddings file: {EMBEDDINGS_FILE} ({EMBEDDINGS_FILE.stat().st_size / 1e6:.1f} MB)"
    )
    print(
        f"  Metadata file: {PARQUET_FILE} ({PARQUET_FILE.stat().st_size / 1e6:.1f} MB)"
    )


if __name__ == "__main__":
    main()
