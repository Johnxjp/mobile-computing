"""Generate synthetic test data for development when HF is unreachable."""

import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

DATA_DIR = Path("data")
IMAGES_DIR = DATA_DIR / "images"
METADATA_FILE = DATA_DIR / "products.jsonl"

NUM_PRODUCTS = 200

COLORS = {
    "Red": (220, 50, 50),
    "Blue": (50, 80, 200),
    "Navy Blue": (20, 40, 100),
    "Black": (30, 30, 30),
    "White": (240, 240, 240),
    "Green": (50, 160, 60),
    "Yellow": (230, 200, 40),
    "Pink": (230, 100, 150),
    "Purple": (130, 50, 180),
    "Brown": (140, 90, 50),
    "Grey": (150, 150, 150),
    "Orange": (230, 130, 40),
    "Beige": (210, 190, 160),
}

CATEGORIES = {
    "Apparel": {
        "Topwear": ["Tshirts", "Shirts", "Tops", "Sweatshirts"],
        "Bottomwear": ["Jeans", "Trousers", "Shorts", "Skirts"],
        "Dress": ["Dresses"],
        "Innerwear": ["Bra", "Briefs"],
    },
    "Footwear": {
        "Shoes": ["Casual Shoes", "Sports Shoes", "Formal Shoes", "Sandals", "Heels"],
    },
    "Accessories": {
        "Watches": ["Watches"],
        "Bags": ["Handbags", "Backpacks", "Clutches"],
        "Jewellery": ["Earrings", "Necklace and Chains", "Ring"],
    },
}

GENDERS = ["Men", "Women", "Boys", "Girls", "Unisex"]
SEASONS = ["Summer", "Winter", "Fall", "Spring"]
USAGES = ["Casual", "Formal", "Sports", "Ethnic", "Party"]


def generate_product_image(color_rgb: tuple, article_type: str, pid: int) -> Image.Image:
    """Generate a simple colored product image with a label."""
    img = Image.new("RGB", (160, 200), color=(245, 245, 245))
    draw = ImageDraw.Draw(img)

    # Draw a colored rectangle as the "product"
    draw.rounded_rectangle([20, 20, 140, 160], radius=10, fill=color_rgb)

    # Add text label
    draw.text((80, 175), article_type[:15], fill=(80, 80, 80), anchor="mm")
    draw.text((80, 190), str(pid), fill=(160, 160, 160), anchor="mm")

    return img


def main():
    random.seed(42)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    products = []
    for pid in range(1, NUM_PRODUCTS + 1):
        master_cat = random.choice(list(CATEGORIES.keys()))
        sub_cat = random.choice(list(CATEGORIES[master_cat].keys()))
        article_type = random.choice(CATEGORIES[master_cat][sub_cat])
        color_name = random.choice(list(COLORS.keys()))
        gender = random.choice(GENDERS)
        season = random.choice(SEASONS)
        usage = random.choice(USAGES)

        product = {
            "id": pid,
            "gender": gender,
            "masterCategory": master_cat,
            "subCategory": sub_cat,
            "articleType": article_type,
            "baseColour": color_name,
            "season": season,
            "year": random.choice([2020, 2021, 2022, 2023, 2024]),
            "usage": usage,
            "productDisplayName": f"{gender} {color_name} {article_type}",
            "image_path": str(IMAGES_DIR / f"{pid}.jpg"),
        }
        products.append(product)

        img = generate_product_image(COLORS[color_name], article_type, pid)
        img.save(IMAGES_DIR / f"{pid}.jpg")

    with open(METADATA_FILE, "w") as f:
        for p in products:
            f.write(json.dumps(p) + "\n")

    print(f"Generated {NUM_PRODUCTS} synthetic products")
    print(f"  Images: {IMAGES_DIR}/")
    print(f"  Metadata: {METADATA_FILE}")


if __name__ == "__main__":
    main()
