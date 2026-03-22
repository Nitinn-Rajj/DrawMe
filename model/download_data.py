"""
DrawMe - Data Download Script
Downloads Google Quick, Draw! .npy bitmap files for selected categories.
"""

import os
import requests

# 15 distinct categories
CATEGORIES = [
    "cloud", "sun", "tree", "car", "fish",
    "cat", "dog", "house", "star", "flower",
    "bird", "bicycle", "guitar", "moon", "hat"
]

BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def download_category(category: str) -> None:
    """Download a single category .npy file from Google Cloud Storage."""
    os.makedirs(DATA_DIR, exist_ok=True)

    filename = f"{category}.npy"
    filepath = os.path.join(DATA_DIR, filename)

    if os.path.exists(filepath):
        print(f"  ✓ {category}.npy already exists, skipping.")
        return

    # The URL uses the category name with spaces replaced by '%20'
    # but our categories don't have spaces, so direct use is fine.
    url = f"{BASE_URL}/{requests.utils.quote(category)}.npy"
    print(f"  ⬇ Downloading {category}.npy ...")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(filepath, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = (downloaded / total_size) * 100
                print(f"\r    {downloaded / 1e6:.1f} MB / {total_size / 1e6:.1f} MB ({pct:.0f}%)", end="", flush=True)

    print(f"\n  ✓ Saved {category}.npy ({downloaded / 1e6:.1f} MB)")


def download_all() -> None:
    """Download all category .npy files."""
    print("=" * 50)
    print("DrawMe — Quick, Draw! Dataset Downloader")
    print("=" * 50)
    print(f"\nDownloading {len(CATEGORIES)} categories to: {DATA_DIR}\n")

    for i, cat in enumerate(CATEGORIES, 1):
        print(f"[{i}/{len(CATEGORIES)}] {cat}")
        try:
            download_category(cat)
        except requests.exceptions.HTTPError as e:
            print(f"  ✗ Failed to download {cat}: {e}")
        except Exception as e:
            print(f"  ✗ Unexpected error for {cat}: {e}")
        print()

    print("=" * 50)
    print("Download complete!")
    print("=" * 50)


if __name__ == "__main__":
    download_all()
