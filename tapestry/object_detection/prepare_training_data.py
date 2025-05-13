import argparse
from pathlib import Path
from tapestry.label_studio.parse_annotations import parse_annotations
from tapestry.utils.image_fetching import download_image

# ─────────────── CONFIG ───────────────
DATA_ROOT = Path("data")
IMAGE_DIR = DATA_ROOT / "images"
LABEL_JSON_DIR = DATA_ROOT / "labels_json"
BATCH_SIZE = 1000

# ─────────────── TRAINING MODE ───────────────
def prepare_training_data(min_group_size: int | None):
    print("🧠 Running in training mode...")
    split_map = parse_annotations(DATA_ROOT, min_group_size=min_group_size)  # or set int for aggregation
    for image_stub, split in split_map.items():
        img_path = IMAGE_DIR / split / f"{image_stub}.png"
        json_path = LABEL_JSON_DIR / split / f"{image_stub}.json"
        img_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        download_image(image_stub, dest_image_path=img_path, dest_json_path=json_path)
    print("✅ Training data ready.")

# ─────────────── ENTRY POINT ───────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregation", type=int, help="Minimum group size for label aggregation (omit to disable)")
    args = parser.parse_args()
    prepare_training_data(min_group_size=args.aggregation)

if __name__ == "__main__":
    main()