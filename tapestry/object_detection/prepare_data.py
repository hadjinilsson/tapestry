import argparse
from pathlib import Path
from tapestry.label_studio.parse_annotations import parse_annotations
from tapestry.utils.image_fetching import download_image
from tapestry.utils.db import get_camera_point_ids

# ─────────────── CONFIG ───────────────
DATA_ROOT = Path("data")
IMAGE_DIR = DATA_ROOT / "images"
LABEL_JSON_DIR = DATA_ROOT / "labels_json"
BATCH_SIZE = 1000

# ─────────────── TRAINING MODE ───────────────
def run_training_mode(min_group_size: int | None):
    print("🧠 Running in training mode...")
    split_map = parse_annotations(DATA_ROOT, min_group_size=min_group_size)  # or set int for aggregation
    for image_stub, split in split_map.items():
        img_path = IMAGE_DIR / split / f"{image_stub}.png"
        json_path = LABEL_JSON_DIR / split / f"{image_stub}.json"
        img_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        download_image(image_stub, dest_image_path=img_path, dest_json_path=json_path)
    print("✅ Training data ready.")

# ─────────────── INFERENCE MODE ───────────────
def run_inference_mode(base_network: str):
    print(f"🔍 Running in inference mode for network: {base_network}")
    camera_point_ids = get_camera_point_ids(base_network)
    print(f"Found {len(camera_point_ids)} camera points to download.")

    infer_dir = IMAGE_DIR / "infer" / base_network
    infer_dir.mkdir(parents=True, exist_ok=True)

    for i in range(0, len(camera_point_ids), BATCH_SIZE):
        batch = camera_point_ids[i:i + BATCH_SIZE]
        print(f"📦 Downloading batch {i // BATCH_SIZE + 1} ({len(batch)} images)...")
        for cp_id in batch:
            img_path = infer_dir / f"{cp_id}.png"
            download_image(cp_id, dest_image_path=img_path)

    print("✅ Inference image download complete.")

# ─────────────── ENTRY POINT ───────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "infer"], required=True)
    parser.add_argument("--base-network", help="Required for inference mode")
    parser.add_argument("--aggregation", type=int, help="Minimum group size for label aggregation (omit to disable)")
    args = parser.parse_args()

    if args.mode == "train":
        run_training_mode(min_group_size=args.aggregation)
    elif args.mode == "infer":
        if not args.base_network:
            raise ValueError("--base-network is required in inference mode")
        run_inference_mode(args.base_network)

if __name__ == "__main__":
    main()