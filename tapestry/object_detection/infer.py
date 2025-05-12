import argparse
import os
from pathlib import Path
import boto3
import pandas as pd
from dotenv import load_dotenv
from ultralytics import YOLO
from tapestry.utils.image_fetching import download_image
from tapestry.utils.db import get_camera_point_ids, get_camera_points_for_annotated_link_segments
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil

load_dotenv()

# S3 config
S3_BUCKET_MODELS = os.getenv("BUCKET_NAME_MODELS")
S3_BUCKET_PREDICTIONS = os.getenv("BUCKET_NAME_PREDICTIONS")
S3_ENDPOINT = os.getenv("AWS_S3_ENDPOINT")
AWS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY")

TEMP_BATCH_DIR = Path("data/images/temp_infer_batch")
TEMP_BATCH_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────── DOWNLOAD MODEL CHECKPOINT ───────────────
def download_checkpoint(run_id: str, local_path: Path):
    s3_key = f"object_detection/{run_id}/weights/best.pt"
    local_path.parent.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3", endpoint_url=S3_ENDPOINT, aws_access_key_id=AWS_KEY, aws_secret_access_key=AWS_SECRET)
    s3.download_file(S3_BUCKET_MODELS, s3_key, str(local_path))
    print(f"✅ Model checkpoint downloaded to {local_path}")

# ─────────────── DOWNLOAD IMAGES ───────────────
def download_one(cp_id: str):
    img_path = TEMP_BATCH_DIR / f"{cp_id}.png"
    if not img_path.exists():
        download_image(cp_id, dest_image_path=img_path)

def download_image_batch(camera_ids: list[str], max_workers: int = 10):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_one, cp_id) for cp_id in camera_ids]
        for i, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
            except Exception as e:
                print(f"⚠️ Failed to download image {camera_ids[i-1]}: {e}")

# ─────────────── RUN INFERENCE ───────────────
def run_inference(model_path: Path, run_id: str, base_network: str, camera_ids: list[str], batch_size: int):
    model = YOLO(str(model_path))
    all_preds = []

    batches = [
        camera_ids[i:i + batch_size]
        for i in range(0, len(camera_ids), batch_size)
    ]

    for batch in tqdm(batches, desc=f"🔍 Inference on {base_network}", unit="batch"):
        batch = camera_ids[i:i + batch_size]
        print(f"📦 Processing batch {i // batch_size + 1} ({len(batch)} images)...")

        # Download batch
        download_image_batch(batch)
        image_paths = [TEMP_BATCH_DIR / f"{cp_id}.png" for cp_id in batch if (TEMP_BATCH_DIR / f"{cp_id}.png").exists()]

        try:

            # Run inference
            results = model.predict(image_paths, save=False, verbose=False)

            for cp_id, result in zip(batch, results):
                for box in result.boxes:
                    cls = int(box.cls.item())
                    conf = float(box.conf.item())
                    xywh = box.xywh[0].tolist()
                    all_preds.append({
                        "camera_point_id": cp_id,
                        "class": cls,
                        "confidence": conf,
                        "x_center": xywh[0],
                        "y_center": xywh[1],
                        "width": xywh[2],
                        "height": xywh[3],
                    })

        finally:

            # Clean up
            for img_path in TEMP_BATCH_DIR.glob("*.png"):
                img_path.unlink(missing_ok=True)

    output_dir = Path("predictions") / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{base_network}.parquet"
    pd.DataFrame(all_preds).to_parquet(out_path, index=False)
    print(f"✅ Saved predictions to {out_path}")
    return out_path

# ─────────────── UPLOAD TO S3 ───────────────
def upload_to_s3(local_file: Path, s3_prefix: str):
    s3 = boto3.client("s3", endpoint_url=S3_ENDPOINT, aws_access_key_id=AWS_KEY, aws_secret_access_key=AWS_SECRET)
    s3_key = f"{s3_prefix}/{local_file.name}"
    s3.upload_file(str(local_file), S3_BUCKET_PREDICTIONS, s3_key)
    print(f"☁️ Uploaded prediction: {s3_key}")

# ─────────────── ENTRY ───────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--base-network")
    group.add_argument("--camera-point-list", help="Path to .txt file with one camera_point_id per line")
    group.add_argument("--use-annotated-link-segments", action="store_true")
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--no-upload", action="store_true", help="Disable S3 upload")
    parser.add_argument("--s3-prefix", default="object_detection")
    args = parser.parse_args()

    checkpoint_path = Path("checkpoints") / args.run_id / "best.pt"
    if not checkpoint_path.exists():
        download_checkpoint(args.run_id, checkpoint_path)

    if args.use_annotated_link_segments:
        camera_ids = get_camera_points_for_annotated_link_segments()
    elif args.camera_point_list:
        with open(args.camera_point_list) as f:
            camera_ids = [line.strip() for line in f if line.strip()]
    else:
        camera_ids = get_camera_point_ids(args.base_network)

    # Group camera points by base network prefix
    grouped = defaultdict(list)
    for cp_id in camera_ids:
        base = cp_id.split("_")[0]
        grouped[base].append(cp_id)

    for base_network, ids in grouped.items():
        output_file = run_inference(checkpoint_path, args.run_id, base_network, ids, args.batch_size)
        if not args.no_upload:
            upload_to_s3(output_file, f"{args.s3_prefix}/{args.run_id}")

if __name__ == "__main__":
    main()