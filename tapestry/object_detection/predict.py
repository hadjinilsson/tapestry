import argparse
import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from ultralytics import YOLO

from tapestry.utils.image_fetching import download_image_batch, delete_downloaded_images
from tapestry.utils import db
from tapestry.utils.s3 import download_dir_from_s3, upload_dir_to_s3
from tapestry.utils.config import save_args
from collections import defaultdict
from tqdm import tqdm

load_dotenv()

S3_BUCKET_MODELS = os.getenv("BUCKET_NAME_MODELS")
S3_BUCKET_PREDICTIONS = os.getenv("BUCKET_NAME_PREDICTIONS")
DATA_DIR = Path("data") / "object_detection" / "predict"
IMAGE_DIR = DATA_DIR / "images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────── RUN INFERENCE ───────────────
def run_inference(model_path: Path, output_dir: Path, base_network: str, camera_ids: list[str], batch_size: int):
    model = YOLO(str(model_path))
    all_preds = []

    batches = [
        camera_ids[i:i + batch_size]
        for i in range(0, len(camera_ids), batch_size)
    ]

    for batch_idx, batch in enumerate(tqdm(batches, desc=f"🔍 Inference on {base_network}", unit="batch"), 1):
        print(f"📦 Processing batch {batch_idx} ({len(batch)} images)...")

        # Download batch
        download_image_batch(batch, IMAGE_DIR)
        image_paths = [IMAGE_DIR / f"{cp_id}.png" for cp_id in batch if (IMAGE_DIR / f"{cp_id}.png").exists()]

        try:
            results = model.predict(image_paths, save=False, verbose=False)
            for cp_id, result in zip(batch, results):
                img_w, img_h = result.orig_shape[1], result.orig_shape[0]
                for box in result.boxes:
                    cls = int(box.cls.item())
                    conf = float(box.conf.item())
                    x_center_norm = box.xywh[0][0].item() / img_w
                    y_center_norm = box.xywh[0][1].item() / img_h
                    width_norm = box.xywh[0][2].item() / img_w
                    height_norm = box.xywh[0][3].item() / img_h
                    all_preds.append({
                        "camera_point_id": cp_id,
                        "class": cls,
                        "confidence": conf,
                        "x_center": x_center_norm,
                        "y_center": y_center_norm,
                        "width": width_norm,
                        "height": height_norm,
                    })
        finally:
            delete_downloaded_images(IMAGE_DIR)

    out_path = output_dir / f"{base_network}.parquet"
    pd.DataFrame(all_preds).to_parquet(out_path, index=False)
    print(f"✅ Saved predictions to {out_path}")


# ─────────────── ENTRY ───────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    filter_type = parser.add_mutually_exclusive_group(required=True)
    filter_type.add_argument("--base-networks", nargs="+", type=str)
    filter_type.add_argument("--annotation-areas", nargs="*", type=str)
    filter_type.add_argument("--annotated-link-segments", action="store_true")
    filter_type.add_argument("--annotated-nodes", action="store_true")
    filter_type.add_argument("--all", action="store_true")
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--s3-prefix", default="object_detection")
    args = parser.parse_args()

    output_dir = DATA_DIR / "runs" / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    save_args(args, output_dir / "predict_config.json")

    checkpoint_path = DATA_DIR / "checkpoints" / args.run_id / "best.pt"
    if not checkpoint_path.exists():
        print(f"⬇️ Downloading checkpoint for run {args.run_id}...")
        download_dir_from_s3(
            s3_prefix=f"object_detection/{args.run_id}/weights",
            local_dir=checkpoint_path.parent,
            bucket=S3_BUCKET_MODELS,
        )

    if args.annotated_link_segments:
        segs = db.get_annotated_link_segments(exclude_geom=True)
    elif args.annotated_nodes:
        segs = db.get_link_segments_for_annotated_nodes(exclude_geom=True)
    elif args.annotation_areas is not None:
        area_names = args.annotation_areas if args.annotation_areas else None
        segs = db.get_link_segments_by_annotation_area(area_names=area_names, exclude_geom=True)
    elif args.base_networks:
        segs = db.get_link_segments_by_base_network(args.base_networks, exclude_geom=True)
    else:
        segs = db.get_all_link_segments(exclude_geom=True)

    camera_ids = segs["camera_point_id"].dropna().unique().tolist()

    grouped = defaultdict(list)
    for cp_id in camera_ids:
        base = cp_id.split("_")[0]
        grouped[base].append(cp_id)

    for base_network, ids in grouped.items():
        run_inference(checkpoint_path, output_dir, base_network, ids, args.batch_size)

    if not args.no_upload:
        s3_key = f"{args.s3_prefix}/{args.run_id}"
        upload_dir_to_s3(output_dir, s3_key, S3_BUCKET_PREDICTIONS)

if __name__ == "__main__":
    main()
