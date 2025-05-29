import os
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

import argparse
import warnings
import logging
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from collections import defaultdict
from tqdm import tqdm

from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar

from tapestry.utils import db
from tapestry.utils.s3 import download_dir_from_s3, upload_dir_to_s3
from tapestry.utils.image_fetching import download_image_batch, delete_downloaded_images
from tapestry.utils.config import save_args
from tapestry.lane_detection.model import LaneDetectionModel
from tapestry.lane_detection.dataset import LaneDetectionDataset

load_dotenv()

S3_BUCKET_MODELS = os.getenv("BUCKET_NAME_MODELS")
S3_BUCKET_PREDICTIONS = os.getenv("BUCKET_NAME_PREDICTIONS")
DATA_DIR = Path("data") / "lane_detection"
IMAGE_DIR = DATA_DIR / "images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("torch").setLevel(logging.ERROR)


def run_inference(checkpoint_path, output_dir, base_network, camera_ids, batch_size):
    model = LaneDetectionModel.load_from_checkpoint(checkpoint_path)
    model.eval()

    all_preds = []
    batches = [camera_ids[i:i + batch_size] for i in range(0, len(camera_ids), batch_size)]

    for batch_idx, batch in enumerate(tqdm(batches, desc=f"🔍 Inference on {base_network}", unit="batch")):
        print(f"📦 Processing batch {batch_idx + 1} ({len(batch)} images)...")
        batch: list[str]
        download_image_batch(batch, IMAGE_DIR)
        dataset = LaneDetectionDataset(data_root=DATA_DIR, mode="predict")
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

        trainer = Trainer(
            accelerator="cpu",
            devices=1,
            logger=False,
            callbacks=[TQDMProgressBar(refresh_rate=20)],
            enable_model_summary=False,
            enable_progress_bar=True,
            log_every_n_steps=50,
        )

        predictions = trainer.predict(model, dataloaders=loader, return_predictions=True)
        all_preds.extend(predictions)
        delete_downloaded_images(IMAGE_DIR)

    df = pd.DataFrame([
        {
            "camera_point_id": pred["numerical_id"],
            "pred_forward": int(pred["predicted_lanes"][0]),
            "pred_backward": int(pred["predicted_lanes"][1]),
            "logit_forward": float(pred["predicted_logits"][0]),
            "logit_backward": float(pred["predicted_logits"][1]),
            "label_forward": int(pred["label"][0]),
            "label_backward": int(pred["label"][1]),
            "has_label": bool(pred["has_label"]),
        }
        for pred in all_preds
    ])

    out_path = output_dir / f"{base_network}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"✅ Saved predictions to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    filter_type = parser.add_mutually_exclusive_group(required=True)
    filter_type.add_argument("--base-networks", nargs="+", type=str)
    filter_type.add_argument("--annotation-areas", nargs="*", type=str)
    filter_type.add_argument("--annotated-link-segments", action="store_true")
    filter_type.add_argument("--annotated-nodes", action="store_true")
    filter_type.add_argument("--all", action="store_true")
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--s3-prefix", default="lane_detection")
    args = parser.parse_args()

    output_dir = DATA_DIR / "runs" / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    save_args(args, output_dir / "predict_config.json")

    checkpoint_path = DATA_DIR / "checkpoints" / args.run_id / "best-checkpoint.ckpt"
    if not checkpoint_path.exists():
        print(f"⬇️ Downloading checkpoint for run {args.run_id}...")
        download_dir_from_s3(
            s3_prefix=f"{args.s3_prefix}/{args.run_id}",
            local_dir=checkpoint_path.parent,
            bucket=S3_BUCKET_MODELS,
        )

    if args.annotated_link_segments:
        df = db.get_annotated_link_segments(exclude_geom=True)
    elif args.annotated_nodes:
        df = db.get_link_segments_for_annotated_nodes(exclude_geom=True)
    elif args.annotation_areas is not None:
        area_names = args.annotation_areas if args.annotation_areas else None
        df = db.get_link_segments_by_annotation_area(area_names=area_names, exclude_geom=True)
    elif args.base_networks:
        df = db.get_link_segments_by_base_network(args.base_networks, exclude_geom=True)
    else:
        df = db.get_all_link_segments(exclude_geom=True)

    camera_ids = df["camera_point_id"].dropna().unique().tolist()
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
