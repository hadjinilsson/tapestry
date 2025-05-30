import os
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

import argparse
import warnings
import logging
from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from dotenv import load_dotenv
from tqdm import tqdm

from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar
import torch

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


def generate_grid(bounds, grid_size, crs):
    minx, miny, maxx, maxy = bounds
    x_coords = list(range(int(minx), int(maxx), grid_size))
    y_coords = list(range(int(miny), int(maxy), grid_size))
    cells = []
    for x in x_coords:
        for y in y_coords:
            geom = box(x, y, x + grid_size, y + grid_size)
            cells.append(geom)
    return gpd.GeoDataFrame(geometry=cells, crs=crs)


def get_within_bounds(df, bounds):
    minx, miny, maxx, maxy = bounds
    bbox = box(minx, miny, maxx, maxy)
    return df[df.geometry.apply(lambda geom: geom.intersects(bbox))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    filter_type = parser.add_mutually_exclusive_group(required=True)
    filter_type.add_argument("--base-networks", nargs="+", type=str)
    filter_type.add_argument("--annotation-areas", nargs="*", type=str)
    filter_type.add_argument("--annotated-nodes", action="store_true")
    filter_type.add_argument("--all", action="store_true")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--s3-prefix", default="lane_detection")
    parser.add_argument("--grid-size", type=int, default=2500)
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

    if args.annotated_nodes:
        segs = db.get_link_segments_for_annotated_nodes(exclude_geom=False)
    elif args.annotation_areas is not None:
        area_names = args.annotation_areas if args.annotation_areas else None
        segs = db.get_link_segments_by_annotation_area(area_names=area_names, exclude_geom=False)
    elif args.base_networks:
        segs = db.get_link_segments_by_base_network(args.base_networks, exclude_geom=False)
    else:
        segs = db.get_all_link_segments(exclude_geom=False)

    segs['base_network_id'] = segs.link_segment_id.str.split('_', expand=True)[0]
    bn_ids = segs["base_network_id"].dropna().unique().tolist()
    pred_seg_ids = set()

    for bn_id in bn_ids:
        segs_bn = segs[segs["base_network_id"] == bn_id]
        bn_bbox = segs_bn.total_bounds  # [minx, miny, maxx, maxy]
        bn_cells = generate_grid(bn_bbox, args.grid_size, crs=segs_bn.crs)
        print(f"📱 {bn_id}: {len(bn_cells)} grid cells")

        bn_preds = []

        for bn_cell in tqdm(bn_cells.geometry, desc=f"🔍 Inference on {bn_id}", unit="cell"):
            segs_not_pred = segs_bn[~segs_bn["link_segment_id"].isin(pred_seg_ids)]
            if segs_not_pred.empty:
                continue

            segs_to_pred = get_within_bounds(segs_not_pred, bn_cell.bounds)
            if segs_to_pred.empty:
                continue

            bn_cell_buff = bn_cell.buffer(100)
            segs_in_cell = get_within_bounds(segs_bn, bn_cell_buff.bounds)
            cap_ids = segs_in_cell['camera_point_id'].unique()
            if not cap_ids.any():
                continue

            download_image_batch(cap_ids, IMAGE_DIR)

            ds = LaneDetectionDataset(data_root=DATA_DIR, mode="predict", seg_ids=list(segs_to_pred.link_segment_id))

            idx_to_keys = {
                i: s for i, s in enumerate(ds.samples)
            }

            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

            trainer = Trainer(
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices=1,
                logger=False,
                callbacks=[TQDMProgressBar(refresh_rate=20)],
                enable_model_summary=False,
                enable_progress_bar=True,
                log_every_n_steps=50,
            )

            model = LaneDetectionModel.load_from_checkpoint(checkpoint_path)
            model.eval()
            batches = trainer.predict(model, dataloaders=loader, return_predictions=True)

            delete_downloaded_images(IMAGE_DIR)

            for batch in batches:
                for pred in batch:
                    numerical_id = int(pred["numerical_id"])
                    key = idx_to_keys.get(numerical_id)
                    if key is None:
                        continue
                    seg_id, dist = key
                    bn_preds.append({
                        "link_segment_id": seg_id,
                        "distance": dist,
                        "pred_forward": int(pred["predicted_lanes"][0]),
                        "pred_backward": int(pred["predicted_lanes"][1]),
                        "prob_forward": float(pred["predicted_probs"][0]),
                        "prob_backward": float(pred["predicted_probs"][1]),
                        "label_forward": int(pred["label"][0]),
                        "label_backward": int(pred["label"][1]),
                        "has_label": bool(pred["has_label"]),
                    })

            pred_seg_ids.update(segs_to_pred["link_segment_id"].tolist())

        if bn_preds:
            bn_preds_df = pd.DataFrame(bn_preds)
            out_path = output_dir / f"{bn_id}.parquet"
            bn_preds_df.to_parquet(out_path, index=False)
            print(f"✅ Saved predictions for {bn_id} to {out_path}")

    if not args.no_upload:
        s3_key = f"{args.s3_prefix}/{args.run_id}"
        upload_dir_to_s3(output_dir, s3_key, S3_BUCKET_PREDICTIONS)


if __name__ == "__main__":
    main()
