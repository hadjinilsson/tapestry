import argparse
import os
import shutil
from datetime import datetime
from dotenv import load_dotenv
from tapestry.utils.s3 import upload_dir_to_s3
from tapestry.utils.config import save_args

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import random_split
from pathlib import Path

from tapestry.lane_detection.dataset import LaneDetectionDataset
from tapestry.lane_detection.model import LaneDetectionModel, LaneDetectionDataModule


load_dotenv()
S3_BUCKET = os.getenv("BUCKET_NAME_MODELS")

# ---- Settings ----
DATA_DIR = Path("data") / "lane_detection"


def train(
        use_prior_preds=False,
        batch_size:int = 16,
        num_workers: int = 4,
        val_split: float = 0.2,
        seed: int = 42,
        max_epochs: int = 50,
        train_config: dict | None = None,
    ):

    torch.manual_seed(seed)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---- Load dataset ----
    full_dataset = LaneDetectionDataset(data_root=DATA_DIR, mode="train", use_prior_preds=use_prior_preds)
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    # ---- Compute statistics ----
    stats = full_dataset.compute_statistics()

    # ---- Init model ----
    model = LaneDetectionModel(
        use_prior_preds=use_prior_preds,
        max_lanes=full_dataset.max_lanes,
        obj_pred_shape=(full_dataset.num_obj_pred_classes, full_dataset.num_bins),
        prior_pred_shape=((full_dataset.max_lanes + 1) * 2 + 5, full_dataset.num_prior_preds),
        lat_vec_len=int(full_dataset.lat_coverage * 2),
        class_weights=stats['lane_weights'],
        obj_mean=stats['object_pred_means'].values,
        obj_std=stats['obj_score_stds'].values,
        lr=1e-3,
        dice_weight=0.5,
        run_id=run_id,
        data_config=full_dataset.data_config,
        obj_pred_config=full_dataset.obj_pred_config,
        train_config=train_config,
    )

    # ---- Data module ----
    data_module = LaneDetectionDataModule(
        train_dataset=train_ds,
        val_dataset=val_ds,
        batch_size=batch_size,
        num_workers=num_workers
    )

    # ---- Logging ----
    tb_logger = TensorBoardLogger(DATA_DIR / "logs", name="lane_detection", version=run_id)

    # ---- Callbacks ----
    checkpoint_cb = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        dirpath=tb_logger.log_dir,
        filename="best-checkpoint"
    )
    early_stop_cb = EarlyStopping(monitor="val_loss", patience=5, mode="min")

    # ---- Trainer ----
    trainer = Trainer(
        logger=tb_logger,
        max_epochs=max_epochs,
        callbacks=[checkpoint_cb, early_stop_cb],
        accelerator="gpu",
        devices=1 if torch.cuda.is_available() else None,
        log_every_n_steps=10
    )

    if use_prior_preds:
        print("🧠 Using prior predictions (second-stage model)")
    else:
        print("🔰 Training without prior predictions (first-stage model)")

    # ---- Train ----
    trainer.fit(model, datamodule=data_module)

    run_dir = Path(tb_logger.log_dir)

    return run_id, run_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-prior-preds", action="store_true", help="Use prior predictions in model input")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--no-upload", action="store_true", help="Disable uploading run to S3")
    parser.add_argument("--s3-prefix", default="lane_detection")
    args = parser.parse_args()

    run_id, run_dir = train(
        args.use_prior_preds,
        args.batch_size,
        args.num_workers,
        args.val_split,
        args.seed,
        args.max_epochs,
        vars(args),
    )
    print(f"🚀 Training complete: Run ID = {run_id}")

    shutil.copy(DATA_DIR / "data_config.json", run_dir)
    shutil.copy(DATA_DIR / "object_predictions" / "object_prediction_config.json", run_dir)
    save_args(args, run_dir / "train_config.json")

    if not args.no_upload:
        print("☁️ Uploading run to S3...")
        upload_dir_to_s3(Path(run_dir), f"{args.s3_prefix}/{run_id}", S3_BUCKET)


if __name__ == "__main__":
    main()