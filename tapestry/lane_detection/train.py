import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import random_split
from pathlib import Path

from tapestry.lane_detection.dataset import LaneDetectionDataset
from tapestry.lane_detection.model import LaneDetectionModel, LaneDetectionDataModule

# ---- Settings ----
data_root = Path("data")  # ← replace with actual path
batch_size = 16
num_workers = 4
val_split = 0.2
seed = 42
max_epochs = 50

torch.manual_seed(seed)

# ---- Load dataset ----
full_dataset = LaneDetectionDataset(data_root=data_root, mode="train")
val_size = int(len(full_dataset) * val_split)
train_size = len(full_dataset) - val_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

# ---- Compute statistics ----
stats = full_dataset.compute_statistics()

# ---- Init model ----
model = LaneDetectionModel(
    max_lanes=full_dataset.max_lanes,
    obj_pred_shape=(full_dataset.num_obj_pred_classes, full_dataset.num_bins),
    lat_vec_len=int(full_dataset.lat_coverage * 2),
    class_weights=stats['lane_weights'],
    obj_mean=stats['object_pred_means'].values,
    obj_std=stats['obj_score_stds'].values,
    lr=1e-3,
    dice_weight=0.5
)

# ---- Data module ----
data_module = LaneDetectionDataModule(
    train_dataset=train_ds,
    val_dataset=val_ds,
    batch_size=batch_size,
    num_workers=num_workers
)

# ---- Logging ----
tb_logger = TensorBoardLogger("logs", name="lane_detection")

# ---- Callbacks ----
checkpoint_cb = ModelCheckpoint(
    monitor="val_loss",
    mode="min",
    save_top_k=1,
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

# ---- Train ----
trainer.fit(model, datamodule=data_module)
