from pathlib import Path

import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import TQDMProgressBar

from tapestry.lane_detection.dataset import LaneDetectionDataset
from tapestry.lane_detection.model import LaneDetectionModel

# ---- Settings ----
data_dir = Path('data') / "lane_detection" # <-- change this to your local test path
max_debug_samples = 10
batch_size = 1

# ---- Load dataset ----
full_dataset = LaneDetectionDataset(data_dir, mode="train")
debug_subset = torch.utils.data.Subset(full_dataset, list(range(min(max_debug_samples, len(full_dataset)))))

# ---- Compute stats ----
stats = full_dataset.compute_statistics()

# ---- Init model ----
model = LaneDetectionModel(
    max_lanes=full_dataset.max_lanes,
    obj_pred_shape=(full_dataset.num_obj_pred_classes, full_dataset.num_bins),
    lat_vec_len=int(full_dataset.lat_coverage * 2),
    class_weights=stats["lane_weights"],
    obj_mean=stats["object_pred_means"].values,
    obj_std=stats["obj_score_stds"].values,
    lr=1e-3,
    dice_weight=0.5,
    data_config=full_dataset.data_config,
    obj_pred_config=full_dataset.obj_pred_config,
)

# ---- Dataloader ----
loader = DataLoader(debug_subset, batch_size=batch_size, shuffle=False)

# ---- Trainer ----
trainer = Trainer(
    accelerator="cpu",
    devices=1,
    logger=CSVLogger("debug_logs"),
    callbacks=[TQDMProgressBar(refresh_rate=1)],
    limit_predict_batches=max_debug_samples,
    log_every_n_steps=1
)

# ---- Run prediction ----
preds = trainer.predict(model, dataloaders=loader, return_predictions=True)
print(preds[0])
