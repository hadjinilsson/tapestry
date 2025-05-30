import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import pytorch_lightning as pl
from torchvision.models import ResNet18_Weights
from torchmetrics import Accuracy, Recall, F1Score


def soft_dice_loss(logits, targets, smooth=1e-5):
    probs = F.softmax(logits, dim=-1)
    targets_one_hot = F.one_hot(targets, num_classes=logits.shape[-1]).float()
    intersection = (probs * targets_one_hot).sum(dim=-1)
    union = probs.sum(dim=-1) + targets_one_hot.sum(dim=-1)
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


class LaneDetectionModel(pl.LightningModule):
    def __init__(
        self,
        max_lanes=5,
        obj_pred_shape=(15, 50),
        lat_vec_len=70,
        image_feat_dim=128,
        obj_feat_dim=64,
        lat_feat_dim=32,
        hidden_dim=128,
        class_weights=None,
        obj_mean=None,
        obj_std=None,
        lr=1e-3,
        dice_weight=0.5,
        freeze_resnet_blocks=True,
        run_id = None,
        data_config = None,
        obj_pred_config = None,
        train_config = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.max_lanes = max_lanes
        self.num_lane_classes = max_lanes + 1
        self.dice_weight = dice_weight

        self.register_buffer("obj_mean", torch.tensor(obj_mean, dtype=torch.float32) if obj_mean is not None else torch.zeros(obj_pred_shape))
        self.register_buffer("obj_std", torch.tensor(obj_std, dtype=torch.float32) if obj_std is not None else torch.ones(obj_pred_shape))
        self.register_buffer("class_weights", class_weights if class_weights is not None else torch.ones((2, self.num_lane_classes)))

        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        if freeze_resnet_blocks:
            for name, param in resnet.named_parameters():
                if "layer1" in name or "layer2" in name:
                    param.requires_grad = False

        self.image_encoder = nn.Sequential(
            *list(resnet.children())[:-2],
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(resnet.fc.in_features, image_feat_dim)
        )

        self.obj_encoder = nn.Sequential(
            nn.Conv1d(obj_pred_shape[0], 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, obj_feat_dim),
            nn.ReLU()
        )

        self.lat_encoder = nn.Sequential(
            nn.Linear(lat_vec_len, lat_feat_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        total_feat_dim = image_feat_dim + obj_feat_dim + lat_feat_dim
        self.classifier = nn.Sequential(
            nn.Linear(total_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 2 * self.num_lane_classes)
        )

        self.val_accuracy_fwd = Accuracy(task="multiclass", num_classes=self.num_lane_classes, average='weighted')
        self.val_recall_fwd = Recall(task="multiclass", num_classes=self.num_lane_classes, average='weighted')
        self.val_f1_fwd = F1Score(task="multiclass", num_classes=self.num_lane_classes, average='weighted')

        self.val_accuracy_bwd = Accuracy(task="multiclass", num_classes=self.num_lane_classes, average='weighted')
        self.val_recall_bwd = Recall(task="multiclass", num_classes=self.num_lane_classes, average='weighted')
        self.val_f1_bwd = F1Score(task="multiclass", num_classes=self.num_lane_classes, average='weighted')

    def forward(self, image, object_scores, lat_neighbours):
        object_scores = (object_scores - self.obj_mean[:, None]) / self.obj_std[:, None]
        object_scores = object_scores.unsqueeze(0) if object_scores.ndim == 2 else object_scores
        img_feat = self.image_encoder(image)
        obj_feat = self.obj_encoder(object_scores)
        lat_feat = self.lat_encoder(lat_neighbours)
        x = torch.cat([img_feat, obj_feat, lat_feat], dim=-1)
        logits = self.classifier(x)
        return logits.view(-1, 2, self.num_lane_classes)

    def compute_loss(self, logits, targets):
        loss_fwd = F.cross_entropy(logits[:, 0], targets[:, 0], weight=self.class_weights[0])
        loss_bwd = F.cross_entropy(logits[:, 1], targets[:, 1], weight=self.class_weights[1])
        dice_fwd = soft_dice_loss(logits[:, 0], targets[:, 0])
        dice_bwd = soft_dice_loss(logits[:, 1], targets[:, 1])
        ce_loss = loss_fwd + loss_bwd
        dice_loss = dice_fwd + dice_bwd
        return ce_loss + self.dice_weight * dice_loss

    def training_step(self, batch, batch_idx):
        logits = self(batch["image"], batch["object_scores"], batch["lat_neighbours"])
        targets = batch["sections"].argmax(dim=-1)
        loss = self.compute_loss(logits, targets)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["image"], batch["object_scores"], batch["lat_neighbours"])
        targets = batch["sections"].argmax(dim=-1)
        loss = self.compute_loss(logits, targets)

        acc_fwd = self.val_accuracy_fwd(logits[:, 0], targets[:, 0])
        acc_bwd = self.val_accuracy_bwd(logits[:, 1], targets[:, 1])
        recall_fwd = self.val_recall_fwd(logits[:, 0], targets[:, 0])
        recall_bwd = self.val_recall_bwd(logits[:, 1], targets[:, 1])
        f1_fwd = self.val_f1_fwd(logits[:, 0], targets[:, 0])
        f1_bwd = self.val_f1_bwd(logits[:, 1], targets[:, 1])

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc_forward", acc_fwd, prog_bar=True)
        self.log("val_acc_backward", acc_bwd, prog_bar=True)
        self.log("val_recall_forward", recall_fwd)
        self.log("val_recall_backward", recall_bwd)
        self.log("val_f1_forward", f1_fwd)
        self.log("val_f1_backward", f1_bwd)

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        logits = self(batch["image"], batch["object_scores"], batch["lat_neighbours"])  # (B, 2, C)
        preds = torch.argmax(logits, dim=-1)  # (B, 2)
        pred_logits = torch.gather(logits, dim=2, index=preds.unsqueeze(-1)).squeeze(-1)  # (B, 2)

        if "label" in batch and "has_label" in batch:
            labels = torch.argmax(batch["label"], dim=-1)  # (B, 2)
            has_label = batch["has_label"]  # (B,)
        else:
            labels = torch.zeros_like(preds)
            has_label = torch.zeros(preds.size(0), dtype=torch.bool)

        batch_size = preds.shape[0]
        results = []

        for i in range(batch_size):
            results.append({
                "numerical_id": batch["numerical_id"][i],  # scalar tensor
                "predicted_lanes": preds[i],  # tensor of shape (2,)
                "predicted_logits": pred_logits[i],  # tensor of shape (2,)
                "label": labels[i],  # tensor of shape (2,)
                "has_label": has_label[i]  # tensor of shape ()
            })

        return results

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class LaneDetectionDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, batch_size=16, num_workers=4):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
