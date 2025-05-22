from pathlib import Path
from tapestry.lane_detection.dataset import LaneDetectionDataset
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def validate_shapes(dataset, max_samples=1000):
    from collections import defaultdict

    shape_tracker = defaultdict(set)

    for i in range(min(len(dataset), max_samples)):
        sample = dataset[i]

        for key, value in sample.items():
            if not isinstance(value, torch.Tensor):
                continue
            shape_tracker[key].add(tuple(value.shape))

    for key, shapes in shape_tracker.items():
        print(f"{key}: {len(shapes)} unique shapes")
        for shape in shapes:
            print(f"  - {shape}")

    print("\n✅ Done. If any tensor has more than one unique shape, it's a batching risk.")


def show_sample_with_predictions(sample):
    img = sample["image"].permute(1, 2, 0).numpy()  # (H, W, 3)
    preds = sample["object_predictions"]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img)

    for _, row in preds.iterrows():
        x, y = row["x_center"], row["y_center"]
        w, h = row["width"], row["height"]
        rect = patches.Rectangle(
            (x - w/2, y - h/2), w, h,
            linewidth=1,
            edgecolor="lime",
            facecolor="none"
        )
        ax.add_patch(rect)

    ax.set_title(f"Link Segment: {sample['numerical_id']}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

train_ds = LaneDetectionDataset(data_root=Path('data'), mode="train")
# inf_ds = LaneDetectionDataset(data_root=Path('data'), mode="inference")

# validate_shapes(train_ds)

stats = train_ds.compute_statistics()

# sample = train_ds[5]
# sample = train_ds[np.random.randint(len(train_ds))]
# show_sample_with_predictions(sample)

# sample = inf_ds[np.random.randint(len(inf_ds))]
# show_sample_with_predictions(sample)
