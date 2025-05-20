from pathlib import Path
from tapestry.lane_detection.dataset import LaneDetectionDataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


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

    ax.set_title(f"Link Segment: {sample['link_segment_id']}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

train_ds = LaneDetectionDataset(data_root=Path('data'), mode="train")
inf_ds = LaneDetectionDataset(data_root=Path('data'), mode="inference")

# Pick an example
# sample = train_ds[102]
sample = train_ds[np.random.randint(len(train_ds))]

show_sample_with_predictions(sample)

sample


# 23823


# import torchvision.transforms.functional as TF
# from torchvision.utils import save_image
#
# # Extract image tensor
# image_tensor = sample["image_tensor"]  # shape: (3, H, W), float [0, 1]
#
# # Optional: clamp values just in case of rounding noise
# image_tensor = image_tensor.clamp(0, 1)
#
# # Save to disk
# save_path = "stitched_image.png"
# save_image(image_tensor, save_path)
#
# # Optional: call only geometry logic
# lon_segments, lat_segments= train_ds.get_context_segments(
#     link_segment_id=sample["link_segment_id"],
#     distance_on_line=sample["sample_distance"],
#     cross_section=sample["cross_section"],
# )
#
# print("LONGITUDINAL:")
# for seg in lon_segments:
#     print(seg["link_segment_id"], seg["used_length"])
#
# print("LATERAL:")
# for neigh in lat_segments:
#     print(neigh["neighbor_segment_id"], neigh["intersection_point"])
#
# lon_segments