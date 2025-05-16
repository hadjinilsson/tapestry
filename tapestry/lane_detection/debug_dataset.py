from pathlib import Path
from tapestry.lane_detection.dataset import LaneDetectionDataset

train_ds = LaneDetectionDataset(data_root=Path('data'), mode="train")
inf_ds = LaneDetectionDataset(data_root=Path('data'), mode="inference")

# Pick an example
sample = train_ds[125]

import torchvision.transforms.functional as TF
from torchvision.utils import save_image

# Extract image tensor
image_tensor = sample["image_tensor"]  # shape: (3, H, W), float [0, 1]

# Optional: clamp values just in case of rounding noise
image_tensor = image_tensor.clamp(0, 1)

# Save to disk
save_path = "stitched_image.png"
save_image(image_tensor, save_path)

# Optional: call only geometry logic
lon_segments, lat_segments= train_ds.get_context_segments(
    link_segment_id=sample["link_segment_id"],
    distance_on_line=sample["sample_distance"],
    cross_section=sample["cross_section"],
)

print("LONGITUDINAL:")
for seg in lon_segments:
    print(seg["link_segment_id"], seg["used_length"])

print("LATERAL:")
for neigh in lat_segments:
    print(neigh["neighbor_segment_id"], neigh["intersection_point"])

lon_segments