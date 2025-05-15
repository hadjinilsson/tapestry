from pathlib import Path
from tapestry.lane_detection.dataset import LaneDetectionDataset

train_ds = LaneDetectionDataset(data_root=Path('data'), mode="train")
inf_ds = LaneDetectionDataset(data_root=Path('data'), mode="inference")

# Pick an example
sample = train_ds[139]

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