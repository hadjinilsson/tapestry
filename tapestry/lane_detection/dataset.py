import torch
from torch.utils.data import Dataset
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point
from shapely.affinity import rotate
from pathlib import Path
import random
import numpy as np


class LaneDetectionDataset(Dataset):
    def __init__(self, data_root: Path, mode: str = "train", sample_every_meters: float = 1.0):
        self.data_root = Path(data_root)
        self.mode = mode
        self.geometry_dir = self.data_root / "lane_detection" / "geometry"
        self.image_dir = self.data_root / "lane_detection" / "images"
        self.predictions_dir = self.data_root / "lane_detection" / "predictions"

        df = pd.read_parquet(self.geometry_dir / "link_segments_projected.parquet")
        self.link_segments = df.set_index("link_segment_id")
        self.sections = pd.read_parquet(self.geometry_dir / "sections_projected.parquet")
        self.neighbours = pd.read_parquet(self.geometry_dir / "neighbours.parquet").set_index(None)
        self.order_uv = pd.read_parquet(self.geometry_dir / "segment_order_uv.parquet")
        self.order_vu = pd.read_parquet(self.geometry_dir / "segment_order_vu.parquet")

        # Build link_id to ordered segments map
        self.link_order = self._build_link_order()

        # Build index of training samples (link_segment_id, geometry length)
        if self.mode == "inference":
            self.samples = self._build_sample_index(sample_every_meters)
        else:
            self.samples = list(self.link_segments.index)

    def _build_link_order(self):
        order = {}
        for direction, df in [("uv", self.order_uv), ("vu", self.order_vu)]:
            for link_id, group in df.groupby("link_id"):
                if link_id not in order:
                    order[link_id] = {}
                order[link_id][direction] = list(group["link_segment_id"])
        return order

    def _build_sample_index(self, spacing_m: float):
        index = []
        infer_ids = self.link_segments.index.tolist()
        for link_segment_id in infer_ids:
            row = self.link_segments.loc[link_segment_id]
            length = row["length_proj"]
            num_samples = max(int(length // spacing_m), 1)
            distances = np.linspace(0, length, num=num_samples, endpoint=False)
            for d in distances:
                index.append((link_segment_id, d))
        return index

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.mode == "inference":
            link_segment_id, distance = self.samples[idx]
        else:
            link_segment_id = self.samples[idx]
            row = self.link_segments.loc[link_segment_id]
            distance = random.uniform(0.0, row["length_proj"])

        # Get context segments
        lon_segments, lat_segments = self.get_context_segments(link_segment_id, distance)

        # Placeholder tensors
        image_tensor = torch.zeros((3, 256, 256))
        heatmap_tensor = torch.zeros((1, 50))
        neighbor_proj = torch.zeros((1, 50))

        label_forward = random.randint(0, 5)
        label_backward = random.randint(0, 5)

        return {
            "image": image_tensor,
            "heatmap": heatmap_tensor,
            "neighbors": neighbor_proj,
            "forward_lane_count": torch.tensor(label_forward, dtype=torch.long),
            "backward_lane_count": torch.tensor(label_backward, dtype=torch.long),
            "link_segment_id": link_segment_id,
            "sample_distance": distance,
            "link_segment": self.link_segments.loc[link_segment_id]
        }

    def get_context_segments(
            self,
            link_segment_id,
            distance_on_line,
            longitudinal_coverage=25.0,
            lateral_radius=25.0,
    ):
        """
        Get neighboring segments in both longitudinal and lateral directions.

        Returns:
            segments_along: list of dicts with keys:
                - link_segment_id
                - geometry (LineString)
                - used_length (float) — how much of this segment will be used
            lateral_segments: list of dicts with keys:
                - neighbor_segment_id
                - intersection_point (Point or MultiPoint)
        """
        # Get the geometry of the training link segment
        row = self.link_segments.loc[link_segment_id]
        geom: LineString = row["geom"]
        sample_point = geom.interpolate(distance_on_line)

        # 1. Cross-sectional line at sample point
        tangent = geom.interpolate(min(distance_on_line + 0.1, geom.length))
        dx = tangent.x - sample_point.x
        dy = tangent.y - sample_point.y
        angle = np.degrees(np.arctan2(dy, dx)) + 90  # perpendicular
        cross_section = LineString([
            (sample_point.x - lateral_radius, sample_point.y),
            (sample_point.x + lateral_radius, sample_point.y)
        ])
        cross_section = rotate(cross_section, angle, origin=sample_point)

        # 2. Lateral neighbors — only use precomputed neighbors
        subset = self.neighbours[self.neighbours["link_segment_id"] == link_segment_id]
        lateral_segments = []
        for _, n_row in subset.iterrows():
            neighbor_id = n_row["neighbor_segment_id"]
            neighbor_geom = self.link_segments.loc[neighbor_id]["geom"]
            intersection = neighbor_geom.intersection(cross_section)
            if not intersection.is_empty:
                lateral_segments.append({
                    "neighbor_segment_id": neighbor_id,
                    "intersection_point": intersection
                })

        # 3. Longitudinal neighbors — always use segment_order_uv
        link_id = row["link_id"]
        ordered_segments = self.link_order[link_id]["uv"]
        current_idx = ordered_segments.index(link_segment_id)

        segments_along = []

        # Preceding (backward)
        remaining_back = longitudinal_coverage - distance_on_line
        i = current_idx - 1
        while remaining_back > 0 and i >= 0:
            seg_id = ordered_segments[i]
            seg_geom = self.link_segments.loc[seg_id]["geom"]
            seg_len = seg_geom.length
            used_len = min(seg_len, remaining_back)
            segments_along.insert(0, {
                "link_segment_id": seg_id,
                "geometry": seg_geom,
                "used_length": used_len
            })
            remaining_back -= seg_len
            i -= 1

        # Current segment (training)
        used_current_len = min(geom.length, 2 * longitudinal_coverage)
        segments_along.append({
            "link_segment_id": link_segment_id,
            "geometry": geom,
            "used_length": used_current_len
        })

        # Proceeding (forward)
        remaining_forward = longitudinal_coverage - (geom.length - distance_on_line)
        i = current_idx + 1
        while remaining_forward > 0 and i < len(ordered_segments):
            seg_id = ordered_segments[i]
            seg_geom = self.link_segments.loc[seg_id]["geom"]
            seg_len = seg_geom.length
            used_len = min(seg_len, remaining_forward)
            segments_along.append({
                "link_segment_id": seg_id,
                "geometry": seg_geom,
                "used_length": used_len
            })
            remaining_forward -= seg_len
            i += 1

        return segments_along, lateral_segments
