import torch
from torch.utils.data import Dataset
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.affinity import rotate
from shapely import wkb
from pathlib import Path
from PIL import Image
import random
import numpy as np


class LaneDetectionDataset(Dataset):
    def __init__(
            self,
            data_root: Path,
            mode: str = "train",
            dim_pixels: int = 256,
            dim_gsdm: float = 50.0,
            lateral_radius: float = 50.0,
            longitudinal_coverage: float = 25.0,
            sample_every_meters: float = 1.0,
    ):
        # Arguments
        self.data_root = Path(data_root)
        self.mode = mode
        self.dim_pixels = dim_pixels
        self.dim_gsdm = dim_gsdm
        self.lateral_radius = lateral_radius
        self.longitudinal_coverage = longitudinal_coverage

        # Derived
        self.pixels_per_meter = dim_pixels / dim_gsdm

        # Directories
        self.geometry_dir = self.data_root / "lane_detection" / "geometry"
        self.image_dir = self.data_root / "lane_detection" / "images"
        self.predictions_dir = self.data_root / "lane_detection" / "predictions"

        # Data
        # Link segments
        df_link_segs = pd.read_parquet(self.geometry_dir / "link_segments_projected.parquet")
        df_link_segs["geom_proj"] = df_link_segs["geom_proj"].apply(wkb.loads)
        self.link_segments = df_link_segs.set_index("link_segment_id")
        # Sections
        df_sections = pd.read_parquet(self.geometry_dir / "sections_projected.parquet")
        df_sections["geom_proj"] = df_sections["geom_proj"].apply(wkb.loads)
        self.sections = df_sections
        # Neighbours
        self.neighbours = pd.read_parquet(self.geometry_dir / "neighbours.parquet")
        self.order_uv = pd.read_parquet(self.geometry_dir / "segment_order_uv.parquet")
        self.order_vu = pd.read_parquet(self.geometry_dir / "segment_order_vu.parquet")
        # Object predictions
        self.predictions = self._load_all_predictions()

        # Build link_id to ordered segments map
        self.link_order = self._build_link_order()

        # Build index of samples (link_segment_id, geometry length)
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

    def _load_all_predictions(self):
        predictions = {}
        for path in self.predictions_dir.glob("*.parquet"):
            df = pd.read_parquet(path)
            for cam_id, group in df.groupby("camera_point_id"):
                predictions[cam_id] = group
        return predictions

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.mode == "inference":
            link_segment_id, distance = self.samples[idx]
            row = self.link_segments.loc[link_segment_id]
        else:
            link_segment_id = self.samples[idx]
            row = self.link_segments.loc[link_segment_id]
            distance = random.uniform(0.0, row["length_proj"])

        # Get link segment data
        geom = row['geom_proj']
        link_id = row['link_id']
        sample_point = geom.interpolate(distance)

        # Placeholder tensors
        heatmap_tensor = torch.zeros((1, 50))
        neighbor_proj = torch.zeros((1, 50))

        # Cross-sectional line to get lateral neighbours and section labels
        tangent = geom.interpolate(min(distance + 0.1, geom.length))
        dx = tangent.x - sample_point.x
        dy = tangent.y - sample_point.y
        angle = np.degrees(np.arctan2(dy, dx)) + 90  # perpendicular
        cross_section = LineString([
            (sample_point.x - self.lateral_radius, sample_point.y),
            (sample_point.x + self.lateral_radius, sample_point.y)
        ])
        cross_section = rotate(cross_section, angle, origin=sample_point)

        # Get context segments
        lon_segments, lat_segments = self.get_context_segments(link_segment_id, link_id, distance, cross_section)

        # Get labels
        label_forward, label_backward = self.label_from_sections(link_segment_id, cross_section)

        # Get composite image
        # Ensure longitudinal continuity
        segment_order = self.link_order[link_id]["uv"]
        indices = [segment_order.index(seg["link_segment_id"]) for seg in lon_segments]
        if any(b - a != 1 for a, b in zip(indices, indices[1:])):
            raise ValueError(f"Discontinuous link segment sequence in longitudinal context: {indices}")
        # Get image slices
        slices = []
        for i, seg in enumerate(lon_segments):
            seg_id = seg["link_segment_id"]
            used_len = seg["used_length"]
            is_current = seg.get("is_current", False)
            should_slice_end = not is_current and used_len < self.link_segments.loc[seg_id]["length_proj"]
            if not should_slice_end:
                slice_tensor = self.load_segment_image_slice(seg_id, used_len)
            else:
                current_index = next(i for i, seg in enumerate(lon_segments) if seg["is_current"])
                if not is_current:
                    slice_from = "top" if i < current_index else "bottom"
                else:
                    slice_from = None
                slice_tensor = self.load_segment_image_slice(seg_id, used_len, slice_from=slice_from)
            slices.append(slice_tensor)
        slices = slices[::-1]

        # Stitch together vertically (along height)
        image_tensor = torch.cat(slices, dim=1)  # (3, H_total, W)

        # Total stitched height in pixels
        total_height = image_tensor.shape[1]

        # check if padding is needed
        if total_height < self.dim_pixels:
            pad_needed = self.dim_pixels - total_height
            pad_top = pad_needed // 2
            padding = torch.zeros((3, pad_top, image_tensor.shape[2]))
            image_tensor = torch.cat([padding, image_tensor, padding.clone()], dim=1)

        # Final fit to pixel dimension
        image_tensor = torch.nn.functional.interpolate(
            image_tensor.unsqueeze(0), size=(self.dim_pixels, self.dim_pixels), mode="bilinear",
            align_corners=False
        ).squeeze(0)

        prediction_slices = []
        offset_y = 0
        for i, seg in enumerate(lon_segments):
            seg_id = seg["link_segment_id"]
            is_current = seg.get("is_current", False)
            used_len = seg["used_length"]
            segment_len = self.link_segments.loc[seg_id]["length_proj"]

            current_index = next(i for i, seg in enumerate(lon_segments) if seg["is_current"])
            if not is_current and used_len < segment_len:
                slice_from = "top" if i < current_index else "bottom"
            else:
                slice_from = None

            df_preds = self.load_segment_predictions_slice(
                link_segment_id=seg_id,
                used_length=used_len,
                segment_length=segment_len,
                slice_from=slice_from
            )
            if not df_preds.empty:
                df_preds["y_center"] += offset_y
                prediction_slices.append(df_preds)

            offset_y += int(round(used_len * self.pixels_per_meter))

        if prediction_slices:
            all_predictions = pd.concat(prediction_slices, ignore_index=True)
        else:
            all_predictions = pd.DataFrame()

        return {
            "image": image_tensor,
            "heatmap": heatmap_tensor,
            "neighbors": neighbor_proj,
            "forward_lane_count": torch.tensor(label_forward),
            "backward_lane_count": torch.tensor(label_backward),
            "link_segment_id": link_segment_id,
            "sample_distance": distance,
            "lon_segments": lon_segments,
            "lat_segments": lat_segments,
            "image_tensor": image_tensor,
            "object_predictions": all_predictions,
            "cross_section": cross_section,
        }

    def get_context_segments(
            self,
            link_segment_id: str,
            link_id: str,
            distance_on_line: float,
            cross_section: LineString,
    ):
        """
        Get neighboring segments in both longitudinal and lateral directions.

        Returns:
            longitudinal_neighbours: list of dicts with keys:
                - link_segment_id
                - geometry (LineString)
                - used_length (float) — how much of this segment will be used
            lateral_neighbours: list of dicts with keys:
                - neighbor_segment_id
                - intersection_point (Point or MultiPoint)
        """
        # Unpack link segment and its length
        row = self.link_segments.loc[link_segment_id]
        length: LineString = row["length_proj"]

        # Longitudinal neighbors — always use segment_order_uv
        ordered_segments = self.link_order[link_id]["uv"]
        try:
            current_idx = ordered_segments.index(link_segment_id)
        except ValueError:
            raise ValueError(f"{link_segment_id} not found in UV ordering for link {link_id}")

        longitudinal_neighbours = []

        # Preceding (backward)
        remaining_back = self.longitudinal_coverage - distance_on_line
        i = current_idx - 1
        while remaining_back > 0 and i >= 0:
            seg_id = ordered_segments[i]
            seg_len = self.link_segments.loc[seg_id]["length_proj"]
            used_len = min(seg_len, remaining_back)
            longitudinal_neighbours.insert(0, {
                "link_segment_id": seg_id,
                "used_length": used_len,
                "is_current": False,
            })
            remaining_back -= seg_len
            i -= 1

        # Current segment
        used_current_len = (
                min(distance_on_line, self.longitudinal_coverage) +
                min(length - distance_on_line, self.longitudinal_coverage)
        )
        longitudinal_neighbours.append({
            "link_segment_id": link_segment_id,
            "used_length": used_current_len,
            "is_current": True,
        })

        # Proceeding (forward)
        remaining_forward = self.longitudinal_coverage - (length - distance_on_line)
        i = current_idx + 1
        while remaining_forward > 0 and i < len(ordered_segments):
            seg_id = ordered_segments[i]
            seg_len = self.link_segments.loc[seg_id]["length_proj"]
            used_len = min(seg_len, remaining_forward)
            longitudinal_neighbours.append({
                "link_segment_id": seg_id,
                "used_length": used_len,
                "is_current": False,
            })
            remaining_forward -= seg_len
            i += 1

        # Lateral neighbors — only use precomputed neighbors
        subset = self.neighbours[self.neighbours["link_segment_id"] == link_segment_id]
        lateral_neighbours = []
        for _, n_row in subset.iterrows():
            neighbor_id = n_row["neighbor_segment_id"]
            neighbor_geom = self.link_segments.loc[neighbor_id]["geom_proj"]
            intersection = neighbor_geom.intersection(cross_section)
            if not intersection.is_empty:
                lateral_neighbours.append({
                    "neighbor_segment_id": neighbor_id,
                    "intersection_point": intersection
                })

        return longitudinal_neighbours, lateral_neighbours

    def label_from_sections(self, link_segment_id: str, cross_section: LineString) -> tuple[int, int]:
        """
        Derives forward and backward lane counts from section geometries.

        Args:
            link_segment_id: the segment we're sampling
            cross_section: the cross-sectional line from the sampled point

        Returns:
            (forward_lane_count, backward_lane_count)
        """
        # Filter to sections belonging to this segment
        candidate_sections = self.sections[
            (self.sections["link_segment_id"] == link_segment_id) &
            (self.sections["component"].isin([
                "general_traffic_lane",
                "general_traffic_lane_two_way",
                "bus_lane",
            ]))
            ]

        # Further filter by intersection with cross-sectional line
        matching_sections = candidate_sections[
            candidate_sections["geom_proj"].apply(lambda geom: geom.intersects(cross_section))
        ]
        # Return zero if no intersections
        if matching_sections.empty:
            return 0, 0
        # Extract direction
        # Get the segment's bearing
        segment_bearing = self.link_segments.loc[link_segment_id]["bearing"]
        # Count by direction
        forward = 0
        backward = 0
        for _, sec in matching_sections.iterrows():
            section_bearing = sec["bearing"]
            delta = (section_bearing - segment_bearing + 360) % 360
            if delta < 90 or delta > 270:
                forward += 1
            else:
                backward += 1

        return forward, backward

    def load_segment_image_slice(
            self,
            link_segment_id: str,
            used_length: float,
            slice_from: str | None = None,
    ) -> torch.Tensor:
        """
        Loads and vertically slices the image for a given link segment.

        Args:
            link_segment_id: segment to load
            used_length: how much of the segment image to use (in meters)
            slice_from: "top" or "bottom" (only applies to non-current segments)

        Returns:
            torch.Tensor of shape (3, H_partial, W), ready to be concatenated
        """
        row = self.link_segments.loc[link_segment_id]
        camera_id = row["camera_point_id"]
        segment_length = row["length_proj"]
        slice_height = int(round(used_length * self.pixels_per_meter))

        # If zero length (shouldn't really happen)
        if used_length <= 0:
            print(f"⚠️ Zero-length slice for {link_segment_id}, skipping")
            return torch.zeros((3, 1, self.dim_pixels))

        # Fill in empty if no camera point (which shouldn't really happen)
        if pd.isnull(camera_id):
            print(f"⚠️ No camera point for {link_segment_id}, inserting blank slice")
            return torch.zeros((3, slice_height, self.dim_pixels))

        # Load image
        image_path = self.image_dir / f"{camera_id}.png"

        # Fill in empty if no image
        if not image_path.exists():
            print(f"⚠️ Missing image {image_path.name}, inserting blank slice")
            return torch.zeros((3, slice_height, self.dim_pixels))

        # Read image
        img = Image.open(image_path).convert("RGB").resize((self.dim_pixels, self.dim_pixels))
        img_np = np.array(img)

        # Step 1: Align full segment — center-crop based on segment_length
        visible_height = int(round(segment_length * self.pixels_per_meter))
        offset = max(0, (self.dim_pixels - visible_height) // 2)
        end = min(self.dim_pixels, offset + visible_height)
        cropped = img_np[offset:end, :, :]

        # Step 2: Slice for context (if not main segment)
        if slice_from is not None:
            if slice_from == "top":
                cropped = cropped[:slice_height, :, :]
            elif slice_from == "bottom":
                cropped = cropped[-slice_height:, :, :]
            else:
                raise ValueError(f"Invalid slice_from: {slice_from}")

        # Convert to torch tensor
        img_crop_tensor = torch.from_numpy(cropped).permute(2, 0, 1).float() / 255.0
        return img_crop_tensor

    def load_segment_predictions_slice(
            self,
            link_segment_id: str,
            used_length: float,
            segment_length: float,
            slice_from: str | None = None,
    ) -> pd.DataFrame:
        """
        Loads prediction rows for a given segment, slices vertically like the image,
        and returns only the predictions in the used part of the segment.

        Args:
            link_segment_id: Link segment ID
            used_length: Number of meters to retain (e.g. 10m)
            segment_length: Total length of the segment (e.g. 30m)
            slice_from: One of None, "top", or "bottom"

        Returns:
            pd.DataFrame with sliced and aligned predictions
        """
        row = self.link_segments.loc[link_segment_id]
        camera_id = row["camera_point_id"]

        # Load object predictions
        o_preds = self.predictions.get(camera_id, pd.DataFrame())
        o_preds["x_center"] *= self.dim_pixels
        o_preds["y_center"] *= self.dim_pixels
        o_preds["width"] *= self.dim_pixels
        o_preds["height"] *= self.dim_pixels

        # 1. Compute the central crop used for this segment
        visible_height = int(round(segment_length * self.pixels_per_meter))
        offset = max(0, (self.dim_pixels - visible_height) // 2)
        end = min(self.dim_pixels, offset + visible_height)

        # 2. Restrict predictions to this visible slice
        o_preds = o_preds[o_preds["y_center"].between(offset, end)]

        if o_preds.empty:
            return o_preds

        # 3. Further slice for partial context segments
        slice_height = int(round(used_length * self.pixels_per_meter))
        if slice_from == "top":
            slice_start = offset
            slice_end = offset + slice_height
        elif slice_from == "bottom":
            slice_end = end
            slice_start = end - slice_height
        else:
            slice_start = offset
            slice_end = end

        o_preds = o_preds[o_preds["y_center"].between(slice_start, slice_end)].copy()
        if o_preds.empty:
            return o_preds

        # 4. Rebase y_center so the cropped region starts at 0
        o_preds["y_center"] -= slice_start

        return o_preds

