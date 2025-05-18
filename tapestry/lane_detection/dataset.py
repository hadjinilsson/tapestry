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
            distance = 10

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
        angle = np.degrees(np.arctan2(dy, dx)) + 90
        cross_section = LineString([
            (sample_point.x - self.lateral_radius, sample_point.y),
            (sample_point.x + self.lateral_radius, sample_point.y)
        ])
        cross_section = rotate(cross_section, angle, origin=sample_point)

        # Get context segments
        lon_neighbours, lat_neighbours = self.get_neighbours(link_segment_id, distance, cross_section)

        # Get labels
        label_forward, label_backward = self.label_from_sections(link_segment_id, cross_section)

        # Get composite image

        # Ensure longitudinal continuity
        segment_order = self.link_order[link_id]["uv"]
        indices = [segment_order.index(seg["link_segment_id"]) for seg in lon_neighbours]
        if any(b - a != -1 for a, b in zip(indices, indices[1:])):
            raise ValueError(f"Discontinuous link segment sequence in longitudinal context: {indices}")

        # Get image slices
        img_slices = []
        for i, lon_neighbour in enumerate(lon_neighbours):
            is_first = True if i == 0 else False
            is_last = True if i == (len(lon_neighbours) - 1) else False
            img_slice = self.load_image_slice(lon_neighbour, is_first, is_last)
            img_slices.append(img_slice)

            # seg_id = lon_neighbour["link_segment_id"]
            # used_len = lon_neighbour["used_length"]
            # is_current = lon_neighbour.get("is_current", False)
            # should_slice_end = not is_current and used_len < self.link_segments.loc[seg_id]["length_proj"]
            # if not should_slice_end:
            #     slice_tensor = self.load_segment_image_slice(seg_id, used_len)
            # else:
            #     current_index = next(i for i, lon_neighbour in enumerate(lon_neighbours) if lon_neighbour["is_current"])
            #     if not is_current:
            #         slice_from = "top" if i > current_index else "bottom"
            #     else:
            #         slice_from = None
            #     slice_tensor = self.load_segment_image_slice(seg_id, used_len, slice_from=slice_from)
            # img_slices.append(slice_tensor)

        # Stitch together vertically (along height)
        image_tensor = torch.cat(img_slices, dim=1)  # (3, H_total, W)

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
        for i, lon_neighbour in enumerate(lon_neighbours):
            seg_id = lon_neighbour["link_segment_id"]
            is_current = lon_neighbour.get("is_current", False)
            used_len = lon_neighbour["used_length"]
            segment_len = self.link_segments.loc[seg_id]["length_proj"]

            current_index = next(i for i, seg in enumerate(lon_neighbours) if seg["is_current"])
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
        prediction_slices = prediction_slices[::-1]

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
            "lon_neighbours": lon_neighbours,
            "lat_neighbours": lat_neighbours,
            "image_tensor": image_tensor,
            "object_predictions": all_predictions,
            "cross_section": cross_section,
        }

    def get_neighbours(
            self,
            current_seg_id: str,
            distance_along: float,
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
        # Unpack
        current_seg = self.link_segments.loc[current_seg_id]
        link_id: LineString = current_seg["link_id"]
        current_seg_len: LineString = current_seg["length_proj"]

        # Longitudinal neighbors — always use segment_order_uv
        uv_seg_order = self.link_order[link_id]["uv"]
        try:
            current_idx = uv_seg_order.index(current_seg_id)
        except ValueError:
            raise ValueError(f"{current_seg_id} not found in UV ordering for link {link_id}")

        # Preceding (bottom of image)
        preceding = []
        remaining_back = self.longitudinal_coverage - distance_along
        i = current_idx - 1
        while remaining_back > 0 and i >= 0:
            pre_seg_id = uv_seg_order[i]
            pre_seg_len = self.link_segments.loc[pre_seg_id]["length_proj"]
            pre_seg_used_len = min(pre_seg_len, remaining_back)
            preceding.insert(0, {
                "link_segment_id": pre_seg_id,
                "used_length": pre_seg_used_len,
                "is_current": False,
                "is_first_uv": self.link_segments.loc[pre_seg_id]["segment_ix_uv"] == 0,
                "is_last_uv": self.link_segments.loc[pre_seg_id]["segment_ix_vu"] == 0,
            })
            remaining_back -= pre_seg_len
            i -= 1

        # Current segment
        current_seg_used_len = (
                min(distance_along, self.longitudinal_coverage) +
                min(current_seg_len - distance_along, self.longitudinal_coverage)
        )
        current = [{
            "link_segment_id": current_seg_id,
            "used_length": current_seg_used_len,
            "is_current": True,
            "is_first_uv": self.link_segments.loc[current_seg_id]["segment_ix_uv"] == 0,
            "is_last_uv": self.link_segments.loc[current_seg_id]["segment_ix_vu"] == 0,
        }]

        # Proceeding (top of image)
        proceeding = []
        remaining_forward = self.longitudinal_coverage - (current_seg_len - distance_along)
        i = current_idx + 1
        while remaining_forward > 0 and i < len(uv_seg_order):
            pro_seg_id = uv_seg_order[i]
            pro_seg_len = self.link_segments.loc[pro_seg_id]["length_proj"]
            pro_seg_used_len = min(pro_seg_len, remaining_forward)
            proceeding.append({
                "link_segment_id": pro_seg_id,
                "used_length": pro_seg_used_len,
                "is_current": False,
                "is_first_uv": self.link_segments.loc[pro_seg_id]["segment_ix_uv"] == 0,
                "is_last_uv": self.link_segments.loc[pro_seg_id]["segment_ix_vu"] == 0,
            })
            remaining_forward -= pro_seg_len
            i += 1

        # Final stack: top-to-bottom = proceeding + current + preceding
        longitudinal_neighbours = proceeding + current + preceding

        # Lateral neighbors — only use precomputed neighbors
        subset = self.neighbours[self.neighbours["link_segment_id"] == current_seg_id]
        lateral_neighbours = []
        for _, lat_seg in subset.iterrows():
            lat_seg_id = lat_seg["neighbor_segment_id"]
            lat_seg_geom = self.link_segments.loc[lat_seg_id]["geom_proj"]
            intersection = lat_seg_geom.intersection(cross_section)
            if not intersection.is_empty:
                lateral_neighbours.append({
                    "neighbor_segment_id": lat_seg_id,
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

    def load_image_slice(
            self,
            lon_neighbour: dict,
            is_first: bool,
            is_last: bool,
    ) -> torch.Tensor:
    # def load_segment_image_slice(
    #         self,
    #         link_segment_id: str,
    #         used_length: float,
    #         slice_from: str | None = None,
    # ) -> torch.Tensor:
        """
        Loads and vertically slices the image for a given link segment, ensuring output slice
        has fixed height corresponding to `used_length`.

        Returns:
            torch.Tensor of shape (3, slice_height, W), ready for stitching
        """

        seg_id = lon_neighbour["link_segment_id"]
        seg = self.link_segments.loc[seg_id]

        is_current = lon_neighbour.get("is_current", False)
        is_first_uv = lon_neighbour.get("is_first_uv", False)
        is_last_uv = lon_neighbour.get("is_last_uv", False)

        seg_len_m = seg["length_proj"]
        seg_used_len_m = lon_neighbour.get("used_length", 0.0)
        camera_id = seg["camera_point_id"]
        camera_offset_m = seg.get("camera_point_offset", 0.0)

        seg_len_px = int(seg_len_m * self.pixels_per_meter)
        seg_used_len_px = int(seg_used_len_m * self.pixels_per_meter)

        if seg_used_len_m <= 0:
            print(f"⚠️ Zero-length slice for {seg_id}, skipping")
            return torch.zeros((3, 1, self.dim_pixels))

        if pd.isnull(camera_id):
            print(f"⚠️ No camera point for {seg_id}, inserting blank slice")
            return torch.zeros((3, seg_used_len_px, self.dim_pixels))

        image_path = self.image_dir / f"{camera_id}.png"
        if not image_path.exists():
            print(f"⚠️ Missing image {image_path.name}, inserting blank slice")
            return torch.zeros((3, seg_used_len_px, self.dim_pixels))

        # Load and resize
        img = Image.open(image_path).convert("RGB").resize((self.dim_pixels, self.dim_pixels))
        img_np = np.array(img)

        # Measurement from bottom
        center_img_m = self.dim_gsdm / 2
        # center_seg_m = center_img_m +
        if True:
            return torch.zeros((3, seg_used_len_px, self.dim_pixels))

        # seg_id = lon_neighbour["link_segment_id"]
        # used_len = lon_neighbour["used_length"]
        # is_current = lon_neighbour.get("is_current", False)
        # should_slice_end = not is_current and used_len < self.link_segments.loc[seg_id]["length_proj"]
        # if not should_slice_end:
        #     slice_tensor = self.load_segment_image_slice(seg_id, used_len)
        # else:
        #     current_index = next(i for i, lon_neighbour in enumerate(lon_neighbours) if lon_neighbour["is_current"])
        #     if not is_current:
        #         slice_from = "top" if i > current_index else "bottom"
        #     else:
        #         slice_from = None
        #     slice_tensor = self.load_segment_image_slice(seg_id, used_len, slice_from=slice_from)
        # img_slices.append(slice_tensor)


        # row = self.link_segments.loc[link_segment_id]
        # camera_id = row["camera_point_id"]
        # segment_length = row["length_proj"]
        # camera_offset = row.get("camera_point_offset", 0.0)
        # pixels_per_meter = self.pixels_per_meter
        # slice_height = int(round(used_length * pixels_per_meter))

        # if used_length <= 0:
        #     print(f"⚠️ Zero-length slice for {link_segment_id}, skipping")
        #     return torch.zeros((3, 1, self.dim_pixels))
        #
        # if pd.isnull(camera_id):
        #     print(f"⚠️ No camera point for {link_segment_id}, inserting blank slice")
        #     return torch.zeros((3, slice_height, self.dim_pixels))
        #
        # image_path = self.image_dir / f"{camera_id}.png"
        # if not image_path.exists():
        #     print(f"⚠️ Missing image {image_path.name}, inserting blank slice")
        #     return torch.zeros((3, slice_height, self.dim_pixels))
        #
        # # Load and resize
        # img = Image.open(image_path).convert("RGB").resize((self.dim_pixels, self.dim_pixels))
        # img_np = np.array(img)

        # # Crop based on camera point offset
        # center_pixel = int((self.dim_gsdm / 2 + camera_offset) * pixels_per_meter)
        # visible_height = int(round(segment_length * pixels_per_meter))
        # start = max(0, center_pixel - visible_height // 2)
        # end = min(self.dim_pixels, start + visible_height)
        # cropped = img_np[start:end, :, :]
        #
        # # Apply top/bottom slicing
        # if slice_from == "top":
        #     sliced = cropped[:slice_height, :, :]
        #     pad = slice_height - sliced.shape[0]
        #     if pad > 0:
        #         sliced = np.vstack([
        #             sliced,
        #             np.zeros((pad, self.dim_pixels, 3), dtype=np.uint8)
        #         ])
        # elif slice_from == "bottom":
        #     sliced = cropped[-slice_height:, :, :]
        #     pad = slice_height - sliced.shape[0]
        #     if pad > 0:
        #         sliced = np.vstack([
        #             np.zeros((pad, self.dim_pixels, 3), dtype=np.uint8),
        #             sliced
        #         ])
        # else:
        #     # For current segment or exact fit
        #     pad = slice_height - cropped.shape[0]
        #     if pad > 0:
        #         pad_top = pad // 2
        #         pad_bottom = pad - pad_top
        #         sliced = np.vstack([
        #             np.zeros((pad_top, self.dim_pixels, 3), dtype=np.uint8),
        #             cropped,
        #             np.zeros((pad_bottom, self.dim_pixels, 3), dtype=np.uint8)
        #         ])
        #     else:
        #         sliced = cropped

        # Convert to tensor
        img_crop_tensor = torch.from_numpy(sliced).permute(2, 0, 1).float() / 255.0
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
        camera_offset = row.get("camera_point_offset", 0.0)
        pixels_per_meter = self.pixels_per_meter

        # Load predictions
        preds = self.predictions.get(camera_id, pd.DataFrame())
        if preds.empty:
            return preds

        # Denormalize coordinates to pixel space
        preds = preds.copy()
        preds["x_center"] *= self.dim_pixels
        preds["y_center"] *= self.dim_pixels
        preds["width"] *= self.dim_pixels
        preds["height"] *= self.dim_pixels

        # Compute camera-centered crop bounds (same as image)
        center_pixel = int((segment_length / 2 - camera_offset) * pixels_per_meter)
        visible_height = int(round(segment_length * pixels_per_meter))
        start = max(0, center_pixel - visible_height // 2)
        end = min(self.dim_pixels, start + visible_height)

        # Filter to those inside cropped image
        preds = preds[preds["y_center"].between(start, end)]
        if preds.empty:
            return preds

        # Apply partial slicing
        slice_height = int(round(used_length * pixels_per_meter))
        if slice_from == "top":
            slice_start = start
            slice_end = start + slice_height
        elif slice_from == "bottom":
            slice_end = end
            slice_start = end - slice_height
        else:
            slice_start = start
            slice_end = end

        # Final filter and rebase y_center
        preds = preds[preds["y_center"].between(slice_start, slice_end)].copy()
        if preds.empty:
            return preds

        preds["y_center"] -= slice_start  # align to top of slice
        return preds

