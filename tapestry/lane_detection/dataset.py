import torch
from torch.utils.data import Dataset
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.affinity import rotate
from shapely import wkb, distance, centroid
from pathlib import Path
from PIL import Image
import random
import numpy as np
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from torchvision.transforms.functional import to_pil_image, to_tensor


class LaneDetectionDataset(Dataset):
    def __init__(
            self,
            data_root: Path,
            mode: str = "train",
            dim_pixels: int = 256,
            dim_gsd: float = 50.0,
            lat_coverage: float = 50.0,
            lon_coverage: float = 25.0,
            train_dist: float = 1.0,
            max_extra_image: float = 10.0,
            max_shift: float = 10.0,
            num_obj_pred_classes: int = 15,
    ):
        # Arguments
        self.data_root = Path(data_root)
        self.mode = mode
        self.dim_pixels = dim_pixels
        self.dim_gsd = dim_gsd
        self.lat_coverage = lat_coverage
        self.lon_coverage = lon_coverage
        self.max_extra_image = max_extra_image
        self.max_shift = max_shift
        self.num_obj_pred_classes = num_obj_pred_classes

        # Derived
        self.pixels_per_meter = dim_pixels / dim_gsd
        self.enable_flip = (self.mode == "train")

        # Directories
        self.geometry_dir = self.data_root / "lane_detection" / "geometry"
        self.image_dir = self.data_root / "lane_detection" / "images"
        self.obj_preds_dir = self.data_root / "lane_detection" / "predictions"

        # Link segments
        df_segs = pd.read_parquet(self.geometry_dir / "link_segments_projected.parquet")
        df_segs["geom_proj"] = df_segs["geom_proj"].apply(wkb.loads)
        self.link_segments = df_segs.set_index("link_segment_id")

        # Sections
        df_sections = pd.read_parquet(self.geometry_dir / "sections_projected.parquet")
        df_sections["geom_proj"] = df_sections["geom_proj"].apply(wkb.loads)
        self.sections = df_sections

        # Lateral neighbours
        self.lat_neighbours = pd.read_parquet(self.geometry_dir / "neighbours.parquet")
        self.order_uv = pd.read_parquet(self.geometry_dir / "segment_order_uv.parquet")
        self.order_vu = pd.read_parquet(self.geometry_dir / "segment_order_vu.parquet")

        # Object predictions
        self.obj_predictions = self._load_all_obj_preds()

        # Build link_id to ordered segments map
        self.link_order = self._build_link_order()

        # Build index of samples (link_segment_id, geometry length)
        if self.mode == "inference":
            self.samples = self._build_sample_index(train_dist)
        else:
            self.samples = list(df_segs.loc[df_segs.is_training, 'link_segment_id'])

    def _build_link_order(self):
        order = {}
        for direction, df in [("uv", self.order_uv), ("vu", self.order_vu)]:
            for link_id, group in df.groupby("link_id"):
                if link_id not in order:
                    order[link_id] = {}
                order[link_id][direction] = list(group["link_segment_id"])
        return order

    def _build_sample_index(self, spacing: float):
        sample_index = []
        infer_ids = self.link_segments.index.tolist()
        for link_segment_id in infer_ids:
            seg = self.link_segments.loc[link_segment_id]
            seg_len = seg["length_proj"]
            num_samples = max(int(seg_len // spacing), 1)
            ds = np.linspace(0, seg_len, num=num_samples, endpoint=False)
            for d in ds:
                sample_index.append((link_segment_id, d))
        return sample_index

    def _load_all_obj_preds(self):
        obj_preds = {}
        for path in self.obj_preds_dir.glob("*.parquet"):
            df = pd.read_parquet(path)
            for cam_id, group in df.groupby("camera_point_id"):
                obj_preds[cam_id] = group
        return obj_preds

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.mode == "inference":
            seg_id, distance_along = self.samples[idx]
            seg = self.link_segments.loc[seg_id]
        else:
            seg_id = self.samples[idx]
            seg = self.link_segments.loc[seg_id]
            distance_along = random.uniform(0.0, seg["length_proj"])

        # Get link segment data
        seg_geom = seg['geom_proj']
        seg_len = seg['length_proj']
        link_id = seg['link_id']
        sample_point = seg_geom.interpolate(distance_along)

        # Placeholder tensors
        heatmap_tensor = torch.zeros((1, 50))
        neighbor_proj = torch.zeros((1, 50))

        # # Cross-sectional line to get lateral neighbours and section labels
        cross_section = self.get_cross_section(seg_geom, distance_along, sample_point)

        #  Get labels
        sections = self.get_sections(seg_id, sample_point, cross_section)

        # Get neighbours
        lat_neighbours = self.get_lat_neighbours(seg_id, sample_point, cross_section)
        lon_neighbours = self.get_lon_neighbours(seg_id, distance_along)
        self.check_lon_continuity(link_id, lon_neighbours)

        # Get used lengths
        used_pre_len, used_pro_len = self.get_used_lengths(lon_neighbours, distance_along, seg_len)

        # Get slice data
        img, obj_preds = self.get_slice_data(lon_neighbours, distance_along, used_pre_len, used_pro_len)

        # Image augmentations
        if self.mode == "train":
            brightness_factor = random.uniform(0.8, 1.2)
            contrast_factor = random.uniform(0.8, 1.2)
            img = TF.adjust_brightness(img, brightness_factor)
            img = TF.adjust_contrast(img, contrast_factor)

        if self.mode == "train":
            noise = torch.randn_like(img) * 0.02
            img = torch.clamp(img + noise, 0.0, 1.0)

        if self.mode == "train" and random.random() < 0.3:
            img = to_pil_image(img)
            img = T.GaussianBlur(kernel_size=3)(img)
            img = to_tensor(img)

        # Lateral shift
        if self.mode == "train" and self.max_shift > 0:
            shift = random.uniform(-self.max_shift, self.max_shift)
        else:
            shift = 0.0

        # Sift image
        img = self.shift_image(img, shift)
        # Shift object predictions
        obj_preds = self.shift_object_predictions(obj_preds, shift)
        # Shift sections
        sections = self.shift_points(sections, shift, self.dim_gsd/2)
        # Sift lateral neighbours
        lat_neighbours = self.shift_points(lat_neighbours, shift, self.lat_coverage)

        # Vertical flip
        flip = self.enable_flip and random.random() < 0.5
        if flip:
            img = torch.flip(img, dims=[1, 2])
            obj_preds["y_center"] = self.dim_pixels - obj_preds["y_center"]
            obj_preds["x_center"] = self.dim_pixels - obj_preds["x_center"]
            sections['direction'] = sections.direction.map({'forward': 'backward', 'backward': 'forward'})
            sections['x_offset'] *= -1
            lat_neighbours['x_offset'] *= -1

        return {
            "image": img,
            "heatmap": heatmap_tensor,
            "neighbors": neighbor_proj,
            "sections": sections,
            "link_segment_id": seg_id,
            "sample_distance": distance_along,
            "lon_neighbours": lon_neighbours,
            "lat_neighbours": lat_neighbours,
            "object_predictions": obj_preds,
            "cross_section": cross_section,
            "shift": shift,
            "flip": flip,
        }

    def get_cross_section(self, seg_geom, distance_along, sample_point):
        tangent = seg_geom.interpolate(min(distance_along + 0.1, seg_geom.length))
        dx = tangent.x - sample_point.x
        dy = tangent.y - sample_point.y
        angle = np.degrees(np.arctan2(dy, dx)) + 90
        cross_section = LineString([
            (sample_point.x - self.lat_coverage, sample_point.y),
            (sample_point.x + self.lat_coverage, sample_point.y)
        ])
        cross_section = rotate(cross_section, angle, origin=sample_point)
        return cross_section

    @staticmethod
    def get_side_of_line(point, line):
        start = Point(line.coords[0])
        end = Point(line.coords[-1])
        cross = (end.x - start.x) * (point.y - start.y) - (end.y - start.y) * (point.x - start.x)
        if cross > 0:
            return -1
        elif cross < 0:
            return 1
        else:
            return 0

    def get_intersecting(self, df, seg_geom, sample_point, cross_section, geom_col='geom_proj'):
        df['intersection_point'] = df[geom_col].apply(lambda x: x.intersection(cross_section))
        # TODO Change to accommodate multi-point?
        df = df[df.intersection_point.apply(lambda x: x.geom_type) == 'Point'].copy()
        df['x_offset_abs'] = df.intersection_point.apply(lambda x: distance(x, sample_point))
        df['x_offset'] = df.intersection_point.apply(lambda x: self.get_side_of_line(x, seg_geom)) * df.x_offset_abs
        return df

    def get_sections(self, seg_id, sample_point, cross_section):

        seg = self.link_segments.loc[seg_id]
        seg_geom = seg['geom_proj']

        sec_types = ["general_traffic_lane", "general_traffic_lane_two_way", "bus_lane"]
        all_secs = self.sections
        can_secs = all_secs[(all_secs["link_segment_id"] == seg_id) & (all_secs["component"].isin(sec_types))].copy()
        seg_secs = self.get_intersecting(can_secs, seg_geom, sample_point, cross_section)
        seg_bear = self.link_segments.loc[seg_id]["bearing"]
        seg_secs['bearing_diff'] = (seg_secs.bearing - seg_bear + 360) % 360
        seg_secs['direction'] = 'backward'
        seg_secs.loc[(seg_secs.bearing_diff < 90) | (seg_secs.bearing_diff > 270), 'direction'] = 'forward'

        return seg_secs

    def get_lat_neighbours(self, seg_id, sample_point, cross_section):

        seg = self.link_segments.loc[seg_id]
        seg_geom = seg['geom_proj']

        lat_neighbour_ids = self.lat_neighbours[self.lat_neighbours["link_segment_id"] == seg_id]
        lat_segs = self.link_segments[self.link_segments.index.isin(lat_neighbour_ids.neighbor_segment_id)].copy()
        lat_neighbours = self.get_intersecting(lat_segs, seg_geom, sample_point, cross_section)
        return lat_neighbours

    def get_lon_neighbours(self, current_seg_id: str, distance_along: float):

        current_seg = self.link_segments.loc[current_seg_id]
        link_id: LineString = current_seg["link_id"]
        current_seg_len: LineString = current_seg["length_proj"]

        uv_seg_order = self.link_order[link_id]["uv"]
        try:
            current_idx = uv_seg_order.index(current_seg_id)
        except ValueError:
            raise ValueError(f"{current_seg_id} not found in UV ordering for link {link_id}")

        lon_neighbours = []

        # Preceding (bottom of image)
        remaining_back = self.lon_coverage - distance_along
        i = current_idx - 1
        while remaining_back > 0 and i >= 0:
            pre_seg_id = uv_seg_order[i]
            pre_seg_len = self.link_segments.loc[pre_seg_id]["length_proj"]
            pre_seg_used_len = min(pre_seg_len, remaining_back)
            lon_neighbours.insert(0, {
                "link_segment_id": pre_seg_id,
                "used_length": pre_seg_used_len,
                "placement": "preceding",
            })
            remaining_back -= pre_seg_len
            i -= 1

        # Current segment
        current_seg_used_len = (
                min(distance_along, self.lon_coverage) +
                min(current_seg_len - distance_along, self.lon_coverage)
        )
        current = {
            "link_segment_id": current_seg_id,
            "used_length": current_seg_used_len,
            "placement": "current",
        }
        lon_neighbours.append(current)

        # Proceeding (top of image)
        remaining_forward = self.lon_coverage - (current_seg_len - distance_along)
        i = current_idx + 1
        while remaining_forward > 0 and i < len(uv_seg_order):
            pro_seg_id = uv_seg_order[i]
            pro_seg_len = self.link_segments.loc[pro_seg_id]["length_proj"]
            pro_seg_used_len = min(pro_seg_len, remaining_forward)
            lon_neighbours.append({
                "link_segment_id": pro_seg_id,
                "used_length": pro_seg_used_len,
                "placement": "proceeding",
            })
            remaining_forward -= pro_seg_len
            i += 1

        return lon_neighbours

    def get_used_lengths(self, lon_neighbours, distance_along, seg_len):
        used_pre_len = 0
        used_pro_len = 0
        for ln in lon_neighbours:
            if ln['placement'] == 'preceding':
                used_pre_len += ln['used_length']
            elif ln['placement'] == 'proceeding':
                used_pro_len += ln['used_length']
            elif ln['placement'] == 'current':
                used_pre_len += min(distance_along, self.lon_coverage)
                used_pro_len += min(seg_len - distance_along, self.lon_coverage)
        return used_pre_len, used_pro_len

    def check_lon_continuity(self, link_id, lon_neighbours):
        ln_order = self.link_order[link_id]["uv"]
        seg_ixs = [ln_order.index(ln["link_segment_id"]) for ln in lon_neighbours]
        if any(b - a != 1 for a, b in zip(seg_ixs, seg_ixs[1:])):
            raise ValueError(f"Discontinuous link segment sequence in longitudinal context: {seg_ixs}")

    def get_slice_data(
            self,
            lon_neighbours: list[dict],
            distance_along: float,
            used_preceding_length: float,
            used_proceeding_length: float,
    ):

        img_slices = []
        obj_preds_slices = []
        slice_anchor_m = 0

        for i, lon_neighbour in enumerate(lon_neighbours):

            is_first = i == 0
            is_last = i == (len(lon_neighbours) - 1)
            is_current = lon_neighbour['placement'] == 'current'

            seg_id = lon_neighbour["link_segment_id"]
            seg = self.link_segments.loc[seg_id]

            is_first_uv = seg["segment_ix_uv"] == 0
            is_last_uv = seg["segment_ix_vu"] == 0

            seg_len_m = seg["length_proj"]
            seg_used_len_m = lon_neighbour.get("used_length", 0.0)
            camera_offset_m = seg.get("camera_point_offset", 0.0)

            # Measurement from bottom
            img_center_m = self.dim_gsd / 2
            seg_center_m = img_center_m - camera_offset_m
            seg_start_m = seg_center_m - seg_len_m / 2
            seg_end_m = seg_center_m + seg_len_m / 2

            pad_above_m = 0
            pad_below_m = 0

            if is_first and is_first_uv:
                extra_needed_m = self.lon_coverage - used_preceding_length
                extra_img_m = min(extra_needed_m, self.max_extra_image)
                if is_current:
                    extra_img_m += min(seg_start_m + distance_along - used_preceding_length - extra_img_m, 0)
                    slice_start_m = seg_start_m + distance_along - used_preceding_length - extra_img_m
                else:
                    extra_img_m += min(seg_end_m - seg_used_len_m - extra_img_m, 0)
                    slice_start_m = seg_end_m - seg_used_len_m - extra_img_m
                pad_below_m = extra_needed_m - extra_img_m
            elif is_first:
                slice_start_m = seg_end_m - seg_used_len_m
            else:
                slice_start_m = seg_start_m

            if is_last and is_last_uv:
                extra_needed_m = self.lon_coverage - used_proceeding_length
                extra_img_m = min(extra_needed_m, self.max_extra_image)
                if is_current:
                    extra_img_m -= max(
                        seg_start_m + distance_along + used_proceeding_length + extra_img_m - self.dim_gsd, 0
                    )
                    slice_end_m = seg_start_m + distance_along + used_proceeding_length + extra_img_m
                else:
                    extra_img_m -= max(seg_start_m + seg_used_len_m + extra_img_m - self.dim_gsd, 0)
                    slice_end_m = seg_start_m + seg_used_len_m + extra_img_m
                pad_above_m = extra_needed_m - extra_img_m
            elif is_last:
                slice_end_m = seg_start_m + seg_used_len_m
            else:
                slice_end_m = seg_end_m

            seg_img_slices = self.get_image_slices(seg_id, slice_start_m, slice_end_m, pad_above_m, pad_below_m)
            img_slices = seg_img_slices + img_slices

            slice_anchor_m += pad_below_m + slice_end_m - slice_start_m
            obj_preds_slice = self.get_obj_preds_slice(seg_id, slice_start_m, slice_end_m, slice_anchor_m)
            obj_preds_slices.append(obj_preds_slice)
            slice_anchor_m += pad_above_m

        img_tensor = torch.cat(img_slices, dim=1)

        if obj_preds_slices:
            obj_preds = pd.concat(obj_preds_slices, ignore_index=True)
        else:
            obj_preds = pd.DataFrame()

        return img_tensor, obj_preds

    def get_image_slices(self, seg_id, slice_start_m, slice_end_m, pad_above_m, pad_below_m):

        seg = self.link_segments.loc[seg_id]
        camera_id = seg["camera_point_id"]
        slice_len_m = slice_end_m - slice_start_m

        slice_start_px = int(round((self.dim_gsd - slice_end_m) * self.pixels_per_meter))
        slice_end_px = int(round((self.dim_gsd - slice_start_m) * self.pixels_per_meter))

        seg_slices = []

        if slice_len_m <= 0:
            print(f"⚠️ Zero-length slice for {seg_id}, skipping")
            return [torch.zeros((3, 1, self.dim_pixels))]

        if pd.isnull(camera_id):
            slice_len_px = int(round((pad_above_m + slice_len_m + pad_below_m) * self.pixels_per_meter))
            print(f"⚠️ No camera point for {seg_id}, inserting blank slice")
            return [torch.zeros((3, slice_len_px, self.dim_pixels))]

        image_path = self.image_dir / f"{camera_id}.png"
        if not image_path.exists():
            slice_len_px = int(round((pad_above_m + slice_len_m + pad_below_m) * self.pixels_per_meter))
            print(f"⚠️ Missing image {image_path.name}, inserting blank slice")
            return [torch.zeros((3, slice_len_px, self.dim_pixels))]

        # Load and resize
        img = Image.open(image_path).convert("RGB").resize((self.dim_pixels, self.dim_pixels))
        img_np = np.array(img)

        if pad_above_m > 0:
            pad_above_px = int(round(pad_above_m * self.pixels_per_meter))
            pad_above = np.zeros((pad_above_px, self.dim_pixels, 3), dtype=np.uint8)
            pad_above = torch.from_numpy(pad_above).permute(2, 0, 1).float() / 255.0
            seg_slices.append(pad_above)

        img_slice = img_np[slice_start_px:slice_end_px, :, :].copy()
        img_slice = torch.from_numpy(img_slice).permute(2, 0, 1).float() / 255.0
        seg_slices.append(img_slice)

        if pad_below_m > 0:
            pad_below_px = int(round(pad_below_m * self.pixels_per_meter))
            pad_below = np.zeros((pad_below_px, self.dim_pixels, 3), dtype=np.uint8)
            pad_below = torch.from_numpy(pad_below).permute(2, 0, 1).float() / 255.0
            seg_slices.append(pad_below)

        return seg_slices

    def get_obj_preds_slice(
            self,
            seg_id: str,
            slice_start_m,
            slice_end_m,
            slice_anchor_m,
    ) -> pd.DataFrame:

        seg = self.link_segments.loc[seg_id]
        camera_id = seg["camera_point_id"]

        # Load predictions
        obj_preds = self.obj_predictions.get(camera_id, pd.DataFrame())
        if obj_preds.empty:
            return obj_preds

        # Denormalize coordinates to pixel space
        obj_preds = obj_preds.copy()
        obj_preds["x_center"] *= self.dim_pixels
        obj_preds["y_center"] *= self.dim_pixels
        obj_preds["width"] *= self.dim_pixels
        obj_preds["height"] *= self.dim_pixels

        # Compute camera-centered crop bounds (same as image)
        slice_start_px = int(round((self.dim_gsd - slice_end_m) * self.pixels_per_meter))
        slice_end_px = int(round((self.dim_gsd - slice_start_m) * self.pixels_per_meter))
        slice_anchor_px = int(round((self.dim_gsd - slice_anchor_m) * self.pixels_per_meter))

        # Filter to those inside cropped image
        obj_preds = obj_preds[obj_preds["y_center"].between(slice_start_px, slice_end_px)]
        if obj_preds.empty:
            return obj_preds

        obj_preds["y_center"] += slice_anchor_px - slice_start_px

        return obj_preds

    def shift_image(self, img, shift_m):
        shift_px = int(round(shift_m * self.pixels_per_meter))
        if shift_px > 0:
            pad = torch.zeros((3, img.shape[1], shift_px), dtype=img.dtype)
            return torch.cat([pad, img[:, :, :-shift_px]], dim=2)
        elif shift_px < 0:
            pad = torch.zeros((3, img.shape[1], -shift_px), dtype=img.dtype)
            return torch.cat([img[:, :, -shift_px:], pad], dim=2)
        return img

    def shift_object_predictions(self, obj_preds, shift_m):
        shift_px = int(round(shift_m * self.pixels_per_meter))
        obj_preds['x_center'] += shift_px
        obj_preds = obj_preds[obj_preds["x_center"].between(0, self.dim_pixels)].copy()
        return obj_preds

    @staticmethod
    def shift_points(df, shift, max_offset, col='x_offset'):
        df[col] += shift
        df = df[df[col].between(-max_offset, max_offset)].copy()
        return df
