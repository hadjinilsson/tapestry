import torch
from torch.utils.data import Dataset
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.affinity import rotate
from shapely import wkb, distance
from pathlib import Path
from PIL import Image
import math
import random
import json
import numpy as np
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from torchvision.transforms.functional import to_pil_image, to_tensor


class LaneDetectionDataset(Dataset):
    def __init__(
            self,
            data_root: Path,
            mode: str = "train",
            use_prior_preds = False,
            dim_pixels: int = 256,
            dim_gsd: float = 50.0,
            dim_bins: float = 1.0,
            lat_coverage: float = 35.0,
            lon_coverage: float = 25.0,
            pred_dist: float = 1.0,
            max_extra_image: float = 10.0,
            max_shift: float = 10.0,
            max_lanes: int = 5,
            max_class_weight: float = 100.0,
            seg_ids: list | None = None,
    ):
        # Arguments
        self.data_dir = Path(data_root)
        self.mode = mode
        self.use_prior_preds = use_prior_preds
        self.dim_pixels = dim_pixels
        self.dim_gsd = dim_gsd
        self.dim_bins = dim_bins
        self.lat_coverage = lat_coverage
        self.lon_coverage = lon_coverage
        self.pred_dist = pred_dist
        self.max_extra_image = max_extra_image
        self.max_shift = max_shift
        self.max_lanes = max_lanes
        self.max_class_weight = max_class_weight
        self.seg_ids = seg_ids

        # Derived
        self.pixels_per_meter = dim_pixels / dim_gsd
        self.num_prior_preds = int(round(2 * lon_coverage / pred_dist))
        self.enable_flip = (self.mode == "train")
        self.num_bins = int(round(dim_gsd / dim_bins))

        # Directories
        self.geometry_dir = self.data_dir / "geometry"
        self.image_dir = self.data_dir / "images"
        self.obj_preds_dir = self.data_dir / "object_predictions"
        self.prior_preds_dir = self.data_dir / "prior_predictions"

        # Data config
        config_path = self.data_dir / "data_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing data_prediction_config.json in {self.data_dir}")
        with open(config_path) as f:
            self.data_config = json.load(f)

        # Link segments
        df_segs = pd.read_parquet(self.geometry_dir / "link_segments_projected.parquet")
        df_segs["geom_proj"] = df_segs["geom_proj"].apply(wkb.loads)
        self.link_segments = df_segs.set_index("link_segment_id")

        # Sections
        sections_path = self.geometry_dir / "sections_projected.parquet"
        if sections_path.exists():
            df_secs = pd.read_parquet(sections_path)
            df_secs["geom_proj"] = df_secs["geom_proj"].apply(wkb.loads)
            self.sections = df_secs
            self.has_labels = True
        else:
            print("⚠️ No sections file found — must run in label-free prediction mode.")
            self.sections = None
            self.has_labels = False

        # Lateral neighbours
        self.lat_neighbours = pd.read_parquet(self.geometry_dir / "neighbours.parquet")
        self.order_uv = pd.read_parquet(self.geometry_dir / "segment_order_uv.parquet")
        self.order_vu = pd.read_parquet(self.geometry_dir / "segment_order_vu.parquet")

        # Build link_id to ordered segments map
        self.link_order = self._build_link_order()

        # Build index of samples (link_segment_id, geometry length)
        if self.mode == "predict":
            self.samples, self.seg_ids = self._build_sample_index()
        else:
            df_samples = df_segs[df_segs.is_training]
            if seg_ids is not None:
                df_samples = df_samples[df_samples.link_segment_id.isin(seg_ids)]
            self.samples = list(df_samples['link_segment_id'])
            self.seg_ids = self.samples

        # Object predictions
        self.obj_predictions, self.num_obj_pred_classes, self.obj_pred_config = self._load_all_obj_preds()

        # Prior predictions
        if self.use_prior_preds:
            self.prior_predictions = self._load_all_prior_preds()

    def _build_link_order(self):
        order = {}
        for direction, df in [("uv", self.order_uv), ("vu", self.order_vu)]:
            for link_id, group in df.groupby("link_id"):
                if link_id not in order:
                    order[link_id] = {}
                order[link_id][direction] = list(group["link_segment_id"])
        return order

    def _build_sample_index(self):
        sample_index = []
        sample_segs = self.link_segments
        if self.seg_ids is not None:
            sample_segs = sample_segs[sample_segs.index.isin(self.seg_ids)]
        seg_ids = sample_segs.index.tolist()
        for seg_id in seg_ids:
            seg = self.link_segments.loc[seg_id]
            seg_len = seg["length_proj"]
            is_last = seg['segment_ix_vu'] == 0
            num_samples = max(int(seg_len // self.pred_dist), 1)
            if not is_last:
                ds = np.linspace(0, seg_len, num=num_samples, endpoint=False)
            else:
                ds = np.linspace(0, seg_len, num=num_samples)
            for d in ds:
                sample_index.append((seg_id, d))
        return sample_index, seg_ids

    def _load_all_obj_preds(self):
        """
        Load combined predictions and full class remapping config.
        Ensure all training camera_point_ids are present in the prediction dict.
        """
        obj_preds = {}
        predictions_path = self.obj_preds_dir / "predictions.parquet"
        config_path = self.obj_preds_dir / "object_prediction_config.json"

        if not predictions_path.exists():
            raise FileNotFoundError(f"Missing predictions.parquet in {self.obj_preds_dir}")
        if not config_path.exists():
            raise FileNotFoundError(f"Missing object_prediction_config.json in {self.obj_preds_dir}")

        # Load class mapping config
        with open(config_path) as f:
            obj_pred_config = json.load(f)

        # Build remapped class ID set
        all_labels = set()
        for run_remap in obj_pred_config.values():
            for entry in run_remap.values():
                all_labels.add(entry["remapped_id"])

        # Load actual predictions
        df = pd.read_parquet(predictions_path)
        for cap_id, group in df.groupby("camera_point_id"):
            obj_preds[cap_id] = group

        # Ensure all training camera point IDs are present
        segs = self.link_segments
        expected_cap_ids = segs[segs.index.isin(self.seg_ids)]["camera_point_id"].dropna().unique()
        empty_df = pd.DataFrame(columns=df.columns, dtype=float)
        num_empty = 0
        for cap_id in expected_cap_ids:
            if cap_id not in obj_preds:
                obj_preds[cap_id] = empty_df.copy()
                num_empty += 1

        percent_empty = int(num_empty / len(expected_cap_ids) * 100)
        print(f'{num_empty} ({percent_empty}%) link segments without object predictions')

        return obj_preds, len(all_labels), obj_pred_config


    def _load_all_prior_preds(self):
        prior_preds = {}
        predictions_path = self.prior_preds_dir / "combined.parquet"

        if not predictions_path.exists():
            raise FileNotFoundError(f"Missing combined.parquet in {self.prior_preds_dir}")

        df = pd.read_parquet(predictions_path)
        keep_cols = ['link_segment_id', 'distance']
        for dr in ['forward', 'backward']:
            logit_cols = [f'{c}_logit_{dr}' for c in range(self.max_lanes + 1)]
            df[logit_cols] = pd.DataFrame(df[f'logits_{dr}'].to_list(), index=df.index)
            keep_cols.extend(logit_cols)
            keep_cols.extend([f'conf_{dr}', f'entropy_{dr}'])
        df = df[keep_cols]

        for seg_id, group in df.groupby("link_segment_id"):
            prior_preds[seg_id] = group.drop(columns='link_segment_id')

        # Ensure all training link segment IDs are present
        empty_df = pd.DataFrame(columns=df.drop(columns='link_segment_id').columns, dtype=float)
        num_empty = 0
        for seg_id in self.seg_ids:
            if seg_id not in prior_preds:
                prior_preds[seg_id] = empty_df.copy()
                num_empty += 1

        percent_empty = int(num_empty / len(self.seg_ids) * 100)
        print(f'{num_empty} ({percent_empty}%) link segments without prior predictions')

        return prior_preds

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx, fast=False):
        if self.mode == "predict":
            seg_id, dist_along = self.samples[idx]
            seg = self.link_segments.loc[seg_id]
            seg_len = seg['length_proj']
        else:
            seg_id = self.samples[idx]
            seg = self.link_segments.loc[seg_id]
            seg_len = seg['length_proj']
            if seg_len >= 3.0:
                margin = 1.0
            else:
                margin = seg_len / 3.0
            dist_along = random.uniform(margin, seg_len - margin)

        # Get link segment data
        seg_geom = seg['geom_proj']
        link_id = seg['link_id']

        sample_point = seg_geom.interpolate(dist_along)

        # # Cross-sectional line to get lateral neighbours and section labels
        cross_section = self.get_cross_section(seg_geom, dist_along, sample_point)

        #  Get labels
        if self.has_labels:
            sections = self.get_sections(seg_id, sample_point, cross_section)
        else:
            sections = None
        has_label = sections is not None

        # Get neighbours
        lat_neighbours = self.get_lat_neighbours(seg_id, sample_point, cross_section) if not fast else None
        lon_neighbours = self.get_lon_neighbours(seg_id, dist_along)
        self.check_lon_continuity(link_id, lon_neighbours)

        # Get used lengths
        used_pre_len, used_pro_len = self.get_used_lengths(lon_neighbours, dist_along, seg_len)

        # Get slice data
        img, obj_preds, prior_preds = self.get_slice_data(
            lon_neighbours,
            dist_along,
            used_pre_len,
            used_pro_len,
            fast
        )

        # Augmentations
        if self.mode == "train" and not fast:

            #  Image
            brightness_factor = random.uniform(0.8, 1.2)
            contrast_factor = random.uniform(0.8, 1.2)
            img = TF.adjust_brightness(img, brightness_factor)
            img = TF.adjust_contrast(img, contrast_factor)

            noise = torch.randn_like(img) * 0.02
            img = torch.clamp(img + noise, 0.0, 1.0)

            img = to_pil_image(img)
            img = T.GaussianBlur(kernel_size=3)(img)
            img = to_tensor(img)

            # Lateral shift
            if self.max_shift > 0:
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
                if self.use_prior_preds: prior_preds["distance"] = self.dim_gsd - prior_preds["distance"]

        else:
            shift = 0
            flip = False

        # Format data
        img = TF.resize(img, size=[self.dim_pixels, self.dim_pixels]) if not fast else None
        obj_scores = self.format_object_predictions(obj_preds)
        lat_neighbours = self.format_lat_neighbours(lat_neighbours) if not fast else None
        if has_label: sections = self.format_sections(sections)
        if self.use_prior_preds: prior_preds = self.format_prior_preds(prior_preds)

        # To tensor
        obj_scores =  torch.tensor(obj_scores.values, dtype=torch.float32)
        lat_neighbours =  torch.tensor(lat_neighbours, dtype=torch.float32) if not fast else None
        if has_label:
            sections = torch.tensor(sections, dtype=torch.float32)
        else:
            sections = torch.zeros((2, self.max_lanes + 1), dtype=torch.int32)
        if self.use_prior_preds: prior_preds = torch.tensor(prior_preds.values, dtype=torch.float32)

        #  Check for nans
        to_check = [obj_scores, sections] if fast else [img, obj_scores, lat_neighbours, sections]
        if self.use_prior_preds: to_check.append(prior_preds)
        for tns in to_check:
            assert not torch.isnan(tns).any(), "NaN detected in input"

        sample = {
            "numerical_id": idx,
            "image": img,
            "object_scores": obj_scores,
            "lat_neighbours": lat_neighbours,
            "sections": sections,
            "shift": shift,
            "flip": flip,
            "has_label": has_label,
        }

        if self.use_prior_preds: sample['prior_predictions'] = prior_preds

        return sample

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

        if seg_id not in self.link_segments.index:
            return None

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

    def get_lon_neighbours(self, current_seg_id: str, dist_along: float):

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
        remaining_back = self.lon_coverage - dist_along
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
                min(dist_along, self.lon_coverage) +
                min(current_seg_len - dist_along, self.lon_coverage)
        )
        current = {
            "link_segment_id": current_seg_id,
            "used_length": current_seg_used_len,
            "placement": "current",
        }
        lon_neighbours.append(current)

        # Proceeding (top of image)
        remaining_forward = self.lon_coverage - (current_seg_len - dist_along)
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
            dist_along: float,
            used_pre_len: float,
            used_pro_len: float,
            fast: False,
    ):

        img_slices = []
        obj_preds_slices = []
        prior_preds_slices = [] if self.use_prior_preds else None
        slice_offset = 0
        slice_offset_img = 0

        for i, lon_neighbour in enumerate(lon_neighbours):

            is_first = i == 0
            is_last = i == (len(lon_neighbours) - 1)
            is_current = lon_neighbour['placement'] == 'current'

            seg_id = lon_neighbour["link_segment_id"]
            seg = self.link_segments.loc[seg_id]

            is_first_uv = seg["segment_ix_uv"] == 0
            is_last_uv = seg["segment_ix_vu"] == 0

            seg_len = seg["length_proj"]
            seg_used_len = lon_neighbour.get("used_length", 0.0)
            camera_offset = seg.get("camera_point_offset", 0.0)

            # Measurement from bottom
            img_c_img = self.dim_gsd / 2
            seg_c_img = img_c_img - camera_offset
            seg_s_img = seg_c_img - seg_len / 2
            seg_e_img = seg_c_img + seg_len / 2

            pad_above = 0
            pad_below = 0
            img_pad_above = 0
            img_pad_below = 0

            if (is_first and is_first_uv) or (is_first and (used_pre_len < self.lon_coverage)):
                pad_below = self.lon_coverage - used_pre_len
                extra_img = min(pad_below, self.max_extra_image)
                if is_current:
                    extra_img += min(seg_s_img + dist_along - used_pre_len - extra_img, 0)
                    slice_s_img = seg_s_img + dist_along - used_pre_len - extra_img
                    slice_s = max(dist_along - self.lon_coverage, 0)
                else:
                    extra_img += min(seg_e_img - seg_used_len - extra_img, 0)
                    slice_s_img = seg_e_img - seg_used_len - extra_img
                    slice_s = seg_len - seg_used_len
                img_pad_below = pad_below - extra_img
            elif is_first:
                slice_s_img = seg_e_img - seg_used_len
                slice_s = seg_len - seg_used_len
            else:
                slice_s_img = seg_s_img
                slice_s = 0

            if (is_last and is_last_uv) or (is_last and (used_pro_len < self.lon_coverage)):
                pad_above = self.lon_coverage - used_pro_len
                extra_img = min(pad_above, self.max_extra_image)
                if is_current:
                    extra_img -= max(seg_s_img + dist_along + used_pro_len + extra_img - self.dim_gsd, 0)
                    slice_e_img = seg_s_img + dist_along + used_pro_len + extra_img
                    slice_e = min(dist_along + self.lon_coverage, seg_len)
                else:
                    extra_img -= max(seg_s_img + seg_used_len + extra_img - self.dim_gsd, 0)
                    slice_e_img = seg_s_img + seg_used_len + extra_img
                    slice_e = seg_used_len
                img_pad_above = pad_above - extra_img
            elif is_last:
                slice_e_img = seg_s_img + seg_used_len
                slice_e = seg_used_len
            else:
                slice_e_img = seg_e_img
                slice_e = seg_len

            if not fast:
                seg_img_slices = self.get_image_slices(seg_id, slice_s_img, slice_e_img, img_pad_below, img_pad_above)
                img_slices = seg_img_slices + img_slices

            slice_offset_img += img_pad_below + slice_e_img - slice_s_img
            obj_preds_slice = self.get_obj_preds_slice(seg_id, slice_s_img, slice_e_img, slice_offset_img)
            obj_preds_slices.append(obj_preds_slice)
            slice_offset_img += img_pad_above

            if self.use_prior_preds:
                slice_offset += pad_below
                prior_preds_slice = self.get_prior_preds_slice(
                    seg_id,
                    slice_s,
                    slice_e,
                    # pad_below,
                    # pad_above,
                    slice_offset,
                )
                prior_preds_slices.append(prior_preds_slice)
                slice_offset += seg_used_len + pad_above

        img_tensor = torch.cat(img_slices, dim=1) if not fast else None

        if obj_preds_slices:
            obj_preds = pd.concat(obj_preds_slices, ignore_index=True)
        else:
            obj_preds = pd.DataFrame()

        if self.use_prior_preds and prior_preds_slices:
            prior_preds = pd.concat(prior_preds_slices, ignore_index=True)
        elif self.use_prior_preds:
            prior_preds = pd.DataFrame()
        else:
            prior_preds = None

        return img_tensor, obj_preds, prior_preds

    def get_image_slices(self, seg_id, slice_start_m, slice_end_m, pad_below_m, pad_above_m):

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
            slice_offset_m,
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
        slice_offset_px = int(round((self.dim_gsd - slice_offset_m) * self.pixels_per_meter))

        # Filter to those inside cropped image
        obj_preds = obj_preds[obj_preds["y_center"].between(slice_start_px, slice_end_px)]
        if obj_preds.empty:
            return obj_preds

        obj_preds["y_center"] += slice_offset_px - slice_start_px

        return obj_preds

    def get_prior_preds_slice(
            self,
            seg_id: str,
            slice_start,
            slice_end,
            # pad_below,
            # pad_above,
            slice_offset,
    ) -> pd.DataFrame:

        prior_pred_slices = []

        # Load predictions
        prior_preds = self.prior_predictions.get(seg_id, pd.DataFrame())
        # Filter to those inside used length and offset
        if not prior_preds.empty:
            prior_preds = prior_preds[prior_preds["distance"].between(slice_start, slice_end)].copy()
            prior_preds["distance"] += slice_offset - slice_start
        # prior_pred_slices.append(prior_pred_slice)

        # if pad_below > 0:
        #     # Create df with shape int(pad_below), prior_pred_slice.shape[1], columns and dtypes from prior_pred_slices
        #     # distance column = range(int(pad_below))
        #     # not sure what logit, conf, and entropy columns should be
        #     # prior_pred_slices.append()
        #     pass

        return prior_preds

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

    def format_object_predictions(self, obj_preds):

        # Normalize to projected space
        obj_scores = obj_preds.copy()
        obj_scores[['x_center', 'y_center', 'width', 'height']] /= (self.pixels_per_meter * self.dim_bins)
        obj_scores['score'] = obj_scores['confidence'] * obj_scores['height']
        obj_scores['score'] /= np.abs(obj_scores['y_center'] - self.lon_coverage).clip(lower=1.0)
        obj_scores['x_start'] = obj_scores['x_center'] - obj_scores['width'] / 2
        obj_scores['x_end'] = obj_scores['x_center'] + obj_scores['width'] / 2

        # Create slot edges
        slots = np.linspace(0, self.lat_coverage, num=self.num_bins + 1)
        slot_centers = (slots[:-1] + slots[1:]) / 2

        # Broadcast x_start and x_end to slots
        x_start = obj_scores['x_start'].values[:, np.newaxis]
        x_end = obj_scores['x_end'].values[:, np.newaxis]
        score = obj_scores['score'].values[:, np.newaxis]

        # Check which slots fall within each object bbox (N objects × 50 slots)
        mask = (slot_centers >= x_start) & (slot_centers <= x_end)
        slot_matrix = mask * score  # Broadcasted score per slot if in range

        # Add as individual columns
        slot_cols = []
        for i in range(self.num_bins):
            col_name = f"slot_{i}"
            obj_scores[col_name] = slot_matrix[:, i]
            slot_cols.append(col_name)

        #  Group by object class and expand
        obj_scores = obj_scores.groupby('class')[slot_cols].sum().reindex(range(self.num_obj_pred_classes), fill_value=0)

        return obj_scores

    def format_prior_preds(self, prior_preds):
        prior_preds = prior_preds.sort_values('distance').reset_index(drop=True)
        prior_preds['pad'] = 0.0

        num_upper_preds = self.num_prior_preds // 2
        num_lower_preds = self.num_prior_preds - num_upper_preds

        upper_preds = prior_preds[prior_preds['distance'] < self.lon_coverage].copy()
        lower_preds = prior_preds[prior_preds['distance'] >= self.lon_coverage].copy().reset_index(drop=True)

        out_cols = [c for c in prior_preds.columns if c != 'distance']
        pad_row = [2.0, 0.0, -2.0, -3.0, -4.0, -5.0, 0.72, 0.95] * 2 + [1]
        pad_row_dict = dict(zip(out_cols, pad_row))

        num_extra = num_upper_preds - len(upper_preds)
        if num_extra > 0:
            pad_df = pd.DataFrame([pad_row_dict] * num_extra)
            upper_preds = pd.concat([pad_df, upper_preds], ignore_index=True)
        elif num_extra < 0:
            upper_preds = upper_preds.tail(num_upper_preds)

        num_extra = num_lower_preds - len(lower_preds)
        if num_extra > 0:
            pad_df = pd.DataFrame([pad_row_dict] * num_extra)
            lower_preds = pd.concat([lower_preds, pad_df], ignore_index=True)
        elif num_extra < 0:
            lower_preds = lower_preds.head(num_lower_preds)

        prior_preds = pd.concat([upper_preds, lower_preds], ignore_index=True)
        prior_preds = prior_preds.drop(columns='distance')

        max_entropy = math.log(self.max_lanes + 1)
        prior_preds['entropy_forward'] /= max_entropy
        prior_preds['entropy_backward'] /= max_entropy

        return prior_preds

    def format_lat_neighbours(self, lat_neighbours):
        lat_neighbours['x_offset'] += self.lat_coverage
        lat_neighbour_pos = lat_neighbours.x_offset.round().astype(int).clip(0, (self.lat_coverage*2) - 1).unique()
        lat_neighbours = np.zeros(round(self.lat_coverage * 2), dtype=int)
        lat_neighbours[lat_neighbour_pos] = 1
        return lat_neighbours

    def format_sections(self, section_df):
        section_mat = np.zeros((2, self.max_lanes + 1), dtype=int)
        for dir_idx, direction in enumerate(['forward', 'backward']):
            count = (section_df.direction == direction).sum()
            if count < self.max_lanes:
                section_mat[dir_idx, count] = 1
            else:
                section_mat[dir_idx, self.max_lanes] = 1
        return section_mat

    def compute_statistics(self):
        obj_scores = []
        section_pos = torch.zeros((2, self.max_lanes + 1), dtype=torch.int32)

        for i in range(len(self)):
            sample = self.__getitem__(i, fast=True)
            sample_obj_scores = pd.DataFrame(sample['object_scores'].detach().cpu().numpy().T)
            if not sample_obj_scores.empty:
                obj_scores.append(sample_obj_scores)

            section_pos += sample['sections'].int()

        # ---- Object predictions ----
        obj_scores = pd.concat(obj_scores)
        obj_score_means = obj_scores.mean(axis=0)
        obj_score_stds = obj_scores.std(axis=0).clip(lower=1e-6)

        # ---- Class balance ----
        section_neg = len(self) - section_pos
        section_weights = (section_neg / section_pos.clamp(min=1e-6)).clamp(max=self.max_class_weight)

        return {
            'object_pred_means': obj_score_means,
            'obj_score_stds': obj_score_stds,
            'lane_weights': section_weights,
        }

