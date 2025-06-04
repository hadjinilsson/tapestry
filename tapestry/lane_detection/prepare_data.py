import os
import math
import json
import argparse
import pandas as pd
import geopandas as gpd
from pathlib import Path
from shapely.geometry import LineString
from dotenv import load_dotenv

from tapestry.utils import db
from tapestry.utils.image_fetching import download_images_for_camera_points
from tapestry.utils.image_fetching import download_images_for_camera_points_threaded
from tapestry.utils.s3 import download_dir_from_s3, download_file_from_s3
from tapestry.utils.config import save_args
from tapestry.lane_detection.utils.neighbours import compute_neighbours
from tapestry.lane_detection.utils.camera_point_offset import get_camera_point_offset

load_dotenv()
S3_BUCKET_PREDS = os.getenv("BUCKET_NAME_PREDICTIONS")
S3_BUCKET_MODELS = os.getenv("BUCKET_NAME_MODELS")
DATA_DIR = Path("data") / "lane_detection"
IMAGE_DIR = DATA_DIR / "images"
GEOM_DIR = DATA_DIR / "geometry"
OBJ_PREDS_DIR = DATA_DIR / "object_predictions"
PRIOR_PREDS_DIR = DATA_DIR / "prior_predictions"

for DIR in [IMAGE_DIR, GEOM_DIR, OBJ_PREDS_DIR]:
    DIR.mkdir(parents=True, exist_ok=True)


def get_crs_lookup() -> dict[str, int]:
    base_networks = db.get_base_networks()
    return dict(zip(base_networks["base_network_id"], base_networks["crs"]))


def compute_bearing(geom: LineString) -> float:
    x1, y1 = geom.coords[0]
    x2, y2 = geom.coords[-1]
    return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 360


def prepare_data(
        segs_all: gpd.GeoDataFrame,
        crs_lookup: dict[str, int],
        object_prediction_run_ids: list[str],
        prior_run_id: str | None = None,
        skip_image_download: bool = False,
        threads: int | None = None,
):

    print("🟡 Preparing data...")

    print("🟡 Calculating camera point offsets...")
    segs_all['base_network_id'] = segs_all.link_segment_id.str.split('_', expand=True)[0]
    caps = db.get_camera_points_by_ids(list(segs_all.dropna(subset='camera_point_id').camera_point_id.unique()))
    segs_all = segs_all.merge(
        caps[['camera_point_id', 'geom']].rename(columns={'geom': 'camera_geom'}),
        how='left',
        on='camera_point_id',
    )
    segs_all = get_camera_point_offset(segs_all, crs_lookup)
    segs_all_path = GEOM_DIR / "link_segments_all.parquet"
    segs_all.to_parquet(segs_all_path)
    print(f"✅ Saved all link segments with camera offsets to {segs_all_path}")

    print("🟡 Filter annotated segments with valid camera point...")
    segs_annotated = segs_all[(segs_all["annotated"] == "Y") & (segs_all["camera_point_id"].notnull())].copy()
    segs_annotated.to_parquet(GEOM_DIR / "link_segments_annotated.parquet")
    print(f"✅ Filtered to {len(segs_annotated)} annotated segments.")

    # Download images for camera points
    cap_ids = segs_all["camera_point_id"].dropna().unique().tolist()
    if not skip_image_download:
        print("🟡 Downloading images...")
        if threads is not None and threads > 0:
            download_images_for_camera_points_threaded(cap_ids, IMAGE_DIR, max_workers=threads)
        else:
            download_images_for_camera_points(cap_ids, IMAGE_DIR)
        downloaded = [f.stem for f in IMAGE_DIR.glob("*.png")]
        missing = sorted(set(cap_ids) - set(downloaded))
        if missing:
            print(f"⚠️ {len(missing)} images were not downloaded.")
        print(f"✅ Image download complete.")
    else:
        print("⏭️ Skipping image download")

    print("🟡 Downloading and combining object predictions...")
    global_preds = []
    class_offset = 0
    obj_pred_conf = {}

    for obj_run_id in object_prediction_run_ids:
        print(f"  ↪ Downloading object predictions from {obj_run_id}...")

        # Download predictions
        run_dir = OBJ_PREDS_DIR / obj_run_id
        s3_prefix = f"object_detection/{obj_run_id}"
        download_dir_from_s3(s3_prefix=s3_prefix, local_dir=run_dir, bucket=S3_BUCKET_PREDS)

        # Download class_info.json separately
        class_info_path = run_dir / "class_info.json"
        if not class_info_path.exists():
            print(f"  ↪ Downloading class_info.json for {obj_run_id}...")
            download_file_from_s3(
                s3_key=f"{s3_prefix}/class_info.json",
                local_path=class_info_path,
                bucket=S3_BUCKET_MODELS,
            )

        if not class_info_path.exists():
            raise FileNotFoundError(f"class_info.json for run {obj_run_id} not found in S3 or local.")

        with open(class_info_path) as f:
            class_info = json.load(f)

        remap_dict = {}
        for entry in class_info:
            old_id = entry["id"]
            remap_dict[old_id] = {
                "remapped_id": old_id + class_offset,
                "class": entry["label"]
            }
        obj_pred_conf[obj_run_id] = remap_dict

        # Apply remapping to each parquet file in the run
        for parquet_path in run_dir.glob("*.parquet"):
            if "class_info" in parquet_path.name:  # skip any accidental extra files
                continue
            try:
                df = pd.read_parquet(parquet_path)
                df["class"] = df["class"].map(lambda x: remap_dict[x]["remapped_id"])
                global_preds.append(df)
            except Exception as e:
                print(f"⚠️ Could not read {parquet_path.name}: {e}")

        class_offset += len(remap_dict)

    # Save config with full remapping info
    with open(OBJ_PREDS_DIR / "object_prediction_config.json", "w") as f:
        json.dump(obj_pred_conf, f, indent=2)

    # Combine and save all predictions
    if not global_preds:
        raise RuntimeError("No valid object detection predictions found.")

    obj_preds = pd.concat(global_preds, ignore_index=True)
    obj_preds.to_parquet(OBJ_PREDS_DIR / "predictions.parquet")
    print(f"✅ Combined predictions from {len(object_prediction_run_ids)} runs.")

    if prior_run_id is not None:
        print(f"🟡 Downloading prior lane predictions from run {prior_run_id}...")
        PRIOR_PREDS_DIR.mkdir(parents=True, exist_ok=True)
        s3_prefix = f"lane_detection/{prior_run_id}"
        download_dir_from_s3(s3_prefix=s3_prefix, local_dir=PRIOR_PREDS_DIR, bucket=S3_BUCKET_PREDS)

        combined_path = PRIOR_PREDS_DIR / "combined.parquet"
        print(f"🟡 Combining prior prediction files into {combined_path}...")

        parquet_paths = sorted(PRIOR_PREDS_DIR.glob("*.parquet"))
        dfs = []
        for path in parquet_paths:
            try:
                df = pd.read_parquet(path)
                dfs.append(df)
            except Exception as e:
                print(f"⚠️ Skipping {path.name}: {e}")

        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            combined_df.to_parquet(combined_path, index=False)
            print(f"✅ Combined prior predictions saved to {combined_path} ({len(combined_df)} rows)")
        else:
            print("⚠️ No valid prior prediction files found to combine.")

    print("🟡 Filtering link segments with images...")
    if not skip_image_download:
        existing_images = set(f.stem for f in IMAGE_DIR.glob("*.png"))
        segs_train = segs_annotated[segs_annotated["camera_point_id"].isin(existing_images)].copy()
    else:
        segs_train = segs_annotated.copy()
    segs_train.to_parquet(GEOM_DIR / "link_segments_training.parquet")
    print(f"✅ Final training set: {len(segs_train)} link segments with image + prediction.")

    # Fetch sections for the final training link segments
    print("🟡 Fetching sections...")
    segs_final_ids = segs_train["link_segment_id"].tolist()
    sections = db.get_sections_by_link_segment_ids(segs_final_ids)
    if not sections.empty:
        sections.to_parquet(GEOM_DIR / "sections.parquet")
        print(f"✅ Sections download complete.")
    else:
        print("⚠️ No sections found for training segments. Continuing without labels.")
        sections = pd.DataFrame()

    # Compute neighbors for all final link segments
    print("🟡 Computing neighbor relationships...")
    neighbours = compute_neighbours(segs_train, crs_lookup)
    neighbours.to_parquet(GEOM_DIR / "neighbours.parquet")
    print(f"✅ Neighbours computed.")

    # Extract link ordering info
    print("🟡 Extract link ordering info...")
    # Extract necessary fields: link_id, segment_ix_uv, segment_ix_vu, link_segment_id
    seg_order = segs_all[[
        "link_segment_id",
        "link_id",
        "segment_ix_uv",
        "segment_ix_vu"
    ]].copy()
    # Sort and group
    seg_order_uv = seg_order.sort_values(["link_id", "segment_ix_uv"])
    seg_order_vu = seg_order.sort_values(["link_id", "segment_ix_vu"])
    # Save both orderings
    seg_order_uv.to_parquet(GEOM_DIR / "segment_order_uv.parquet", index=False)
    seg_order_vu.to_parquet(GEOM_DIR / "segment_order_vu.parquet", index=False)
    print(f"✅ Saved ordered link segment indices to: \n  - segment_order_uv.parquet\n  - segment_order_vu.parquet")

    # Reproject geometries to local CRS (per base network)
    print("🟡 Reprojecting link segments and sections by base network...")
    segs_all['is_training'] = True
    segs_all.loc[~segs_all.link_segment_id.isin(segs_train.link_segment_id), 'is_training'] = False
    segs_proj = []
    secs_proj = []
    for bn_id, epsg in crs_lookup.items():
        print(f"  ↪ Reprojecting base network {bn_id} to EPSG:{epsg}...")

        # --- Link Segments ---
        seg_subset = segs_all[segs_all["base_network_id"] == bn_id].copy()
        if not seg_subset.empty:
            gdf_seg = seg_subset.to_crs(epsg=epsg)
            gdf_seg = gdf_seg[gdf_seg.is_valid]
            gdf_seg["geom_proj"] = gdf_seg.geometry
            gdf_seg["length_proj"] = gdf_seg.geometry.length
            df_seg = {}
            for col in gdf_seg.drop(columns='geom').columns:
                df_seg[col] = list(gdf_seg[col])
            df_seg = pd.DataFrame(df_seg)
            segs_proj.append(df_seg)

        # --- Sections ---
        sec_subset = sections[sections["base_network_id"] == bn_id].copy() if not sections.empty else pd.DataFrame()
        if not sec_subset.empty:
            gdf_sec = sec_subset.to_crs(epsg=epsg)
            gdf_sec = gdf_sec[gdf_sec.is_valid]
            gdf_sec["geom_proj"] = gdf_sec.geometry
            gdf_sec["length_proj"] = gdf_sec.geometry.length
            df_sec = {}
            for col in gdf_sec.drop(columns='geom').columns:
                df_sec[col] = list(gdf_sec[col])
            df_sec = pd.DataFrame(df_sec)
            secs_proj.append(df_sec)

    if segs_proj:
        segs = pd.concat(segs_proj, ignore_index=True)
        segs["bearing"] = segs["geom_proj"].apply(compute_bearing)
        segs["geom_proj"] = segs["geom_proj"].apply(lambda g: g.wkb)
        segs.to_parquet(GEOM_DIR / "link_segments_projected.parquet")
        print(f"✅ Saved projected link segments ({len(segs)} rows)")

    if secs_proj:
        secs = pd.concat(secs_proj, ignore_index=True)
        secs["bearing"] = secs["geom_proj"].apply(compute_bearing)
        secs["geom_proj"] = secs["geom_proj"].apply(lambda g: g.wkb)
        secs.to_parquet(GEOM_DIR / "sections_projected.parquet")
        print(f"✅ Saved projected sections ({len(secs)} rows)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--object-detection-run-ids",
        required=True,
        nargs="+",
        help="One or more run IDs for object detection predictions",
    )
    parser.add_argument(
        "--prior-run-id",
        type=str,
        default=None,
        help="Run ID for previous lane detection model predictions (used in second-stage pseudo-recurrent training)",
    )
    filter_type = parser.add_mutually_exclusive_group(required=True)
    filter_type.add_argument("--base-networks", nargs="+", type=str)
    filter_type.add_argument("--annotation-areas", nargs="*", type=str)
    filter_type.add_argument("--annotated-link-segments", action="store_true")
    filter_type.add_argument("--annotated-nodes", action="store_true")
    filter_type.add_argument("--all", action="store_true")
    parser.add_argument("--skip-image-download", action="store_true")
    parser.add_argument("--threads", type=int)
    args = parser.parse_args()

    save_args(args, DATA_DIR / "data_config.json")

    # Fetch link segments
    if args.annotated_link_segments:
        link_segments = db.get_annotated_link_segments()
    elif args.annotated_nodes:
        link_segments = db.get_link_segments_for_annotated_nodes()
    elif args.annotation_areas is not None:
        area_names = args.annotation_areas if args.annotation_areas else None
        link_segments = db.get_link_segments_by_annotation_area(area_names=area_names)
    elif args.base_networks:
        link_segments = db.get_link_segments_by_base_network(args.base_networks)
    else:
        link_segments = db.get_all_link_segments()

    # Construct CRS lookup from base_networks table
    crs_lookup = get_crs_lookup()

    prepare_data(
        segs_all=link_segments,
        crs_lookup=crs_lookup,
        object_prediction_run_ids=args.object_detection_run_ids,
        prior_run_id=args.prior_run_id,
        skip_image_download=args.skip_image_download,
        threads=args.threads,
    )


if __name__ == "__main__":
    main()
