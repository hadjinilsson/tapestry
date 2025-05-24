import os
import math
import argparse
import pandas as pd
import geopandas as gpd
from pathlib import Path
from shapely.geometry import LineString
from dotenv import load_dotenv

from tapestry.utils import db
from tapestry.utils.image_fetching import download_images_for_camera_points
from tapestry.utils.image_fetching import download_images_for_camera_points_threaded
from tapestry.utils.s3 import download_dir_from_s3
from tapestry.lane_detection.utils.neighbours import compute_neighbours
from tapestry.lane_detection.utils.camera_point_offset import get_camera_point_offset

load_dotenv()
DATA_ROOT = Path("data")
GEOMETRY_DIR = DATA_ROOT / "lane_detection" / "geometry"
GEOMETRY_DIR.mkdir(parents=True, exist_ok=True)
S3_BUCKET = os.getenv("BUCKET_NAME_PREDICTIONS")


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
        object_prediction_run_id: str,
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
    segs_all_path = GEOMETRY_DIR / "link_segments_all.parquet"
    segs_all.to_parquet(segs_all_path)
    print(f"✅ Saved all link segments with camera offsets to {segs_all_path}")

    print("🟡 Filter annotated segments with valid camera point...")
    segs_annotated = segs_all[(segs_all["annotated"] == "Y") & (segs_all["camera_point_id"].notnull())].copy()
    segs_annotated.to_parquet(GEOMETRY_DIR / "link_segments_annotated.parquet")
    print(f"✅ Filtered to {len(segs_annotated)} annotated segments.")

    # Download images for camera points
    cap_ids = segs_all["camera_point_id"].dropna().unique().tolist()
    images_dir = DATA_ROOT / "lane_detection" / "images"
    if not skip_image_download:
        print("🟡 Downloading images...")
        if threads and threads > 0:
            download_images_for_camera_points_threaded(cap_ids, images_dir, max_workers=threads)
        else:
            download_images_for_camera_points(cap_ids, images_dir)
        downloaded = [f.stem for f in images_dir.glob("*.png")]
        missing = sorted(set(cap_ids) - set(downloaded))
        if missing:
            print(f"⚠️ {len(missing)} images were not downloaded.")
        print(f"✅ Image download complete.")
    else:
        print("⏭️ Skipping image download")

    print("🟡 Downloading object detection predictions...")
    predictions_dir = DATA_ROOT / "lane_detection" / "predictions"
    s3_prefix = f"object_detection/{object_prediction_run_id}"
    download_dir_from_s3(s3_prefix=s3_prefix, local_dir=predictions_dir, bucket=S3_BUCKET)
    pred_cap_ids = set()
    for parquet_path in predictions_dir.glob("*.parquet"):
        try:
            obj_preds = pd.read_parquet(parquet_path)
            pred_cap_ids.update(obj_preds["camera_point_id"].unique())
        except Exception as e:
            print(f"⚠️ Could not read {parquet_path.name}: {e}")
    print(f"✅ Object predictions download complete.")

    print("🟡 Filtering link segments with images...")
    if not skip_image_download:
        existing_images = set(f.stem for f in images_dir.glob("*.png"))
        valid_cap_ids = pred_cap_ids & existing_images
    else:
        valid_cap_ids = pred_cap_ids
    segs_train = segs_annotated[segs_annotated["camera_point_id"].isin(valid_cap_ids)].copy()
    segs_train.to_parquet(GEOMETRY_DIR / "link_segments_training.parquet")
    print(f"✅ Final training set: {len(segs_train)} link segments with image + prediction.")

    # Fetch sections for the final training link segments
    print("🟡 Fetching sections...")
    segs_final_ids = segs_train["link_segment_id"].tolist()
    sections = db.get_sections_by_link_segment_ids(segs_final_ids)
    sections.to_parquet(GEOMETRY_DIR / "sections.parquet")
    print(f"✅ Sections download complete.")

    # Compute neighbors for all final link segments
    print("🟡 Computing neighbor relationships...")
    neighbours = compute_neighbours(segs_train, crs_lookup)
    neighbours.to_parquet(GEOMETRY_DIR / "neighbours.parquet")
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
    seg_order_uv.to_parquet(GEOMETRY_DIR / "segment_order_uv.parquet", index=False)
    seg_order_vu.to_parquet(GEOMETRY_DIR / "segment_order_vu.parquet", index=False)
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
        sec_subset = sections[sections["base_network_id"] == bn_id].copy()
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
        segs.to_parquet(GEOMETRY_DIR / "link_segments_projected.parquet")
        print(f"✅ Saved projected link segments ({len(segs)} rows)")

    if secs_proj:
        secs = pd.concat(secs_proj, ignore_index=True)
        secs["bearing"] = secs["geom_proj"].apply(compute_bearing)
        secs["geom_proj"] = secs["geom_proj"].apply(lambda g: g.wkb)
        secs.to_parquet(GEOMETRY_DIR / "sections_projected.parquet")
        print(f"✅ Saved projected sections ({len(secs)} rows)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--object-detection-run-id", required=True, help="S3 run ID for object detection predictions")
    filter_type = parser.add_mutually_exclusive_group(required=True)
    filter_type.add_argument("--base-networks", nargs="+", type=str)
    filter_type.add_argument("--annotation-areas", nargs="*", type=str)
    filter_type.add_argument("--annotated-link-segments", action="store_true")
    filter_type.add_argument("--annotated-nodes", action="store_true")
    filter_type.add_argument("--all", action="store_true")
    parser.add_argument("--skip-image-download", action="store_true")
    parser.add_argument("--threads", type=int)
    args = parser.parse_args()

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
        object_prediction_run_id=args.object_detection_run_id,
        skip_image_download=args.skip_image_download,
        threads=args.threads,
    )


if __name__ == "__main__":
    main()
