import os
import argparse
import pandas as pd
import geopandas as gpd
from pathlib import Path
from dotenv import load_dotenv

from tapestry.utils.db import get_link_segments_near_annotation_areas, get_sections_by_link_segment_ids
from tapestry.utils.db import get_base_network_crs
from tapestry.utils.image_fetching import download_images_for_camera_points
from tapestry.utils.image_fetching import download_images_for_camera_points_threaded
from tapestry.utils.s3 import download_dir_from_s3
from tapestry.lane_detection.utils.neighbours import compute_neighbours

load_dotenv()
DATA_ROOT = Path("data")
GEOMETRY_DIR = DATA_ROOT / "lane_detection" / "geometry"
GEOMETRY_DIR.mkdir(parents=True, exist_ok=True)
S3_BUCKET = os.getenv("BUCKET_NAME_PREDICTIONS")

def main(
        annotation_area_ids: list[str] | None = None,
        object_prediction_run_id: str = None,
        threads: int | None = None
):
    # Step 1: Fetch all link segments within 25m of annotation areas
    print("🟡 Step 1: Fetching link segments near annotation areas...")
    all_link_segments = get_link_segments_near_annotation_areas(
        buffer_meters=25,
        annotation_area_names=annotation_area_ids
    )
    print(f"✅ Retrieved {len(all_link_segments)} candidate link segments.")

    all_link_path = GEOMETRY_DIR / "link_segments_all.parquet"
    all_link_segments.to_parquet(all_link_path)
    print(f"📦 Saved all link segments to {all_link_path}")

    # Step 2: Filter annotated segments with valid camera point
    annotated_link_segments = all_link_segments[
        (all_link_segments["annotated"] == "Y") &
        (all_link_segments["camera_point_id"].notnull())
    ].copy()
    annotated_link_segments.to_parquet(GEOMETRY_DIR / "link_segments_annotated.parquet")

    camera_point_ids = annotated_link_segments["camera_point_id"].unique().tolist()
    print(f"✅ Filtered to {len(annotated_link_segments)} annotated segments with {len(camera_point_ids)} unique camera points.")

    # Step 3: Download images for those camera points
    images_dir = DATA_ROOT / "lane_detection" / "images"
    if threads and threads > 0:
        download_images_for_camera_points_threaded(camera_point_ids, images_dir, max_workers=args.threads)
    else:
        download_images_for_camera_points(camera_point_ids, images_dir)
    print(f"✅ Images downloaded to {images_dir}")

    downloaded = [f.stem for f in images_dir.glob("*.png")]
    missing = sorted(set(camera_point_ids) - set(downloaded))
    if missing:
        print(f"⚠️ {len(missing)} images were not downloaded.")

    # Step 4: Download and validate object detection predictions
    print("🟡 Step 4: Downloading object detection predictions...")
    predictions_dir = DATA_ROOT / "lane_detection" / "predictions"
    s3_prefix = f"object_detection/{object_prediction_run_id}"
    download_dir_from_s3(s3_prefix=s3_prefix, local_dir=predictions_dir, bucket=S3_BUCKET)

    pred_camera_ids = set()
    for parquet_path in predictions_dir.glob("*.parquet"):
        try:
            df_pred = pd.read_parquet(parquet_path)
            pred_camera_ids.update(df_pred["camera_point_id"].unique())
        except Exception as e:
            print(f"⚠️ Could not read {parquet_path.name}: {e}")

    existing_images = set(f.stem for f in images_dir.glob("*.png"))
    valid_cp_ids = pred_camera_ids & existing_images

    training_link_segments = annotated_link_segments[
        annotated_link_segments["camera_point_id"].isin(valid_cp_ids)
    ].copy()

    training_link_segments.to_parquet(GEOMETRY_DIR / "link_segments_training.parquet")
    print(f"✅ Final training set: {len(training_link_segments)} link segments with image + prediction.")

    # Step 5: Fetch sections for the final training link segments
    print("🟡 Step 5: Fetching sections...")
    final_segment_ids = training_link_segments["link_segment_id"].tolist()
    sections = get_sections_by_link_segment_ids(final_segment_ids)
    sections_path = GEOMETRY_DIR / "sections.parquet"
    sections.to_parquet(sections_path)
    print(f"✅ Saved {len(sections)} sections to {sections_path}")

    # Step 6: Compute neighbors for all final link segments
    print("🟡 Step 6: Computing neighbor relationships...")
    crs_lookup = get_base_network_crs()
    neighbours = compute_neighbours(training_link_segments, crs_lookup)
    neighbours_path = GEOMETRY_DIR / "neighbours.parquet"
    neighbours.to_parquet(neighbours_path)
    print(f"✅ Saved neighbor list to {neighbours_path} ({len(neighbours)} rows)")

    # Step 7: Extract link ordering info
    print("🟡 Step 7: Extract link ordering info...")

    # Extract necessary fields: link_id, segment_ix_uv, segment_ix_vu, link_segment_id
    order_df = training_link_segments[[
        "link_segment_id",
        "link_id",
        "segment_ix_uv",
        "segment_ix_vu"
    ]].copy()

    # Sort and group
    order_df_uv = order_df.sort_values(["link_id", "segment_ix_uv"])
    order_df_vu = order_df.sort_values(["link_id", "segment_ix_vu"])

    # Save both orderings
    order_df_uv.to_parquet(GEOMETRY_DIR / "segment_order_uv.parquet", index=False)
    order_df_vu.to_parquet(GEOMETRY_DIR / "segment_order_vu.parquet", index=False)

    print(f"✅ Saved ordered link segment indices to: \n  - segment_order_uv.parquet\n  - segment_order_vu.parquet")

    # Step 8: Reproject geometries to local CRS (per base network)
    print("🟡 Step 8: Reprojecting link segments and sections by base network...")

    all_link_segments['is_training'] = True
    all_link_segments.loc[
        ~all_link_segments.link_segment_id.isin(training_link_segments.link_segment_id),
        'is_training'
    ] = False

    projected_links = []
    projected_sections = []

    for base_network_id, epsg in crs_lookup.items():
        print(f"  ↪ Reprojecting base network {base_network_id} to EPSG:{epsg}...")

        # --- Link Segments ---
        ls_subset = all_link_segments[all_link_segments["base_network_id"] == base_network_id].copy()
        if not ls_subset.empty:
            gdf_ls = gpd.GeoDataFrame(ls_subset, geometry="geom", crs="EPSG:3857")
            gdf_ls_proj = gdf_ls.to_crs(epsg=epsg)
            gdf_ls_proj["geom_proj"] = gdf_ls_proj.geometry
            gdf_ls_proj["length_proj"] = gdf_ls_proj.geometry.length
            gdf_ls_proj = gdf_ls_proj.drop(columns=["geom"])
            projected_links.append(gdf_ls_proj)

        # --- Sections ---
        s_subset = sections[sections["base_network_id"] == base_network_id].copy()
        if not s_subset.empty:
            gdf_sec = gpd.GeoDataFrame(s_subset, geometry="geom", crs="EPSG:4326")
            gdf_sec_proj = gdf_sec.to_crs(epsg=epsg)
            gdf_sec_proj["geom_proj"] = gdf_sec_proj.geometry
            gdf_sec_proj["length_proj"] = gdf_sec_proj.geometry.length
            gdf_sec_proj = gdf_sec_proj.drop(columns=["geom"])
            projected_sections.append(gdf_sec_proj)

    if projected_links:
        df_links_proj = pd.concat(projected_links, ignore_index=True)
        df_links_proj.to_parquet(GEOMETRY_DIR / "link_segments_projected.parquet")
        print(f"✅ Saved projected link segments ({len(df_links_proj)} rows)")

    if projected_sections:
        df_sections_proj = pd.concat(projected_sections, ignore_index=True)
        df_sections_proj.to_parquet(GEOMETRY_DIR / "sections_projected.parquet")
        print(f"✅ Saved projected sections ({len(df_sections_proj)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--area_names",
        nargs="+",
        type=str,
        help="List of AnnotationArea namess to include. If omitted, all areas are used.",
    )
    parser.add_argument(
        "--object-prediction-run-id",
        required=True,
        help="Run ID for object detection predictions to download from S3."
    )
    parser.add_argument(
        "--threads",
        type=int,
        help="Enable threaded image download (specify number of threads)"
    )
    args = parser.parse_args()
    main(
        annotation_area_ids=args.area_names,
        object_prediction_run_id=args.object_prediction_run_id,
        threads=args.threads,
    )