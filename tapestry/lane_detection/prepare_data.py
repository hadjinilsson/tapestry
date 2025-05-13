import os
import argparse
import pandas as pd
import geopandas as gpd
from pathlib import Path
from dotenv import load_dotenv

from tapestry.utils.db import get_link_segments_near_annotation_areas, get_sections_by_link_segment_ids
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
    candidate_link_segments = get_link_segments_near_annotation_areas(
        buffer_meters=25,
        annotation_area_names=annotation_area_ids
    )
    print(f"✅ Retrieved {len(candidate_link_segments)} candidate link segments.")

    candidate_path = GEOMETRY_DIR / "link_segments_all.parquet"
    candidate_link_segments.to_parquet(candidate_path)
    print(f"📦 Saved all link segments to {candidate_path}")

    # Step 2: Filter annotated segments with valid camera point
    annotated_segments = candidate_link_segments[
        (candidate_link_segments["annotated"] == "Y") &
        (candidate_link_segments["camera_point_id"].notnull())
    ].copy()
    annotated_segments.to_parquet(GEOMETRY_DIR / "link_segments_annotated.parquet")

    camera_point_ids = annotated_segments["camera_point_id"].unique().tolist()
    print(f"✅ Filtered to {len(annotated_segments)} annotated segments with {len(camera_point_ids)} unique camera points.")

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

    final_segments = annotated_segments[
        annotated_segments["camera_point_id"].isin(valid_cp_ids)
    ].copy()

    final_segments.to_parquet(GEOMETRY_DIR / "link_segments_final.parquet")
    print(f"✅ Final training set: {len(final_segments)} link segments with image + prediction.")

    # Step 5: Fetch sections for the final training link segments
    print("🟡 Step 5: Fetching sections...")
    final_segment_ids = final_segments["link_segment_id"].tolist()
    sections = get_sections_by_link_segment_ids(final_segment_ids)
    sections_path = GEOMETRY_DIR / "sections.parquet"
    sections.to_parquet(sections_path)
    print(f"✅ Saved {len(sections)} sections to {sections_path}")

    # Step 6: Compute neighbors for all final link segments
    print("🟡 Step 6: Computing neighbor relationships...")
    neighbours = compute_neighbours(final_segments)
    neighbours_path = GEOMETRY_DIR / "neighbours.parquet"
    neighbours.to_parquet(neighbours_path)
    print(f"✅ Saved neighbor list to {neighbours_path} ({len(neighbours)} rows)")

    # Step 7: Extract link ordering info
    print("🟡 Step 7: Extract link ordering info...")

    # Extract necessary fields: link_id, segment_ix_uv, segment_ix_vu, link_segment_id
    order_df = final_segments[[
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