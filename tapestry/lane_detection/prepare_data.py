import os
import argparse
import pandas as pd
import geopandas as gpd
from pathlib import Path
from dotenv import load_dotenv
from glob import glob

from tapestry.utils.db import get_link_segments_near_annotation_areas
from tapestry.utils.image_fetching import download_images_for_camera_points
from tapestry.utils.s3 import download_dir_from_s3
from tapestry.utils.db import get_sections_by_link_segment_ids
from tapestry.lane_detection.utils.neighbours import compute_neighbours

load_dotenv()
DATA_ROOT = Path("data")
GEOMETRY_DIR = DATA_ROOT / "lane_detection" / "geometry"
GEOMETRY_DIR.mkdir(parents=True, exist_ok=True)
S3_BUCKET = os.getenv("BUCKET_NAME_PREDICTIONS")


def main(annotation_area_ids: list[str] | None = None):
    # Step 1: Fetch all link segments within 25 m of annotation areas
    print("🟡 Step 1: Fetching link segments near annotation areas...")
    candidate_link_segments = get_link_segments_near_annotation_areas(
        buffer_meters=25,
        annotation_area_ids=annotation_area_ids
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
    print(f"""
    ✅ Filtered to {len(annotated_segments)}
    annotated segments with {len(camera_point_ids)}
    unique camera points.
    """)

    # Step 3: Download images
    images_dir = DATA_ROOT / "lane_detection" / "images"
    download_images_for_camera_points(camera_point_ids, images_dir)
    print(f"✅ Images downloaded to {images_dir}")

    downloaded = [f.stem for f in images_dir.glob("*.png")]
    missing = sorted(set(camera_point_ids) - set(downloaded))
    print(f"⚠️ {len(missing)} images were not downloaded.") if missing else None

    # Step 4: Download object predictions
    run_id = args.prediction_run_id
    predictions_dir = DATA_ROOT / "lane_detection" / "predictions"
    s3_prefix = f"object_detection/{run_id}"
    download_dir_from_s3(s3_prefix=s3_prefix, local_dir=predictions_dir, bucket=S3_BUCKET)

    pred_camera_ids = set()

    for parquet_path in predictions_dir.glob("*.parquet"):
        try:
            df_pred = pd.read_parquet(parquet_path)
            pred_camera_ids.update(df_pred["camera_point_id"].unique())
        except Exception as e:
            print(f"⚠️ Could not read {parquet_path.name}: {e}")

    # Determine which camera points are fully available
    existing_images = set(f.stem for f in images_dir.glob("*.png"))
    valid_cp_ids = pred_camera_ids & existing_images

    final_segments = annotated_segments[
        annotated_segments["camera_point_id"].isin(valid_cp_ids)
    ].copy()

    final_segments.to_parquet(GEOMETRY_DIR / "link_segments_final.parquet")
    print(f"✅ Final training set: {len(final_segments)} link segments with image + prediction.")

    print("🟡 Step 5: Fetching sections...")
    final_segment_ids = final_segments["link_segment_id"].tolist()
    sections = get_sections_by_link_segment_ids(final_segment_ids)

    sections_path = GEOMETRY_DIR / "sections.parquet"
    sections.to_parquet(sections_path)
    print(f"✅ Saved {len(sections)} sections to {sections_path}")

    neighbours = compute_neighbours(final_segments)
    neighbours_path = GEOMETRY_DIR / "neighbours.parquet"
    neighbours.to_parquet(neighbours_path)
    print(f"✅ Saved neighbor list to {neighbours_path} ({len(neighbours)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--areas",
        nargs="+",
        type=str,
        help="List of AnnotationArea IDs to include. If omitted, all areas are used.",
    )
    args = parser.parse_args()
    main(annotation_area_ids=args.areas)