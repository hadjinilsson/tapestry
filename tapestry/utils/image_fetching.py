import os
import boto3
import requests
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

# ─────────────────────────────────────────────
# ENVIRONMENT
# ─────────────────────────────────────────────
AWS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_ENDPOINT = os.getenv("AWS_S3_ENDPOINT")
BUCKET_NAME_IMAGES = os.getenv("BUCKET_NAME_IMAGERY")

# ─────────────────────────────────────────────
# UTILITY: Key construction
# ─────────────────────────────────────────────
def get_image_key(camera_point_id: str) -> str:
    base, numeric = camera_point_id.split("_")
    x = (int(numeric) // 10000) * 10000
    y = x + 10000
    return f"{base}/{x}_{y}/{base}_{numeric}.png"

# ─────────────────────────────────────────────
# PRESIGNED URL GENERATION
# ─────────────────────────────────────────────
def generate_presigned_url(key: str, expires_in: int = 3600) -> str:
    s3 = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=AWS_KEY,
        aws_secret_access_key=AWS_SECRET,
    )
    return s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": BUCKET_NAME_IMAGES, "Key": key},
        ExpiresIn=expires_in,
    )

# ─────────────────────────────────────────────
# DOWNLOAD IMAGE + (OPTIONAL) LABEL JSON
# ─────────────────────────────────────────────
def download_image(camera_point_id: str, dest_image_path: Path, dest_json_path: Path = None):
    key = get_image_key(camera_point_id)

    # Download image
    try:
        image_url = generate_presigned_url(key)
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        with open(dest_image_path, "wb") as f:
            f.write(response.content)
        # print(f"✅ Image downloaded: {camera_point_id}")
    except Exception as e:
        print(f"❌ Failed to download image for {camera_point_id}: {e}")

    # Optional: download JSON label
    if dest_json_path is not None:
        json_key = key.replace(".png", ".json")
        try:
            json_url = generate_presigned_url(json_key)
            response = requests.get(json_url, timeout=30)
            response.raise_for_status()
            with open(dest_json_path, "wb") as f:
                f.write(response.content)
            # print(f"✅ JSON label downloaded: {camera_point_id}")
        except Exception as e:
            print(f"⚠️  Failed to download JSON for {camera_point_id}: {e}")


def download_one(cp_id: str, img_dir: Path):
    img_path = img_dir / f"{cp_id}.png"
    if not img_path.exists():
        download_image(cp_id, dest_image_path=img_path)


def download_image_batch(camera_ids: list[str], img_dir: Path, max_workers: int = 10):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_one, cp_id, img_dir) for cp_id in camera_ids]
        for i, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
            except Exception as e:
                print(f"⚠️ Failed to download image {camera_ids[i-1]}: {e}")


def delete_downloaded_images(img_dir):
    for img_path in img_dir.glob("*.png"):
        img_path.unlink(missing_ok=True)


def download_images_for_camera_points(camera_point_ids, image_dir: Path, batch_size: int = 100, overwrite=False):
    image_dir.mkdir(parents=True, exist_ok=True)

    for i in range(0, len(camera_point_ids), batch_size):
        batch = camera_point_ids[i:i + batch_size]
        print(f"📥 Downloading batch {i // batch_size + 1} ({len(batch)} images)...")
        for cp_id in batch:
            image_path = image_dir / f"{cp_id}.png"
            if image_path.exists() and not overwrite:
                continue
            download_image(cp_id, dest_image_path=image_path)


def download_images_for_camera_points_threaded(
        camera_point_ids,
        image_dir: Path,
        max_workers: int = 16,
        overwrite=False
):
    """
    Download images using multithreading.

    Args:
        camera_point_ids: List of camera_point_id strings.
        image_dir: Directory to save images.
        max_workers: Number of threads to use.
        overwrite: If False, skips images that already exist.
    """
    image_dir.mkdir(parents=True, exist_ok=True)

    def download_one(cp_id):
        img_path = image_dir / f"{cp_id}.png"
        if not overwrite and img_path.exists():
            return
        download_image(cp_id, dest_image_path=img_path)

    print(f"📥 Downloading {len(camera_point_ids)} images with {max_workers} threads...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_one, cp_id): cp_id for cp_id in camera_point_ids}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
            except Exception as e:
                cp_id = futures[future]
                print(f"⚠️ Failed to download image {cp_id}: {e}")
