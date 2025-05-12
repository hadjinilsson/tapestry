import os
import boto3
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# ENVIRONMENT
# ─────────────────────────────────────────────
AWS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_ENDPOINT = os.getenv("AWS_S3_ENDPOINT")
BUCKET_NAME_IMAGES = os.getenv("BUCKET_NAME_IMAGES")

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
        print(f"✅ Image downloaded: {camera_point_id}")
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
            print(f"✅ JSON label downloaded: {camera_point_id}")
        except Exception as e:
            print(f"⚠️  Failed to download JSON for {camera_point_id}: {e}")
