import os
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Environment
IMAGE_PROXY_URL = os.getenv("IMAGE_PROXY_URL")
IMAGE_PROXY_TOKEN = os.getenv("IMAGE_PROXY_TOKEN")

# ─────────────────────────────────────────────
# UTILITY: Key construction
# ─────────────────────────────────────────────
def get_image_key(camera_point_id: str) -> str:
    base, numeric = camera_point_id.split("_")
    x = (int(numeric) // 10000) * 10000
    y = x + 10000
    return f"{base}/{x}_{y}/{base}_{numeric}.png"

# ─────────────────────────────────────────────
# DOWNLOAD IMAGE + (OPTIONAL) LABEL JSON
# ─────────────────────────────────────────────
def download_image(camera_point_id: str, dest_image_path: Path, dest_json_path: Path = None):
    key = get_image_key(camera_point_id)

    # Download image
    image_url = f"{IMAGE_PROXY_URL}?key={key}&token={IMAGE_PROXY_TOKEN}"
    try:
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        with open(dest_image_path, "wb") as f:
            f.write(response.content)
        print(f"✅ Image downloaded: {camera_point_id}")
    except Exception as e:
        print(f"❌ Failed to download image for {camera_point_id}: {e}")

    # Optional: download JSON label from same key path but with .json
    if dest_json_path is not None:
        json_key = key.replace(".png", ".json")
        json_url = f"{IMAGE_PROXY_URL}?key={json_key}&token={IMAGE_PROXY_TOKEN}"
        try:
            response = requests.get(json_url, timeout=30)
            response.raise_for_status()
            with open(dest_json_path, "wb") as f:
                f.write(response.content)
            print(f"✅ JSON label downloaded: {camera_point_id}")
        except Exception as e:
            print(f"⚠️  Failed to download JSON for {camera_point_id}: {e}")
