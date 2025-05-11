import argparse
from pathlib import Path
from ultralytics import YOLO
import os
import boto3
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ─────────────── CONFIG ───────────────
DATA_YAML_PATH = Path("data/data.yaml")
DEFAULT_OUTPUT_DIR = Path("runs/object_detection")
S3_BUCKET = os.getenv("MODELS_BUCKET_NAME")
S3_ENDPOINT = os.getenv("AWS_S3_ENDPOINT")
AWS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY")

# ─────────────── TRAINING ───────────────
def train(model_type: str, epochs: int, imgsz: int, output_dir: Path) -> tuple[str, str]:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    model = YOLO(model_type)
    results = model.train(
        data=str(DATA_YAML_PATH.resolve()),
        epochs=epochs,
        imgsz=imgsz,
        project=str(output_dir.parent),
        name=run_id,
    )
    return run_id, results.save_dir

# ─────────────── UPLOAD TO S3 ───────────────
def upload_to_s3(local_dir: Path, s3_prefix: str):
    s3 = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=AWS_KEY,
        aws_secret_access_key=AWS_SECRET,
    )
    for file in local_dir.rglob("*"):
        if file.is_file():
            s3_key = f"{s3_prefix}/{file.relative_to(local_dir)}"
            s3.upload_file(str(file), S3_BUCKET, s3_key)
            print(f"✅ Uploaded: {s3_key}")

# ─────────────── CLI ───────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8s.pt", help="YOLOv8 model type")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--upload", action="store_true", help="Upload run to S3")
    parser.add_argument("--s3-prefix", default="object_detection_runs")
    args = parser.parse_args()

    run_id, run_dir = train(args.model, args.epochs, args.imgsz, DEFAULT_OUTPUT_DIR)
    print(f"🚀 Training complete: Run ID = {run_id}")

    if args.upload:
        print("☁️ Uploading run to S3...")
        upload_to_s3(Path(run_dir), f"{args.s3_prefix}/{run_id}")

if __name__ == "__main__":
    main()
