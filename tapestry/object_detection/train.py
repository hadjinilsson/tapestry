import argparse
from pathlib import Path
from ultralytics import YOLO
import os
from datetime import datetime
from dotenv import load_dotenv
from tapestry.utils.s3 import upload_dir_to_s3

load_dotenv()

# ─────────────── CONFIG ───────────────
DATA_YAML_PATH = Path("data/data.yaml")
DEFAULT_OUTPUT_DIR = Path("runs/object_detection")
S3_BUCKET = os.getenv("BUCKET_NAME_MODELS")

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

# ─────────────── CLI ───────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8s.pt", help="YOLOv8 model type")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--upload", action="store_true", help="Upload run to S3")
    parser.add_argument("--s3-prefix", default="object_detection")
    args = parser.parse_args()

    run_id, run_dir = train(args.model, args.epochs, args.imgsz, DEFAULT_OUTPUT_DIR)
    print(f"🚀 Training complete: Run ID = {run_id}")

    if args.upload:
        print("☁️ Uploading run to S3...")
        upload_dir_to_s3(Path(run_dir), f"{args.s3_prefix}/{run_id}", S3_BUCKET)

if __name__ == "__main__":
    main()
