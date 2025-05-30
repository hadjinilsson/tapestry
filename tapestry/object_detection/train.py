import argparse
import json
import yaml
import shutil
from pathlib import Path
from ultralytics import YOLO
import os
from datetime import datetime
from dotenv import load_dotenv
from tapestry.utils.s3 import upload_dir_to_s3
from tapestry.utils.config import save_args


load_dotenv()

# ─────────────── CONFIG ───────────────
DATA_DIR = Path("data") / "object_detection" / "train"
S3_BUCKET = os.getenv("BUCKET_NAME_MODELS")

# ─────────────── TRAINING ───────────────
def train(model_type: str, epochs: int, imgsz: int, output_dir: Path) -> tuple[str, Path]:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    model = YOLO(model_type)
    data_yaml_path = DATA_DIR / "data.yaml"
    data_yaml_str = str(data_yaml_path.resolve())
    results = model.train(
        data=data_yaml_str,
        epochs=epochs,
        imgsz=imgsz,
        project=str(output_dir),
        name=run_id,
    )

    save_dir = Path(results.save_dir)

    with open(data_yaml_path) as f:
        data_cfg = yaml.safe_load(f)
        class_names = data_cfg.get("names", [])

    with open(save_dir / "class_info.json", "w") as f:
        json.dump(
            [{"id": i, "label": name} for i, name in enumerate(class_names)],
            f,
            indent=2
        )

    return run_id, save_dir

# ─────────────── CLI ───────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8s.pt", help="YOLOv8 model type")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--no-upload", action="store_true", help="Upload run to S3")
    parser.add_argument("--s3-prefix", default="object_detection")
    args = parser.parse_args()

    output_dir = DATA_DIR / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id, run_dir = train(args.model, args.epochs, args.imgsz, output_dir)
    print(f"🚀 Training complete: Run ID = {run_id}")

    shutil.copy(DATA_DIR / "data_config.json", run_dir)
    save_args(args, run_dir / "train_config.json")

    if not args.no_upload:
        print("☁️ Uploading run to S3...")
        upload_dir_to_s3(Path(run_dir), f"{args.s3_prefix}/{run_id}", S3_BUCKET)

if __name__ == "__main__":
    main()
