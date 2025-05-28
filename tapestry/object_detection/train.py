import argparse
import json
import yaml
from pathlib import Path
from ultralytics import YOLO
import os
from datetime import datetime
from dotenv import load_dotenv
from tapestry.utils.s3 import upload_dir_to_s3

load_dotenv()

# ─────────────── CONFIG ───────────────
DATA_ROOT = Path("data")
S3_BUCKET = os.getenv("BUCKET_NAME_MODELS")

# ─────────────── TRAINING ───────────────
def train(model_type: str, epochs: int, imgsz: int, output_dir: Path) -> tuple[str, str]:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    model = YOLO(model_type)
    data_yaml_path = DATA_ROOT / "object_detection" / "data.yaml"
    data_yaml_str = str(data_yaml_path.resolve())
    results = model.train(
        data=data_yaml_str,
        epochs=epochs,
        imgsz=imgsz,
        project=str(output_dir.parent),
        name=run_id,
    )

    save_dir = Path(results.save_dir)

    with open(data_yaml_path) as f:
        data_cfg = yaml.safe_load(f)
        class_names = data_cfg.get("names", [])

    val_results = model.val(data=data_yaml_str, imgsz=imgsz)

    class_result = getattr(val_results, "class_result", None)
    if class_result is not None and isinstance(class_result, (list, np.ndarray)):
        class_metrics = []
        for class_id, row in enumerate(class_result):
            # Ensure row has the correct shape
            if isinstance(row, (list, np.ndarray)) and len(row) >= 4:
                precision, recall, mAP50, mAP50_95 = row[:4]
                class_metrics.append({
                    "id": class_id,
                    "label": class_names[class_id] if class_id < len(class_names) else f"class_{class_id}",
                    "precision": float(precision),
                    "recall": float(recall),
                    "mAP50": float(mAP50),
                    "mAP50_95": float(mAP50_95),
                })
        with open(save_dir / "class_metrics.json", "w") as f:
            json.dump(class_metrics, f, indent=2)
        print(f"✅ Saved class_metrics.json with {len(class_metrics)} entries")
    else:
        print("⚠️ No valid per-class metrics available; skipping class_metrics.json")

    with open(save_dir / "class_info.json", "w") as f:
        json.dump(
            [{"id": i, "label": name} for i, name in enumerate(class_names)],
            f,
            indent=2
        )

    return run_id, results.save_dir

# ─────────────── CLI ───────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8s.pt", help="YOLOv8 model type")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--no-upload", action="store_true", help="Upload run to S3")
    parser.add_argument("--s3-prefix", default="object_detection")
    args = parser.parse_args()

    output_dir = DATA_ROOT / "object_detection" / "runs"
    run_id, run_dir = train(args.model, args.epochs, args.imgsz, output_dir)
    print(f"🚀 Training complete: Run ID = {run_id}")

    if not args.no_upload:
        print("☁️ Uploading run to S3...")
        upload_dir_to_s3(Path(run_dir), f"{args.s3_prefix}/{run_id}", S3_BUCKET)

if __name__ == "__main__":
    main()
