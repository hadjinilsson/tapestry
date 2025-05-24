import argparse
import shutil
from pathlib import Path
from tapestry.label_studio.parse_annotations import parse_annotations
from tapestry.utils.image_fetching import download_image
from collections import Counter

# ───────────── CONFIG ─────────────
DATA_ROOT = Path("data")
MODULE_ROOT = Path("tapestry/object_detection")
counter = Counter()

# ───────────── TRAINING MODE ─────────────

def prepare_training_data(remap_mode: str, remap_configs: list[Path] | None, min_group_size: int | None):
    print("🧠 Preparing data")

    if (DATA_ROOT / "object_detection" / "images").exists():
        shutil.rmtree(DATA_ROOT / "object_detection" / "images")

    if (DATA_ROOT / "object_detection" / "labels").exists():
        shutil.rmtree(DATA_ROOT / "object_detection" / "labels")

    split_map = parse_annotations(
        DATA_ROOT / "object_detection",
        remap_mode=remap_mode,
        remap_configs=remap_configs,
        min_group_size=min_group_size
    )
    for image_stub, split in split_map.items():
        img_path = DATA_ROOT / "object_detection" / "images" / split / f"{image_stub}.png"
        img_path.parent.mkdir(parents=True, exist_ok=True)
        download_image(image_stub, dest_image_path=img_path)
    print("✅ Data ready.")

    label_dir = DATA_ROOT / "object_detection" / "labels" / "train"
    for txt_file in label_dir.glob("*.txt"):
        with open(txt_file) as f:
            for line in f:
                class_id = int(line.strip().split()[0])
                counter[class_id] += 1

    yaml_path = DATA_ROOT / "object_detection" / "data.yaml"
    with open(yaml_path) as f:
        for line in f:
            if line.startswith("names:"):
                names_line = line
                break
        else:
            names_line = "names: []"

    label_names = eval(names_line.split(":", 1)[1].strip())

    print("\n📊 Training class frequencies:")
    for class_id, count in sorted(counter.items()):
        label = label_names[class_id] if class_id < len(label_names) else f"class_{class_id}"
        print(f"  {label:25} → {count}")

# ───────────── ENTRY POINT ─────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--remap",
        choices=["none", "auto", "manual"],
        default="none",
        help="How to remap raw rectanglelabels to training classes"
    )
    parser.add_argument(
        "--remap-configs",
        nargs="+",
        type=str,
        default=[],
        help="One or more remap config names (omit '.json'), e.g. 'cars markings'"
    )
    parser.add_argument(
        "--min-group-size",
        type=int,
        default=None,
        help="Minimum group size for automatic aggregation (used only if --remap auto)"
    )
    args = parser.parse_args()

    remap_paths = [MODULE_ROOT / "remap_configs" / f"{name}.json" for name in args.remap_configs]

    prepare_training_data(
        remap_mode=args.remap,
        remap_configs=remap_paths,
        min_group_size=args.min_group_size
    )

if __name__ == "__main__":
    main()
