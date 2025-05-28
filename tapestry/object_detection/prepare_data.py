import argparse
import shutil
from pathlib import Path
from tapestry.label_studio.parse_annotations import parse_annotations
from tapestry.utils.image_fetching import download_image
from tapestry.utils.config import save_args
from collections import Counter


# ───────────── CONFIG ─────────────
DATA_DIR = Path("data") / "object_detection" / "train"
MODULE_DIR = Path("tapestry/object_detection")
counter = Counter()

DATA_DIR.mkdir(parents=True, exist_ok=True)

# ───────────── TRAINING MODE ─────────────

def prepare_data(remap_mode: str, remap_configs: list[Path] | None, min_group_size: int | None):
    print("🧠 Preparing data")

    if (DATA_DIR/ "images").exists():
        shutil.rmtree(DATA_DIR / "images")

    if (DATA_DIR / "labels").exists():
        shutil.rmtree(DATA_DIR / "labels")

    split_map = parse_annotations(
        DATA_DIR,
        remap_mode=remap_mode,
        remap_configs=remap_configs,
        min_group_size=min_group_size
    )
    for image_stub, split in split_map.items():
        img_path = DATA_DIR / "images" / split / f"{image_stub}.png"
        img_path.parent.mkdir(parents=True, exist_ok=True)
        download_image(image_stub, dest_image_path=img_path)
    print("✅ Data ready.")

    label_dir = DATA_DIR / "labels" / "train"
    for txt_file in label_dir.glob("*.txt"):
        with open(txt_file) as f:
            for line in f:
                class_id = int(line.strip().split()[0])
                counter[class_id] += 1

    yaml_path = DATA_DIR / "data.yaml"
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

    save_args(args, DATA_DIR / "data_config.json")

    remap_paths = [MODULE_DIR / "remap_configs" / f"{name}.json" for name in args.remap_configs]

    prepare_data(
        remap_mode=args.remap,
        remap_configs=remap_paths,
        min_group_size=args.min_group_size
    )

if __name__ == "__main__":
    main()
