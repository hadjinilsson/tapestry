import os
import requests
import random
from pathlib import Path
from dotenv import load_dotenv
from tapestry.label_studio.aggregate import aggregate_labels

load_dotenv()

LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL")
LABEL_STUDIO_PROJECT_ID = os.getenv("LABEL_STUDIO_PROJECT_ID", "3")
LABEL_STUDIO_API_TOKEN = os.getenv("LABEL_STUDIO_API_TOKEN")

TRAIN_RATIO = 0.8


def fetch_annotations():
    url = f"{LABEL_STUDIO_URL}/api/projects/{LABEL_STUDIO_PROJECT_ID}/export?exportType=JSON"
    headers = {"Authorization": f"Token {LABEL_STUDIO_API_TOKEN}"}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch annotations: {response.text}")
    return response.json()


def parse_annotations(output_dir: Path, min_group_size: int | None = None):
    annotations = fetch_annotations()
    if min_group_size is not None:
        annotations = aggregate_labels(annotations, min_group_size=min_group_size)

    label_map = {}
    label_id_counter = 0
    split_map = {}

    all_image_stubs = [
        Path(item["data"]["image"].split("key=")[-1].split("&")[0]).stem
        for item in annotations if "image" in item["data"]
    ]
    random.shuffle(all_image_stubs)
    split_index = int(len(all_image_stubs) * TRAIN_RATIO)

    for i, stub in enumerate(all_image_stubs):
        split_map[stub] = "train" if i < split_index else "val"

    for item in annotations:
        image_url = item["data"].get("image")
        if not image_url:
            continue

        key = image_url.split("key=")[-1].split("&")[0]
        filename = os.path.basename(key)
        image_stub = Path(filename).stem
        split = split_map[image_stub]

        label_dir = output_dir / "labels" / split
        label_dir.mkdir(parents=True, exist_ok=True)
        label_path = label_dir / f"{image_stub}.txt"

        lines = []
        for result in item.get("annotations", [])[0].get("result", []):
            if result["type"] != "rectanglelabels":
                continue
            label = result["value"]["rectanglelabels"][0]
            if label not in label_map:
                label_map[label] = label_id_counter
                label_id_counter += 1

            x = result["value"]["x"] / 100
            y = result["value"]["y"] / 100
            w = result["value"]["width"] / 100
            h = result["value"]["height"] / 100
            cx = x + w / 2
            cy = y + h / 2

            lines.append(f"{label_map[label]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        if lines:
            with open(label_path, "w") as f:
                f.write("\n".join(lines))

    label_names = [None] * len(label_map)
    for name, idx in label_map.items():
        label_names[idx] = name

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"train: {output_dir.resolve()}/images/train\n")
        f.write(f"val: {output_dir.resolve()}/images/val\n\n")
        f.write(f"nc: {len(label_names)}\n")
        f.write("names: [")
        f.write(", ".join([f"'{name}'" for name in label_names]))
        f.write("]\n")

    print("✅ Annotations parsed and data.yaml written.")
    return split_map  # Dict[str, str]: image_stub → 'train' or 'val'


if __name__ == "__main__":
    output_dir = Path("data")
    split_map = parse_annotations(output_dir)
    print(f"Split map generated for {len(split_map)} images.")