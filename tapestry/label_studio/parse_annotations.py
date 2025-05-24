import os
import json
import requests
import random
from pathlib import Path
from dotenv import load_dotenv
from tapestry.label_studio.aggregate import aggregate_labels
from collections import defaultdict

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

def group_result_items(result):
    grouped = defaultdict(lambda: {"choices": defaultdict(set)})

    for item in result:
        region_id = item["id"]

        if item["type"] == "rectanglelabels":
            grouped[region_id]["rectangle"] = item

        elif item["type"] == "choices":
            name = item["from_name"]
            values = item["value"].get("choices", [])
            if not isinstance(values, list):
                values = [values]
            for v in values:
                grouped[region_id]["choices"][name].add(v)

    merged = []
    for group in grouped.values():
        if "rectangle" not in group:
            continue
        rect_item = group["rectangle"].copy()
        rect_item["value"] = rect_item["value"].copy()
        rect_item["value"]["choices"] = {
            k: sorted(list(v)) for k, v in group["choices"].items()
        }
        merged.append(rect_item)

    return merged

def matches_class(item, definition):
    if item["value"]["rectanglelabels"][0] != definition["label"]:
        return False

    choices = item["value"].get("choices", {})

    for k, v in definition.items():
        if k in {"name", "label"}:
            continue

        actual = choices.get(k)
        if actual is None:
            return False

        # Special strict set match for multi-choice (e.g. 'turn')
        if k == "turn":
            if not isinstance(actual, list) or not isinstance(v, list):
                return False
            if set(actual) != set(v):
                return False
            continue  # Already matched

        # Normalize 1-element lists to scalar
        if isinstance(actual, list) and len(actual) == 1:
            actual = actual[0]

        if isinstance(v, list):
            if actual not in v:
                return False
        else:
            if actual != v:
                return False

    return True

def manual_remap_labels(annotations, remap_paths):
    merged_classes = []
    for path in remap_paths:
        with open(path, "r") as f:
            config = json.load(f)
            merged_classes.extend(config.get("classes", []))

    remapped = []
    for item in annotations:
        merged_results = group_result_items(item.get("annotations", [])[0].get("result", []))
        new_results = []
        for result in merged_results:
            for definition in merged_classes:
                if matches_class(result, definition):
                    new_result = result.copy()
                    new_result["value"] = new_result["value"].copy()
                    new_result["value"]["rectanglelabels"] = [definition["name"]]
                    new_results.append(new_result)
                    break
        if new_results:
            new_item = item.copy()
            new_item["annotations"] = [{"result": new_results}]
            remapped.append(new_item)
    return remapped

def parse_annotations(output_dir: Path, remap_mode: str = "none", remap_configs: list[Path] | None = None, min_group_size: int | None = None):
    annotations = fetch_annotations()
    if remap_mode == "auto":
        annotations = aggregate_labels(annotations, min_group_size=min_group_size)
    elif remap_mode == "manual":
        if not remap_configs:
            raise ValueError("Manual remap mode requires --remap-configs")
        annotations = manual_remap_labels(annotations, remap_configs)

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
    return split_map

if __name__ == "__main__":
    output_dir = Path("data")
    split_map = parse_annotations(output_dir)
    print(f"Split map generated for {len(split_map)} images.")
