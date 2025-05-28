import json
from argparse import Namespace
from pathlib import Path


def save_args(args: Namespace, output_path: Path):
    """Save argparse.Namespace to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(vars(args), f, indent=2)
