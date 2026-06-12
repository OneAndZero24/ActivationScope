#!/usr/bin/env python3
"""Generate pyproject.toml from the template and matrix.yml.

Usage:
    python utils/generate-pyproject.py
    python utils/generate-pyproject.py --matrix /path/to/matrix.yml
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from matrix import load  # noqa: E402


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEMPLATE = os.path.join(ROOT, "pyproject.toml.template")
OUTPUT = os.path.join(ROOT, "pyproject.toml")


from typing import List, Dict

def generate(template: str, combos: List[Dict[str, str]]) -> str:
    """Fill the template with values derived from *combos*."""
    # Minimum Python and Torch across the matrix.
    pythons = sorted(
        {c["python_version"] for c in combos},
        key=lambda v: tuple(int(x) for x in v.split(".")),
    )
    torches = sorted(
        {c["torch_version"] for c in combos},
        key=lambda v: tuple(int(x) for x in v.split(".")),
    )

    python_min = pythons[0]  # oldest Python is the floor.
    torch_floor = torches[0] # oldest Torch is the dependency floor.

    classifiers = ",\n".join(
        f'    "Programming Language :: Python :: {py}"' for py in pythons
    )

    text = template
    text = text.replace("__PYTHON_MIN__", python_min)
    text = text.replace("__TORCH_VERSION__", torch_floor)
    text = text.replace(
        "__VERSION__",
        os.environ.get("ASCOPE_VERSION", "0.1.0"),
    )
    text = text.replace("__PYTHON_CLASSIFIERS__", classifiers + ",\n")

    return text


def main() -> None:
    p = argparse.ArgumentParser(description="Generate pyproject.toml from matrix.yml.")
    p.add_argument(
        "--matrix",
        default=os.path.join(ROOT, "matrix.yml"),
        help="Path to the matrix YAML file.",
    )
    args = p.parse_args()

    combos = load(args.matrix)
    with open(TEMPLATE) as f:
        template = f.read()

    content = generate(template, combos)
    with open(OUTPUT, "w") as f:
        f.write(content)

    print(f"Generated {OUTPUT}")


if __name__ == "__main__":
    main()
