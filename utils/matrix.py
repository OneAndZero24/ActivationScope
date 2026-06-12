"""Unified matrix reader — returns a list of dicts with 'python_version' and 'torch_version'."""

import yaml
import json
import argparse
from typing import List, Dict

def load(path: str) -> List[Dict[str, str]]:
    """Read *path* (matrix.yml) and return a list of combos.

    Each combo is ``{'python_version': '3.X', 'torch_version': 'Y.Y.Z'}``.
    """
    with open(path) as f:
        return yaml.safe_load(f)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a matrix YAML file.")
    parser.add_argument("path", type=str, help="Path to the matrix YAML file.")
    args = parser.parse_args()

    combos = load(args.path)
    print(json.dumps({
        'include': [
            {
                'python_version': c['python_version'],
                'torch_version':  c['torch_version'],
                'label': f"py{c['python_version'].replace('.', '')}-torch{c['torch_version'].replace('.', '')}"
            }
            for c in combos
        ]
    }))