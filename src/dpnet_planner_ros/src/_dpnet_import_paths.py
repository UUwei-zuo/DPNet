"""Utility helpers for local imports used by DPNet planner modules."""

from pathlib import Path
import sys


def ensure_dpnet_import_paths(file_path):
    file_dir = Path(file_path).resolve().parent

    source_dir = str(file_dir)
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)

    for candidate_root in (file_dir, *file_dir.parents):
        scripts_dir = candidate_root / "scripts"
        if scripts_dir.is_dir():
            scripts_path = str(scripts_dir)
            if scripts_path not in sys.path:
                sys.path.insert(0, scripts_path)
            break
