#!/usr/bin/env python3

import json
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    spawn_json = repo_root / "src/dpnet_bridge_overlay/config/DPNet_spawn.json"
    scan_pattern_file = repo_root / "examples/Doppler_ScanPatterns.yaml"

    if not spawn_json.is_file():
        raise FileNotFoundError(f"Missing spawn config: {spawn_json}")
    if not scan_pattern_file.is_file():
        raise FileNotFoundError(f"Missing scan pattern file: {scan_pattern_file}")

    with spawn_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    updated = False
    for obj in data.get("objects", []):
        for sensor in obj.get("sensors", []):
            if sensor.get("type") == "sensor.lidar.doppler":
                sensor["pattern_file"] = str(scan_pattern_file)
                updated = True

    if not updated:
        raise RuntimeError("No sensor.lidar.doppler found in DPNet_spawn.json")

    with spawn_json.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
        f.write("\n")

    print(f"[DPNet] Updated pattern_file: {scan_pattern_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
