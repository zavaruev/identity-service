"""Quality ranking for the face photo library. Pure functions, no I/O beyond a file hash."""

import hashlib
import os
from pathlib import Path
import shutil
import time

import photos as photos_mod

CAP = 24
MIN_IMPROVEMENT = 0.10


def frame_score(det_score: float, bbox) -> float:
    box = list(bbox or [])
    if len(box) < 4:
        return 0.0
    width = max(0.0, float(box[2]) - float(box[0]))
    height = max(0.0, float(box[3]) - float(box[1]))
    return max(0.0, float(det_score)) * width * height


def content_sha1(path: str) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def normalise_entries(manifest: dict) -> list:
    entries = []
    for name, value in (manifest or {}).items():
        if isinstance(value, dict):
            entries.append({
                "name": name,
                "source": str(value.get("source", "")),
                "score": float(value.get("score", 0) or 0),
                "sha1": str(value.get("sha1", "")),
            })
        else:
            entries.append({"name": name, "source": str(value), "score": 0.0, "sha1": ""})
    return entries


def select_top(entries: list, cap: int = CAP) -> tuple:
    ordered = sorted(entries, key=lambda e: -float(e.get("score", 0)))
    return ordered[:cap], ordered[cap:]


def archive_removed(data_file: str, key: str, names: list) -> str:
    """Move dropped frames to a private, timestamped archive before deleting."""
    d = photos_mod.directory(data_file, key)
    o = photos_mod.orig_dir(data_file, key)
    dest = d / "_pruned" / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    dest.mkdir(parents=True, exist_ok=True)
    for name in names:
        safe = photos_mod._safe_name(name)
        for source in (d / safe, o / safe):
            if not source.is_file():
                continue
            target = dest / source.name
            if not target.exists():
                shutil.copy2(source, target)
                os.chmod(target, 0o600)
            source.unlink()
    return str(dest)
