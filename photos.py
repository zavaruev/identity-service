"""Photo library for the face training set.

The face service stores only embeddings, so the identity service keeps the
source images it has enrolled. A thumbnail in this library is a face sample
that can be deleted individually by rebuilding the embedding set.
"""
import hashlib
import json
import re
from pathlib import Path

ALLOWED = {".jpg", ".jpeg", ".png", ".webp"}
SAFE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


def root(data_file: str) -> Path:
    return Path(data_file).resolve().parent / "photos"


def directory(data_file: str, key: str) -> Path:
    key = str(key or "")
    if not SAFE_KEY.match(key) or key in (".", ".."):
        raise ValueError(f"unsafe profile key: {key!r}")
    d = root(data_file) / key
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe_name(name: str) -> str:
    name = Path(str(name or "")).name
    if Path(name).suffix.lower() not in ALLOWED:
        raise ValueError("unsupported image type")
    return name


def list_photos(data_file: str, key: str) -> list:
    try:
        d = directory(data_file, key)
    except ValueError:
        return []
    return sorted(p.name for p in d.iterdir()
                  if p.is_file() and p.suffix.lower() in ALLOWED)


def orig_dir(data_file: str, key: str) -> Path:
    d = directory(data_file, key) / "orig"
    d.mkdir(exist_ok=True)
    return d


def read_original(data_file: str, key: str, name: str) -> bytes | None:
    try:
        target = orig_dir(data_file, key) / _safe_name(name)
    except ValueError:
        return None
    return target.read_bytes() if target.is_file() else None


def save_photos(data_file: str, key: str, files: list, thumbs: list | None = None) -> list:
    """Store the ORIGINAL image for face enrollment and a display thumbnail."""
    d = directory(data_file, key)
    saved = []
    for i, (name, content) in enumerate(files):
        name = _safe_name(name)
        target = d / name
        n = 1
        while target.exists():
            target = d / f"{n}_{name}"
            n += 1
        target = d / f"{n}_{name}" if n > 1 else target
        target.write_bytes((thumbs or [None] * len(files))[i] or content)
        (orig_dir(data_file, key) / target.name).write_bytes(content)
        saved.append(target.name)
    return saved


def delete_photo(data_file: str, key: str, name: str) -> bool:
    try:
        target = directory(data_file, key) / _safe_name(name)
    except ValueError:
        return False
    if not target.is_file():
        return False
    target.unlink()
    original = orig_dir(data_file, key) / target.name
    if original.is_file():
        original.unlink()
    provenance = manifest(data_file, key)
    if name in provenance:
        del provenance[name]
        (directory(data_file, key) / "manifest.json").write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2))
    return True


def read_photo(data_file: str, key: str, name: str) -> bytes | None:
    try:
        target = directory(data_file, key) / _safe_name(name)
    except ValueError:
        return None
    if not target.is_file():
        return None
    return target.read_bytes()


def manifest(data_file: str, key: str) -> dict:
    try:
        return json.loads((directory(data_file, key) / "manifest.json").read_text())
    except Exception:
        return {}


def backfill_originals(data_file: str, key: str) -> int:
    """Libraries imported before the crop change have thumbnails but no orig/."""
    added = 0
    for name, source in manifest(data_file, key).items():
        path = Path(source)
        target = orig_dir(data_file, key) / _safe_name(name)
        if not target.exists() and path.is_file():
            target.write_bytes(path.read_bytes())
            added += 1
    return added


def crop_face(data: bytes, bbox, padding: float = 0.2) -> bytes:
    """Crop the detected face out of a full camera frame (Pillow, stdlib-safe)."""
    from io import BytesIO
    from PIL import Image

    image = Image.open(BytesIO(data)).convert("RGB")
    x1, y1, x2, y2 = [int(v) for v in bbox]
    w, h = x2 - x1, y2 - y1
    px, py = w * padding, h * padding
    box = (max(0, int(x1 - px)), max(0, int(y1 - py)),
           min(image.width, int(x2 + px)), min(image.height, int(y2 + py)))
    out = BytesIO()
    image.crop(box).save(out, format="JPEG", quality=88)
    return out.getvalue()


async def import_pending(data_file: str, key: str, source_root: str,
                         cap: int = 24, accept=None) -> list:
    """Seed the library from face-captures' archived/enrollment source frames.

    `accept(bytes)` (awaitable) returns cropped image bytes for a frame that
    really contains THIS person's face, or None to skip it. Without it the
    raw frames are copied (tests only). A marker prevents re-import after a
    restart, so deleted frames stay deleted.
    """
    d = directory(data_file, key)
    if (d / ".seeded").exists() or list_photos(data_file, key):
        return []
    src = Path(source_root)
    candidates = []
    patterns = list(src.glob(f"_archive/*/{key}")) + [
        src / "_enrollment" / key, src / key]
    for directory_path in patterns:
        if directory_path.is_dir():
            candidates.extend(
                p for p in directory_path.iterdir()
                if p.is_file() and p.suffix.lower() in ALLOWED
            )
    candidates.sort(key=lambda p: (p.stat().st_mtime, p.name))
    saved = []
    provenance = {}
    for source in candidates:
        original = source.read_bytes()
        thumb = original
        if accept is not None:
            accepted = await accept(original)
            if accepted is None:
                continue
            if isinstance(accepted, tuple):
                thumb, original = accepted
            else:
                thumb = accepted
        tag = hashlib.sha1(str(source).encode()).hexdigest()[:8]
        name = f"src_{tag}_{source.name}"
        (d / name).write_bytes(thumb)
        (orig_dir(data_file, key) / name).write_bytes(original)
        saved.append(name)
        provenance[name] = str(source)
        if len(saved) >= cap:
            break
    (d / "manifest.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2))
    (d / ".seeded").write_text("")
    return saved
