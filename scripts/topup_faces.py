#!/usr/bin/env python3
"""One-time top-up of every identity gallery to the best 24 frames.

Dry-run by default; --apply performs writes through identity's curate endpoint.
"""
import asyncio
import base64
import json
import os
from pathlib import Path
import sys

import aiohttp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import curator  # noqa: E402
import profiles as profiles_mod  # noqa: E402

CAP = curator.CAP
MIN_CONF = 0.6
DATA_FILE = os.getenv("IDENTITY_DATA_FILE", str(ROOT / "data" / "profiles.json"))
PENDING_DIR = os.getenv("FACE_PENDING_DIR", str(ROOT / "face_pending"))
CAPTURES_DIR = os.getenv("FACE_CAPTURES_DIR", str(ROOT / "captures"))
IDENTITY_URL = os.getenv("IDENTITY_URL", "http://localhost:18794").rstrip("/")
FACE_API = os.getenv("FACE_API_URL", "http://localhost:8090")
ADMIN_USER = os.getenv("IDENTITY_ADMIN_USER", "")
ADMIN_PASSWORD = os.getenv("IDENTITY_ADMIN_PASSWORD", "")


def raw_candidates(captures_dir: str) -> list:
    """Full frames the capture daemon marked quality_ok, newest first."""
    out = []
    for jsonl in Path(captures_dir).rglob("*_frames.jsonl"):
        for line in jsonl.read_text(errors="replace").splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if record.get("quality_ok") and record.get("path"):
                out.append(record["path"])
    return sorted(set(out), reverse=True)


def archived_candidates(pending_dir: str, key: str) -> list:
    root = Path(pending_dir)
    found = []
    for pattern in ("_archive/*/" + key + "/*.jpg", "_enrollment/" + key + "/*.jpg", key + "/*.jpg"):
        found.extend(str(p) for p in root.glob(pattern) if p.is_file())
    return sorted(set(found), reverse=True)


def score_one(path: str, recognizer) -> dict | None:
    """Only keep a frame recognised as this person with conf >= 0.6."""
    record = recognizer(path)
    if not record:
        return None
    if str(record.get("name", "")).lower() != record.get("wanted", "").lower():
        return None
    if float(record.get("confidence", 0)) < MIN_CONF:
        return None
    score = curator.frame_score(record.get("det_score", 0), record.get("bbox"))
    if score <= 0:
        return None
    return {"path": path, "score": score, "sha1": curator.content_sha1(path)}


def engine_recognizer(key: str, face_api: str):
    """Real /recognize call, matching only this person's frames."""
    import asyncio
    import aiohttp

    def call(path):
        async def go():
            with open(path, "rb") as handle:
                image = handle.read()
            form = aiohttp.FormData()
            form.add_field("file", image, filename=os.path.basename(path),
                           content_type="image/jpeg")
            form.add_field("threshold", "0.35")
            async with aiohttp.ClientSession() as session:
                async with session.post(
                        face_api.rstrip("/") + "/recognize", data=form,
                        timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status != 200:
                        return None
                    faces = (await response.json()).get("faces", [])
            face = next((f for f in faces
                         if str(f.get("name", "")).lower() == key.lower()), None)
            return {**face, "wanted": key} if face else None
        return asyncio.run(go())

    return call


def topup_one(key: str, existing: list, raw: list, *, poster=None,
              cap: int = CAP, dry_run: bool = False) -> dict:
    """Post the best candidates until the gallery reaches cap."""
    have = len(existing)
    worst = min([e.get("score", 0) for e in existing] or [0])
    margin = 1 + curator.MIN_IMPROVEMENT
    eligible = [r for r in raw if float(r.get("score", 0)) > worst * margin]
    eligible.sort(key=lambda r: -float(r.get("score", 0)))
    needed = max(0, cap - have)
    picked = eligible[:needed]
    result = {"key": key, "gallery": have, "candidates": len(eligible),
              "picked": len(picked), "dry_run": dry_run, "posted": []}
    if dry_run or poster is None:
        return result
    for candidate in picked:
        response = poster(key, candidate["path"], None)
        if response and response.get("status") in ("ok", "duplicate"):
            result["gallery"] = min(cap, result["gallery"] + 1)
            result["posted"].append(candidate["path"])
            if result["gallery"] >= cap:
                break
    return result


async def post_curate(session, key: str, path: str) -> dict:
    if not ADMIN_USER or not ADMIN_PASSWORD:
        raise RuntimeError("IDENTITY_ADMIN_USER/PASSWORD are required for --apply")
    with open(path, "rb") as handle:
        image = handle.read()
    form = aiohttp.FormData()
    form.add_field("file", image, filename=os.path.basename(path), content_type="image/jpeg")
    auth = aiohttp.BasicAuth(ADMIN_USER, ADMIN_PASSWORD)
    url = f"{IDENTITY_URL}/api/profiles/{key}/face/curate"
    async with session.post(url, data=form, auth=auth,
                            timeout=aiohttp.ClientTimeout(total=60)) as response:
        return await response.json()


def main() -> int:
    apply = "--apply" in sys.argv
    store = profiles_mod.load_store(DATA_FILE)
    keys = sorted(store.get("profiles", {}))
    print(f"topup: {len(keys)} profiles, mode={'apply' if apply else 'dry-run'}")
    for key in keys:
        manifest = json.loads((Path(DATA_FILE).parent / "photos" / key / "manifest.json").read_text()
                              if (Path(DATA_FILE).parent / "photos" / key / "manifest.json").is_file() else "{}")
        existing = curator.normalise_entries(manifest)
        raw = []
        for path in archived_candidates(PENDING_DIR, key) + raw_candidates(CAPTURES_DIR):
            if any(e.get("sha1") == curator.content_sha1(path) for e in existing):
                continue
            candidate = score_one(path, engine_recognizer(key, FACE_API))
            if candidate:
                raw.append(candidate)
        poster = None
        if apply:
            import asyncio
            def post_sync(name, path, _gallery):
                async def call():
                    async with aiohttp.ClientSession() as session:
                        return await post_curate(session, name, path)
                return asyncio.run(call())
            poster = post_sync
        result = topup_one(key, existing, raw, poster=poster, dry_run=not apply)
        print(f"  {key}: gallery={result['gallery']} candidates={result['candidates']} "
              f"picked={result['picked']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
