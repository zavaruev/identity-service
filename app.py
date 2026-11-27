"""Identity service: one profile store over the two existing engines."""
import array
import asyncio
import base64
import hashlib
import io
import json
from pathlib import Path
import math
import os
import secrets
import wave
from html import escape as html_escape

import aiohttp
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response

import curator as curator_mod
import engines as engines_mod
import photos as photos_mod
import profiles as profiles_mod

DATA_FILE = os.getenv("IDENTITY_DATA_FILE", "/app/data/profiles.json")
STATIC_DIR = os.path.dirname(os.path.abspath(__file__))


def _token() -> str:
    return os.getenv("IDENTITY_TOKEN", "")


def _admin() -> tuple:
    return os.getenv("IDENTITY_ADMIN_USER", "identityadmin"), os.getenv("IDENTITY_ADMIN_PASSWORD", "")


def _basic_ok(request: Request) -> bool:
    user, password = _admin()
    if not password:
        return False
    header = request.headers.get("authorization", "")
    if not header.startswith("Basic "):
        return False
    try:
        raw = base64.b64decode(header[6:]).decode()
        got_user, got_password = raw.split(":", 1)
    except Exception:
        return False
    return (secrets.compare_digest(got_user, user)
            and secrets.compare_digest(got_password, password))


CHALLENGE = {"WWW-Authenticate": 'Basic realm="Identity"'}


async def require_token(request: Request):
    # The operator page logs in with Basic; its JS then calls these read
    # endpoints with the SAME credentials. Accept both, fail closed if
    # neither secret is configured.
    if _basic_ok(request):
        return request
    token = _token()
    if not token:
        raise HTTPException(503, "IDENTITY_TOKEN is not configured")
    header = request.headers.get("authorization", "")
    if not header.startswith("Bearer ") or not secrets.compare_digest(header[7:], token):
        raise HTTPException(401, "invalid identity token", headers=CHALLENGE)
    return request


async def require_admin(request: Request):
    if not _admin()[1]:
        raise HTTPException(503, "IDENTITY_ADMIN_PASSWORD is not configured")
    if not _basic_ok(request):
        # Without WWW-Authenticate the browser shows JSON instead of a login
        # prompt — that is why the interface looked broken in the field.
        raise HTTPException(401, "admin credentials required", headers=CHALLENGE)
    return request


DEFAULT_ROOMS = ("livingroom", "kitchen", "entrance1", "camera-corridor1", "balcony")


def allowed_rooms() -> list:
    """The fixed camera list; IDENTITY_ROOMS can narrow it, never widen it by
    accident — the UI and the API validate against the SAME values."""
    raw = os.getenv("IDENTITY_ROOMS", ",".join(DEFAULT_ROOMS))
    return [r.strip() for r in raw.split(",") if r.strip()]


def wav_level(data: bytes) -> tuple:
    """(rms, peak, duration, clipped_fraction) of a mono s16le WAV, stdlib only."""
    with wave.open(io.BytesIO(data), "rb") as w:
        rate = w.getframerate()
        frames = w.getnframes()
        raw = w.readframes(frames)
    samples = array.array("h")
    samples.frombytes(raw[:len(raw) // 2 * 2])
    duration = frames / rate if rate else 0.0
    if not samples:
        return 0.0, 0, duration, 0.0
    peak = max(abs(v) for v in samples)
    rms = math.sqrt(sum(v * v for v in samples) / len(samples))
    clipped = sum(1 for v in samples if abs(v) >= 32700) / len(samples)
    return rms, peak, duration, clipped


def quality(rms: float, peak: int, duration: float, clipped: float = 0.0) -> dict:
    """Human-readable verdict after a sample is added, from measured levels."""
    if duration < 2.0:
        verdict, advice = "too short", "record at least 2 seconds"
    elif rms < 800 or peak < 4000:
        verdict, advice = "too quiet", "move closer to the camera and speak louder"
    elif clipped > 0.01:
        verdict, advice = "clipping", "speak softer or step back to 0.7–1 m"
    else:
        verdict, advice = "good", "this sample can be used"
    return {"verdict": verdict, "advice": advice, "rms": round(rms),
            "peak": peak, "duration": round(duration, 1),
            "clip_pct": round(clipped * 100, 2)}


async def capture_from_gateway(room: str, seconds: float) -> bytes:
    """Fetch a fresh WAV from the gateway's live camera-audio tap."""
    url = os.getenv("GATEWAY_URL", "http://localhost:18792").rstrip("/")
    token = os.getenv("IDENTITY_CAPTURE_TOKEN", "")
    if not token:
        raise RuntimeError("IDENTITY_CAPTURE_TOKEN is not configured")
    timeout = aiohttp.ClientTimeout(total=max(30, int(seconds) + 20))
    async with aiohttp.ClientSession(timeout=timeout) as sess:
        async with sess.post(
            f"{url}/api/identity/capture",
            json={"room": room, "seconds": seconds},
            headers={"Authorization": f"Bearer {token}"},
        ) as r:
            if r.status != 200:
                raise engines_mod.EngineError(r.status, await r.text())
            return await r.read()


def create_app(store: dict | None = None, engines=None) -> FastAPI:
    app = FastAPI(title="identity")
    data_file = os.getenv("IDENTITY_DATA_FILE", DATA_FILE)
    data = store if store is not None else profiles_mod.load_store(data_file)

    def save():
        if store is None:
            profiles_mod.save_store(data_file, data)

    client = engines or engines_mod.Engines(
        os.getenv("SPEAKER_ID_URL", "http://localhost:8001"),
        os.getenv("FACE_API_URL", "http://localhost:8090"),
        os.getenv("API_KEY", ""),
        voice_gallery_dir=os.getenv("VOICE_GALLERY_DIR", ""),
    )

    async def rosters():
        try:
            return await client.rosters()
        except Exception:
            return [], []

    pending_dir = os.getenv("FACE_PENDING_DIR", "").strip()

    @app.on_event("startup")
    async def start_photo_seed():
        # Background so the port opens immediately; a full camera frame has to
        # be recognized before it earns a thumbnail, and InsightFace is slower
        # than a boot probe should wait for.
        app.state.photo_seed_task = asyncio.create_task(seed_face_photos())

    async def seed_face_photos():
        if not pending_dir:
            return
        min_conf = float(os.getenv("IDENTITY_PHOTO_MIN_CONF", "0.35"))
        for key in sorted(data["profiles"]):
            backfilled = photos_mod.backfill_originals(data_file, key)
            if backfilled:
                print(f"[identity] restored {backfilled} originals for {key}",
                      flush=True)
            async def accept(raw, key=key):
                try:
                    faces = await client.recognize(raw, min_conf)
                except Exception as exc:
                    print(f"[identity] face check failed for {key}: {exc!r}", flush=True)
                    return None
                for face in faces:
                    if (str(face.get("name", "")).lower() == key
                            and float(face.get("confidence", 0)) >= min_conf
                            and face.get("bbox")):
                        # Thumbnail = crop; enrollment = ORIGINAL frame, which
                        # is what the face engine's quality gates accepted.
                        return photos_mod.crop_face(raw, face["bbox"]), raw
                return None
            try:
                imported = await photos_mod.import_pending(
                    data_file, key, pending_dir, accept=accept)
                if imported:
                    print(f"[identity] imported {len(imported)} face crops for {key}",
                          flush=True)
            except Exception as exc:
                print(f"[identity] photo import failed for {key}: {exc!r}", flush=True)

    @app.get("/health")
    async def health():
        return {"status": "ok", "admin": bool(_admin()[1]),
                "token": bool(_token()), "profiles": len(data["profiles"])}

    @app.get("/api/profiles", dependencies=[Depends(require_token)])
    async def list_profiles():
        voice, face = await rosters()
        out = []
        for k in sorted(data["profiles"]):
            view = profiles_mod.profile_view(data, k, voice, face)
            view["photos"] = photos_mod.list_photos(data_file, k)
            out.append(view)
        return {"profiles": out}

    @app.post("/api/profiles/{key}", dependencies=[Depends(require_admin)])
    async def upsert(key: str, req: Request):
        body = await req.json()
        fields = {k: body[k] for k in (
            "display_name", "honorific", "telegram_chat_ids", "tts_voice", "notes"
        ) if k in body}
        view = profiles_mod.upsert_profile(data, key, **fields)
        save()
        return profiles_mod.profile_view(data, key)

    @app.delete("/api/profiles/{key}", dependencies=[Depends(require_admin)])
    async def delete_profile(key: str):
        """Remove the profile record only. Engine galleries are left alone:
        speaker-id has no delete API, and removing someone's voice/face data
        is a destructive act that needs an explicit instruction."""
        key = str(key).strip().lower()
        if key not in data["profiles"]:
            raise HTTPException(404, f"unknown profile: {key}")
        del data["profiles"][key]
        save()
        return {"deleted": key,
                "detail": "profiles.json record only; engine samples were not deleted"}

    async def _enroll(key: str, files: list[UploadFile], voice: bool):
        if profiles_mod.profile_view(data, key) is None:
            raise HTTPException(404, f"unknown profile: {key}")
        payload = [(f.filename or "sample", await f.read()) for f in files]
        if not payload:
            raise HTTPException(400, "no files uploaded")
        q = None
        if voice:
            levels = []
            for _name, content in payload:
                try:
                    levels.append(wav_level(content))
                except Exception:
                    continue    # ogg/mp3 uploads: no WAV header to measure
            if levels:
                q = quality(
                    sum(v[0] for v in levels) / len(levels),
                    max(v[1] for v in levels),
                    sum(v[2] for v in levels),
                    max(v[3] for v in levels),
                )
            else:
                q = {"verdict": "not measured",
                     "advice": "not a WAV file — level and clipping unchecked",
                     "rms": None, "peak": None, "duration": None, "clip_pct": None}
        try:
            if voice:
                return {**await client.enroll_voice(key, payload), "quality": q}
            return await client.enroll_face(key, payload)
        except engines_mod.EngineError as exc:
            return JSONResponse(status_code=502, content={
                "status": "error",
                "detail": f"engine HTTP {exc.status}: {exc.detail}",
            })

    @app.post("/api/profiles/{key}/voice/enroll", dependencies=[Depends(require_admin)])
    async def enroll_voice(key: str, files: list[UploadFile] = File(...)):
        return await _enroll(key, files, True)

    @app.post("/api/profiles/{key}/face/enroll", dependencies=[Depends(require_admin)])
    async def enroll_face(key: str, files: list[UploadFile] = File(...)):
        if profiles_mod.profile_view(data, key) is None:
            raise HTTPException(404, f"unknown profile: {key}")
        payload = [(f.filename or "photo", await f.read()) for f in files]
        if not payload:
            raise HTTPException(400, "no files uploaded")
        saved = []
        try:
            min_conf = float(os.getenv("IDENTITY_PHOTO_MIN_CONF", "0.35"))
            thumbs = []
            for _name, content in payload:
                thumb = content
                try:
                    faces = await client.recognize(content, min_conf)
                    for face in faces:
                        if (str(face.get("name", "")).lower() == key
                                and face.get("bbox")):
                            thumb = photos_mod.crop_face(content, face["bbox"])
                            break
                except Exception:
                    pass
                thumbs.append(thumb)
            saved = photos_mod.save_photos(data_file, key, payload, thumbs)
            # The engine gets ORIGINALS: a tight crop can fail its blur/size gate.
            to_send = list(zip(saved, [content for _name, content in payload]))
            result = await client.enroll_face(key, to_send, replace=False)
            accepted = {str(a.get("file")) for a in result.get("accepted", [])}
            for name in saved:
                if name not in accepted:
                    photos_mod.delete_photo(data_file, key, name)
            return {**result, "photos": photos_mod.list_photos(data_file, key)}
        except engines_mod.EngineError as exc:
            for name in saved:
                photos_mod.delete_photo(data_file, key, name)
            return JSONResponse(status_code=502, content={
                "status": "error",
                "detail": f"engine HTTP {exc.status}: {exc.detail}"})

    @app.post("/api/profiles/{key}/face/curate", dependencies=[Depends(require_admin)])
    async def curate_face(key: str, file: UploadFile = File(...)):
        """Keep the best CAP frames: add a better arrival frame, rebuild the
        engine from the best originals, archive and audit what was removed."""
        if profiles_mod.profile_view(data, key) is None:
            raise HTTPException(404, f"unknown profile: {key}")
        content = await file.read()
        min_conf = float(os.getenv("IDENTITY_PHOTO_MIN_CONF", "0.35"))
        faces = await client.recognize(content, min_conf)
        face = next((f for f in faces
                     if str(f.get("name", "")).lower() == key and f.get("bbox")
                     and float(f.get("confidence", 0)) >= 0.6), None)
        if face is None:
            raise HTTPException(400, f"frame is not a confident {key} face")

        score = curator_mod.frame_score(face.get("det_score", 0), face["bbox"])
        sha1 = hashlib.sha1(content).hexdigest()
        d = photos_mod.directory(data_file, key)
        o = photos_mod.orig_dir(data_file, key)
        old_manifest = photos_mod.manifest(data_file, key)
        entries = curator_mod.normalise_entries(old_manifest)
        if any(e["sha1"] == sha1 for e in entries):
            return {"status": "duplicate", "kept": len(entries), "removed": [],
                    "engine_samples": len(entries), "score": score}
        if len(entries) >= curator_mod.CAP and entries:
            worst = min(float(e.get("score", 0) or 0) for e in entries)
            if score < worst * (1 + curator_mod.MIN_IMPROVEMENT):
                return {"status": "unchanged", "kept": len(entries), "removed": [],
                        "engine_samples": len(entries), "score": score}

        saved = []
        try:
            thumb = photos_mod.crop_face(content, face["bbox"])
            saved = photos_mod.save_photos(
                data_file, key, [(file.filename or "frame.jpg", content)], [thumb])
            new_name = saved[0]
            new_entry = {"name": new_name, "source": str(o / new_name),
                         "score": score, "sha1": sha1}
            keep, drop = curator_mod.select_top(entries + [new_entry], curator_mod.CAP)
            if len(entries) < curator_mod.CAP:
                # A partial gallery must not rebuild the engine from those few
                # files — it would throw away embeddings learned from pending/.
                # Append the new original until the gallery is full.
                result = await client.enroll_face(key, [(new_name, content)], replace=False)
                accepted = {str(a.get("file")) for a in result.get("accepted", [])} \
                    if result.get("accepted") is not None else {new_name}
                final_keep = entries + ([new_entry] if new_name in accepted else [])
                removed = [e["name"] for e in drop]
                out = {e["name"]: {"source": e["source"], "score": e["score"], "sha1": e["sha1"]}
                       for e in final_keep}
                (d / "manifest.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                                 encoding="utf-8")
                archived = curator_mod.archive_removed(data_file, key, removed)
                with (d / "curate_audit.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"score": score, "kept": len(final_keep),
                                             "removed": removed, "archived": archived,
                                             "mode": "append"}) + "\n")
                return {"status": "ok", "kept": len(final_keep), "removed": removed,
                        "engine_samples": int(result.get("samples", len(final_keep))),
                        "score": score, "archived": archived, "mode": "append"}
            originals = []
            for entry in keep:
                source = o / entry["name"]
                originals.append((entry["name"],
                                  source.read_bytes() if source.is_file()
                                  else Path(entry["source"]).read_bytes()))
            result = await client.enroll_face(key, originals, replace=True)
            listed = result.get("accepted")
            accepted = ({str(a.get("file")) for a in listed} if listed is not None
                        else {name for name, _ in originals})
            final_keep = [e for e in keep if e["name"] in accepted]
            rejected = [e for e in keep if e["name"] not in accepted]
            removed = [e["name"] for e in drop + rejected]
            out = {e["name"]: {"source": e["source"], "score": e["score"], "sha1": e["sha1"]}
                   for e in final_keep}
            (d / "manifest.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
            archived = curator_mod.archive_removed(data_file, key, removed)
            with (d / "curate_audit.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"score": score, "kept": len(final_keep),
                                         "removed": removed, "archived": archived}) + "\n")
            return {"status": "ok", "kept": len(final_keep), "removed": removed,
                    "engine_samples": int(result.get("samples", len(final_keep))),
                    "score": score, "archived": archived}
        except engines_mod.EngineError as exc:
            for name in saved:
                photos_mod.delete_photo(data_file, key, name)
            (d / "manifest.json").write_text(json.dumps(old_manifest, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
            return JSONResponse(status_code=502, content={
                "status": "error", "detail": f"engine HTTP {exc.status}: {exc.detail}"})

    @app.delete("/api/profiles/{key}/face/photos/{name}",
                dependencies=[Depends(require_admin)])
    async def delete_face_photo(key: str, name: str):
        """Remove ONE training photo: rebuild the embedding from the rest."""
        if profiles_mod.profile_view(data, key) is None:
            raise HTTPException(404, f"unknown profile: {key}")
        before = photos_mod.list_photos(data_file, key)
        if name not in before:
            raise HTTPException(404, f"unknown photo: {name}")
        rest = [n for n in before if n != name]
        # Rebuild from ORIGINAL frames, not the display crops (quality gates).
        remaining_files = [
            (n, photos_mod.read_original(data_file, key, n)
             or photos_mod.read_photo(data_file, key, n)) for n in rest]
        try:
            # Engine first: if the rebuild fails, the local photo stays.
            result = (await client.enroll_face(key, remaining_files, replace=True)
                      if rest else await client.delete_face(key))
        except engines_mod.EngineError as exc:
            return JSONResponse(status_code=502, content={
                "status": "error",
                "detail": f"engine HTTP {exc.status}: {exc.detail}"})
        photos_mod.delete_photo(data_file, key, name)
        return {"deleted": name, "photos": photos_mod.list_photos(data_file, key), **result}

    @app.get("/api/photos/{key}/{name}", dependencies=[Depends(require_token)])
    async def get_photo(key: str, name: str):
        content = photos_mod.read_photo(data_file, key, name)
        if content is None:
            raise HTTPException(404, "photo not found")
        suffix = os.path.splitext(name)[1].lower()
        media = {".png": "image/png", ".webp": "image/webp"}.get(suffix, "image/jpeg")
        return Response(content=content, media_type=media)

    @app.post("/api/identify", dependencies=[Depends(require_token)])
    async def identify(file: UploadFile = File(...)):
        try:
            uid, conf = await client.identify(await file.read())
        except engines_mod.EngineError as exc:
            return JSONResponse(status_code=502, content={"detail": str(exc)})
        voice, face = await rosters()
        view = profiles_mod.profile_view(data, uid, voice, face)
        return {
            "person": uid,
            "display_name": (view or {}).get("display_name", uid),
            "honorific": (view or {}).get("honorific", ""),
            "confidence": conf,
        }

    @app.post("/api/recognize", dependencies=[Depends(require_token)])
    async def recognize(file: UploadFile = File(...), threshold: float = 0.35):
        try:
            faces = await client.recognize(await file.read(), threshold)
        except engines_mod.EngineError as exc:
            return JSONResponse(status_code=502, content={"detail": str(exc)})
        voice, face = await rosters()
        out = []
        for item in faces:
            name = str(item.get("name", "unknown"))
            view = profiles_mod.profile_view(data, name, voice, face) or {}
            out.append({**item, "name": name,
                        "display_name": view.get("display_name", name),
                        "honorific": view.get("honorific", "")})
        return {"faces": out}

    @app.post("/api/record", dependencies=[Depends(require_admin)])
    async def record(room: str = Form(...), profile: str = Form(...),
                     seconds: float = Form(6)):
        """Record from the named camera and replace this person's voice sample."""
        if room not in allowed_rooms():
            raise HTTPException(400, f"unknown room {room!r}; choose one of {allowed_rooms()}")
        if profiles_mod.profile_view(data, profile) is None:
            raise HTTPException(404, f"unknown profile: {profile}")
        try:
            wav = await capture_from_gateway(room, seconds)
        except Exception as exc:
            detail = f"camera capture failed: {getattr(exc, 'status', type(exc).__name__)}: {exc}"
            return JSONResponse(status_code=502, content={"status": "error", "detail": detail})
        if len(wav) < 44:
            return JSONResponse(status_code=502, content={
                "status": "error", "detail": "camera returned no audio; nothing enrolled"})
        rms, peak, duration, clipped = wav_level(wav)
        # Measured 25.11.2026: a quiet living-room capture read rms 414 /
        # peak 2366; field speech peaks ran 4725..13139. The bar sits above
        # the measured room floor and below the weakest observed voice, and
        # enrollment is destructive (voice /enroll REPLACES the centroid), so
        # the error is deliberately on the side of refusing a quiet sample.
        min_rms = float(os.getenv("IDENTITY_CAPTURE_MIN_RMS", "800"))
        min_peak = float(os.getenv("IDENTITY_CAPTURE_MIN_PEAK", "4000"))
        if rms < min_rms or peak < min_peak or duration < 2.0:
            return JSONResponse(status_code=502, content={
                "status": "error",
                "detail": (f"no speech in the capture (rms={rms:.0f} peak={peak} "
                           f"dur={duration:.1f}s); nothing enrolled")})
        try:
            result = await client.enroll_voice(profile, [("sample.wav", wav)])
        except engines_mod.EngineError as exc:
            return JSONResponse(status_code=502, content={
                "status": "error",
                "detail": f"voice enrollment failed: HTTP {exc.status}: {exc.detail}"})
        return {
            "status": "ok", "room": room, "profile": profile,
            "seconds": seconds, "bytes": len(wav), **result,
            "quality": quality(rms, peak, duration, clipped),
            "detail": "voice sample replaced completely (append is not supported)",
        }

    @app.get("/", response_class=HTMLResponse, dependencies=[Depends(require_admin)])
    async def ui():
        with open(os.path.join(STATIC_DIR, "static", "index.html")) as f:
            html = f.read()
        options = "".join(
            f'<option value="{html_escape(r)}">{html_escape(r)}</option>'
            for r in allowed_rooms()
        )
        # One source of truth: the same allowed_rooms() the API validates.
        page = html.replace("<!--ROOM_OPTIONS-->", options)
        # The operator must see the NEW page immediately after a deploy; a
        # cached Russian/old layout cost a whole round of field confusion.
        return Response(content=page, media_type="text/html",
                        headers={"Cache-Control": "no-store"})

    return app


app = create_app()
