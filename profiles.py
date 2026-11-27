"""One atomic JSON profile store for the identity service.

Sample presence is NOT stored: it is derived from the galleries at read time,
so a file written by an engine can never drift from what the UI shows.
"""
import json
import os
import tempfile
from datetime import datetime, timezone


class StoreError(Exception):
    """The store file exists but cannot be trusted."""


def empty_store() -> dict:
    return {"version": 1, "profiles": {}}


def load_store(path: str) -> dict:
    if not os.path.exists(path):
        return empty_store()
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as exc:
        raise StoreError(f"cannot parse {path}: {exc!r}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("profiles"), dict):
        raise StoreError(f"bad profile store shape in {path}")
    return data


def save_store(path: str, store: dict) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix=".profiles-", dir=directory)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(store, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def seed_store(names: dict, honorifics: dict, chat_ids: dict,
               voice_names: list, face_names: list) -> dict:
    """Seed from the Phase A config. Gallery names are accepted only to keep
    the interface stable; a gallery user without an honorific is not a person."""
    store = empty_store()
    reverse = {}
    for chat, person in (chat_ids or {}).items():
        reverse.setdefault(str(person).lower(), []).append(str(chat))
    now = datetime.now(timezone.utc).isoformat()
    for key, honorific in (honorifics or {}).items():
        key = str(key).lower()
        if key.startswith("test"):
            continue
        store["profiles"][key] = {
            "display_name": str((names or {}).get(key, key)),
            "honorific": str(honorific or ""),
            "telegram_chat_ids": reverse.get(key, []),
            "tts_voice": "",
            "notes": "",
            "updated": now,
        }
    return store


def upsert_profile(store: dict, key: str, **fields) -> dict:
    key = str(key).strip().lower()
    profile = dict(store["profiles"].get(key, {}))
    profile.update(fields)
    profile["updated"] = datetime.now(timezone.utc).isoformat()
    store["profiles"][key] = profile
    return profile


def profile_view(store: dict, key: str, voice_names: list = (),
                 face_names: list = ()) -> dict | None:
    profile = store["profiles"].get(str(key).lower())
    if profile is None:
        return None
    view = dict(profile)
    view["key"] = str(key).lower()
    view["voice_enrolled"] = str(key).lower() in set(voice_names or ())
    view["face_enrolled"] = str(key).lower() in set(face_names or ())
    return view


def chat_map(store: dict) -> dict:
    out = {}
    for person, profile in store["profiles"].items():
        for chat in profile.get("telegram_chat_ids") or []:
            out[str(chat)] = person
    return out
