"""Identity API: auth fail-closed, identity joins, honest errors, usable UI."""
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as identity_app  # noqa: E402
import engines  # noqa: E402


class FakeEngines:
    def __init__(self, uid="person1", confidence=0.9, error=None):
        self.uid = uid
        self.confidence = confidence
        self.error = error

    async def identify(self, audio):
        if self.error:
            raise self.error
        return self.uid, self.confidence

    async def recognize(self, image, threshold):
        return [{"name": self.uid, "confidence": 0.8}]

    async def enroll_voice(self, key, files):
        if self.error:
            raise self.error
        return {"status": "ok", "user_id": key}

    async def enroll_face(self, key, files):
        return {"status": "ok", "name": key}

    async def rosters(self):
        return ["person1"], ["person1", "person2", "person3"]


def make_client(monkeypatch, fake=None, token="T", admin="pw"):
    monkeypatch.setenv("IDENTITY_TOKEN", token)
    monkeypatch.setenv("IDENTITY_ADMIN_USER", "admin")
    monkeypatch.setenv("IDENTITY_ADMIN_PASSWORD", admin)
    import profiles
    store = profiles.seed_store(
        {"person1": "Person 1"},
        {"person1": "сэр", "person2": "мэм", "person3": "мэм"},
        {"100001": "person1", "100002": "person2", "100003": "person3"},
        [], [],
    )
    app = identity_app.create_app(store=store, engines=fake or FakeEngines())
    return TestClient(app)


def test_ui_fetch_reads_profiles_with_basic(monkeypatch):
    """The page has Basic credentials from its login; its JS fetch to
    /api/profiles must work with the SAME credentials, not a Bearer token."""
    client = make_client(monkeypatch)
    assert client.get("/api/profiles", auth=("admin", "pw")).status_code == 200
    assert client.get("/api/profiles",
                      headers={"Authorization": "Bearer T"}).status_code == 200
    assert client.get("/api/profiles", auth=("admin", "bad")).status_code == 401


def test_voice_upload_reports_sample_quality(monkeypatch):
    import array
    import io
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(array.array("h", [4000] * 32000).tobytes())
    client = make_client(monkeypatch)
    r = client.post("/api/profiles/person1/voice/enroll", auth=("admin", "pw"),
                    files={"files": ("s.wav", buf.getvalue(), "audio/wav")})
    assert r.status_code == 200
    assert r.json()["quality"]["verdict"] == "good"


def test_delete_profile_removes_only_the_metadata(monkeypatch):
    client = make_client(monkeypatch)
    assert client.delete("/api/profiles/person2", auth=("admin", "pw")).status_code == 200
    names = [p["key"] for p in client.get(
        "/api/profiles", headers={"Authorization": "Bearer T"}).json()["profiles"]]
    assert names == ["person1", "person3"]
    assert client.delete("/api/profiles/person2",
                         auth=("admin", "pw")).status_code == 404
    assert client.delete("/api/profiles/person2").status_code == 401


def test_ui_is_a_compact_dashboard_and_not_cached(monkeypatch):
    client = make_client(monkeypatch)
    r = client.get("/", auth=("admin", "pw"))
    html = r.text
    assert r.headers.get("cache-control") == "no-store"
    for needle in ("grid-template-columns", "minmax(320px", "max-width:1200px",
                   "collapsible", "48px"):
        assert needle in html, needle
    assert "padding:80px 24px" not in html
    for needle in ("min-width:0", "max-width:100%", "overflow:hidden"):
        assert needle in html, needle
    assert "--parchment:#2a2a2c" in html
    assert "color:var(--card-fg)" in html


def test_ui_uses_the_approved_design_tokens_and_crud(monkeypatch):
    client = make_client(monkeypatch)
    html = client.get("/", auth=("admin", "pw")).text
    for needle in ("#0066cc", "SF Pro Display", "backdrop-filter: blur(20px)",
                   "9999px", "Delete", "Edit", "Add profile",
                   "profiles.json record only"):
        assert needle in html, needle
    assert "linear-gradient" not in html
    assert html.count("box-shadow") == 1  # the single product-style shadow


def test_ui_401_challenges_the_browser(monkeypatch):
    """A bare 401 without WWW-Authenticate makes Chrome show JSON, not a login
    prompt — that is exactly what the user saw at 18794."""
    client = make_client(monkeypatch)
    for kwargs in ({}, {"auth": ("admin", "wrong")}):
        r = client.get("/", **kwargs)
        assert r.status_code == 401
        assert r.headers["www-authenticate"].startswith("Basic")


def test_empty_secrets_fail_closed(monkeypatch):
    client = make_client(monkeypatch, token="", admin="")
    assert client.get("/api/profiles", headers={"Authorization": "Bearer x"}).status_code == 503
    assert client.post("/api/profiles/person1", auth=("admin", "pw"), json={}).status_code == 503
    assert client.get("/", auth=("admin", "x")).status_code == 503


def test_identify_joins_person_and_honorific(monkeypatch):
    client = make_client(monkeypatch)
    r = client.post("/api/identify", headers={"Authorization": "Bearer T"},
                    files={"file": ("a.ogg", b"x", "audio/ogg")})
    assert r.status_code == 200
    body = r.json()
    assert body["person"] == "person1"
    assert body["display_name"] == "Person 1"
    assert body["honorific"] == "сэр"
    assert body["confidence"] == 0.9
    assert client.post("/api/identify", files={"file": ("a", b"x")}).status_code in (401, 403)


def test_unknown_uid_gives_empty_honorific(monkeypatch):
    client = make_client(monkeypatch, fake=FakeEngines(uid="testuser"))
    body = client.post("/api/identify", headers={"Authorization": "Bearer T"},
                       files={"file": ("a", b"x")}).json()
    assert body["person"] == "testuser"
    assert body["honorific"] == ""
    assert body["display_name"] == "testuser"


def test_enroll_error_is_returned_with_status(monkeypatch):
    client = make_client(monkeypatch, fake=FakeEngines(
        error=engines.EngineError(500, "boom")))
    r = client.post("/api/profiles/person1/voice/enroll", auth=("admin", "pw"),
                    files={"files": ("s.wav", b"x")})
    assert r.status_code == 502
    assert "500" in r.text and "boom" in r.text


def test_ui_room_picker_is_a_fixed_list_of_the_five_cameras(monkeypatch):
    client = make_client(monkeypatch)
    html = client.get("/", auth=("admin", "pw")).text
    assert '<select name="room"' in html
    for room in ("livingroom", "kitchen", "entrance1", "camera-corridor1", "balcony"):
        assert f'<option value="{room}"' in html, room
    assert 'name="room" value=' not in html
    for needle in ("What to say in front of the camera", "0.7–1 m", "Computer, turn on the light",
                   "Dobbi, turn off the light", "10–15", "Sample quality"):
        assert needle in html, needle


def test_ui_has_a_light_dark_theme_toggle(monkeypatch):
    client = make_client(monkeypatch)
    html = client.get("/", auth=("admin", "pw")).text
    for needle in ("data-action=\"toggle-theme\"", "localStorage",
                   "prefers-color-scheme", "data-theme", "Dark theme", "Light theme"):
        assert needle in html, needle
    assert "linear-gradient" not in html
    assert html.count("box-shadow") == 1


def test_ui_has_voice_enroll_and_camera_record_forms(monkeypatch):
    client = make_client(monkeypatch)
    html = client.get("/", auth=("admin", "pw")).text
    for needle in ("/api/record", "voice/enroll", "face/enroll", "replaces",
                   "room", "seconds", "files"):
        assert needle in html, needle
