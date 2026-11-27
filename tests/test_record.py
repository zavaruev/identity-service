"""/api/record: capture from a camera, then enroll; no silent failure."""
import array
import io
import os
import sys
import wave

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as identity_app  # noqa: E402
import engines  # noqa: E402
import profiles  # noqa: E402


def wav(amplitude: int, frames: int = 32000) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(array.array("h", [amplitude] * frames).tobytes())
    return out.getvalue()


WAV = wav(4000)      # 2 s, clearly above the room floor
SILENT = wav(0)      # a perfectly valid WAV containing no speech


class FakeEngines:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    async def enroll_voice(self, key, files):
        self.calls.append((key, files))
        if self.error:
            raise self.error
        return {"status": "ok", "user_id": key}


def make_client(monkeypatch, fake):
    monkeypatch.setenv("IDENTITY_TOKEN", "T")
    monkeypatch.setenv("IDENTITY_ADMIN_USER", "admin")
    monkeypatch.setenv("IDENTITY_ADMIN_PASSWORD", "pw")
    store = profiles.seed_store(
        {"person1": "Person 1"}, {"person1": "сэр"}, {}, [], [])
    return TestClient(identity_app.create_app(store=store, engines=fake))


async def _async_wav(payload):
    return payload


def test_record_captures_then_enrolls(monkeypatch):
    fake = FakeEngines()
    client = make_client(monkeypatch, fake)
    monkeypatch.setattr(identity_app, "capture_from_gateway",
                        lambda room, seconds: _async_wav(WAV))
    r = client.post("/api/record", auth=("admin", "pw"),
                    data={"room": "livingroom", "profile": "person1", "seconds": 6})
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert fake.calls == [("person1", [("sample.wav", WAV)])]


def test_silence_is_refused_before_enroll(monkeypatch):
    fake = FakeEngines()
    client = make_client(monkeypatch, fake)
    monkeypatch.setattr(identity_app, "capture_from_gateway",
                        lambda room, seconds: _async_wav(SILENT))
    r = client.post("/api/record", auth=("admin", "pw"),
                    data={"room": "livingroom", "profile": "person1", "seconds": 6})
    assert r.status_code == 502
    assert "no speech" in r.json()["detail"]
    assert fake.calls == []


def test_record_response_reports_sample_quality(monkeypatch):
    client = make_client(monkeypatch, FakeEngines())
    monkeypatch.setattr(identity_app, "capture_from_gateway",
                        lambda room, seconds: _async_wav(WAV))
    body = client.post("/api/record", auth=("admin", "pw"),
                       data={"room": "livingroom", "profile": "person1"}).json()
    assert body["quality"]["verdict"] == "good"
    assert body["quality"]["peak"] == 4000
    assert body["quality"]["duration"] == 2.0

    monkeypatch.setattr(identity_app, "capture_from_gateway",
                        lambda room, seconds: _async_wav(wav(32760)))
    body = client.post("/api/record", auth=("admin", "pw"),
                       data={"room": "livingroom", "profile": "person1"}).json()
    assert "clipping" in body["quality"]["verdict"]


def test_gateway_error_is_surfaced_not_silenced(monkeypatch):
    fake = FakeEngines()
    client = make_client(monkeypatch, fake)
    def boom(room, seconds):
        raise engines.EngineError(502, "camera did not answer")
    monkeypatch.setattr(identity_app, "capture_from_gateway", boom)
    r = client.post("/api/record", auth=("admin", "pw"),
                    data={"room": "livingroom", "profile": "person1"})
    assert r.status_code == 502
    assert "502" in r.text and "camera did not answer" in r.text
    assert fake.calls == []


def test_unknown_room_refused_before_capture(monkeypatch):
    fake = FakeEngines()
    client = make_client(monkeypatch, fake)
    called = []
    monkeypatch.setattr(identity_app, "capture_from_gateway",
                        lambda room, seconds: called.append(room))
    r = client.post("/api/record", auth=("admin", "pw"),
                    data={"room": "not-a-camera", "profile": "person1"})
    assert r.status_code == 400
    assert "room" in r.json()["detail"]
    assert called == []


def test_unknown_profile_refused_before_capture(monkeypatch):
    fake = FakeEngines()
    client = make_client(monkeypatch, fake)
    called = []
    monkeypatch.setattr(identity_app, "capture_from_gateway",
                        lambda room, seconds: called.append(room))
    r = client.post("/api/record", auth=("admin", "pw"),
                    data={"room": "livingroom", "profile": "nobody"})
    assert r.status_code == 404
    assert called == []


def test_response_names_replacement(monkeypatch):
    client = make_client(monkeypatch, FakeEngines())
    monkeypatch.setattr(identity_app, "capture_from_gateway",
                        lambda room, seconds: _async_wav(WAV))
    r = client.post("/api/record", auth=("admin", "pw"),
                    data={"room": "livingroom", "profile": "person1"})
    assert "replaced" in r.json()["detail"]
