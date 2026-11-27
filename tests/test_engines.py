"""Engine clients: all writes go through the engines' HTTP APIs."""
import builtins
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import engines  # noqa: E402


class Resp:
    def __init__(self, status, payload):
        self.status = status
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def json(self):
        return self._payload


class StubSession:
    def __init__(self, payloads):
        self.payloads = payloads
        self.calls = []

    def post(self, url, data=None, timeout=None, headers=None):
        self.calls.append(("POST", url, data, headers))
        status, payload = self.payloads.get(url, (200, {}))
        return Resp(status, payload)

    def get(self, url, timeout=None):
        self.calls.append(("GET", url, None, None))
        status, payload = self.payloads.get(url, (200, {}))
        return Resp(status, payload)


def form_field(form, name):
    for headers, _opts, value in form._fields:
        if headers.get("name") == name:
            return value
    return None


def make(voice="http://v", face="http://f", key="K", payloads=None, gallery=""):
    session = StubSession(payloads or {})
    return engines.Engines(voice, face, key, session=session,
                           voice_gallery_dir=gallery), session


@pytest.mark.asyncio
async def test_all_writes_go_through_engine_http(monkeypatch):
    real_open = builtins.open

    def guarded(file, *a, **kw):
        if str(file).endswith(".npy"):
            raise AssertionError("engines must never write a .npy directly")
        return real_open(file, *a, **kw)

    monkeypatch.setattr(builtins, "open", guarded)
    client, sess = make(payloads={
        "http://v/enroll": (200, {"status": "ok", "user_id": "person1"}),
        "http://f/enroll": (200, {"status": "ok", "name": "person2"}),
    })
    await client.enroll_voice("person1", [("s.wav", b"wav")])
    await client.enroll_face("person2", [("p.jpg", b"jpg")])
    assert len(sess.calls) == 2
    method, url, form, headers = sess.calls[0]
    assert (method, url) == ("POST", "http://v/enroll")
    assert headers["X-API-Key"] == "K"
    assert form_field(form, "user_id") == "person1"
    assert sess.calls[1][1] == "http://f/enroll"
    assert form_field(sess.calls[1][2], "name") == "person2"
    assert sess.calls[1][3] is None


@pytest.mark.asyncio
async def test_identify_passthrough():
    client, _ = make(payloads={
        "http://v/identify": (200, {"user_id": "person1", "confidence": 0.9}),
    })
    assert await client.identify(b"ogg") == ("person1", 0.9)


@pytest.mark.asyncio
async def test_recognize_passthrough():
    faces = [{"name": "person2", "confidence": 0.7}]
    client, _ = make(payloads={"http://f/recognize": (200, {"faces": faces})})
    assert await client.recognize(b"jpg", 0.35) == faces


@pytest.mark.asyncio
async def test_rosters_from_voice_gallery_and_face_health(tmp_path):
    for name in ("person1", "testuser"):
        (tmp_path / f"{name}.npy").write_bytes(b"x")
    client, _ = make(
        payloads={"http://f/health": (200, {"known_faces": ["person1", "person2"]})},
        gallery=str(tmp_path),
    )
    voice, face = await client.rosters()
    assert voice == ["person1", "testuser"]
    assert face == ["person1", "person2"]


@pytest.mark.asyncio
async def test_non_200_raises_engine_error_with_status():
    client, _ = make(payloads={"http://v/enroll": (500, {"detail": "boom"})})
    with pytest.raises(engines.EngineError) as exc:
        await client.enroll_voice("person1", [("s.wav", b"wav")])
    assert exc.value.status == 500
    assert "boom" in str(exc.value)
