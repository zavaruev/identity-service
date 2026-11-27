import base64
import hashlib
import io
import json
import os
import sys

from PIL import Image
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as identity_app  # noqa: E402
import engines  # noqa: E402
import photos as photos_mod  # noqa: E402
import profiles  # noqa: E402

AUTH = {"Authorization": "Basic " + base64.b64encode(b"admin:pw").decode()}
def _jpeg(color=(120, 120, 120)):
    buf = io.BytesIO()
    Image.new("RGB", (300, 300), color).save(buf, format="JPEG")
    return buf.getvalue()

FRAME = _jpeg()


class CurateEngines:
    def __init__(self, error=None, reject_all=False):
        self.error = error
        self.reject_all = reject_all
        self.calls = []

    async def recognize(self, image, threshold):
        return [{"name": "person3", "confidence": 0.91,
                 "bbox": [0, 0, 200, 200], "det_score": 0.9}]

    async def enroll_face(self, key, files, replace=False):
        self.calls.append((key, list(files), replace))
        if self.error:
            raise self.error
        if self.reject_all:
            return {"accepted": [], "rejected": [{"file": n} for n, _ in files]}
        return {"accepted": [{"file": n} for n, _ in files], "rejected": []}

    async def rosters(self):
        return [], ["person3"]


def setup(monkeypatch, tmp_path, fake, count=24):
    data_file = tmp_path / "profiles.json"
    monkeypatch.setenv("IDENTITY_DATA_FILE", str(data_file))
    monkeypatch.setenv("IDENTITY_ADMIN_USER", "admin")
    monkeypatch.setenv("IDENTITY_ADMIN_PASSWORD", "pw")
    monkeypatch.setenv("IDENTITY_TOKEN", "T")
    store = profiles.seed_store({"person3": "Person 3"},
                                {"person3": "мэм"}, {}, [], ["person3"])
    d = photos_mod.directory(str(data_file), "person3")
    o = photos_mod.orig_dir(str(data_file), "person3")
    manifest = {}
    for i in range(count):
        name = f"old{i:02d}.jpg"
        content = b"old" + bytes([i])
        (d / name).write_bytes(content)
        (o / name).write_bytes(content)
        manifest[name] = {"source": str(o / name), "score": 5000 + i,
                          "sha1": hashlib.sha1(content).hexdigest()}
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    client = TestClient(identity_app.create_app(store=store, engines=fake))
    return client, d, o, manifest


def post(client, content=FRAME, name="new.jpg"):
    return client.post("/api/profiles/person3/face/curate", headers=AUTH,
                       files={"file": (name, content, "image/jpeg")})


def test_curate_adds_a_better_frame_and_archives_the_worst(monkeypatch, tmp_path):
    fake = CurateEngines()
    client, d, o, before = setup(monkeypatch, tmp_path, fake)
    result = post(client)
    assert result.status_code == 200, result.text
    body = result.json()
    assert body["kept"] == 24 and body["removed"] == ["old00.jpg"]
    assert fake.calls and fake.calls[-1][2] is True and len(fake.calls[-1][1]) == 24
    assert any(p.name.startswith("old00") for p in (d / "_pruned").rglob("*"))
    after = json.loads((d / "manifest.json").read_text())
    assert "old00.jpg" not in after and len(after) == 24


def test_a_partial_gallery_appends_with_the_profile_key(monkeypatch, tmp_path):
    fake = CurateEngines()
    client, d, o, before = setup(monkeypatch, tmp_path, fake, count=8)
    result = post(client)
    assert result.status_code == 200, result.text
    key, files, replace = fake.calls[-1]
    assert key == "person3" and replace is False
    assert len(files) == 1  # only the new frame reaches the engine
    assert result.json()["kept"] == 9


def test_a_partial_gallery_appends_instead_of_rebuilding(monkeypatch, tmp_path):
    fake = CurateEngines()
    client, d, o, before = setup(monkeypatch, tmp_path, fake, count=8)
    result = post(client)
    assert result.status_code == 200, result.text
    assert fake.calls[-1][2] is False          # replace=False
    assert len(fake.calls[-1][1]) == 1          # only the new original
    assert result.json()["kept"] == 9
    assert len(json.loads((d / "manifest.json").read_text())) == 9


def test_a_worse_frame_does_not_replace_a_known_good_one(monkeypatch, tmp_path):
    class LowEngines(CurateEngines):
        async def recognize(self, image, threshold):
            return [{"name": "person3", "confidence": 0.9,
                     "bbox": [0, 0, 10, 10], "det_score": 0.1}]
    fake = LowEngines()
    client, _, _, _ = setup(monkeypatch, tmp_path, fake)
    result = post(client)
    assert result.status_code == 200 and result.json()["status"] == "unchanged"
    assert fake.calls == []


def test_engine_samples_come_from_the_engine_result(monkeypatch, tmp_path):
    class CountingEngines(CurateEngines):
        async def enroll_face(self, key, files, replace=False):
            await super().enroll_face(key, files, replace)
            return {"accepted": [{"file": n} for n, _ in files], "samples": 21}
    fake = CountingEngines()
    client, _, _, _ = setup(monkeypatch, tmp_path, fake, count=8)
    result = post(client)
    assert result.json()["engine_samples"] == 21


def test_engine_error_rolls_back_the_local_write(monkeypatch, tmp_path):
    fake = CurateEngines(error=engines.EngineError(400, "No usable faces"))
    client, d, o, before = setup(monkeypatch, tmp_path, fake)
    result = post(client)
    assert result.status_code == 502
    after = json.loads((d / "manifest.json").read_text())
    assert after == before
    assert not list(d.glob("new*.jpg"))
    assert not list((d / "_pruned").rglob("*")) if (d / "_pruned").exists() else True


def test_content_sha1_dedupe_prevents_a_local_duplicate(monkeypatch, tmp_path):
    fake = CurateEngines()
    client, d, o, before = setup(monkeypatch, tmp_path, fake)
    duplicate = (o / "old00.jpg").read_bytes()
    result = post(client, content=duplicate, name="copy.jpg")
    assert result.status_code == 200 and result.json()["status"] == "duplicate"
    assert json.loads((d / "manifest.json").read_text()) == before
    assert not fake.calls


def test_a_second_curate_does_not_rearchive_the_same_frame(monkeypatch, tmp_path):
    fake = CurateEngines()
    client, d, o, before = setup(monkeypatch, tmp_path, fake)
    post(client)
    archived = {p.name for p in (d / "_pruned").rglob("*") if p.is_file()}
    post(client, content=_jpeg((90, 90, 90)), name="new2.jpg")
    after = {p.name for p in (d / "_pruned").rglob("*") if p.is_file()}
    assert archived <= after
    assert sum(1 for n in after if n.startswith("old01")) <= 1


def test_curate_requires_admin_basic(monkeypatch, tmp_path):
    fake = CurateEngines()
    client, _, _, _ = setup(monkeypatch, tmp_path, fake)
    r = client.post("/api/profiles/person3/face/curate",
                    files={"file": ("f.jpg", FRAME, "image/jpeg")})
    assert r.status_code == 401
    assert "Basic" in r.headers.get("www-authenticate", "")
