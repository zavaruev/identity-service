"""Photo library: thumbnails of the faces actually in the training set."""
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as identity_app  # noqa: E402
import photos as photos_mod  # noqa: E402
import profiles  # noqa: E402


class FakeEngines:
    def __init__(self):
        self.calls = []
        self.deleted = []

    async def enroll_face(self, key, files, replace=False):
        self.calls.append((key, [n for n, _ in files], replace,
                           [c for _n, c in files]))
        return {"name": key, "accepted": [{"file": n} for n, _ in files],
                "samples": len(files)}

    async def delete_face(self, key):
        self.deleted.append(key)
        return {"deleted": key}


def make(monkeypatch, tmp_path, fake=None):
    monkeypatch.setenv("IDENTITY_DATA_FILE", str(tmp_path / "profiles.json"))
    monkeypatch.setenv("IDENTITY_ADMIN_USER", "admin")
    monkeypatch.setenv("IDENTITY_ADMIN_PASSWORD", "pw")
    monkeypatch.setenv("IDENTITY_TOKEN", "T")
    store = profiles.seed_store({}, {"person1": "сэр"}, {}, [], [])
    return TestClient(identity_app.create_app(
        store=store, engines=fake or FakeEngines())), tmp_path


def test_face_upload_saves_thumbnails_and_reports_them(monkeypatch, tmp_path):
    client, root = make(monkeypatch, tmp_path)
    r = client.post("/api/profiles/person1/face/enroll", auth=("admin", "pw"),
                    files=[("files", ("a.jpg", b"AAA", "image/jpeg")),
                           ("files", ("b.jpg", b"BBB", "image/jpeg"))])
    assert r.status_code == 200
    assert sorted(r.json()["photos"]) == ["a.jpg", "b.jpg"]
    assert (root / "photos" / "person1" / "a.jpg").read_bytes() == b"AAA"


def test_delete_one_photo_rebuilds_the_rest(monkeypatch, tmp_path):
    fake = FakeEngines()
    client, root = make(monkeypatch, tmp_path, fake)
    client.post("/api/profiles/person1/face/enroll", auth=("admin", "pw"),
                files=[("files", ("a.jpg", b"AAA", "image/jpeg")),
                       ("files", ("b.jpg", b"BBB", "image/jpeg"))])
    r = client.delete("/api/profiles/person1/face/photos/a.jpg",
                      auth=("admin", "pw"))
    assert r.status_code == 200
    assert r.json()["photos"] == ["b.jpg"]
    assert fake.calls[-1] == ("person1", ["b.jpg"], True, [b"BBB"])
    assert not (root / "photos" / "person1" / "a.jpg").exists()


def test_photo_endpoint_needs_auth_and_rejects_traversal(monkeypatch, tmp_path):
    client, _ = make(monkeypatch, tmp_path)
    client.post("/api/profiles/person1/face/enroll", auth=("admin", "pw"),
                files={"files": ("a.jpg", b"AAA", "image/jpeg")})
    assert client.get("/api/photos/person1/a.jpg").status_code == 401
    r = client.get("/api/photos/person1/a.jpg", auth=("admin", "pw"))
    assert r.status_code == 200 and r.content == b"AAA"
    assert client.get("/api/photos/person1/../profiles.json",
                      auth=("admin", "pw")).status_code in (400, 404)


@pytest.mark.asyncio
async def test_import_pending_keeps_only_frames_with_the_right_face(tmp_path):
    src = tmp_path / "face_pending"
    (src / "_enrollment" / "person1").mkdir(parents=True)
    (src / "_enrollment" / "person1" / "a.jpg").write_bytes(b"AAA")
    (src / "_enrollment" / "person1" / "b.jpg").write_bytes(b"BBB")
    data_file = str(tmp_path / "profiles.json")
    seen = []

    async def accept(data):
        seen.append(data)
        return b"CROP" if data == b"AAA" else None   # BBB has no face

    saved = await photos_mod.import_pending(
        data_file, "person1", str(src), cap=24, accept=accept)
    assert len(saved) == 1
    assert photos_mod.read_photo(data_file, "person1", saved[0]) == b"CROP"
    assert seen == [b"AAA", b"BBB"]


@pytest.mark.asyncio
async def test_rebuild_sends_originals_not_the_display_crops(tmp_path):
    src = tmp_path / "face_pending" / "_enrollment" / "person1"
    src.mkdir(parents=True)
    (src / "a.jpg").write_bytes(b"FULL_FRAME")
    (src / "b.jpg").write_bytes(b"FULL_FRAME_2")
    data_file = str(tmp_path / "profiles.json")

    async def accept(raw):
        return (b"CROP", raw)   # crop for display, original for enrollment

    await photos_mod.import_pending(
        data_file, "person1", str(tmp_path / "face_pending"), accept=accept)
    assert photos_mod.read_photo(
        data_file, "person1", photos_mod.list_photos(data_file, "person1")[0]) == b"CROP"

    import os
    os.environ["IDENTITY_DATA_FILE"] = data_file
    os.environ["IDENTITY_ADMIN_USER"] = "admin"
    os.environ["IDENTITY_ADMIN_PASSWORD"] = "pw"
    fake = FakeEngines()
    client = TestClient(identity_app.create_app(
        store=profiles.seed_store({}, {"person1": "сэр"}, {}, [], []),
        engines=fake))
    name = photos_mod.list_photos(data_file, "person1")[0]
    r = client.delete(f"/api/profiles/person1/face/photos/{name}",
                      auth=("admin", "pw"))
    assert r.status_code == 200
    sent = fake.calls[-1][3]
    assert sent and sent[0] != b"CROP"
    assert b"FULL_FRAME" in sent[0]
    for key in ("IDENTITY_DATA_FILE", "IDENTITY_ADMIN_USER", "IDENTITY_ADMIN_PASSWORD"):
        os.environ.pop(key, None)


@pytest.mark.asyncio
async def test_import_pending_seeds_real_source_frames_once(tmp_path):
    src = tmp_path / "face_pending"
    (src / "_archive" / "2026-11-15" / "person1").mkdir(parents=True)
    (src / "_archive" / "2026-11-15" / "person1" / "a.jpg").write_bytes(b"AAA")
    (src / "_enrollment" / "person1").mkdir(parents=True)
    (src / "_enrollment" / "person1" / "b.jpg").write_bytes(b"BBB")
    data_file = str(tmp_path / "profiles.json")

    first = await photos_mod.import_pending(data_file, "person1", str(src), cap=24)
    assert len(first) == 2
    assert sorted(photos_mod.list_photos(data_file, "person1")) == sorted(first)
    assert await photos_mod.import_pending(
        data_file, "person1", str(src), cap=24) == []
    manifest = photos_mod.manifest(data_file, "person1")
    assert len(manifest) == 2
    assert all("face_pending" in v for v in manifest.values())


def test_ui_shows_photo_thumbnails_and_delete_controls(monkeypatch, tmp_path):
    client, _ = make(monkeypatch, tmp_path)
    html = client.get("/", auth=("admin", "pw")).text
    for needle in ("Training set thumbnails", 'data-action="photo-delete"', "/api/photos/",
                   "source frames from the face-captures archive"):
        assert needle in html, needle
