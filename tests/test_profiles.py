"""Profile store: one atomic JSON file, seeded from the Phase A config files."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import profiles  # noqa: E402


HONORIFICS = {"person1": "сэр", "person2": "мэм", "person3": "мэм"}
NAMES = {"person1": "Person 1"}
CHAT_IDS = {
    "100001": "person1",
    "100002": "person2",
    "100003": "person3",
}


def test_seed_ignores_test_profiles():
    store = profiles.seed_store(
        NAMES, HONORIFICS, CHAT_IDS,
        voice_names=["person1", "testuser", "testcurl"],
        face_names=["person1", "person2", "person3"],
    )
    assert set(store["profiles"]) == {"person1", "person2", "person3"}
    a = store["profiles"]["person1"]
    assert a["display_name"] == "Person 1"
    assert a["honorific"] == "сэр"
    assert a["telegram_chat_ids"] == ["100001"]
    assert store["profiles"]["person2"]["honorific"] == "мэм"
    assert store["profiles"]["person2"]["telegram_chat_ids"] == ["100002"]
    assert store["profiles"]["person3"]["telegram_chat_ids"] == ["100003"]


def test_atomic_save_leaves_no_partial_file(tmp_path, monkeypatch):
    path = str(tmp_path / "profiles.json")
    with open(path, "w") as f:
        f.write("ORIGINAL")
    real_replace = profiles.os.replace

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(profiles.os, "replace", boom)
    with pytest.raises(OSError):
        profiles.save_store(path, profiles.empty_store())
    with open(path) as f:
        assert f.read() == "ORIGINAL"
    assert list(tmp_path.iterdir()) == [tmp_path / "profiles.json"]

    monkeypatch.setattr(profiles.os, "replace", real_replace)
    profiles.save_store(path, profiles.empty_store())
    assert profiles.load_store(path) == profiles.empty_store()


def test_chat_map_uses_confirmed_ids():
    store = profiles.seed_store(NAMES, HONORIFICS, CHAT_IDS, [], [])
    assert profiles.chat_map(store) == CHAT_IDS


def test_unknown_profile_returns_none():
    store = profiles.seed_store(NAMES, HONORIFICS, CHAT_IDS, [], [])
    assert profiles.profile_view(store, "nobody", [], []) is None


def test_corrupt_store_raises_instead_of_wiping(tmp_path):
    path = str(tmp_path / "profiles.json")
    with open(path, "w") as f:
        f.write("{not json")
    with pytest.raises(profiles.StoreError):
        profiles.load_store(path)
    with open(path) as f:
        assert f.read() == "{not json"
