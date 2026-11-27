import hashlib

import curator
import pytest


def test_score_is_intrinsic_det_score_times_bbox_area():
    assert curator.frame_score(0.7, [0, 0, 100, 200]) == 14000
    assert curator.frame_score(0.6, [0, 0, 100, 200]) == 12000


def test_content_sha1_ignores_the_filename(tmp_path):
    a = tmp_path / "one.jpg"; a.write_bytes(b"same bytes")
    b = tmp_path / "two.jpg"; b.write_bytes(b"same bytes")
    assert curator.content_sha1(str(a)) == curator.content_sha1(str(b))
    assert curator.content_sha1(str(a)) == hashlib.sha1(b"same bytes").hexdigest()


def test_select_top_keeps_the_highest_scores():
    entries = [{"name": f"f{i}", "score": float(i), "sha1": str(i)} for i in range(26)]
    keep, drop = curator.select_top(entries, cap=24)
    assert len(keep) == 24 and len(drop) == 2
    assert {e["name"] for e in drop} == {"f0", "f1"}
    assert {e["name"] for e in keep} >= {"f2", "f25"}


def test_ties_do_not_displace_an_existing_frame():
    entries = [
        {"name": "old", "score": 100.0, "sha1": "a"},
        {"name": "new", "score": 100.0, "sha1": "b"},
    ]
    keep, drop = curator.select_top(entries, cap=1)
    assert keep[0]["name"] == "old" and drop[0]["name"] == "new"


def test_manifest_entries_can_be_normalised_from_legacy_strings():
    entries = curator.normalise_entries({"a.jpg": "/orig/a.jpg",
                                         "b.jpg": {"source": "/orig/b.jpg", "score": 42.0}})
    assert entries[0]["score"] == 0.0 and entries[1]["score"] == 42.0
    assert entries[0]["source"] == "/orig/a.jpg"


def test_cap_and_margin_match_the_spec():
    assert curator.CAP == 24 and curator.MIN_IMPROVEMENT == 0.10
