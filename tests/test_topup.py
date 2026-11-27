import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

import topup_faces as topup  # noqa: E402


def test_candidates_come_only_from_quality_ok_frames(tmp_path):
    captures = tmp_path / "captures"
    day = captures / "2026-11-20"; day.mkdir(parents=True)
    good = day / "08-00-00_entrance1_face01.jpg"; good.write_bytes(b"good")
    bad = day / "08-00-00_entrance1_face02.jpg"; bad.write_bytes(b"bad")
    (day / "08-00-00_frames.jsonl").write_text(
        '{"path": "%s", "quality_ok": true}\n{"path": "%s", "quality_ok": false}\n' % (good, bad))
    found = topup.raw_candidates(str(captures))
    assert [str(good)] == found


def test_topup_fills_person3_from_raw_captures():
    existing = [{"name": f"old{i}", "score": 100.0 + i} for i in range(8)]
    raw = [{"path": f"/raw/{i}.jpg", "score": 900.0 + i,
            "sha1": f"h{i}"} for i in range(20)]
    posted = []
    counts = {"person3": 8}

    def poster(key, path, gallery):
        posted.append((key, path))
        counts[key] = min(24, counts[key] + 1)
        return {"status": "ok"}

    result = topup.topup_one("person3", existing, raw, poster=poster, cap=24)
    assert result["gallery"] == 24 and len(posted) == 16
    assert result["dry_run"] is False


def test_topup_uses_the_same_improvement_margin():
    existing = [{"name": "a", "score": 100.0} for _ in range(23)]
    raw = [{"path": "/marginal.jpg", "score": 105.0, "sha1": "x"}]
    posted = []
    topup.topup_one("person1", existing, raw, poster=lambda *a: posted.append(a))
    assert posted == []


def test_topup_dry_run_posts_nothing():
    posted = []
    result = topup.topup_one("person3", [], [{"path": "/a.jpg", "score": 1}],
                             poster=lambda *a: posted.append(a), cap=24, dry_run=True)
    assert result["dry_run"] is True and posted == []


def test_topup_does_not_remove_a_better_existing_frame():
    existing = [{"name": f"keep{i}", "score": 5000.0 + i} for i in range(24)]
    raw = [{"path": "/worse.jpg", "score": 10.0, "sha1": "x"}]
    posted = []
    topup.topup_one("person1", existing, raw, poster=lambda *a: posted.append(a), cap=24)
    assert posted == []
