import json


from batch.lock_claim import atomic_write_json, now_iso, release_claim, try_claim


def test_now_iso_format():
    iso = now_iso()
    assert iso.endswith("+00:00")
    assert "T" in iso


def test_try_claim_and_release(tmp_path):
    video_path = tmp_path / "test_video.m4a"
    video_path.touch()
    worker_id = "worker-123"
    claim = try_claim(video_path, worker_id)
    assert claim is not None
    lock_path, info = claim
    assert lock_path.exists()
    assert info["worker_id"] == worker_id
    assert "claimed_at" in info
    assert "job_id" in info
    # Try to claim again, should fail
    claim2 = try_claim(video_path, worker_id)
    assert claim2 is None
    # Release claim
    release_claim(lock_path)
    assert not lock_path.exists()


def test_release_claim_nonexistent(tmp_path):
    lock_path = tmp_path / "nonexistent.lock"
    # Should not raise
    release_claim(lock_path)


def test_atomic_write_json(tmp_path):
    path = tmp_path / "data.json"
    data = {"a": 1, "b": "test"}
    atomic_write_json(path, data)
    assert path.exists()
    with path.open() as f:
        loaded = json.load(f)
    assert loaded == data
