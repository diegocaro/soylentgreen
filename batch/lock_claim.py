#!/usr/bin/env python3
import json
import os
import socket
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def now_iso() -> str:
    """Return current UTC time in ISO 8601 with tzinfo=UTC."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def now_iso_safe() -> str:
    return now_iso().replace(":", "").replace("-", "")


class FileClaimLockError(Exception):
    pass


class FileClaimLock:
    def __init__(
        self, file_path: Path, worker_id: str | None = None, lock_suffix: str = ".lock"
    ):
        self.file_path = file_path
        self.lock_suffix = lock_suffix
        self.lock_path = file_path.with_name(file_path.name + lock_suffix)
        self.fd = None
        self.info = None
        # Default worker name is the hostname
        self.worker_id = worker_id or socket.gethostname()

    def __enter__(self):
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            self.fd = os.open(self.lock_path, flags, 0o600)
        except FileExistsError:
            raise FileClaimLockError(f"Lock already claimed: {self.lock_path}")
        except Exception as e:
            raise FileClaimLockError(f"Failed to claim lock: {e}")

        self.info = {
            "worker_id": self.worker_id,
            "claimed_at": now_iso(),
            "job_id": uuid.uuid4().hex,
        }
        data = json.dumps(self.info).encode("utf-8")
        os.write(self.fd, data)
        os.fsync(self.fd)
        return self.lock_path, self.info

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.fd is not None:
            try:
                os.close(self.fd)
            except Exception:
                pass
        try:
            self.lock_path.unlink()
        except FileNotFoundError:
            pass


def try_claim(
    video_path: Path, worker_id: str, lock_suffix: str = ".lock"
) -> tuple[Path, dict[str, str]] | None:
    """
    Try to create an exclusive claim file next to `video_path` by appending `lock_suffix`.

    Returns:
      (lock_path: Path, info: dict[str, str]) on success
      None if already claimed or on failure
    """

    lock_path: Path = video_path.with_name(
        video_path.name + lock_suffix
    )  # /path/to/file.m4a.lock
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        # os.open accepts path-like objects (Path) so we don't convert to str
        fd: int = os.open(lock_path, flags, 0o600)
    except FileExistsError:
        return None  # already claimed
    except Exception:
        return None

    try:
        info: dict[str, str] = {
            "worker_id": worker_id,
            "claimed_at": now_iso(),
            "job_id": uuid.uuid4().hex,
        }
        data = json.dumps(info).encode("utf-8")
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    return lock_path, info


def release_claim(lock_path: Path) -> None:
    """Remove the claim file if it exists (no error if already removed)."""
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def atomic_write_json(path: Path, data: Any) -> None:
    """
    Atomically write JSON to `path` using a temporary file next to it.
    Accepts only pathlib.Path. Ensures parent directory exists.
    """

    tmp: Path = path.with_name(path.name + ".tmp")
    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf8") as f:
        json.dump(data, f, separators=(",", ":"), ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    # Atomic replace using Path.replace
    tmp.replace(path)


class DummyModel:
    def predict(self, video_path: Path) -> dict[str, object]:
        return {
            "job_id": uuid.uuid4().hex,
            "labels": {"person": 3},
            "model_name": "my_model_name",  # Replace with actual model name
            "model_version": "v1",  # Replace with actual model version
            "video_path": str(video_path),
        }


def run_lock_claim_example(
    video_path: Path,
    worker: str,
    model: Any,
    prefix_file: str = "labels",
    suffix_folder: str = "results",
):
    claimed = try_claim(video_path, worker)
    if claimed:
        lock_path, _info = claimed
        try:
            # run inference on video (replace with real code)
            result: dict[str, object] = model.predict(video_path)
            timestamp = now_iso_safe()
            # create a sibling directory named "<original_filename>.results"
            result_dir: Path = video_path.with_name(
                f"{video_path.name}.{suffix_folder}"
            )
            result_dir.mkdir(parents=True, exist_ok=True)
            result_path: Path = (
                result_dir
                / f"{prefix_file}-{result['model_name']}-{result['model_version']}-{timestamp}.json"
            )
            atomic_write_json(result_path, result)
        finally:
            release_claim(lock_path)
    else:
        # someone else claimed it, skip
        pass


def run_lock_with_try_example(
    video_path: Path,
    worker: str,
    model: Any,
    prefix_file: str = "labels",
    suffix_folder: str = "results",
):
    try:
        with FileClaimLock(video_path, worker) as _claim:
            # run inference on video (replace with real code)
            result: dict[str, object] = model.predict(video_path)
            timestamp = now_iso_safe()
            result_dir: Path = video_path.with_name(
                f"{video_path.name}.{suffix_folder}"
            )
            result_dir.mkdir(parents=True, exist_ok=True)
            result_path: Path = (
                result_dir
                / f"{prefix_file}-{result['model_name']}-{result['model_version']}-{timestamp}.json"
            )
            atomic_write_json(result_path, result)
    except FileClaimLockError:
        # someone else claimed it, skip
        pass


# usage example
if __name__ == "__main__":
    video: Path = Path(
        "sample-videos/aqara_video/lumi1.54ef44603857/20251021/000000.mp4"
    )
    worker: str = "gpu-1"
    model = DummyModel()
    run_lock_claim_example(video, worker, model)

    run_lock_with_try_example(video, worker, model)
