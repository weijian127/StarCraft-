"""Public-demo safeguards that do not need the SC2 engine or model."""

import re
import tempfile
from pathlib import Path

from server.app import Config, UploadRateLimiter, support_payload
from server.store import JOB_ID_PATTERN, JobStore, display_name, new_job_id


def test_public_job_id_has_128_random_bits():
    job_id = new_job_id(public=True)
    assert JOB_ID_PATTERN.match(job_id)
    assert len(job_id.rsplit("-", 1)[1]) == 32
    assert len(job_id.rsplit("-", 1)[1]) * 4 >= 128


def test_local_job_id_stays_compatible():
    job_id = new_job_id(public=False)
    assert JOB_ID_PATTERN.match(job_id)
    assert len(job_id.rsplit("-", 1)[1]) == 6


def test_client_filename_cannot_be_a_server_path():
    assert display_name("..\\..\\evil.SC2Replay") == "evil.SC2Replay"
    assert display_name("C:\\Windows\\x.SC2Replay") == "x.SC2Replay"


def test_public_store_uses_public_ids_and_reports_size():
    details = {"sha256": "0" * 64, "replay_id": "r", "game_version": "5.0.4.82457",
               "map_title": "Submerged LE", "duration_seconds": 1.0, "label": 1}
    with tempfile.TemporaryDirectory() as directory:
        store = JobStore(directory, public_mode=True)
        record = store.create("client replay.SC2Replay", b"abc", details, {"frames": 50})
        assert re.search(r"-[0-9a-f]{32}$", record["job_id"])
        assert store.size_bytes() >= 3
        assert store.read(record["job_id"])["original_name"] == "client replay.SC2Replay"
        assert store.upload_path(record["job_id"]).read_bytes() == b"abc"


def test_upload_rate_limiter_is_per_client():
    limiter = UploadRateLimiter(2, 60)
    assert limiter.allow("a")
    assert limiter.can_attempt("a")
    assert limiter.allow("a")
    assert not limiter.can_attempt("a")
    assert not limiter.allow("a")
    assert limiter.allow("b")


def test_public_support_describes_demo_and_limits():
    root = Path("unused")
    config = Config(sc2_path=root, team_dir=root, checkpoint=root, jobs_root=root,
                    public_mode=True, uploads_per_hour=6, rate_window_seconds=3600,
                    job_timeout=1800, max_jobs_bytes=1024)
    payload = support_payload(config)
    assert payload["public_demo"]["enabled"] is True
    assert payload["limits"]["public_mode"] is True
    assert payload["limits"]["max_jobs_bytes"] == 1024
    assert payload["jobs"]["timeout_seconds"] == 1800
    assert "6 uploads" in payload["upload"]["rate_limit"]
