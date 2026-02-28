from __future__ import annotations

import atexit
import multiprocessing as mp
import os
import signal
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FALLBACK_SESSION_KEY = "__default__"
_TERMINAL_JOB_STATUSES = {"succeeded", "failed", "cancelled"}
_JOB_LOG_LIMIT = 5000
_MAX_EVENT_DRAIN = 2000
_HEARTBEAT_INTERVAL_SECONDS = 10.0
_CANCEL_GRACE_SECONDS = 30.0
_TERMINATE_GRACE_SECONDS = 10.0


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_worker_mp_context() -> mp.context.BaseContext:
    try:
        return mp.get_context("fork")
    except ValueError:
        return mp.get_context("spawn")


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class PredictionJob:
    job_id: str
    status: str                                     # queued|running|succeeded|failed|cancelled|cancelling
    created_at: str                                 # UTC ISO timestamp
    prediction_kind: str                            # "structure" or "affinity"
    input_yaml_path: str                            # Generated YAML path
    output_dir: str                                 # boltz_results_* path
    resolved_args: dict[str, Any]                   # Full resolved config
    model: str = "boltz2"
    progress: dict[str, Any] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    cancel_requested: bool = False
    cancel_requested_at: str | None = None
    cancel_grace_deadline_monotonic: float | None = None
    terminate_sent_at_monotonic: float | None = None
    last_event: dict[str, Any] | None = None
    last_update_at: str = field(default_factory=lambda: _utc_now_iso())
    last_update_monotonic: float = field(default_factory=time.monotonic)
    worker_pid: int | None = None
    worker_exit_code: int | None = None
    worker_log_path: str | None = None
    worker_error_log_path: str | None = None
    worker_log_read_offset: int = 0
    backend: str = "local"
    scheduler_job_id: str | None = None
    process: mp.Process | None = None
    cancel_flag_path: str | None = None
    event_conn: Any | None = None
    # Boltz-specific
    record_ids: list[str] = field(default_factory=list)
    has_affinity: bool = False


# ---------------------------------------------------------------------------
# Exception class
# ---------------------------------------------------------------------------

class PredictionCancelledError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# State containers + locks
# ---------------------------------------------------------------------------

_PREDICTION_JOBS: dict[str, dict[str, PredictionJob]] = {}
_PREDICTION_JOBS_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def _get_session_jobs_locked(session_key: str) -> dict[str, PredictionJob]:
    jobs = _PREDICTION_JOBS.get(session_key)
    if jobs is None:
        jobs = {}
        _PREDICTION_JOBS[session_key] = jobs
    return jobs


def _get_job_locked(
    job_id: str,
    jobs_by_id: dict[str, PredictionJob],
) -> PredictionJob:
    job = jobs_by_id.get(job_id)
    if job is None:
        raise KeyError(f"No prediction job found for job_id={job_id!r}")
    return job


# ---------------------------------------------------------------------------
# Job lifecycle
# ---------------------------------------------------------------------------

def _touch_job(job: PredictionJob) -> None:
    job.last_update_at = _utc_now_iso()
    job.last_update_monotonic = time.monotonic()


def _append_job_log(job: PredictionJob, message: str) -> None:
    _touch_job(job)
    line = f"{_utc_now_iso()} {message}"
    job.logs.append(line)
    if len(job.logs) > _JOB_LOG_LIMIT:
        del job.logs[: len(job.logs) - _JOB_LOG_LIMIT]


# ---------------------------------------------------------------------------
# Atexit cleanup for non-daemonic worker processes
# ---------------------------------------------------------------------------

def _cleanup_all_workers() -> None:
    """Terminate all alive worker processes at interpreter shutdown.

    Called via atexit. We intentionally do NOT acquire
    ``_PREDICTION_JOBS_LOCK`` here — during interpreter teardown another
    thread may hold the lock and never release it, which would deadlock.
    This is safe because no new jobs can arrive once atexit handlers run.
    """
    # Collect all alive worker processes and their pipe connections.
    alive_procs: list[mp.Process] = []
    conns: list[Any] = []
    for session_jobs in _PREDICTION_JOBS.values():
        for job in session_jobs.values():
            if job.process is not None and job.process.is_alive():
                alive_procs.append(job.process)
            if job.event_conn is not None:
                conns.append(job.event_conn)

    if not alive_procs:
        return

    # Phase 1: send SIGTERM to all alive workers in parallel so they can
    # begin shutting down concurrently.
    for proc in alive_procs:
        try:
            os.kill(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass

    # Phase 2: give each process up to 5 s to exit gracefully.
    for proc in alive_procs:
        proc.join(timeout=5)

    # Phase 3: forcefully kill any survivors.
    for proc in alive_procs:
        if proc.is_alive():
            try:
                proc.kill()
            except OSError:
                pass
            proc.join(timeout=2)

    # Phase 4: close pipe connections.
    for conn in conns:
        try:
            conn.close()
        except OSError:
            pass


atexit.register(_cleanup_all_workers)
