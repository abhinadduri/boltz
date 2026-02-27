from __future__ import annotations

import multiprocessing as mp
import signal
import time
from pathlib import Path
from typing import Any

from ._jobs import (
    PredictionJob,
    _CANCEL_GRACE_SECONDS,
    _MAX_EVENT_DRAIN,
    _TERMINAL_JOB_STATUSES,
    _TERMINATE_GRACE_SECONDS,
    _append_job_log,
    _touch_job,
    _utc_now_iso,
)
from ._slurm import _map_slurm_state_to_job_status, _query_slurm_state


# ---------------------------------------------------------------------------
# Log ingestion
# ---------------------------------------------------------------------------

def _ingest_worker_log_lines(job: PredictionJob) -> None:
    log_path = job.worker_log_path
    if not isinstance(log_path, str) or not log_path.strip():
        return
    path_obj = Path(log_path)
    if not path_obj.is_file():
        return

    try:
        with path_obj.open("rb") as f:
            f.seek(job.worker_log_read_offset)
            chunk = f.read()
            job.worker_log_read_offset = f.tell()
    except Exception:
        return

    if not chunk:
        return
    try:
        text = chunk.decode("utf-8", errors="replace")
    except Exception:
        return
    for line in text.splitlines():
        line = line.rstrip()
        if not line:
            continue
        _append_job_log(job, f"[worker] {line}")


# ---------------------------------------------------------------------------
# Event recording
# ---------------------------------------------------------------------------

def _record_job_event(job: PredictionJob, event: dict[str, Any]) -> None:
    _touch_job(job)
    job.last_event = dict(event)
    progress: dict[str, Any] = {
        "phase": event.get("phase"),
        "message": event.get("message"),
    }
    job.progress = progress

    message = event.get("message")
    if isinstance(message, str) and message.strip():
        kind = str(event.get("kind") or "event")
        _append_job_log(job, f"[{kind}] {message}")


# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------

def _process_alive(process: mp.Process | None) -> bool:
    if process is None:
        return False
    try:
        return process.is_alive()
    except Exception:
        return False


def _signal_name_for_exit_code(exit_code: int) -> str | None:
    if exit_code >= 0:
        return None
    signal_number = -exit_code
    try:
        return signal.Signals(signal_number).name
    except Exception:
        return None


def _release_job_runtime_resources(job: PredictionJob) -> None:
    process = job.process
    if process is not None:
        try:
            if process.exitcode is not None:
                job.worker_exit_code = int(process.exitcode)
        except Exception:
            pass
        try:
            process.join(timeout=0)
        except Exception:
            pass
    event_conn = job.event_conn
    if event_conn is not None:
        try:
            event_conn.close()
        except Exception:
            pass
    cancel_flag_path = job.cancel_flag_path
    if isinstance(cancel_flag_path, str) and cancel_flag_path:
        try:
            Path(cancel_flag_path).unlink(missing_ok=True)
        except Exception:
            pass
    job.process = None
    job.event_conn = None
    job.cancel_flag_path = None


# ---------------------------------------------------------------------------
# Worker message handling
# ---------------------------------------------------------------------------

def _apply_worker_message(job: PredictionJob, message: dict[str, Any]) -> None:
    if not isinstance(message, dict):
        return

    msg_type = message.get("type")
    if msg_type == "event":
        event = message.get("event")
        if isinstance(event, dict):
            _record_job_event(job, event)
        return

    if msg_type == "heartbeat":
        _touch_job(job)
        current_progress = dict(job.progress)
        current_progress["worker_heartbeat_at"] = job.last_update_at
        job.progress = current_progress
        return

    if msg_type == "status":
        _touch_job(job)
        status_value = message.get("status")
        if isinstance(status_value, str) and status_value:
            if not (status_value == "running" and job.cancel_requested):
                job.status = status_value
        started_at = message.get("started_at")
        if isinstance(started_at, str) and started_at:
            job.started_at = started_at
        finished_at = message.get("finished_at")
        if isinstance(finished_at, str) and finished_at:
            job.finished_at = finished_at
        error = message.get("error")
        if isinstance(error, str) and error.strip():
            job.error = error
        note = message.get("message")
        if isinstance(note, str) and note.strip():
            _append_job_log(job, note)
        return

    if msg_type == "traceback":
        trace = message.get("traceback")
        if isinstance(trace, str) and trace.strip():
            _append_job_log(job, trace.strip())
        return


# ---------------------------------------------------------------------------
# Event draining
# ---------------------------------------------------------------------------

def _drain_job_events_locked(job: PredictionJob) -> None:
    event_conn = job.event_conn
    if event_conn is None:
        return

    drained = 0
    while drained < _MAX_EVENT_DRAIN:
        try:
            if not event_conn.poll(0):
                break
            message = event_conn.recv()
        except (EOFError, OSError, ValueError):
            break
        drained += 1
        _apply_worker_message(job, message)

    if drained >= _MAX_EVENT_DRAIN:
        _append_job_log(job, f"[warning] Drained {_MAX_EVENT_DRAIN} worker events; additional events remain queued.")


# ---------------------------------------------------------------------------
# Poll intervals
# ---------------------------------------------------------------------------

def _recommend_poll_interval_seconds(job: PredictionJob, *, for_logs: bool = False) -> float | None:
    if job.status in _TERMINAL_JOB_STATUSES:
        return None

    if job.status == "queued":
        interval = 3.0
    elif job.status == "cancelling":
        interval = 4.0
    else:
        phase = str(job.progress.get("phase") or "")
        if phase in {"model_download", "input_processing"}:
            interval = 10.0
        elif phase == "msa_generation":
            interval = 15.0
        elif phase in {"structure_prediction", "affinity_prediction"}:
            interval = 5.0
        else:
            interval = 8.0

    seconds_since_update = max(0.0, time.monotonic() - float(job.last_update_monotonic))
    if job.status == "running" and seconds_since_update >= 120.0:
        interval = max(interval, 20.0)

    if for_logs:
        interval = max(interval * 1.5, 6.0)

    return round(interval, 1)


# ---------------------------------------------------------------------------
# State sync
# ---------------------------------------------------------------------------

def _sync_job_state_locked(job: PredictionJob) -> None:
    if job.backend == "slurm":
        _ingest_worker_log_lines(job)
        if isinstance(job.scheduler_job_id, str) and job.scheduler_job_id.strip():
            state = _query_slurm_state(job.scheduler_job_id)
            if isinstance(state, str) and state:
                mapped = _map_slurm_state_to_job_status(state)
                current_progress = dict(job.progress)
                current_progress["scheduler_state"] = state
                job.progress = current_progress
                _touch_job(job)

                if mapped == "running" and job.started_at is None:
                    job.started_at = _utc_now_iso()

                if mapped in _TERMINAL_JOB_STATUSES:
                    if job.status not in _TERMINAL_JOB_STATUSES:
                        job.status = mapped
                    if mapped == "failed" and not job.error:
                        job.error = f"Slurm job ended in state {state}."
                    if mapped == "cancelled" and not job.error:
                        job.error = f"Slurm job cancelled (state={state})."
                    if job.finished_at is None:
                        job.finished_at = _utc_now_iso()
                elif job.status != "cancelling":
                    job.status = mapped
        return

    _drain_job_events_locked(job)

    process = job.process
    if process is None:
        return

    if job.worker_pid is None and process.pid is not None:
        job.worker_pid = int(process.pid)

    now = time.monotonic()
    if job.cancel_requested and _process_alive(process):
        deadline = job.cancel_grace_deadline_monotonic
        if deadline is not None and now >= deadline and job.terminate_sent_at_monotonic is None:
            try:
                process.terminate()
                job.terminate_sent_at_monotonic = now
                _append_job_log(job, "[cancelling] Cancellation grace elapsed; sent SIGTERM to worker process.")
            except Exception as exc:
                _append_job_log(job, f"[warning] Failed to terminate worker process cleanly: {type(exc).__name__}: {exc}")
        elif (
            job.terminate_sent_at_monotonic is not None
            and now >= (job.terminate_sent_at_monotonic + _TERMINATE_GRACE_SECONDS)
            and _process_alive(process)
        ):
            try:
                process.kill()
                job.terminate_sent_at_monotonic = now
                _append_job_log(job, "[cancelling] Worker did not exit after SIGTERM; sent SIGKILL.")
            except Exception as exc:
                _append_job_log(job, f"[warning] Failed to force-kill worker process: {type(exc).__name__}: {exc}")

    if _process_alive(process):
        return

    try:
        exit_code = process.exitcode
    except Exception:
        exit_code = None
    if isinstance(exit_code, int):
        job.worker_exit_code = exit_code

    if job.status not in _TERMINAL_JOB_STATUSES:
        if job.cancel_requested:
            job.status = "cancelled"
            if not job.error:
                if isinstance(exit_code, int) and exit_code < 0:
                    signal_name = _signal_name_for_exit_code(exit_code)
                    if signal_name:
                        job.error = f"Prediction cancelled by request (worker terminated by {signal_name})."
                    else:
                        job.error = f"Prediction cancelled by request (worker exit code {exit_code})."
                else:
                    job.error = "Prediction cancelled by request."
            _append_job_log(job, "[cancelled] Prediction cancelled.")
        elif exit_code == 0:
            job.status = "succeeded"
            _append_job_log(job, "[succeeded] Prediction job completed.")
        else:
            job.status = "failed"
            if not job.error:
                job.error = f"Worker process exited unexpectedly with code {exit_code}."
            _append_job_log(job, f"[failed] {job.error}")

    if job.finished_at is None:
        job.finished_at = _utc_now_iso()

    _release_job_runtime_resources(job)
