"""Worker process that wraps ``boltz.main.predict()`` for local execution."""

from __future__ import annotations

import os
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ._jobs import _HEARTBEAT_INTERVAL_SECONDS, PredictionCancelledError


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _send(conn: Any, msg: dict[str, Any]) -> None:
    try:
        conn.send(msg)
    except Exception:
        pass


def _heartbeat_loop(conn: Any, cancel_flag_path: str, interval: float) -> None:
    while True:
        time.sleep(interval)
        _send(conn, {"type": "heartbeat", "at": _utc_now_iso()})
        # Check cancellation in heartbeat too
        if cancel_flag_path and Path(cancel_flag_path).exists():
            return


def _check_cancel(cancel_flag_path: str) -> None:
    if cancel_flag_path and Path(cancel_flag_path).exists():
        raise PredictionCancelledError("Prediction cancelled by user request.")


def _run_prediction_worker(
    yaml_path: str,
    predict_kwargs: dict[str, Any],
    cancel_flag_path: str,
    event_conn: Any,
    worker_log_path: str,
    cuda_devices: str | None = None,
) -> None:
    """Run boltz.main.predict() in a forked child process.

    Parameters
    ----------
    yaml_path : str
        Path to the YAML input file.
    predict_kwargs : dict
        Keyword arguments to pass to ``predict()``, minus ``data``.
    cancel_flag_path : str
        Path to a file that, when created, signals cancellation.
    event_conn : Any
        Parent-side of a multiprocessing Pipe for sending status/event messages.
    worker_log_path : str
        Path to redirect stdout/stderr to.
    cuda_devices : str | None
        CUDA_VISIBLE_DEVICES value, or None to leave unchanged.
    """
    try:
        # Set CUDA devices
        if cuda_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices

        # Redirect stdout/stderr to log file
        log_dir = Path(worker_log_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = open(worker_log_path, "w")
        sys.stdout = log_file
        sys.stderr = log_file

        # Start heartbeat thread
        hb_thread = threading.Thread(
            target=_heartbeat_loop,
            args=(event_conn, cancel_flag_path, _HEARTBEAT_INTERVAL_SECONDS),
            daemon=True,
        )
        hb_thread.start()

        # Emit running status
        _send(event_conn, {
            "type": "status",
            "status": "running",
            "started_at": _utc_now_iso(),
        })

        # Phase: model download (handled inside predict)
        _send(event_conn, {
            "type": "event",
            "event": {"kind": "phase", "phase": "model_download", "message": "Downloading/verifying model weights..."},
        })

        _check_cancel(cancel_flag_path)

        # Phase: input processing + structure prediction
        _send(event_conn, {
            "type": "event",
            "event": {"kind": "phase", "phase": "input_processing", "message": "Processing inputs and generating MSA..."},
        })

        _check_cancel(cancel_flag_path)

        _send(event_conn, {
            "type": "event",
            "event": {"kind": "phase", "phase": "structure_prediction", "message": "Running structure prediction..."},
        })

        # Call predict() – use .callback() to invoke the underlying function
        # directly, bypassing Click's CLI context machinery.
        from boltz.main import predict
        predict.callback(data=yaml_path, **predict_kwargs)

        _check_cancel(cancel_flag_path)

        # Success
        _send(event_conn, {
            "type": "status",
            "status": "succeeded",
            "finished_at": _utc_now_iso(),
        })

    except PredictionCancelledError:
        _send(event_conn, {
            "type": "status",
            "status": "cancelled",
            "finished_at": _utc_now_iso(),
            "error": "Prediction cancelled by user request.",
        })
    except Exception as exc:
        tb = traceback.format_exc()
        _send(event_conn, {"type": "traceback", "traceback": tb})
        _send(event_conn, {
            "type": "status",
            "status": "failed",
            "finished_at": _utc_now_iso(),
            "error": f"{type(exc).__name__}: {exc}",
        })
    finally:
        try:
            event_conn.close()
        except Exception:
            pass
