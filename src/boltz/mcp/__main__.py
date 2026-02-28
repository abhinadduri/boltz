"""Boltz MCP server — all tool definitions and entry point."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
import uuid
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from ._gpu import find_free_devices, query_gpu_devices
from ._jobs import (
    PredictionJob,
    _CANCEL_GRACE_SECONDS,
    _PREDICTION_JOBS,
    _PREDICTION_JOBS_LOCK,
    _FALLBACK_SESSION_KEY,
    _TERMINAL_JOB_STATUSES,
    _append_job_log,
    _get_job_locked,
    _get_session_jobs_locked,
    _get_worker_mp_context,
    _touch_job,
    _utc_now_iso,
)
from ._results import (
    read_affinity_json,
    read_confidence_json,
    read_structure_text,
    scan_prediction_outputs,
)
from ._slurm import _resolve_backend_mode, _submit_slurm_job
from ._sync import _recommend_poll_interval_seconds, _sync_job_state_locked
from ._workers import _run_prediction_worker
from ._yaml_builder import build_yaml, validate_input as _validate_input_impl

mcp = FastMCP("boltz")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_current_session_key() -> str:
    return _FALLBACK_SESSION_KEY


def _resolve_cuda_devices_for_local(devices: int = 1) -> str | None:
    free = find_free_devices(n=devices, max_utilization=50, min_free_memory_mb=4000)
    if free:
        return ",".join(str(d) for d in free)
    return None


def _build_predict_cli_args(kwargs: dict[str, Any]) -> list[str]:
    """Build CLI arguments for ``boltz predict`` (used by SLURM backend)."""
    args: list[str] = []
    mapping = {
        "out_dir": "--out_dir",
        "cache": "--cache",
        "model": "--model",
        "recycling_steps": "--recycling_steps",
        "sampling_steps": "--sampling_steps",
        "diffusion_samples": "--diffusion_samples",
        "step_scale": "--step_scale",
        "output_format": "--output_format",
        "devices": "--devices",
        "accelerator": "--accelerator",
        "seed": "--seed",
        "max_msa_seqs": "--max_msa_seqs",
        "sampling_steps_affinity": "--sampling_steps_affinity",
        "diffusion_samples_affinity": "--diffusion_samples_affinity",
    }
    for key, flag in mapping.items():
        val = kwargs.get(key)
        if val is not None:
            args.extend([flag, str(val)])

    # Boolean flags
    if kwargs.get("use_msa_server"):
        args.append("--use_msa_server")
    if kwargs.get("use_potentials"):
        args.append("--use_potentials")
    if kwargs.get("write_full_pae"):
        args.append("--write_full_pae")
    if kwargs.get("write_full_pde"):
        args.append("--write_full_pde")
    if kwargs.get("affinity_mw_correction"):
        args.append("--affinity_mw_correction")
    if kwargs.get("no_kernels"):
        args.append("--no_kernels")
    return args


def _kernels_available() -> bool:
    """Check if cuequivariance_torch is importable."""
    try:
        import cuequivariance_torch  # noqa: F401
        return True
    except Exception:
        return False


def _launch_prediction(
    yaml_path: str,
    predict_kwargs: dict[str, Any],
    prediction_kind: str,
    backend: str,
    job_name: str | None,
    slurm_partition: str | None,
    slurm_gpus: int | None,
    slurm_cpus_per_task: int | None,
    slurm_mem: str | None,
    slurm_time: str | None,
    devices: int,
    has_affinity: bool = False,
) -> dict[str, Any]:
    """Common logic for launching a prediction job (local or SLURM)."""
    # Auto-detect kernel availability
    if "no_kernels" not in predict_kwargs:
        predict_kwargs["no_kernels"] = not _kernels_available()
    session_key = _get_current_session_key()
    job_id = str(uuid.uuid4())
    now_iso = _utc_now_iso()

    resolved_mode, mode_reason = _resolve_backend_mode(backend)
    out_dir = predict_kwargs.get("out_dir", "./")

    job = PredictionJob(
        job_id=job_id,
        status="queued",
        created_at=now_iso,
        prediction_kind=prediction_kind,
        input_yaml_path=yaml_path,
        output_dir=str(Path(out_dir).expanduser() / f"boltz_results_{Path(yaml_path).stem}"),
        resolved_args=predict_kwargs,
        model=predict_kwargs.get("model", "boltz2"),
        backend=resolved_mode,
        has_affinity=has_affinity,
        last_update_at=now_iso,
        last_update_monotonic=time.monotonic(),
    )
    _append_job_log(job, f"[queued] Job created. Backend: {resolved_mode} ({mode_reason})")

    if resolved_mode == "slurm":
        cli_args = _build_predict_cli_args(predict_kwargs)
        try:
            result = _submit_slurm_job(
                job_id=job_id,
                job_name_prefix=job_name or "boltz_predict",
                yaml_path=yaml_path,
                predict_args=cli_args,
                slurm_partition=slurm_partition,
                slurm_gpus=slurm_gpus,
                slurm_cpus_per_task=slurm_cpus_per_task,
                slurm_mem=slurm_mem,
                slurm_time=slurm_time,
                default_gpus=devices,
            )
            job.scheduler_job_id = result["scheduler_job_id"]
            job.worker_log_path = result.get("worker_log_path")
            job.worker_error_log_path = result.get("worker_error_log_path")
            _append_job_log(job, f"[slurm] Submitted as SLURM job {job.scheduler_job_id}.")
        except Exception as exc:
            job.status = "failed"
            job.error = f"SLURM submission failed: {exc}"
            job.finished_at = _utc_now_iso()
            _append_job_log(job, f"[failed] {job.error}")
    else:
        # Local execution
        ctx = _get_worker_mp_context()
        parent_conn, child_conn = ctx.Pipe(duplex=False)

        cancel_flag = tempfile.NamedTemporaryFile(
            prefix="boltz_cancel_", suffix=".flag", delete=False
        )
        cancel_flag_path = cancel_flag.name
        cancel_flag.close()
        os.unlink(cancel_flag_path)  # Remove so it doesn't exist yet

        worker_log_dir = Path(out_dir).expanduser() / ".boltz_mcp_logs"
        worker_log_dir.mkdir(parents=True, exist_ok=True)
        worker_log_path = str(worker_log_dir / f"worker_{job_id}.log")

        cuda_devices = _resolve_cuda_devices_for_local(devices)

        process = ctx.Process(
            target=_run_prediction_worker,
            args=(yaml_path, predict_kwargs, cancel_flag_path, child_conn, worker_log_path, cuda_devices),
            daemon=True,
        )
        process.start()

        job.process = process
        job.event_conn = parent_conn
        job.cancel_flag_path = cancel_flag_path
        job.worker_log_path = worker_log_path
        job.worker_pid = process.pid
        _append_job_log(job, f"[local] Started worker process (pid={process.pid}).")

    with _PREDICTION_JOBS_LOCK:
        jobs = _get_session_jobs_locked(session_key)
        jobs[job_id] = job

    poll_interval = _recommend_poll_interval_seconds(job) or 5.0
    return {
        "job_id": job_id,
        "status": job.status,
        "backend": job.backend,
        "output_dir": job.output_dir,
        "recommended_initial_poll_interval_seconds": poll_interval,
    }


# ===================================================================
# Group 1: Prediction Launch (4 tools)
# ===================================================================

@mcp.tool()
def predict_structure(
    # --- Molecule definitions ---
    proteins: list[dict[str, Any]] | None = None,
    ligands: list[dict[str, Any]] | None = None,
    nucleic_acids: list[dict[str, Any]] | None = None,
    # --- Constraints ---
    pocket_constraints: list[dict[str, Any]] | None = None,
    bond_constraints: list[dict[str, Any]] | None = None,
    contact_constraints: list[dict[str, Any]] | None = None,
    # --- Model / sampling ---
    model: str = "boltz2",
    recycling_steps: int = 3,
    sampling_steps: int = 200,
    diffusion_samples: int = 1,
    step_scale: float | None = None,
    use_msa_server: bool = True,
    use_potentials: bool = False,
    max_msa_seqs: int = 8192,
    seed: int | None = None,
    # --- Output ---
    output_format: str = "mmcif",
    out_dir: str | None = None,
    job_name: str | None = None,
    write_full_pae: bool = False,
    write_full_pde: bool = False,
    # --- Backend ---
    backend: str = "auto",
    slurm_partition: str | None = None,
    slurm_gpus: int | None = None,
    slurm_cpus_per_task: int | None = None,
    slurm_mem: str | None = None,
    slurm_time: str | None = None,
    devices: int = 1,
    accelerator: str = "gpu",
) -> dict[str, Any]:
    """Predict the 3D structure of a biomolecular complex.

    Accepts structured molecule definitions (proteins, ligands, nucleic acids)
    and optional constraints. Builds the required YAML input internally and
    launches a Boltz structure prediction job.

    Returns {job_id, status, recommended_initial_poll_interval_seconds}.
    Use get_prediction_status() to poll for completion.
    """
    # Build YAML
    yaml_content = build_yaml(
        proteins=proteins,
        ligands=ligands,
        nucleic_acids=nucleic_acids,
        pocket_constraints=pocket_constraints,
        bond_constraints=bond_constraints,
        contact_constraints=contact_constraints,
    )

    # Write YAML to temp file
    out_dir = out_dir or "./"
    yaml_dir = Path(out_dir).expanduser() / ".boltz_mcp_inputs"
    yaml_dir.mkdir(parents=True, exist_ok=True)
    stem = job_name or f"predict_{uuid.uuid4().hex[:8]}"
    yaml_path = str(yaml_dir / f"{stem}.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    predict_kwargs = {
        "out_dir": out_dir,
        "model": model,
        "recycling_steps": recycling_steps,
        "sampling_steps": sampling_steps,
        "diffusion_samples": diffusion_samples,
        "step_scale": step_scale,
        "use_msa_server": use_msa_server,
        "use_potentials": use_potentials,
        "max_msa_seqs": max_msa_seqs,
        "seed": seed,
        "output_format": output_format,
        "write_full_pae": write_full_pae,
        "write_full_pde": write_full_pde,
        "devices": devices,
        "accelerator": accelerator,
    }

    return _launch_prediction(
        yaml_path=yaml_path,
        predict_kwargs=predict_kwargs,
        prediction_kind="structure",
        backend=backend,
        job_name=job_name,
        slurm_partition=slurm_partition,
        slurm_gpus=slurm_gpus,
        slurm_cpus_per_task=slurm_cpus_per_task,
        slurm_mem=slurm_mem,
        slurm_time=slurm_time,
        devices=devices,
    )


@mcp.tool()
def predict_affinity(
    # --- Molecule definitions ---
    proteins: list[dict[str, Any]] | None = None,
    ligands: list[dict[str, Any]] | None = None,
    nucleic_acids: list[dict[str, Any]] | None = None,
    # --- Affinity-specific ---
    binder_id: str = "",
    affinity_mw_correction: bool = False,
    sampling_steps_affinity: int = 200,
    diffusion_samples_affinity: int = 5,
    # --- Constraints ---
    pocket_constraints: list[dict[str, Any]] | None = None,
    bond_constraints: list[dict[str, Any]] | None = None,
    contact_constraints: list[dict[str, Any]] | None = None,
    # --- Model / sampling ---
    model: str = "boltz2",
    recycling_steps: int = 3,
    sampling_steps: int = 200,
    diffusion_samples: int = 1,
    step_scale: float | None = None,
    use_msa_server: bool = True,
    use_potentials: bool = False,
    max_msa_seqs: int = 8192,
    seed: int | None = None,
    # --- Output ---
    output_format: str = "mmcif",
    out_dir: str | None = None,
    job_name: str | None = None,
    write_full_pae: bool = False,
    write_full_pde: bool = False,
    # --- Backend ---
    backend: str = "auto",
    slurm_partition: str | None = None,
    slurm_gpus: int | None = None,
    slurm_cpus_per_task: int | None = None,
    slurm_mem: str | None = None,
    slurm_time: str | None = None,
    devices: int = 1,
    accelerator: str = "gpu",
) -> dict[str, Any]:
    """Predict structure + binding affinity for a protein-ligand complex.

    Same as predict_structure but additionally computes affinity for the
    specified binder chain. Requires model='boltz2'.

    The binder_id must correspond to a ligand chain ID.
    """
    if not binder_id:
        return {"error": "binder_id is required for affinity prediction."}
    if model != "boltz2":
        return {"error": "Affinity prediction requires model='boltz2'."}

    yaml_content = build_yaml(
        proteins=proteins,
        ligands=ligands,
        nucleic_acids=nucleic_acids,
        pocket_constraints=pocket_constraints,
        bond_constraints=bond_constraints,
        contact_constraints=contact_constraints,
        affinity_binder=binder_id,
    )

    out_dir = out_dir or "./"
    yaml_dir = Path(out_dir).expanduser() / ".boltz_mcp_inputs"
    yaml_dir.mkdir(parents=True, exist_ok=True)
    stem = job_name or f"affinity_{uuid.uuid4().hex[:8]}"
    yaml_path = str(yaml_dir / f"{stem}.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    predict_kwargs = {
        "out_dir": out_dir,
        "model": model,
        "recycling_steps": recycling_steps,
        "sampling_steps": sampling_steps,
        "diffusion_samples": diffusion_samples,
        "step_scale": step_scale,
        "use_msa_server": use_msa_server,
        "use_potentials": use_potentials,
        "max_msa_seqs": max_msa_seqs,
        "seed": seed,
        "output_format": output_format,
        "write_full_pae": write_full_pae,
        "write_full_pde": write_full_pde,
        "devices": devices,
        "accelerator": accelerator,
        "affinity_mw_correction": affinity_mw_correction,
        "sampling_steps_affinity": sampling_steps_affinity,
        "diffusion_samples_affinity": diffusion_samples_affinity,
    }

    return _launch_prediction(
        yaml_path=yaml_path,
        predict_kwargs=predict_kwargs,
        prediction_kind="affinity",
        backend=backend,
        job_name=job_name,
        slurm_partition=slurm_partition,
        slurm_gpus=slurm_gpus,
        slurm_cpus_per_task=slurm_cpus_per_task,
        slurm_mem=slurm_mem,
        slurm_time=slurm_time,
        devices=devices,
        has_affinity=True,
    )


@mcp.tool()
def predict_from_yaml(
    yaml_content: str,
    # --- Model / sampling ---
    model: str = "boltz2",
    recycling_steps: int = 3,
    sampling_steps: int = 200,
    diffusion_samples: int = 1,
    step_scale: float | None = None,
    use_msa_server: bool = True,
    use_potentials: bool = False,
    max_msa_seqs: int = 8192,
    seed: int | None = None,
    # --- Output ---
    output_format: str = "mmcif",
    out_dir: str | None = None,
    job_name: str | None = None,
    write_full_pae: bool = False,
    write_full_pde: bool = False,
    # --- Backend ---
    backend: str = "auto",
    slurm_partition: str | None = None,
    slurm_gpus: int | None = None,
    slurm_cpus_per_task: int | None = None,
    slurm_mem: str | None = None,
    slurm_time: str | None = None,
    devices: int = 1,
    accelerator: str = "gpu",
) -> dict[str, Any]:
    """Run prediction from a raw YAML string.

    Escape hatch for advanced inputs (templates, modifications, cyclic peptides)
    that aren't easily expressed through predict_structure's parameters.
    """
    out_dir = out_dir or "./"
    yaml_dir = Path(out_dir).expanduser() / ".boltz_mcp_inputs"
    yaml_dir.mkdir(parents=True, exist_ok=True)
    stem = job_name or f"yaml_{uuid.uuid4().hex[:8]}"
    yaml_path = str(yaml_dir / f"{stem}.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    predict_kwargs = {
        "out_dir": out_dir,
        "model": model,
        "recycling_steps": recycling_steps,
        "sampling_steps": sampling_steps,
        "diffusion_samples": diffusion_samples,
        "step_scale": step_scale,
        "use_msa_server": use_msa_server,
        "use_potentials": use_potentials,
        "max_msa_seqs": max_msa_seqs,
        "seed": seed,
        "output_format": output_format,
        "write_full_pae": write_full_pae,
        "write_full_pde": write_full_pde,
        "devices": devices,
        "accelerator": accelerator,
    }

    return _launch_prediction(
        yaml_path=yaml_path,
        predict_kwargs=predict_kwargs,
        prediction_kind="structure",
        backend=backend,
        job_name=job_name,
        slurm_partition=slurm_partition,
        slurm_gpus=slurm_gpus,
        slurm_cpus_per_task=slurm_cpus_per_task,
        slurm_mem=slurm_mem,
        slurm_time=slurm_time,
        devices=devices,
    )


@mcp.tool()
def predict_from_file(
    input_path: str,
    # --- Model / sampling ---
    model: str = "boltz2",
    recycling_steps: int = 3,
    sampling_steps: int = 200,
    diffusion_samples: int = 1,
    step_scale: float | None = None,
    use_msa_server: bool = True,
    use_potentials: bool = False,
    max_msa_seqs: int = 8192,
    seed: int | None = None,
    # --- Output ---
    output_format: str = "mmcif",
    out_dir: str | None = None,
    job_name: str | None = None,
    write_full_pae: bool = False,
    write_full_pde: bool = False,
    # --- Backend ---
    backend: str = "auto",
    slurm_partition: str | None = None,
    slurm_gpus: int | None = None,
    slurm_cpus_per_task: int | None = None,
    slurm_mem: str | None = None,
    slurm_time: str | None = None,
    devices: int = 1,
    accelerator: str = "gpu",
) -> dict[str, Any]:
    """Run prediction on an existing YAML/FASTA file on disk.

    The file must be a .yaml, .yml, .fasta, or .fa file that Boltz can parse.
    """
    p = Path(input_path).expanduser()
    if not p.is_file():
        return {"error": f"Input file not found: {input_path}"}
    if p.suffix.lower() not in {".yaml", ".yml", ".fasta", ".fa"}:
        return {"error": f"Unsupported file type: {p.suffix}. Expected .yaml, .yml, .fasta, or .fa"}

    out_dir = out_dir or "./"
    predict_kwargs = {
        "out_dir": out_dir,
        "model": model,
        "recycling_steps": recycling_steps,
        "sampling_steps": sampling_steps,
        "diffusion_samples": diffusion_samples,
        "step_scale": step_scale,
        "use_msa_server": use_msa_server,
        "use_potentials": use_potentials,
        "max_msa_seqs": max_msa_seqs,
        "seed": seed,
        "output_format": output_format,
        "write_full_pae": write_full_pae,
        "write_full_pde": write_full_pde,
        "devices": devices,
        "accelerator": accelerator,
    }

    return _launch_prediction(
        yaml_path=str(p),
        predict_kwargs=predict_kwargs,
        prediction_kind="structure",
        backend=backend,
        job_name=job_name,
        slurm_partition=slurm_partition,
        slurm_gpus=slurm_gpus,
        slurm_cpus_per_task=slurm_cpus_per_task,
        slurm_mem=slurm_mem,
        slurm_time=slurm_time,
        devices=devices,
    )


# ===================================================================
# Group 2: Job Management (4 tools)
# ===================================================================

@mcp.tool()
def get_prediction_status(job_id: str) -> dict[str, Any]:
    """Get the current status and progress of a prediction job.

    Returns job_id, status, progress, timestamps, error info, output_dir,
    and recommended_poll_interval_seconds.
    """
    session_key = _get_current_session_key()
    with _PREDICTION_JOBS_LOCK:
        jobs = _get_session_jobs_locked(session_key)
        job = _get_job_locked(job_id, jobs)
        _sync_job_state_locked(job)
        poll = _recommend_poll_interval_seconds(job)
        return {
            "job_id": job.job_id,
            "status": job.status,
            "prediction_kind": job.prediction_kind,
            "model": job.model,
            "backend": job.backend,
            "progress": dict(job.progress),
            "created_at": job.created_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "error": job.error,
            "output_dir": job.output_dir,
            "recommended_poll_interval_seconds": poll,
        }


@mcp.tool()
def get_prediction_logs(
    job_id: str,
    from_line: int = 0,
    max_lines: int = 200,
) -> dict[str, Any]:
    """Retrieve log lines from a prediction job.

    Supports pagination via from_line and max_lines.
    """
    session_key = _get_current_session_key()
    with _PREDICTION_JOBS_LOCK:
        jobs = _get_session_jobs_locked(session_key)
        job = _get_job_locked(job_id, jobs)
        _sync_job_state_locked(job)

        total = len(job.logs)
        end = min(from_line + max_lines, total)
        lines = job.logs[from_line:end]
        poll = _recommend_poll_interval_seconds(job, for_logs=True)

        return {
            "job_id": job.job_id,
            "lines": lines,
            "from_line": from_line,
            "next_line": end,
            "total_lines": total,
            "recommended_poll_interval_seconds": poll,
        }


@mcp.tool()
def cancel_prediction(job_id: str, force: bool = False) -> dict[str, Any]:
    """Cancel a running or queued prediction job.

    For local jobs: sets a cancel flag file, then SIGTERM after 30s grace.
    For SLURM jobs: sends scancel.
    """
    session_key = _get_current_session_key()
    with _PREDICTION_JOBS_LOCK:
        jobs = _get_session_jobs_locked(session_key)
        job = _get_job_locked(job_id, jobs)

        if job.status in _TERMINAL_JOB_STATUSES:
            return {
                "job_id": job_id,
                "status": job.status,
                "message": f"Job already in terminal state: {job.status}",
            }

        if job.cancel_requested and not force:
            return {
                "job_id": job_id,
                "status": job.status,
                "message": "Cancellation already requested. Use force=True to escalate.",
            }

        job.cancel_requested = True
        job.cancel_requested_at = _utc_now_iso()
        job.status = "cancelling"

        if job.backend == "slurm" and job.scheduler_job_id:
            try:
                subprocess.run(
                    ["scancel", job.scheduler_job_id],
                    check=False,
                    capture_output=True,
                    timeout=10,
                )
                _append_job_log(job, f"[cancelling] Sent scancel for SLURM job {job.scheduler_job_id}.")
            except Exception as exc:
                _append_job_log(job, f"[warning] scancel failed: {exc}")
        elif job.backend == "local":
            # Set cancel flag file
            if job.cancel_flag_path:
                try:
                    Path(job.cancel_flag_path).touch()
                    _append_job_log(job, "[cancelling] Cancel flag file created.")
                except Exception as exc:
                    _append_job_log(job, f"[warning] Failed to create cancel flag: {exc}")

            job.cancel_grace_deadline_monotonic = time.monotonic() + _CANCEL_GRACE_SECONDS

            if force and job.process is not None:
                try:
                    job.process.terminate()
                    job.terminate_sent_at_monotonic = time.monotonic()
                    _append_job_log(job, "[cancelling] Force: sent immediate SIGTERM.")
                except Exception as exc:
                    _append_job_log(job, f"[warning] Force terminate failed: {exc}")

        return {
            "job_id": job_id,
            "status": job.status,
            "message": "Cancellation requested.",
        }


@mcp.tool()
def list_prediction_jobs(
    status: str | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    """List all prediction jobs in the current session.

    Optionally filter by status (queued, running, succeeded, failed, cancelled, cancelling).
    """
    session_key = _get_current_session_key()
    with _PREDICTION_JOBS_LOCK:
        jobs = _get_session_jobs_locked(session_key)

        result: list[dict[str, Any]] = []
        for job in jobs.values():
            if job.status not in _TERMINAL_JOB_STATUSES:
                _sync_job_state_locked(job)
            if status is not None and job.status != status:
                continue
            result.append({
                "job_id": job.job_id,
                "status": job.status,
                "prediction_kind": job.prediction_kind,
                "model": job.model,
                "backend": job.backend,
                "created_at": job.created_at,
                "started_at": job.started_at,
                "finished_at": job.finished_at,
            })
            if len(result) >= limit:
                break

        return {
            "jobs": result,
            "total": len(jobs),
            "showing": len(result),
        }


# ===================================================================
# Group 3: Result Retrieval (4 tools)
# ===================================================================

@mcp.tool()
def get_prediction_results(job_id: str) -> dict[str, Any]:
    """Get a summary of all outputs for a completed prediction job.

    Returns record IDs, model file paths, top-level confidence scores,
    affinity results, and available auxiliary files.
    """
    session_key = _get_current_session_key()
    with _PREDICTION_JOBS_LOCK:
        jobs = _get_session_jobs_locked(session_key)
        job = _get_job_locked(job_id, jobs)
        _sync_job_state_locked(job)

        if job.status != "succeeded":
            return {
                "job_id": job_id,
                "status": job.status,
                "error": job.error,
                "message": f"Job is not yet complete (status={job.status}). Results unavailable.",
            }

        output_summary = scan_prediction_outputs(job.output_dir)
        return {
            "job_id": job_id,
            "status": job.status,
            "output_dir": job.output_dir,
            **output_summary,
        }


@mcp.tool()
def get_confidence_scores(
    job_id: str,
    record_id: str | None = None,
    model_idx: int = 0,
) -> dict[str, Any]:
    """Get detailed confidence scores for a specific model prediction.

    Returns confidence_score, ptm, iptm, ligand_iptm, protein_iptm,
    complex_plddt, complex_iplddt, complex_pde, complex_ipde,
    chains_ptm, pair_chains_iptm.
    """
    session_key = _get_current_session_key()
    with _PREDICTION_JOBS_LOCK:
        jobs = _get_session_jobs_locked(session_key)
        job = _get_job_locked(job_id, jobs)

    if job.status != "succeeded":
        return {"error": f"Job not complete (status={job.status})."}

    predictions_dir = Path(job.output_dir) / "predictions"
    if not predictions_dir.is_dir():
        return {"error": "Predictions directory not found."}

    # Resolve record_id
    if record_id is None:
        records = sorted(d.name for d in predictions_dir.iterdir() if d.is_dir())
        if not records:
            return {"error": "No prediction records found."}
        record_id = records[0]

    confidence_path = predictions_dir / record_id / f"confidence_{record_id}_model_{model_idx}.json"
    if not confidence_path.is_file():
        return {"error": f"Confidence file not found: {confidence_path}"}

    scores = read_confidence_json(str(confidence_path))
    return {
        "job_id": job_id,
        "record_id": record_id,
        "model_idx": model_idx,
        **scores,
    }


@mcp.tool()
def get_affinity_results(
    job_id: str,
    record_id: str | None = None,
) -> dict[str, Any]:
    """Get affinity prediction results for a completed job.

    Returns affinity_pred_value (log10 IC50), affinity_probability_binary,
    and optionally per-ensemble values.
    """
    session_key = _get_current_session_key()
    with _PREDICTION_JOBS_LOCK:
        jobs = _get_session_jobs_locked(session_key)
        job = _get_job_locked(job_id, jobs)

    if job.status != "succeeded":
        return {"error": f"Job not complete (status={job.status})."}

    predictions_dir = Path(job.output_dir) / "predictions"
    if not predictions_dir.is_dir():
        return {"error": "Predictions directory not found."}

    if record_id is None:
        records = sorted(d.name for d in predictions_dir.iterdir() if d.is_dir())
        if not records:
            return {"error": "No prediction records found."}
        record_id = records[0]

    affinity_path = predictions_dir / record_id / f"affinity_{record_id}.json"
    if not affinity_path.is_file():
        return {"error": f"Affinity file not found: {affinity_path}. Was this an affinity prediction job?"}

    affinity = read_affinity_json(str(affinity_path))
    return {
        "job_id": job_id,
        "record_id": record_id,
        **affinity,
    }


@mcp.tool()
def read_structure_file(
    job_id: str,
    record_id: str | None = None,
    model_idx: int = 0,
    max_lines: int = 500,
) -> dict[str, Any]:
    """Read the predicted structure file (mmCIF/PDB) for client inspection.

    Returns truncated text content of the structure file.
    """
    session_key = _get_current_session_key()
    with _PREDICTION_JOBS_LOCK:
        jobs = _get_session_jobs_locked(session_key)
        job = _get_job_locked(job_id, jobs)

    if job.status != "succeeded":
        return {"error": f"Job not complete (status={job.status})."}

    predictions_dir = Path(job.output_dir) / "predictions"
    if not predictions_dir.is_dir():
        return {"error": "Predictions directory not found."}

    if record_id is None:
        records = sorted(d.name for d in predictions_dir.iterdir() if d.is_dir())
        if not records:
            return {"error": "No prediction records found."}
        record_id = records[0]

    record_dir = predictions_dir / record_id
    # Find structure file
    for ext in (".cif", ".pdb"):
        path = record_dir / f"{record_id}_model_{model_idx}{ext}"
        if path.is_file():
            text = read_structure_text(str(path), max_lines=max_lines)
            return {
                "job_id": job_id,
                "record_id": record_id,
                "model_idx": model_idx,
                "format": ext.lstrip("."),
                "path": str(path),
                "content": text,
            }

    return {"error": f"Structure file not found for record_id={record_id}, model_idx={model_idx}."}


# ===================================================================
# Group 4: Utilities (2 tools)
# ===================================================================

@mcp.tool()
def validate_input(
    proteins: list[dict[str, Any]] | None = None,
    ligands: list[dict[str, Any]] | None = None,
    nucleic_acids: list[dict[str, Any]] | None = None,
    pocket_constraints: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Pre-flight validation of prediction input.

    Checks sequences, SMILES, CCD codes, and constraint references.
    Returns estimated complexity without running a prediction.
    """
    return _validate_input_impl(
        proteins=proteins,
        ligands=ligands,
        nucleic_acids=nucleic_acids,
        pocket_constraints=pocket_constraints,
    )


@mcp.tool()
def get_boltz_info() -> dict[str, Any]:
    """Get information about the Boltz installation and environment.

    Returns boltz version, cache directory, which model weights are cached,
    and GPU availability.
    """
    # Version
    try:
        import boltz
        version = getattr(boltz, "__version__", None)
        if version is None:
            from importlib.metadata import version as get_version
            version = get_version("boltz")
    except Exception:
        version = "unknown"

    # Cache directory
    cache_dir = os.environ.get("BOLTZ_CACHE", str(Path.home() / ".boltz"))
    cache_path = Path(cache_dir)

    # Check cached models
    cached_models: dict[str, bool] = {}
    for model_file in ["boltz1_conf.ckpt", "boltz2_conf.ckpt", "boltz2_aff.ckpt", "ccd.pkl"]:
        cached_models[model_file] = (cache_path / model_file).is_file()

    # GPU info
    gpu_devices = query_gpu_devices()

    return {
        "version": version,
        "cache_dir": cache_dir,
        "cached_models": cached_models,
        "gpu_count": len(gpu_devices),
        "gpu_devices": gpu_devices,
        "gpu_available": len(gpu_devices) > 0,
        "slurm_available": shutil.which("sbatch") is not None,
    }


# ===================================================================
# Entry point
# ===================================================================

def main() -> None:
    parser = ArgumentParser(description="Boltz MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
    )
    parser.add_argument("--host", default=None, help="Host interface for HTTP transports.")
    parser.add_argument("--port", type=int, default=None, help="Port for HTTP transports.")
    parser.add_argument("--mount-path", default=None, help="Optional mount path for SSE transport.")
    args = parser.parse_args()

    if args.host:
        mcp.settings.host = args.host
    if args.port is not None:
        mcp.settings.port = args.port

    mount_path = args.mount_path if args.mount_path else None
    mcp.run(transport=args.transport, mount_path=mount_path)


if __name__ == "__main__":
    main()
