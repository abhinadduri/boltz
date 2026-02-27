from __future__ import annotations

import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _resolve_backend_mode(backend: str) -> tuple[str, str]:
    mode = str(backend or "auto").strip().lower()
    if mode not in {"auto", "local", "slurm"}:
        raise ValueError("`backend` must be one of: 'auto', 'local', 'slurm'.")

    if mode == "local":
        return "local", "Explicit backend override."
    if mode == "slurm":
        if shutil.which("sbatch") is None:
            raise ValueError("`backend='slurm'` requested, but `sbatch` was not found in PATH.")
        return "slurm", "Explicit backend override."

    if shutil.which("sbatch") is not None:
        return "slurm", "Detected `sbatch` in PATH; using slurm backend."
    return "local", "No `sbatch` detected; using local backend."


def _parse_sbatch_job_id(raw_stdout: str) -> str | None:
    if not isinstance(raw_stdout, str):
        return None
    for raw_line in raw_stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        token = line.split(";", 1)[0].strip().split()[0]
        if token:
            return token
    return None


def _submit_slurm_job(
    *,
    job_id: str,
    job_name_prefix: str,
    yaml_path: str,
    predict_args: list[str],
    slurm_partition: str | None = None,
    slurm_gpus: int | None = None,
    slurm_cpus_per_task: int | None = None,
    slurm_mem: str | None = None,
    slurm_time: str | None = None,
    default_gpus: int = 1,
) -> dict[str, Any]:
    if shutil.which("sbatch") is None:
        raise RuntimeError("`sbatch` was not found in PATH.")

    run_dir_path = (Path.home() / ".boltz" / "mcp" / "runs" / f"{job_name_prefix}_{job_id}").resolve()
    run_dir_path.mkdir(parents=True, exist_ok=True)

    resource_args: list[str] = []
    if slurm_partition is not None:
        resource_args.extend(["--partition", slurm_partition])
    effective_gpus = slurm_gpus if slurm_gpus is not None else default_gpus
    if effective_gpus is not None:
        resource_args.extend(["--gres", f"gpu:{effective_gpus}"])
    if slurm_cpus_per_task is not None:
        resource_args.extend(["--cpus-per-task", str(slurm_cpus_per_task)])
    if slurm_mem is not None:
        resource_args.extend(["--mem", slurm_mem])
    if slurm_time is not None:
        resource_args.extend(["--time", slurm_time])

    src_root = str(Path(__file__).resolve().parents[2])
    venv_python = Path(src_root) / ".venv" / "bin" / "python"
    if not venv_python.is_file():
        venv_python = Path(__file__).resolve().parents[3] / ".venv" / "bin" / "python"
    if venv_python.is_file():
        python_exe = str(venv_python.absolute())
    else:
        python_exe = str(Path(sys.executable).resolve())

    command = [python_exe, "-m", "boltz", "predict", yaml_path, *predict_args]
    full_cmd_str = " ".join(shlex.quote(arg) for arg in command)
    wrapped_command = f"PYTHONPATH={shlex.quote(src_root)}:${{PYTHONPATH:-}} {full_cmd_str}"

    out_template = str(run_dir_path / "slurm-%j.out")
    err_template = str(run_dir_path / "slurm-%j.err")
    sbatch_cmd = [
        "sbatch",
        "--parsable",
        "--job-name",
        f"{job_name_prefix}_{job_id[:8]}",
        "--chdir",
        str(run_dir_path),
        "--output",
        out_template,
        "--error",
        err_template,
        *resource_args,
        "--wrap",
        wrapped_command,
    ]

    result = subprocess.run(
        sbatch_cmd,
        check=False,
        capture_output=True,
        text=True,
    )
    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    if result.returncode != 0:
        raise RuntimeError(
            f"sbatch submission failed with exit code {result.returncode}. stdout={stdout!r} stderr={stderr!r}"
        )

    scheduler_job_id = _parse_sbatch_job_id(stdout)
    if scheduler_job_id is None:
        raise RuntimeError(f"Unable to parse slurm job id from sbatch output: {stdout!r}")

    return {
        "scheduler_job_id": scheduler_job_id,
        "worker_log_path": str(run_dir_path / f"slurm-{scheduler_job_id}.out"),
        "worker_error_log_path": str(run_dir_path / f"slurm-{scheduler_job_id}.err"),
        "submission_command": sbatch_cmd,
    }


def _query_slurm_state(scheduler_job_id: str) -> str | None:
    if not scheduler_job_id:
        return None

    if shutil.which("sacct") is not None:
        try:
            result = subprocess.run(
                ["sacct", "-j", scheduler_job_id, "--format=State", "-n", "-P"],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0 and isinstance(result.stdout, str) and result.stdout.strip():
                for raw_line in result.stdout.splitlines():
                    line = raw_line.strip()
                    if not line:
                        continue
                    state = line.split("|", 1)[0].strip()
                    if not state:
                        continue
                    return state.split()[0].strip().upper()
        except Exception:
            pass

    if shutil.which("squeue") is not None:
        try:
            result = subprocess.run(
                ["squeue", "-h", "-j", scheduler_job_id, "-o", "%T"],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0 and isinstance(result.stdout, str) and result.stdout.strip():
                return result.stdout.strip().splitlines()[0].strip().upper()
        except Exception:
            pass

    return None


def _map_slurm_state_to_job_status(state: str) -> str:
    normalized = str(state or "").strip().upper()
    if normalized in {
        "PENDING",
        "CONFIGURING",
        "RESIZING",
        "SUSPENDED",
        "REQUEUE_FED",
        "REQUEUED",
    }:
        return "queued"
    if normalized in {"RUNNING", "COMPLETING", "STAGE_OUT"}:
        return "running"
    if normalized in {"COMPLETED"}:
        return "succeeded"
    if normalized in {"CANCELLED", "PREEMPTED"}:
        return "cancelled"
    if normalized in {
        "FAILED",
        "BOOT_FAIL",
        "DEADLINE",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "TIMEOUT",
    }:
        return "failed"
    return "running"
