"""Read prediction outputs: confidence JSON, affinity JSON, structure files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def scan_prediction_outputs(output_dir: str) -> dict[str, Any]:
    """Walk the predictions/ directory and return a summary of all outputs.

    Expected structure::

        {output_dir}/predictions/{record_id}/
            {record_id}_model_0.cif
            confidence_{record_id}_model_0.json
            affinity_{record_id}.json  (optional)
            plddt_{record_id}_model_0.npz
            pae_{record_id}_model_0.npz  (optional)
            pde_{record_id}_model_0.npz  (optional)
            embeddings_{record_id}.npz  (optional)
    """
    predictions_dir = Path(output_dir) / "predictions"
    if not predictions_dir.is_dir():
        return {"records": [], "predictions_dir": str(predictions_dir), "found": False}

    records: list[dict[str, Any]] = []
    for record_dir in sorted(predictions_dir.iterdir()):
        if not record_dir.is_dir():
            continue
        record_id = record_dir.name
        record_info: dict[str, Any] = {
            "record_id": record_id,
            "models": [],
            "has_affinity": False,
            "affinity_path": None,
            "auxiliary_files": [],
        }

        # Find model structure files
        model_files: list[dict[str, Any]] = []
        for f in sorted(record_dir.iterdir()):
            name = f.name
            if name.startswith(f"{record_id}_model_") and (name.endswith(".cif") or name.endswith(".pdb")):
                # Extract model index
                stem = f.stem  # e.g. "rec_model_0"
                parts = stem.rsplit("_", 1)
                model_idx = int(parts[-1]) if len(parts) > 1 and parts[-1].isdigit() else 0
                confidence_path = record_dir / f"confidence_{record_id}_model_{model_idx}.json"
                confidence_scores = None
                if confidence_path.is_file():
                    try:
                        confidence_scores = _read_json(str(confidence_path))
                    except Exception:
                        pass
                model_files.append({
                    "model_idx": model_idx,
                    "structure_path": str(f),
                    "structure_format": f.suffix.lstrip("."),
                    "confidence_path": str(confidence_path) if confidence_path.is_file() else None,
                    "top_confidence_score": confidence_scores.get("confidence_score") if confidence_scores else None,
                })
        record_info["models"] = sorted(model_files, key=lambda m: m["model_idx"])

        # Check for affinity
        affinity_path = record_dir / f"affinity_{record_id}.json"
        if affinity_path.is_file():
            record_info["has_affinity"] = True
            record_info["affinity_path"] = str(affinity_path)

        # Auxiliary files
        aux_patterns = ["plddt_", "pae_", "pde_", "embeddings_", "pre_affinity_"]
        for f in sorted(record_dir.iterdir()):
            if any(f.name.startswith(p) for p in aux_patterns):
                record_info["auxiliary_files"].append(str(f))

        records.append(record_info)

    return {
        "records": records,
        "predictions_dir": str(predictions_dir),
        "found": True,
    }


def _read_json(path: str) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def read_confidence_json(path: str) -> dict[str, Any]:
    """Parse a confidence_*.json file.

    Returns dict with keys: confidence_score, ptm, iptm, ligand_iptm, protein_iptm,
    complex_plddt, complex_iplddt, complex_pde, complex_ipde, chains_ptm, pair_chains_iptm.
    """
    if not Path(path).is_file():
        raise FileNotFoundError(f"Confidence file not found: {path}")
    return _read_json(path)


def read_affinity_json(path: str) -> dict[str, Any]:
    """Parse an affinity_*.json file.

    Returns dict with keys: affinity_pred_value, affinity_probability_binary,
    and optionally per-ensemble values.
    """
    if not Path(path).is_file():
        raise FileNotFoundError(f"Affinity file not found: {path}")
    return _read_json(path)


def read_structure_text(path: str, max_lines: int = 500) -> str:
    """Read and return truncated mmCIF/PDB text for client inspection."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Structure file not found: {path}")

    lines: list[str] = []
    with p.open() as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                lines.append(f"\n... [truncated at {max_lines} lines] ...")
                break
            lines.append(line.rstrip())
    return "\n".join(lines)
