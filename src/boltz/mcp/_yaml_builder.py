"""Convert structured tool parameters into Boltz YAML input format.

The generated YAML must be consumable by ``boltz.data.parse.yaml.parse_yaml``
(which delegates to ``boltz.data.parse.schema.parse_boltz_schema``).
"""

from __future__ import annotations

import re
from typing import Any

import yaml


_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
_NUCLEOTIDES_RNA = set("ACGU")
_NUCLEOTIDES_DNA = set("ACGT")


def _validate_protein_sequence(seq: str) -> None:
    invalid = set(seq.upper()) - _AMINO_ACIDS
    if invalid:
        raise ValueError(f"Invalid amino acid(s) in protein sequence: {sorted(invalid)}")


def _validate_nucleic_acid_sequence(seq: str, na_type: str) -> None:
    allowed = _NUCLEOTIDES_RNA if na_type == "rna" else _NUCLEOTIDES_DNA
    invalid = set(seq.upper()) - allowed
    if invalid:
        raise ValueError(f"Invalid nucleotide(s) in {na_type} sequence: {sorted(invalid)}")


def _validate_smiles(smiles: str) -> None:
    if not smiles or not smiles.strip():
        raise ValueError("SMILES string cannot be empty.")


def _coerce_id(raw_id: Any) -> Any:
    """Return a string or list-of-strings suitable for the YAML ``id`` field."""
    if isinstance(raw_id, list):
        return [str(i) for i in raw_id]
    return str(raw_id)


def build_yaml(
    proteins: list[dict[str, Any]] | None = None,
    ligands: list[dict[str, Any]] | None = None,
    nucleic_acids: list[dict[str, Any]] | None = None,
    pocket_constraints: list[dict[str, Any]] | None = None,
    bond_constraints: list[dict[str, Any]] | None = None,
    contact_constraints: list[dict[str, Any]] | None = None,
    affinity_binder: str | None = None,
) -> str:
    """Build a Boltz-compatible YAML string from structured parameters.

    Parameters
    ----------
    proteins : list[dict] | None
        Each dict: {id, sequence, msa?, modifications?, cyclic?}
    ligands : list[dict] | None
        Each dict: {id, smiles?|ccd?}
    nucleic_acids : list[dict] | None
        Each dict: {id, sequence, type: "rna"|"dna"}
    pocket_constraints : list[dict] | None
        Each dict: {binder, contacts, max_distance?}
    bond_constraints : list[dict] | None
        Each dict: {atom1, atom2}
    contact_constraints : list[dict] | None
        Each dict: {token1, token2, max_distance?}
    affinity_binder : str | None
        If set, adds a ``properties`` section for affinity prediction.

    Returns
    -------
    str
        YAML string ready to be written to a file.
    """
    proteins = proteins or []
    ligands = ligands or []
    nucleic_acids = nucleic_acids or []

    if not proteins and not ligands and not nucleic_acids:
        raise ValueError("At least one molecule (protein, ligand, or nucleic acid) must be provided.")

    sequences: list[dict[str, Any]] = []

    # Proteins
    for prot in proteins:
        seq = prot.get("sequence", "")
        _validate_protein_sequence(seq)
        entry: dict[str, Any] = {
            "id": _coerce_id(prot["id"]),
            "sequence": seq,
        }
        if "msa" in prot and prot["msa"] is not None:
            entry["msa"] = prot["msa"]
        if "modifications" in prot and prot["modifications"]:
            entry["modifications"] = prot["modifications"]
        if prot.get("cyclic"):
            entry["cyclic"] = True
        sequences.append({"protein": entry})

    # Nucleic acids
    for na in nucleic_acids:
        na_type = na.get("type", "dna").lower()
        if na_type not in ("rna", "dna"):
            raise ValueError(f"Nucleic acid type must be 'rna' or 'dna', got: {na_type!r}")
        seq = na.get("sequence", "")
        _validate_nucleic_acid_sequence(seq, na_type)
        entry = {
            "id": _coerce_id(na["id"]),
            "sequence": seq,
        }
        if "msa" in na and na["msa"] is not None:
            entry["msa"] = na["msa"]
        sequences.append({na_type: entry})

    # Ligands
    for lig in ligands:
        entry = {"id": _coerce_id(lig["id"])}
        has_smiles = "smiles" in lig and lig["smiles"]
        has_ccd = "ccd" in lig and lig["ccd"]
        if has_smiles and has_ccd:
            raise ValueError("Ligand must have either 'smiles' or 'ccd', not both.")
        if not has_smiles and not has_ccd:
            raise ValueError("Ligand must have either 'smiles' or 'ccd'.")
        if has_smiles:
            _validate_smiles(lig["smiles"])
            entry["smiles"] = lig["smiles"]
        else:
            entry["ccd"] = lig["ccd"]
        sequences.append({"ligand": entry})

    data: dict[str, Any] = {
        "version": 1,
        "sequences": sequences,
    }

    # Constraints
    constraints: list[dict[str, Any]] = []
    for pc in (pocket_constraints or []):
        c: dict[str, Any] = {
            "binder": pc["binder"],
            "contacts": pc["contacts"],
        }
        if "max_distance" in pc and pc["max_distance"] is not None:
            c["max_distance"] = pc["max_distance"]
        constraints.append({"pocket": c})

    for bc in (bond_constraints or []):
        constraints.append({
            "bond": {
                "atom1": bc["atom1"],
                "atom2": bc["atom2"],
            }
        })

    for cc in (contact_constraints or []):
        c = {
            "token1": cc["token1"],
            "token2": cc["token2"],
        }
        if "max_distance" in cc and cc["max_distance"] is not None:
            c["max_distance"] = cc["max_distance"]
        constraints.append({"contact": c})

    if constraints:
        data["constraints"] = constraints

    # Affinity property
    if affinity_binder is not None:
        data["properties"] = [{"affinity": {"binder": affinity_binder}}]

    try:
        return yaml.dump(data, default_flow_style=False, sort_keys=False)
    except TypeError:
        # Older PyYAML versions don't support sort_keys
        return yaml.dump(data, default_flow_style=False)


def validate_input(
    proteins: list[dict[str, Any]] | None = None,
    ligands: list[dict[str, Any]] | None = None,
    nucleic_acids: list[dict[str, Any]] | None = None,
    pocket_constraints: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Pre-flight validation of input parameters.

    Returns a dict with validation results and estimated complexity.
    """
    errors: list[str] = []
    warnings: list[str] = []
    proteins = proteins or []
    ligands = ligands or []
    nucleic_acids = nucleic_acids or []

    if not proteins and not ligands and not nucleic_acids:
        errors.append("At least one molecule (protein, ligand, or nucleic acid) must be provided.")

    total_residues = 0
    chain_ids: list[str] = []

    for i, prot in enumerate(proteins):
        if "id" not in prot:
            errors.append(f"Protein {i}: missing 'id'.")
        else:
            pid = prot["id"]
            ids = pid if isinstance(pid, list) else [pid]
            chain_ids.extend(str(x) for x in ids)

        if "sequence" not in prot:
            errors.append(f"Protein {i}: missing 'sequence'.")
        else:
            seq = prot["sequence"]
            try:
                _validate_protein_sequence(seq)
            except ValueError as e:
                errors.append(f"Protein {i}: {e}")
            total_residues += len(seq)

    for i, na in enumerate(nucleic_acids):
        if "id" not in na:
            errors.append(f"Nucleic acid {i}: missing 'id'.")
        else:
            nid = na["id"]
            ids = nid if isinstance(nid, list) else [nid]
            chain_ids.extend(str(x) for x in ids)

        na_type = na.get("type", "dna").lower()
        if na_type not in ("rna", "dna"):
            errors.append(f"Nucleic acid {i}: type must be 'rna' or 'dna', got: {na_type!r}")

        if "sequence" not in na:
            errors.append(f"Nucleic acid {i}: missing 'sequence'.")
        else:
            seq = na["sequence"]
            try:
                _validate_nucleic_acid_sequence(seq, na_type)
            except ValueError as e:
                errors.append(f"Nucleic acid {i}: {e}")
            total_residues += len(seq)

    for i, lig in enumerate(ligands):
        if "id" not in lig:
            errors.append(f"Ligand {i}: missing 'id'.")
        else:
            lid = lig["id"]
            ids = lid if isinstance(lid, list) else [lid]
            chain_ids.extend(str(x) for x in ids)

        has_smiles = "smiles" in lig and lig["smiles"]
        has_ccd = "ccd" in lig and lig["ccd"]
        if has_smiles and has_ccd:
            errors.append(f"Ligand {i}: must have either 'smiles' or 'ccd', not both.")
        elif not has_smiles and not has_ccd:
            errors.append(f"Ligand {i}: must have either 'smiles' or 'ccd'.")

    # Check constraint references
    for i, pc in enumerate(pocket_constraints or []):
        binder = pc.get("binder")
        if binder and str(binder) not in chain_ids:
            warnings.append(f"Pocket constraint {i}: binder '{binder}' not found in chain IDs.")
        contacts = pc.get("contacts", [])
        for j, contact in enumerate(contacts):
            if isinstance(contact, list) and len(contact) >= 1:
                if str(contact[0]) not in chain_ids:
                    warnings.append(f"Pocket constraint {i}, contact {j}: chain '{contact[0]}' not found in chain IDs.")

    # Duplicate chain IDs
    seen: set[str] = set()
    for cid in chain_ids:
        if cid in seen:
            warnings.append(f"Duplicate chain ID: '{cid}'.")
        seen.add(cid)

    # Complexity estimate
    complexity = "low"
    if total_residues > 1000:
        complexity = "high"
    elif total_residues > 300:
        complexity = "medium"

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "num_chains": len(chain_ids),
        "total_residues": total_residues,
        "estimated_complexity": complexity,
    }
