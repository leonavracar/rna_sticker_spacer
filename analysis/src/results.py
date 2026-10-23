"""Atomic replicate checkpoints and wide, sweep-local analysis tables.

Every row in replicates.csv is one trajectory. Every row in summary.csv is one
experimental condition. Scientific metrics retain readable, stable column names.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# Run provenance stays in SQLite. These are the minimal columns needed when
# analyzing a sweep's CSV without joining a separate metadata table.
REPLICATE_METADATA = [
    "folder_name", "traj_replicate", "molecule_file", "sequence_class",
    "repeat_count", "scramble_id", "E_att", "angle_k_ppp", "density",
    "n_molecules", "n_particles", "nm_per_sim_length_unit",
    "conc_mol_per_L", "conc_nuc_per_L", "late_fraction", "min_contacts",
]

CONDITION_METADATA = [
    "experiment", "molecule_file", "sequence_class", "repeat_count", "E_att",
    "angle_k_ppp", "density", "n_molecules", "n_particles",
    "nm_per_sim_length_unit", "late_fraction", "min_contacts",
]

# Long-format bins contain punctuation such as "<2%"; these labels are made
# valid, readable pandas column names with explicit stable mappings.
BIN_NAMES = {
    "mono": "monomer", "di": "dimer", "tri": "trimer",
    "4-9": "4_to_9", "10-19": "10_to_19", "20-49": "20_to_49",
    "50+": "50_plus",
    "<2%": "lt_2pct", "2-<3%": "2_to_lt_3pct",
    "3-<10%": "3_to_lt_10pct", "10-<20%": "10_to_lt_20pct",
    "20-<50%": "20_to_lt_50pct", ">=50%": "ge_50pct",
}

EXCLUDED_METADATA = set(REPLICATE_METADATA) | {
    "run_id", "source_sweep", "system_id", "experiment",
}

# Already available under the names n_inter_contacts and n_edges.
DUPLICATE_SCALARS = {"inter_particle_contacts_total", "mol_contacts_total"}
STATS = ("mean", "std", "p10", "p90")


def _name(value: Any) -> str:
    """Normalize a cluster/species label for a column name."""
    raw = str(value)
    if raw in BIN_NAMES:
        return BIN_NAMES[raw]
    return re.sub(r"_+", "_", re.sub(r"[^a-zA-Z0-9]+", "_", raw)).strip("_").lower()


def _finite(value: Any) -> float | None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _store(target: dict[str, float], key: str, value: Any) -> None:
    number = _finite(value)
    if number is not None:
        if key in target and not np.isclose(target[key], number):
            raise ValueError(f"Conflicting metric values for {key}")
        target[key] = number


def metrics_from_analysis(
    summary: dict[str, Any],
    bins: pd.DataFrame,
    extra_tables: dict[str, pd.DataFrame],
) -> dict[str, float]:
    """Flatten a completed trajectory into uniquely named scalar columns.

    Pairing/contact calculations are upstream; this only reshapes their
    late-window averages and preserves ranked-cluster and species labels.
    """
    metrics: dict[str, float] = {}

    for key, value in summary.items():
        if key not in EXCLUDED_METADATA and key not in DUPLICATE_SCALARS:
            _store(metrics, key, value)

    if not bins.empty:
        for row in bins.to_dict("records"):
            key = f"cluster_bin_{_name(row['scheme'])}_{_name(row['bin'])}"
            _store(metrics, key, row["fraction_particles"])

    for table_name, table in extra_tables.items():
        if table.empty:
            continue
        for row in table.to_dict("records"):
            if table_name == "ranked_clusters":
                labels = ["rank", _name(row["rank"])]
                if "species" in row and pd.notna(row["species"]):
                    labels.append(_name(row["species"]))
            elif table_name == "cluster_contacts":
                labels = ["largest_contacts", _name(row["type"])]
            elif table_name == "largest_cluster":
                labels = ["largest"]
            else:
                labels = [_name(table_name)]
                for label in ("rank", "type", "species"):
                    if label in row and pd.notna(row[label]):
                        labels.append(_name(row[label]))
            prefix = "_".join(labels)
            for field, value in row.items():
                if field in {"frame", "rank", "type", "species"}:
                    continue
                if field in EXCLUDED_METADATA and not (
                    table_name in {"ranked_clusters", "largest_cluster"}
                    and field in {"n_molecules", "n_particles"}
                ):
                    continue
                _store(metrics, f"{prefix}_{_name(field)}", value)

    return metrics


def save_checkpoint(
    db_path: Path,
    experiment: str,
    source_sweep: str,
    metadata: dict[str, Any],
    metrics: dict[str, float],
    part: str = "analysis",
) -> None:
    """Atomically upsert one replicate; energy is a separate optional part."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS replicate_results (
                experiment TEXT NOT NULL,
                source_sweep TEXT NOT NULL,
                folder_name TEXT NOT NULL,
                traj_replicate TEXT NOT NULL,
                part TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                PRIMARY KEY (experiment, source_sweep, folder_name, traj_replicate, part)
            )"""
        )
        connection.execute(
            """INSERT OR REPLACE INTO replicate_results
               (experiment, source_sweep, folder_name, traj_replicate,
                part, metadata_json, metrics_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                experiment, source_sweep, str(metadata["folder_name"]),
                str(metadata["traj_replicate"]), part,
                json.dumps(metadata), json.dumps(metrics),
            ),
        )


def _checkpoint_metrics(obj: Any) -> dict[str, float]:
    """Read current wide checkpoints and earlier tidy checkpoints."""
    if isinstance(obj, dict):
        return {str(key): float(value) for key, value in obj.items()}
    if not isinstance(obj, list):
        raise ValueError("Unsupported checkpoint metrics payload")

    # Allows re-export of existing SQLite results without reading old CSVs.
    metrics: dict[str, float] = {}
    for row in obj:
        group, field = row["metric_group"], row["metric"]
        category = row.get("category") or ""
        species = row.get("species") or ""
        if group == "scalar":
            key = field
        elif group == "species_metrics":
            key = f"{field}_{species}"
        elif group == "contact_types":
            key = f"{field}_{category}"
        elif group.startswith("cluster_bins_"):
            key = f"cluster_bin_{_name(group.removeprefix('cluster_bins_'))}_{_name(category)}"
        elif group == "ranked_clusters":
            key = f"rank_{_name(category)}"
            if species:
                key += f"_{_name(species)}"
            key += f"_{_name(field)}"
        elif group == "cluster_contacts":
            key = f"largest_contacts_{_name(category)}_{_name(field)}"
        elif group == "largest_cluster":
            key = f"largest_{_name(field)}"
            if species:
                key += f"_{_name(species)}"
        elif group == "energy":
            key = field
        else:
            key = f"{_name(group)}_{_name(category)}_{_name(field)}"
        if key not in DUPLICATE_SCALARS:
            _store(metrics, key, row["value"])
    return metrics


def _replicate_rows(db_path: Path) -> pd.DataFrame:
    if not db_path.is_file():
        return pd.DataFrame(columns=REPLICATE_METADATA)
    with sqlite3.connect(db_path) as connection:
        records = connection.execute(
            """SELECT experiment, source_sweep, folder_name, traj_replicate,
                      part, metadata_json, metrics_json
               FROM replicate_results
               ORDER BY experiment, source_sweep, folder_name, traj_replicate, part"""
        ).fetchall()

    combined: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for experiment, sweep, folder, trajectory, part, meta_json, metrics_json in records:
        identity = (experiment, sweep, folder, trajectory)
        meta = json.loads(meta_json)
        row = combined.setdefault(
            identity, {field: meta.get(field) for field in REPLICATE_METADATA}
        )
        row["experiment"] = experiment
        row.update(_checkpoint_metrics(json.loads(metrics_json)))

    data = pd.DataFrame(combined.values())
    if data.empty:
        return pd.DataFrame(columns=REPLICATE_METADATA)

    # The sweep and scientific experiment are fixed by the output folder / CLI.
    # Scramble columns are only useful for scrambled RQ2 data.
    if data["scramble_id"].isna().all():
        data = data.drop(columns=["scramble_id"])
    if not data["sequence_class"].isin(["scrambled", "repeat"]).any():
        data = data.drop(columns=["sequence_class", "repeat_count"])
    elif not (data["experiment"] == "length_patterning").any():
        data = data.drop(columns=["sequence_class"])
    if data["experiment"].nunique() == 1:
        data = data.drop(columns=["experiment"])

    identifier_columns = [
        name for name in ["experiment", *REPLICATE_METADATA] if name in data.columns
    ]
    metric_columns = sorted(set(data.columns) - set(identifier_columns))
    return data[identifier_columns + metric_columns].sort_values(
        ["folder_name", "traj_replicate"], kind="stable"
    ).reset_index(drop=True)


def _condition_columns(data: pd.DataFrame) -> list[str]:
    """Separate physical conditions without splitting independent scrambles."""
    columns = [
        field for field in CONDITION_METADATA
        if field in data.columns
    ]
    if "sequence_class" in data:
        scrambled_or_repeat = data["sequence_class"].isin(["scrambled", "repeat"])
        # RQ2: same length / sequence class but different scramble file names
        # are independent replicates of the *same* experimental condition.
        if scrambled_or_repeat.any():
            columns.remove("molecule_file")
    return columns


def _aggregate(replicates: pd.DataFrame) -> pd.DataFrame:
    if replicates.empty:
        return pd.DataFrame()

    conditions = _condition_columns(replicates)
    metadata = [
        col for col in ["experiment", *REPLICATE_METADATA]
        if col in replicates.columns
    ]
    metrics = [
        col for col in replicates.columns if col not in metadata
    ]
    data = replicates.copy()
    data["_trajectory_id"] = (
        data["folder_name"].astype(str) + "/" +
        data["traj_replicate"].astype(str)
    )
    data["_unit_id"] = data["_trajectory_id"]
    if "sequence_class" in data:
        is_scrambled = data["sequence_class"].eq("scrambled")
        # A sequence is the independent unit; trajectories are subsamples.
        data.loc[is_scrambled, "_unit_id"] = (
            data.loc[is_scrambled, "molecule_file"].astype(str)
        )
    # Missing metadata must form a group, not cause a condition to disappear.
    units = (
        data.groupby(conditions + ["_unit_id"], dropna=False, sort=True)[metrics]
        .mean().reset_index()
    )
    grouped = units.groupby(conditions, dropna=False, sort=True)
    summary = grouped.size().rename("n").reset_index()
    trajectory_counts = (
        data.groupby(conditions, dropna=False, sort=True)["_trajectory_id"]
        .nunique().rename("n_trajectories").reset_index()
    )
    summary = summary.merge(trajectory_counts, on=conditions, how="left")

    # Keep the calculation explicit and robust to pandas' naming of lambdas.
    means = grouped[metrics].mean().add_suffix("_mean")
    stds = grouped[metrics].std().add_suffix("_std")
    p10 = grouped[metrics].quantile(0.10).add_suffix("_p10")
    p90 = grouped[metrics].quantile(0.90).add_suffix("_p90")
    stats = pd.concat([means, stds, p10, p90], axis=1).reset_index()
    result = summary.merge(stats, on=conditions, how="left")
    columns = [
        f"{metric}_{stat}" for metric in sorted(metrics) for stat in STATS
    ]
    return result[conditions + ["n", "n_trajectories"] + columns]


def export_sweep(db_path: Path, output_sweep: Path) -> tuple[Path, int]:
    """Write one wide replicate table and one wide condition summary."""
    replicates = _replicate_rows(db_path)
    output_sweep.mkdir(parents=True, exist_ok=True)
    replicates.to_csv(output_sweep / "replicates.csv", index=False)
    _aggregate(replicates).to_csv(output_sweep / "summary.csv", index=False)
    return output_sweep, len(replicates)
