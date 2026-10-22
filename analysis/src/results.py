"""Checkpoint per-replicate results and export tidy, plotting-ready CSVs."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EXPERIMENT_NAMES = {
    "single_chain": "validation",
    "reentrance": "rq1_reentrance",
    "length_patterning": "rq2_length_patterning",
    "mixture": "rq3_mixtures",
}

METADATA_FIELDS = {
    "run_id", "folder_name", "traj_replicate", "source_sweep",
    "E_att", "angle_k_ppp", "density", "molecule_file",
    "n_molecules", "n_particles", "nm_per_sim_length_unit",
    "conc_mol_per_L", "conc_nuc_per_L", "late_fraction",
    "min_contacts", "sequence_class", "repeat_count",
}

GROUP_FIELDS = [
    "experiment", "sequence_class", "repeat_count",
    "E_att", "angle_k_ppp", "density", "n_molecules",
    "n_particles", "late_fraction", "min_contacts",
    "metric_group", "metric", "category", "species",
]

REPLICATE_FIELDS = [
    "experiment", "source_sweep", "folder_name", "traj_replicate",
    "molecule_file", "sequence_class", "repeat_count",
    "E_att", "angle_k_ppp", "density", "n_molecules", "n_particles",
    "nm_per_sim_length_unit", "conc_mol_per_L", "conc_nuc_per_L",
    "late_fraction", "min_contacts", "metric_group", "metric",
    "category", "species", "value",
]

SUMMARY_FIELDS = GROUP_FIELDS + [
    "mean", "std", "p10", "p90", "n", "n_trajectories",
]


def metrics_from_analysis(
    summary: dict[str, Any],
    bins: pd.DataFrame,
    extra_tables: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    """Convert compact trajectory summaries into a common long-form schema."""
    metrics: list[dict[str, Any]] = []

    def add(group: str, metric: str, value: Any,
            category: str = "", species: str = "") -> None:
        if value is None or not pd.api.types.is_number(value):
            return
        number = float(value)
        if not np.isfinite(number):
            return
        metrics.append({
            "metric_group": group,
            "metric": metric,
            "category": str(category),
            "species": str(species),
            "value": number,
        })

    for name, value in summary.items():
        if name in METADATA_FIELDS:
            continue
        pair = re.search(r"_(cag\d+_cag\d+)$", name)
        if pair:
            add("contact_types", name[:pair.start()], value, category=pair.group(1))
            continue
        sp = re.search(r"_(cag\d+)$", name)
        if sp:
            add("species_metrics", name[:sp.start()], value, species=sp.group(1))
            continue
        add("scalar", name, value)

    if not bins.empty:
        for row in bins.to_dict("records"):
            add(
                f"cluster_bins_{row['scheme']}", "fraction_particles",
                row["fraction_particles"], category=row["bin"],
            )

    for table_name, table in extra_tables.items():
        if table.empty:
            continue
        for row in table.to_dict("records"):
            category = str(row.get("rank", row.get("type", "")))
            species = str(row.get("species", ""))
            for name, value in row.items():
                if name in METADATA_FIELDS or name in {"frame", "rank", "type", "species"}:
                    continue
                sp = re.search(r"_(cag\d+)$", name)
                if sp:
                    add(table_name, name[:sp.start()], value, category, sp.group(1))
                else:
                    add(table_name, name, value, category, species)

    return metrics


def save_checkpoint(
    db_path: Path,
    experiment: str,
    source_sweep: str,
    metadata: dict[str, Any],
    metrics: list[dict[str, Any]],
    part: str = "analysis",
) -> None:
    """Atomically replace one replicate's results; retain optional energy separately."""
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


def _replicate_rows(db_path: Path, experiment: str) -> pd.DataFrame:
    if not db_path.is_file():
        return pd.DataFrame(columns=REPLICATE_FIELDS)
    with sqlite3.connect(db_path) as connection:
        records = connection.execute(
            """SELECT source_sweep, metadata_json, metrics_json
               FROM replicate_results WHERE experiment = ?
               ORDER BY source_sweep, folder_name, traj_replicate, part""",
            (experiment,),
        ).fetchall()

    rows: list[dict[str, Any]] = []
    for source_sweep, metadata_json, metrics_json in records:
        metadata = json.loads(metadata_json)
        base = {
            field: metadata.get(field)
            for field in REPLICATE_FIELDS
            if field not in {"metric_group", "metric", "category", "species", "value"}
        }
        base["experiment"] = experiment
        base["source_sweep"] = source_sweep
        for measurement in json.loads(metrics_json):
            rows.append({**base, **measurement})

    return pd.DataFrame(rows, columns=REPLICATE_FIELDS)


def _aggregate(replicates: pd.DataFrame) -> pd.DataFrame:
    """Give each scrambled sequence equal weight, irrespective of trajectory count."""
    if replicates.empty:
        return pd.DataFrame(columns=SUMMARY_FIELDS)

    data = replicates.copy()
    scrambled = data["sequence_class"].eq("scrambled")
    data["unit_id"] = (
        data["source_sweep"].astype(str) + "/" +
        data["folder_name"].astype(str) + "/" +
        data["traj_replicate"].astype(str)
    )
    data.loc[scrambled, "unit_id"] = (
        data.loc[scrambled, "source_sweep"].astype(str) + "/" +
        data.loc[scrambled, "molecule_file"].astype(str)
    )
    data["trajectory_id"] = (
        data["source_sweep"].astype(str) + "/" +
        data["folder_name"].astype(str) + "/" +
        data["traj_replicate"].astype(str)
    )

    # Each scramble's trajectories are averaged first; the scramble, not the
    # individual trajectory, is the independent unit in the comparison.
    unit = (
        data.groupby(GROUP_FIELDS + ["unit_id"], dropna=False)["value"]
        .mean().reset_index()
    )
    grouped = unit.groupby(GROUP_FIELDS, dropna=False)["value"]
    summary = grouped.agg(mean="mean", std="std", n="count").reset_index()
    percentiles = grouped.agg(
        p10=lambda x: float(np.percentile(x, 10)),
        p90=lambda x: float(np.percentile(x, 90)),
    ).reset_index()
    summary = summary.merge(percentiles, on=GROUP_FIELDS, how="left")
    trajectory_counts = (
        data.groupby(GROUP_FIELDS, dropna=False)["trajectory_id"]
        .nunique().rename("n_trajectories").reset_index()
    )
    summary = summary.merge(trajectory_counts, on=GROUP_FIELDS, how="left")
    return summary[SUMMARY_FIELDS]


def export_experiment(
    db_path: Path, output_root: Path, experiment: str
) -> tuple[Path, int]:
    """Write exactly two tidy tables per experiment from the checkpoint DB."""
    if experiment not in EXPERIMENT_NAMES:
        raise ValueError(f"Unknown experiment: {experiment}")
    replicates = _replicate_rows(db_path, experiment)
    destination = output_root / "compiled" / EXPERIMENT_NAMES[experiment]
    destination.mkdir(parents=True, exist_ok=True)
    replicates.to_csv(destination / "replicates.csv", index=False)
    _aggregate(replicates).to_csv(destination / "summary.csv", index=False)
    return destination, len(replicates)
