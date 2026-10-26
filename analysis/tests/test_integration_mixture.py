"""Real-GSD integration regression for the checked-in short mixture example.

Run separately with: python -m pytest -m integration -q

The CSVs under example_results/ are a recorded baseline generated from
the research pipeline, *not* an independently validated scientific truth.
Additional physical/conservation assertions guard against tautological checks.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "mixture" / "eatt2"
REFERENCE = ROOT / "example_results" / "mixture"


def _compare_csv_tables(actual: Path, reference: Path) -> pd.DataFrame:
    """Compare all published columns, not merely the existence of CSV files."""
    expected_df = pd.read_csv(reference)
    actual_df = pd.read_csv(actual)
    assert not expected_df.empty, reference
    assert set(actual_df.columns) == set(expected_df.columns), (
        f"Changed exported columns in {actual.name}: "
        f"missing={set(expected_df) - set(actual_df)}, "
        f"new={set(actual_df) - set(expected_df)}"
    )
    # Stable identifiers permit deterministic comparison even if row order changes.
    order_by = [
        field for field in ("folder_name", "traj_replicate", "molecule_file",
                            "E_att", "angle_k_ppp", "density")
        if field in actual_df.columns
    ]
    if order_by:
        actual_df = actual_df.sort_values(order_by).reset_index(drop=True)
        expected_df = expected_df.sort_values(order_by).reset_index(drop=True)
    actual_df = actual_df[expected_df.columns]
    assert_frame_equal(
        actual_df, expected_df, check_dtype=False,
        rtol=1e-6, atol=1e-7,
        check_exact=False,
    )
    return actual_df


@pytest.mark.integration
def test_mixture_gsd_end_to_end(tmp_path: Path) -> None:
    """Rebuild pairing, graph, checkpoints and CSVs from real trajectory frames."""
    if importlib.util.find_spec("gsd") is None or importlib.util.find_spec("freud") is None:
        pytest.skip("Install analysis/requirements-trajectory.txt (gsd and freud)")
    for filename in ("params.json", "trajectory_0.gsd"):
        assert (EXAMPLE / filename).is_file(), f"Missing tracked fixture: {filename}"
    for filename in ("replicates.csv", "summary.csv"):
        assert (REFERENCE / filename).is_file(), f"Missing golden CSV: {filename}"

    # The published example holds files flat in eatt2/, whereas the CLI
    # expects a sweep directory containing one or more run subdirectories.
    sweep = tmp_path / "mixture_example"
    run = sweep / "eatt2"
    run.mkdir(parents=True)
    for filename in ("params.json", "trajectory_0.gsd"):
        shutil.copy2(EXAMPLE / filename, run / filename)

    output_root = tmp_path / "results"
    command = [
        sys.executable,
        str(ROOT / "analysis" / "run_pipeline.py"),
        "--sweep", str(sweep),
        "--output-root", str(output_root),
        "--experiment", "mixture",
        "--stage", "all",
        "--overwrite-npz",
    ]
    run_result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert run_result.returncode == 0, (
        f"Pipeline failed ({run_result.returncode})\n"
        f"STDOUT:\n{run_result.stdout}\nSTDERR:\n{run_result.stderr}"
    )

    generated = output_root / sweep.name
    assert (generated / "analysis.sqlite").is_file()
    assert (generated / "eatt2" / "computed_metrics" / "pairing" /
            "trajectory_0_pairing.npz").is_file()
    assert (generated / "eatt2" / "computed_metrics" / "mol_graph" /
            "trajectory_0_molecule_contact_edges.npz").is_file()

    replicates = _compare_csv_tables(
        generated / "replicates.csv", REFERENCE / "replicates.csv"
    )
    summary = _compare_csv_tables(
        generated / "summary.csv", REFERENCE / "summary.csv"
    )
    assert len(replicates) == len(summary) == 1
    row = replicates.iloc[0]
    assert row["n_molecules"] == 285  # 235 CAG10 + 50 CAG47
    assert row["n_particles"] == 235 * 30 + 50 * 141
    assert row["E_att"] == pytest.approx(2.0)
    assert row["late_fraction"] == pytest.approx(0.1)
    assert row["min_contacts"] == 2
    assert row["n_base_pairs"] >= row["n_inter_contacts"] >= 0
    assert row["n_base_pairs"] >= row["n_intra_contacts"] >= 0
    assert row["n_base_pairs"] == pytest.approx(
        row["n_intra_contacts"] + row["n_inter_contacts"]
    )
    assert row["n_edges"] >= 0
    assert row["mean_degree"] >= 0
    assert summary.iloc[0]["n"] == 1
    assert summary.iloc[0]["n_trajectories"] == 1
    # All particles, irrespective of RNA species, must be accounted for.
    assert row["frac_mol_cag10"] + row["frac_mol_cag47"] == pytest.approx(1.0)
    assert row["largest_fraction_total_particles"] <= 1.0
    for metric in ("n_base_pairs", "n_edges", "n_inter_contacts", "n_intra_contacts"):
        assert np.isfinite(row[metric]), metric
