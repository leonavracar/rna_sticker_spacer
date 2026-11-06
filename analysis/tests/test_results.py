"""Regression tests for wide exports and experiment-specific measurements.

Run from the repository root: python -m pytest analysis/tests -q
No real trajectories or GSD/freud dependencies are needed for these tests.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.framewise import component_frame_rows, summarize_numeric
from src.mixtures import mixture_cluster_rows
from src.results import export_sweep, metrics_from_analysis, save_checkpoint
from src.single_chain import single_chain_framewise


def metadata(trajectory: str = "trajectory_0", sequence: str = "mixed", molecule_file: str = "cag10x235_cag47x50.json") -> dict[str, object]:
    return {
        "folder_name": "0_E_att_2_density_0.005",
        "traj_replicate": trajectory,
        "molecule_file": molecule_file,
        "sequence_class": sequence,
        "repeat_count": 10 if sequence in {"repeat", "scrambled"} else None,
        "scramble_id": None,
        "E_att": 2.0,
        "angle_k_ppp": 2.5,
        "density": .005,
        "n_molecules": 285,
        "n_particles": 14100,
        "nm_per_sim_length_unit": .6,
        "conc_mol_per_L": .00077,
        "conc_nuc_per_L": .0384,
        "late_fraction": .10,
        "min_contacts": 2,
    }


def test_wide_columns_preserve_ranked_species_and_contact_type() -> None:
    info = metadata()
    summary = {**info, "n_base_pairs": 1477.76, "n_edges": 309.695,
               "inter_particle_contacts_total": 1245.935,
               "mol_contacts_total": 309.695}
    bins = pd.DataFrame([
        {"scheme": "particle_percentage", "bin": "mono", "fraction_particles": .15},
        {"scheme": "particle_percentage", "bin": ">=50%", "fraction_particles": .85},
    ])
    ranked = pd.DataFrame([
        {"rank": "largest", "species": "cag10", "n_molecules": 31,
         "n_particles": 930, "fraction_total_particles": 930 / 14100},
        {"rank": "largest", "species": "cag47", "n_molecules": 32,
         "n_particles": 4512, "fraction_total_particles": 4512 / 14100},
    ])
    contacts = pd.DataFrame([
        {"type": "cag10__cag47", "n_inter_contacts": 455,
         "fraction_inter_contacts": .375}
    ])
    largest = pd.DataFrame([
        {"n_molecules": 63, "n_particles": 5442,
         "fraction_total_particles": 5442 / 14100,
         "n_edges": 190, "mean_degree": 380 / 63}
    ])
    metrics = metrics_from_analysis(
        summary, bins, {
            "ranked_clusters": ranked, "cluster_contacts": contacts,
            "largest_cluster": largest,
        }
    )
    assert metrics["n_base_pairs"] == 1477.76
    assert metrics["cluster_bin_particle_percentage_monomer"] == .15
    assert metrics["cluster_bin_particle_percentage_ge_50pct"] == .85
    assert metrics["rank_largest_cag10_n_molecules"] == 31
    assert metrics["rank_largest_cag47_n_particles"] == 4512
    assert metrics["largest_contacts_cag10_cag47_n_inter_contacts"] == 455
    assert metrics["largest_n_molecules"] == 63
    assert metrics["largest_mean_degree"] == pytest.approx(380 / 63)
    assert "inter_particle_contacts_total" not in metrics
    assert "mol_contacts_total" not in metrics


def test_checkpoint_export_one_row_per_replicate_and_energy_join(tmp_path: Path) -> None:
    sweep = tmp_path / "52b_from_frame"
    db = sweep / "analysis.sqlite"
    for i, pairs in enumerate((100., 110., 120.)):
        info = metadata(trajectory=f"trajectory_{i}")
        save_checkpoint(db, "mixture", "52b_from_frame", info,
                        {"n_base_pairs": pairs, "n_edges": 10 + i})
    save_checkpoint(db, "mixture", "52b_from_frame", metadata(),
                    {"E_pot": 1234.}, part="energy")

    folder, count = export_sweep(db, sweep)
    replicates = pd.read_csv(folder / "replicates.csv")
    summary = pd.read_csv(folder / "summary.csv")
    assert count == 3
    assert len(replicates) == 3
    assert len(summary) == 1
    assert "metric_group" not in replicates
    assert "source_sweep" not in replicates
    assert "system_id" not in replicates
    assert "scramble_id" not in replicates
    assert "sequence_class" not in replicates
    assert replicates.loc[0, "E_pot"] == 1234.
    assert np.isnan(replicates.loc[1, "E_pot"])
    assert summary.loc[0, "n"] == 3
    assert summary.loc[0, "n_trajectories"] == 3
    assert summary.loc[0, "n_base_pairs_mean"] == pytest.approx(110.)
    assert summary.loc[0, "n_base_pairs_std"] == pytest.approx(10.)
    assert summary.loc[0, "n_base_pairs_p10"] == pytest.approx(102.)
    assert summary.loc[0, "n_base_pairs_p90"] == pytest.approx(118.)
    assert not (tmp_path / "compiled").exists()


def test_rq2_scrambles_are_independent_units(tmp_path: Path) -> None:
    sweep = tmp_path / "51b_repeat_number_rand"
    db = sweep / "analysis.sqlite"
    for i, (sequence, value) in enumerate((
        ("scramble_A.json", 0.),
        ("scramble_A.json", 20.),
        ("scramble_B.json", 30.),
    )):
        info = metadata(
            trajectory=f"trajectory_{i}", sequence="scrambled",
            molecule_file=sequence
        )
        save_checkpoint(db, "length_patterning", sweep.name, info,
                        {"mean_edge_weight": value})

    export_sweep(db, sweep)
    replicate = pd.read_csv(sweep / "replicates.csv")
    summary = pd.read_csv(sweep / "summary.csv")
    assert len(replicate) == 3
    assert "molecule_file" in replicate
    assert "sequence_class" in summary
    assert "molecule_file" not in summary
    assert len(summary) == 1
    assert summary.loc[0, "n"] == 2
    assert summary.loc[0, "n_trajectories"] == 3
    assert summary.loc[0, "mean_edge_weight_mean"] == pytest.approx(20.)


def test_late_window_is_ten_percent_of_frames() -> None:
    frames = pd.DataFrame({"frame": list(range(20)),
                           "n_base_pairs": list(range(20))})
    result = summarize_numeric(frames, .10, ["n_base_pairs"])
    assert result["n_base_pairs"] == 18.5


def test_rq1_rq2_cluster_bins_and_ranks_conserve_material() -> None:
    lengths = np.array([30, 30, 30, 30])
    frames = {0: (
        np.array([0, 1]), np.array([1, 2]), np.array([2, 1])
    )}
    graph, bins, ranks = component_frame_rows(
        frames, lengths, min_contacts=2,
        with_recurrence=False, with_bins=True, with_ranks=True
    )
    assert graph.loc[0, "n_edges"] == 1
    for _, group in bins.groupby("scheme"):
        assert group["fraction_particles"].sum() == pytest.approx(1.)
    largest = ranks.loc[ranks["rank"] == "largest"].iloc[0]
    assert largest["n_molecules"] == 2
    assert largest["fraction_particles"] == pytest.approx(.5)


def test_single_chain_mfe_precision_and_recall() -> None:
    npz = {"frame_indices": np.array([0]),
           "qi_0": np.array([0, 1]),
           "pi_0": np.array([9, 8])}
    reference = {(1, 10), (2, 7)}
    measured = single_chain_framewise(npz, reference, index_offset=1)
    assert measured.loc[0, "mfe_matches"] == 1
    assert measured.loc[0, "extra_pairs"] == 1
    assert measured.loc[0, "precision"] == .5
    assert measured.loc[0, "recall"] == .5


def test_mixture_contacts_and_largest_cluster() -> None:
    lengths = np.array([30, 30, 141, 141])
    species = np.array(["cag10", "cag10", "cag47", "cag47"])
    graph = {
        "frame_indices": np.array([0]),
        "frame": np.array([0, 0, 0]),
        "mol_a": np.array([0, 1, 2]),
        "mol_b": np.array([2, 3, 3]),
        "n_contacts": np.array([3, 2, 4]),
    }
    intra = {
        "frame_indices": np.array([0]),
        "frame": np.array([], dtype=int),
        "mol_id": np.array([], dtype=int),
        "n_intra_contacts": np.array([], dtype=int),
    }
    ranked, contacts, largest, system = mixture_cluster_rows(
        graph, lengths, species, min_contacts=2, intra_npz=intra,
        pairable_sites_by_species={"cag10": 20, "cag47": 94}
    )
    assert largest.loc[0, "n_molecules"] == 4
    assert largest.loc[0, "n_edges"] == 3
    assert largest.loc[0, "mean_degree"] == pytest.approx(1.5)
    mixed = contacts[contacts["type"] == "cag10__cag47"].iloc[0]
    assert mixed["n_inter_contacts"] == 5
    assert mixed["fraction_inter_contacts"] == pytest.approx(5 / 9)
    assert system.loc[0, "n_clusters_mixed"] == 1
    assert "drop_frac_cag10" not in system.columns
    assert ranked["fraction_total_particles"].sum() == pytest.approx(1.)
