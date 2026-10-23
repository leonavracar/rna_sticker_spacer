"""Small unit tests for axes-first plotting and RQ1/RQ2 projection logic."""
from __future__ import annotations

from functools import partial
from pathlib import Path
from types import SimpleNamespace
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.plots.style import PlotStyle, MOLECULE_BINS, RANKS
from src.plots.data import melt_metrics, filter_rows
from src.plots.figures import (
    plot_cluster_bins, plot_largest_contact_composition,
    plot_metric_series, plot_mixture_ranked_bars,
)
from src.plots.panels import panels_by
from src.plots.projections import repeat_count, projection_xy


STYLE = PlotStyle()


def test_labels_and_melt():
    assert STYLE.label("E_att").startswith("$")
    assert STYLE.label("Rg_nm_mean") == STYLE.label("Rg_nm")
    source = pd.DataFrame({"E_att": [2., 4.], "n_edges": [3, 7],
                           "mean_degree": [1., 2.]})
    melted = melt_metrics(source, ["n_edges", "mean_degree"], id_vars=["E_att"])
    assert len(melted) == 4
    assert set(melted["metric"]) == {"n_edges", "mean_degree"}
    assert list(filter_rows(source, E_att=2.)["n_edges"]) == [3]


def test_partial_panels_plot_summary_series():
    summary = pd.DataFrame({
        "angle_k_ppp": [0., 0., 2.5, 2.5],
        "E_att": [2., 4., 2., 4.],
        "n_edges_mean": [10., 12., 17., 20.],
        "n_edges_std": [1., 2., 1., 3.],
    })
    fig, axes = panels_by(
        summary, by="angle_k_ppp",
        plot_fn=partial(plot_metric_series, style=STYLE, metric="n_edges"),
        style=STYLE,
    )
    assert len(axes[0, 0].lines) == 1
    assert len(axes[0, 1].lines) == 1
    plt.close(fig)


def test_rq1_bins_stack_to_one():
    d = {"E_att": [2., 4.]}
    for idx, category in enumerate(MOLECULE_BINS):
        d[f"cluster_bin_molecule_count_{category}_mean"] = (
            [.5, .3] if idx == 0 else
            [.5, .7] if idx == 6 else [0., 0.]
        )
    fig, ax = plt.subplots()
    plot_cluster_bins(pd.DataFrame(d), ax, scheme="molecule_count", style=STYLE)
    assert len(ax.patches) == 14
    plt.close(fig)


def test_mixture_contacts_are_largest_cluster_fractions():
    d = pd.DataFrame({
        "E_att": [2., 4.],
        "largest_contacts_cag10_cag10_fraction_inter_contacts_mean": [.1, .2],
        "largest_contacts_cag10_cag47_fraction_inter_contacts_mean": [.5, .5],
        "largest_contacts_cag47_cag47_fraction_inter_contacts_mean": [.4, .3],
        "largest_contacts_cag10_cag10_n_inter_contacts_mean": [100, 200],
        "largest_contacts_cag10_cag47_n_inter_contacts_mean": [500, 500],
        "largest_contacts_cag47_cag47_n_inter_contacts_mean": [400, 300],
    })
    fig, ax = plt.subplots()
    plot_largest_contact_composition(d, ax, style=STYLE, annotate_counts=True)
    assert ax.get_ylim()[1] == 1
    plt.close(fig)


def test_ranked_mixture_material_uses_whole_system_denominator():
    values = {}
    for rank in RANKS:
        for species in ("cag10", "cag47"):
            frac = .5 if rank == "largest" else 0.
            values[f"rank_{rank}_{species}_fraction_total_particles_mean"] = [frac, frac]
            values[f"rank_{rank}_{species}_n_molecules_mean"] = [2., 2.]
    df = pd.DataFrame({"E_att": [2., 4.], **values})
    fig, ax = plt.subplots()
    plot_mixture_ranked_bars(df, ax, style=STYLE, conditions=(2., 4.))
    assert len(ax.patches) == len(RANKS) * 2 * 2
    plt.close(fig)


def test_final_frame_projection_and_repeat_labels():
    frame = SimpleNamespace(
        configuration=SimpleNamespace(box=np.array([10., 10., 10., 0., 0., 0.])),
        particles=SimpleNamespace(
            position=np.array([[0., 0., 0.], [1., 1., 0.], [-1., -1., 0.]])
        ),
    )
    hist, x_edges, y_edges = projection_xy(frame, bins=10, sigma=0)
    assert hist.sum() == 3
    assert len(x_edges) == 11
    assert repeat_count("cag47x100.json") == 47
    assert repeat_count("CAG10_scramble.json") == 10



def test_eatt_series_uses_evenly_spaced_categorical_ticks():
    from src.plots.figures import plot_metric_series, plot_contact_balance
    source = pd.DataFrame({
        "E_att": [1., 2., 4., 6., 8., 12., 16., 32.],
        "n_edges_mean": [0, 1, 2, 3, 4, 5, 6, 7],
        "n_edges_std": [1.] * 8,
    })
    fig, ax = plt.subplots()
    plot_metric_series(source, ax, metric="n_edges", style=STYLE)
    assert ax.get_xticks().tolist() == list(range(8))
    assert [tick.get_text() for tick in ax.get_xticklabels()] == [
        "1", "2", "4", "6", "8", "12", "16", "32"
    ]
    assert ax.lines[0].get_xdata().tolist() == list(range(8))
    plt.close(fig)

    contacts = pd.DataFrame({"E_att": [2., 4., 8.]})
    for metric in ("n_base_pairs", "n_inter_contacts", "n_intra_contacts"):
        contacts[f"{metric}_mean"] = [100., 110., 120.]
        contacts[f"{metric}_std"] = [5., 5., 5.]
    fig, ax = plt.subplots()
    plot_contact_balance(contacts, ax, style=STYLE)
    assert len(ax.lines) == 3
    assert {line.get_label() for line in ax.lines} == {
        "Total", "Intermolecular", "Intramolecular"
    }
    assert ax.get_xticks().tolist() == [0., 1., 2.]
    plt.close(fig)


def test_faceted_cluster_bin_legend_appears_once():
    from src.plots.style import MOLECULE_BINS
    source = {
        "angle_k_ppp": [0., 2.5, 5.],
        "E_att": [2., 2., 2.],
    }
    for i, c in enumerate(MOLECULE_BINS):
        source[f"cluster_bin_molecule_count_{c}_mean"] = (
            [1., 1., 1.] if i == 0 else [0., 0., 0.]
        )
    fig, axes = panels_by(
        pd.DataFrame(source), by="angle_k_ppp",
        plot_fn=partial(plot_cluster_bins, scheme="molecule_count", style=STYLE),
        style=STYLE, legend="last",
    )
    assert sum(ax.get_legend() is not None for ax in axes.flat) == 1
    assert axes.flat[2].get_legend() is not None
    plt.close(fig)


def test_thesis_largest_cluster_figure_is_categorical_bars():
    from src.plots.figures import plot_largest_cluster_metric
    df = pd.DataFrame({
        "E_att": [2., 4., 8.],
        "largest_mean_degree_mean": [5., 4., 7.],
        "largest_n_edges_mean": [310., 653., 688.],
    })
    fig, ax = plt.subplots()
    plot_largest_cluster_metric(
        df, ax, metric="largest_mean_degree",
        annotation="largest_n_edges", annotation_label="edges", style=STYLE,
    )
    assert len(ax.patches) == 3
    assert ax.get_xticks().tolist() == [0., 1., 2.]
    assert len(ax.texts) == 3
    plt.close(fig)


def test_pure_and_mixed_ranked_condition_plot_contract():
    from src.plots.figures import plot_ranked_condition
    from src.plots.style import RANKS
    pure = {"E_att": [2.]}
    mix = {"E_att": [2.]}
    for r in RANKS:
        pure[f"rank_{r}_fraction_particles_mean"] = [
            1. if r == "largest" else 0.
        ]
        for sp in ("cag10", "cag47"):
            mix[f"rank_{r}_{sp}_fraction_total_particles_mean"] = [
                .5 if r == "largest" else 0.
            ]
            mix[f"rank_{r}_{sp}_n_molecules_mean"] = [2.]
    fig, axes = plt.subplots(1, 2)
    plot_ranked_condition(pd.DataFrame(pure), axes[0], style=STYLE)
    plot_ranked_condition(pd.DataFrame(mix), axes[1], style=STYLE,
                          species=("cag10", "cag47"), annotate_molecules=True)
    assert len(axes[0].patches) == 5
    assert len(axes[1].patches) == 10
    plt.close(fig)
