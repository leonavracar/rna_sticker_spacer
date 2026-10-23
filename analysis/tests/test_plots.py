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
