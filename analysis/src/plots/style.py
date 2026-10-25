"""Style settings and labels shared by the plotting functions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Any

# Preserve the original seven-bin palette from analysis_scripts/plots_ps.py.
BIN_COLORS = (
    "#2166ac", "#67a9cf", "#d1e5f0", "#fff176",
    "#f4a582", "#d6604d", "#b2182b",
)
# Ranked-cluster colors (largest to monomers).
RANK_COLORS = ("#EF3659", "#FF695C", "#FFA08B", "#E7BC93", "#59A8D6")
ATTRACTION_COLORS = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
)
SPECIES_COLORS = {"cag10": "#2FB5D9", "cag47": "#F53F58"}
CONTACT_COLORS = {
    "cag10_cag10": "#2FB5D9",
    "cag10_cag47": "#A881C2",
    "cag47_cag47": "#F53F58",
}
MOLECULE_BINS = (
    "monomer", "dimer", "trimer", "4_to_9",
    "10_to_19", "20_to_49", "50_plus",
)
PARTICLE_BINS = (
    "monomer", "lt_2pct", "2_to_lt_3pct", "3_to_lt_10pct",
    "10_to_lt_20pct", "20_to_lt_50pct", "ge_50pct",
)
RANKS = ("largest", "second", "third", "other_multimers", "monomers")

LABELS: dict[str, str] = {
    "E_att": r"$E_{\mathrm{att}}$",
    "angle_k_ppp": r"$k_{\mathrm{bbb}}$",
    "density": r"$\rho$",
    "repeat_count": "CAG repeats",
    "n_base_pairs": "Base-pair contacts",
    "n_intra_contacts": "Intramolecular contacts",
    "n_inter_contacts": "Intermolecular contacts",
    "n_edges": "Molecular graph edges",
    "mean_degree": "Mean molecular degree",
    "mean_edge_weight": "Mean contacts per qualifying edge",
    "molecule_recurrence": "Molecular partner recurrence",
    "particle_recurrence": "Particle-contact recurrence",
    "mfe_matches": "MFE-matching base pairs",
    "extra_pairs": "Non-MFE base pairs",
    "precision": "Precision",
    "recall": "Recall",
    "Rg_nm": r"$R_g$ (nm)",
    "sequence_class": "Sequence class",
    "fraction": "Fraction of system particles",
    "largest_fraction_total_particles": "Largest-cluster particle fraction",
    "largest_mean_degree": "Largest-cluster mean degree",
    "largest_n_edges": "Largest-cluster edges",
    "largest_n_molecules": "Largest-cluster molecules",
    "largest": "Top 1",
    "second": "Top 2",
    "third": "Top 3",
    "other_multimers": "Other multi",
    "monomers": "Monomers",
    "monomer": "Monomers",
    "dimer": "Dimers",
    "trimer": "Trimers",
    "4_to_9": "4–9",
    "10_to_19": "10–19",
    "20_to_49": "20–49",
    "50_plus": "50+",
    "lt_2pct": "<2%",
    "2_to_lt_3pct": "2–<3%",
    "3_to_lt_10pct": "3–<10%",
    "10_to_lt_20pct": "10–<20%",
    "20_to_lt_50pct": "20–<50%",
    "ge_50pct": "≥50%",
    "cag10": "(CAG)10",
    "cag47": "(CAG)47",
    "repeat": "Periodic",
    "scrambled": "Scrambled",
}


@dataclass(frozen=True)
class PlotStyle:
    """Plot colors, sizes and labels."""
    font_size: float = 10
    title_size: float = 13
    panel_title_size: float = 11
    label_size: float = 11
    tick_size: float = 9
    legend_size: float = 9
    show_grid: bool = True
    grid_alpha: float = 0.22
    legend_loc: str = "best"
    point_size: float = 28
    point_alpha: float = 0.65
    mean_lw: float = 2.2
    band_alpha: float = 0.20
    stack_alpha: float = 0.92
    jitter: float = 0.08
    dpi: int = 180
    cmap: str = "magma"
    box_facecolor: str = "white"
    box_edgecolor: str = "#444444"
    box_median_color: str = "#2C2C2C"
    box_whisker_color: str = "#555555"
    box_point_color: str = "#616161"
    box_point_alpha: float = 0.6
    box_point_size: float = 12
    box_width: float = 0.56
    stack_colors: tuple[str, ...] = BIN_COLORS
    rank_colors: tuple[str, ...] = RANK_COLORS
    attraction_colors: tuple[str, ...] = ATTRACTION_COLORS
    species_colors: Mapping[str, str] = field(default_factory=lambda: dict(SPECIES_COLORS))
    contact_colors: Mapping[str, str] = field(default_factory=lambda: dict(CONTACT_COLORS))
    label_map: Mapping[str, str] = field(default_factory=lambda: dict(LABELS))

    def mpl_rc(self) -> dict[str, Any]:
        return {
            "font.size": self.font_size,
            "figure.titlesize": self.title_size,
            "axes.titlesize": self.panel_title_size,
            "axes.labelsize": self.label_size,
            "xtick.labelsize": self.tick_size,
            "ytick.labelsize": self.tick_size,
            "legend.fontsize": self.legend_size,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": self.dpi,
        }

    def attraction_palette(self, values) -> dict:
        """Use one consistent attraction color across every validation panel."""
        return {
            value: self.attraction_colors[i % len(self.attraction_colors)]
            for i, value in enumerate(values)
        }

    def label(self, field: str) -> str:
        if field in self.label_map:
            return self.label_map[field]
        for suffix in ("_mean", "_std", "_p10", "_p90"):
            if field.endswith(suffix):
                return self.label(field[:-len(suffix)])
        return field.replace("_", " ").capitalize()
