"""Optional standalone HTML molecular-contact networks from cached graph NPZs.

Requires pyvis (not imported unless called). Plots molecular graphs as undirected
networks: nodes are RNA molecules, weighted edges represent qualifying molecule
pairs with at least min_contacts accepted particle contacts.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .style import PlotStyle


def save_pyvis_network(
    graph_npz: str | Path,
    molecule_lookup_npy: str | Path,
    output_html: str | Path, *,
    style: PlotStyle,
    frame: int = -1,
    min_contacts: int = 2,
    species_by_length: dict[int, str] | None = None,
    max_nodes: int = 500,
) -> Path:
    """Write a portable HTML network viewer, colored by RNA species.

    This uses molecule_contact_edges.npz and molecule_lookup.npy only.
    It does NOT read GSDs or any framewise CSV. The selected frame is a saved
    GSD *frame index*, not a timestep.
    """
    try:
        from pyvis.network import Network
    except ImportError as exc:
        raise ImportError("Install optional dependency pyvis to export HTML networks") from exc

    if min_contacts < 1:
        raise ValueError("min_contacts must be positive")
    lookup = np.load(molecule_lookup_npy)
    lengths = np.bincount(np.asarray(lookup, dtype=int))
    n_molecules = len(lengths)
    if n_molecules > max_nodes:
        raise ValueError(
            f"Network contains {n_molecules} nodes (limit {max_nodes}); "
            "select a smaller demo or explicitly increase max_nodes."
        )
    with np.load(graph_npz) as graph:
        frames = graph["frame_indices"].astype(int)
        if not len(frames):
            raise ValueError("Graph archive has no saved frames")
        chosen = int(frames[frame]) if frame < 0 else int(frame)
        if chosen not in frames:
            raise ValueError(f"Frame {chosen} not present in graph archive")
        mask = (graph["frame"] == chosen) & (graph["n_contacts"] >= min_contacts)
        edges = list(zip(
            graph["mol_a"][mask].astype(int),
            graph["mol_b"][mask].astype(int),
            graph["n_contacts"][mask].astype(int),
        ))

    degrees = np.zeros(n_molecules, dtype=int)
    contact_counts = np.zeros(n_molecules, dtype=int)
    for a, b, weight in edges:
        degrees[a] += 1
        degrees[b] += 1
        contact_counts[a] += weight
        contact_counts[b] += weight

    net = Network(
        height="720px", width="100%", directed=False, notebook=False,
        bgcolor="#111827", font_color="#E6EDF5", cdn_resources="in_line",
    )
    net.barnes_hut(gravity=-6500, central_gravity=.25, spring_length=115)
    for molecule in range(n_molecules):
        length = int(lengths[molecule])
        species = (species_by_length or {}).get(length, f"{length} nt RNA")
        color = style.species_colors.get(species, "#9AA6B6")
        net.add_node(
            int(molecule),
            label="", color=color,
            size=5 + min(15, float(np.sqrt(degrees[molecule])) * 3),
            title=(f"RNA {molecule} · {species}<br>"
                   f"Chain length: {length} nt<br>"
                   f"Graph degree: {degrees[molecule]}<br>"
                   f"Intermolecular contacts: {contact_counts[molecule]}"),
        )
    for a, b, weight in edges:
        net.add_edge(
            int(a), int(b), width=1 + min(7, np.sqrt(weight)),
            color="#8693A9",
            title=f"{weight} accepted particle contacts",
            value=int(weight),
        )
    net.show_buttons(filter_=["physics"])
    target = Path(output_html)
    target.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(target), open_browser=False)
    return target
