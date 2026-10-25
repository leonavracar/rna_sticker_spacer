"""PyVis visualizations of molecular contact graphs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .style import PlotStyle

COMPONENT_COLORS = {
    "monomer": "#57A8F0",
    "dimer": "#AE8AC9",
    "oligomer": "#45B58B",
    "large_cluster": "#FF5968",
}


def _component_groups(n_molecules: int, edges: list[tuple[int, int, int]]):
    """Classify molecules by the size of their connected component."""
    import networkx as nx

    graph = nx.Graph()
    graph.add_nodes_from(range(n_molecules))
    graph.add_weighted_edges_from(edges)

    groups = {}
    sizes = {}
    for members in nx.connected_components(graph):
        size = len(members)
        group = ("monomer" if size == 1 else "dimer" if size == 2
                 else "oligomer" if size <= 4 else "large_cluster")
        for molecule in members:
            groups[molecule] = group
            sizes[molecule] = size
    return graph, groups, sizes


def _frame_edges(
    graph_npz: str | Path, frame: int, min_contacts: int,
) -> tuple[int, list[tuple[int, int, int]]]:
    """Load qualifying molecular contacts for one saved frame."""
    with np.load(graph_npz) as data:
        frames = data["frame_indices"].astype(int)
        if not len(frames):
            raise ValueError("Graph archive contains no frames")
        chosen = int(frames[frame]) if frame < 0 else int(frame)
        if chosen not in frames:
            raise ValueError(f"Frame {chosen} is not in the graph archive")
        mask = (data["frame"] == chosen) & (data["n_contacts"] >= min_contacts)
        edges = [
            (int(a), int(b), int(w))
            for a, b, w in zip(data["mol_a"][mask],
                               data["mol_b"][mask], data["n_contacts"][mask])
        ]
    return chosen, edges

def save_pyvis_network(
    graph_npz: str | Path,
    molecule_lookup_npy: str | Path,
    output_html: str | Path, *,
    style: PlotStyle,
    frame: int = -1,
    min_contacts: int = 2,
    species_by_length: dict[int, str] | None = None,
    trajectory_gsd: str | Path | None = None,
    color_by: str = "component",
) -> Path:
    """Export one frame of a trajectory's molecular contact network.

    The final saved frame is used by default. Supply the GSD path to check
    that the cached last frame is also the trajectory's actual last frame.
    """
    try:
        from pyvis.network import Network
    except ImportError as exc:
        raise ImportError("Install pyvis to export interactive molecular graphs") from exc

    if min_contacts < 1:
        raise ValueError("min_contacts must be positive")
    if color_by not in {"component", "species"}:
        raise ValueError("color_by must be 'component' or 'species'")

    lookup = np.load(molecule_lookup_npy)
    lengths = np.bincount(np.asarray(lookup, dtype=int))
    n_molecules = len(lengths)

    chosen, edges = _frame_edges(graph_npz, frame, min_contacts)

    if trajectory_gsd is not None and frame == -1:
        import gsd.hoomd
        with gsd.hoomd.open(str(trajectory_gsd), "r") as trajectory:
            last_frame = len(trajectory) - 1
        if chosen != last_frame:
            raise ValueError(
                f"Last cached frame is {chosen}, but the GSD ends at {last_frame}. "
                "Recompute this trajectory's graph using stride=1."
            )

    graph, groups, component_sizes = _component_groups(n_molecules, edges)
    degrees = dict(graph.degree())

    net = Network(
        height="850px", width="100%", bgcolor="white",
        font_color="#222222", notebook=False,
        directed=False, cdn_resources="in_line",
    )
    net.barnes_hut(
        gravity=-1000, central_gravity=.35, spring_length=70,
        spring_strength=.08, damping=.35, overlap=.25,
    )
    for molecule in graph.nodes:
        degree = degrees[molecule]
        group = groups[molecule]
        length = int(lengths[molecule])
        species = (species_by_length or {}).get(length, f"{length} nt")
        color = (COMPONENT_COLORS[group] if color_by == "component"
                 else style.species_colors.get(species, "#AAAAAA"))
        base = {"monomer": 12, "dimer": 18,
                "oligomer": 22, "large_cluster": 24}[group]
        net.add_node(
            int(molecule), label="",
            size=min(55, base + 3 * degree) if degree else 12,
            color={"background": color, "border": "#222222"},
            title=(f"Molecule {molecule}<br>"
                   f"RNA: {species}<br>"
                   f"Degree: {degree}<br>"
                   f"Component: {component_sizes[molecule]} molecules"),
        )

    for a, b, weight in edges:
        color = ("rgba(219,74,89,0.45)"
                 if groups[a] == "large_cluster" else "rgba(90,90,90,0.35)")
        net.add_edge(
            a, b, value=weight, width=1 + weight,
            color=color, title=f"{weight} base-pair contacts",
        )

    net.set_options(json.dumps({
        "nodes": {
            "shape": "dot", "borderWidth": 1,
            "font": {"size": 14, "color": "#222222"},
        },
        "edges": {"smooth": {"enabled": True, "type": "continuous"}},
        "physics": {
            "enabled": True,
            "stabilization": {"enabled": True, "iterations": 500},
            "barnesHut": {
                "gravitationalConstant": -1000,
                "centralGravity": .35,
                "springLength": 70,
                "springConstant": .08,
                "damping": .35,
                "avoidOverlap": .25,
            },
        },
        "interaction": {
            "hover": True, "navigationButtons": True, "keyboard": True,
            "dragNodes": True,
        },
    }))

    target = Path(output_html)
    target.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(target), open_browser=False)
    return target
