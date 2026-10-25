"""Export molecular contact networks from saved trajectory frames."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .style import PlotStyle

NODE_COLORS = {"cag10": "#56A9D5", "cag47": "#F05252"}
EDGE_COLORS = {
    "cag10_cag10": "#56A9D5",
    "cag10_cag47": "#AC79C3",
    "cag47_cag47": "#F05252",
}


def _frame_edges(
    graph_npz: str | Path, frame: int = -1, min_contacts: int = 2,
) -> tuple[int, list[tuple[int, int, int]]]:
    """Read molecular contacts for one saved frame."""
    with np.load(graph_npz) as data:
        frames = data["frame_indices"].astype(int)
        if len(frames) == 0:
            raise ValueError("Graph archive has no saved frames")
        chosen = int(frames[frame]) if frame < 0 else int(frame)
        if chosen not in frames:
            raise ValueError(f"Frame {chosen} is not in the graph archive")
        mask = (data["frame"] == chosen) & (data["n_contacts"] >= min_contacts)
        edges = [
            (int(a), int(b), int(weight))
            for a, b, weight in zip(
                data["mol_a"][mask], data["mol_b"][mask], data["n_contacts"][mask]
            )
        ]
    return chosen, edges


def _contact_type(species_a: str, species_b: str) -> str:
    """Classify an intermolecular contact by RNA species."""
    if species_a == species_b:
        return f"{species_a}_{species_a}"
    return "cag10_cag47"


def _cluster_graph(edges: list[tuple[int, int, int]]):
    """Build a contact graph containing only molecules with edges."""
    import networkx as nx

    graph = nx.Graph()
    graph.add_weighted_edges_from(edges)
    return graph


def save_pyvis_network(
    graph_npz: str | Path,
    molecule_lookup_npy: str | Path,
    output_html: str | Path, *,
    style: PlotStyle,
    frame: int = -1,
    min_contacts: int = 2,
    species_by_length: dict[int, str] | None = None,
    trajectory_gsd: str | Path | None = None,
) -> Path:
    """Save a single frame's molecular clusters as standalone HTML."""
    import networkx as nx
    from pyvis.network import Network

    if min_contacts < 1:
        raise ValueError("min_contacts must be positive")

    lengths = np.bincount(np.load(molecule_lookup_npy).astype(int))
    species_by_length = species_by_length or {30: "cag10", 141: "cag47"}
    chosen, edges = _frame_edges(graph_npz, frame, min_contacts)

    if trajectory_gsd is not None and frame == -1:
        import gsd.hoomd

        with gsd.hoomd.open(str(trajectory_gsd), "r") as trajectory:
            if chosen != len(trajectory) - 1:
                raise ValueError("The cached graph does not include the final GSD frame")

    graph = _cluster_graph(edges)
    if not graph:
        raise ValueError(f"No molecular clusters at frame {chosen}")

    positions = nx.spring_layout(graph, seed=42, iterations=250, scale=550)
    net = Network(
        height="850px", width="100%", bgcolor="white",
        notebook=False, directed=False, cdn_resources="in_line",
    )

    for molecule in graph:
        kind = species_by_length.get(int(lengths[molecule]), "unknown")
        color = NODE_COLORS.get(kind, "#888888")
        x, y = positions[molecule]
        net.add_node(
            int(molecule), label="", x=float(x), y=float(y),
            shape="dot", size=7 if kind == "cag10" else 11,
            physics=False, fixed=True, borderWidth=1,
            color={
                "background": color, "border": "#FFFFFF",
                "highlight": {"background": color, "border": "#222222"},
            },
            title=(f"Molecule: {molecule}<br>Type: {kind.upper()}<br>"
                   f"Degree: {graph.degree[molecule]}"),
        )

    for a, b, data in graph.edges(data=True):
        kind_a = species_by_length.get(int(lengths[a]), "unknown")
        kind_b = species_by_length.get(int(lengths[b]), "unknown")
        contacts = int(data["weight"])
        color = EDGE_COLORS.get(_contact_type(kind_a, kind_b), "#888888")
        net.add_edge(
            int(a), int(b), smooth=False,
            color={"color": color, "opacity": .72},
            width=1.6 * np.sqrt(contacts),
            title=f"{kind_a.upper()}-{kind_b.upper()}<br>{contacts} particle contacts",
        )

    net.set_options(
        '{"physics":{"enabled":false},'
        '"interaction":{"hover":true,"tooltipDelay":80,"zoomView":true,'
        '"dragView":true},"nodes":{"font":{"size":0}},'
        '"edges":{"smooth":false}}'
    )
    output = Path(output_html)
    output.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(output), open_browser=False)
    return output
