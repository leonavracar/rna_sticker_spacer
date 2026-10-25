"""Export molecular contact networks as standalone HTML files."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..cluster_distribution import ranked_components
from .style import PlotStyle

RANK_LABELS = {
    "largest": "Top 1",
    "second": "Top 2",
    "third": "Top 3",
    "other_multimers": "Other multimers",
    "monomers": "Monomers",
}


def _frame_edges(
    graph_npz: str | Path, frame: int = -1, min_contacts: int = 2,
) -> tuple[int, list[tuple[int, int, int]]]:
    """Return the contacts in one cached trajectory frame."""
    with np.load(graph_npz) as data:
        frames = data["frame_indices"].astype(int)
        if not len(frames):
            raise ValueError("Graph archive contains no saved frames")
        chosen = int(frames[frame]) if frame < 0 else int(frame)
        if chosen not in frames:
            raise ValueError(f"Frame {chosen} is not in the graph archive")
        mask = (data["frame"] == chosen) & (data["n_contacts"] >= min_contacts)
        edges = [
            (int(a), int(b), int(w))
            for a, b, w in zip(
                data["mol_a"][mask],
                data["mol_b"][mask],
                data["n_contacts"][mask],
            )
        ]
    return chosen, edges


def _component_groups(n_molecules: int, edges: list[tuple[int, int, int]]):
    """Return the contact graph, component classes and component sizes."""
    import networkx as nx

    graph = nx.Graph()
    graph.add_nodes_from(range(n_molecules))
    graph.add_weighted_edges_from(edges)

    categories, sizes = {}, {}
    for component in nx.connected_components(graph):
        n = len(component)
        category = ("monomer" if n == 1 else "dimer" if n == 2
                    else "oligomer" if n <= 4 else "large_cluster")
        for molecule in component:
            categories[molecule] = category
            sizes[molecule] = n
    return graph, categories, sizes


def _rank_members(graph: nx.Graph, lengths: np.ndarray) -> dict[str, set[int]]:
    """Rank connected components by particle count, as in the cluster plots."""
    import networkx as nx

    components = [np.asarray(sorted(c), dtype=int)
                  for c in nx.connected_components(graph)]
    return {rank: set(ids.tolist()) for rank, ids in ranked_components(components, lengths)}


def available_network_ranks(
    graph_npz: str | Path,
    molecule_lookup_npy: str | Path, *,
    frame: int = -1,
    min_contacts: int = 2,
) -> list[str]:
    """List nonempty cluster ranks in a saved frame."""
    lengths = np.bincount(np.load(molecule_lookup_npy).astype(int))
    _, edges = _frame_edges(graph_npz, frame, min_contacts)
    graph, _, _ = _component_groups(len(lengths), edges)
    return [rank for rank, members in _rank_members(graph, lengths).items()
            if members]


def _contact_type(species_a: str, species_b: str) -> str:
    """Return the base-pair contact class for a molecular edge."""
    if species_a == species_b:
        return f"{species_a}_{species_a}"
    if {species_a, species_b} == {"cag10", "cag47"}:
        return "cag10_cag47"
    return "other"


def _legend(style: PlotStyle, title: str, frame: int,
            molecule_count: int, edge_count: int,
            rank_summary: list[str]) -> str:
    """Render the color key above the network."""
    species = style.species_colors
    contacts = style.contact_colors
    dot = lambda color: (
        f'<span style="display:inline-block;width:11px;height:11px;'
        f'border-radius:50%;background:{color};margin-right:6px"></span>'
    )
    bond = lambda color: (
        f'<span style="display:inline-block;width:21px;height:3px;'
        f'background:{color};margin:0 6px 3px 0"></span>'
    )
    return (
        '<div style="position:absolute;top:12px;left:12px;z-index:100;'
        'background:rgba(255,255,255,.94);padding:12px 15px;'
        'border:1px solid #ddd;border-radius:6px;'
        'font:13px Arial,sans-serif;color:#333;line-height:1.8">'
        f'<b>{title}</b> · frame {frame}<br>'
        f'{molecule_count} molecules · {edge_count} edges<br>'
        f'{dot(species["cag10"])}CAG10&nbsp;&nbsp;'
        f'{dot(species["cag47"])}CAG47<br>'
        f'{bond(contacts["cag10_cag10"])}10–10&nbsp;&nbsp;'
        f'{bond(contacts["cag10_cag47"])}10–47&nbsp;&nbsp;'
        f'{bond(contacts["cag47_cag47"])}47–47'
        + (('<hr style="border:0;border-top:1px solid #ddd">'
            + '<br>'.join(rank_summary)) if rank_summary else '')
        + '</div>'
    )


def save_pyvis_network(
    graph_npz: str | Path,
    molecule_lookup_npy: str | Path,
    output_html: str | Path, *,
    style: PlotStyle,
    frame: int = -1,
    min_contacts: int = 2,
    species_by_length: dict[int, str] | None = None,
    trajectory_gsd: str | Path | None = None,
    rank: str = "all",
) -> Path:
    """Save a stable PyVis network for a frame and optional cluster rank.

    Molecule colors identify RNA species; edge colors identify contact types.
    Connected components are ranked by their nucleotide counts.
    """
    from pyvis.network import Network
    import networkx as nx

    if min_contacts < 1:
        raise ValueError("min_contacts must be positive")
    if rank != "all" and rank not in RANK_LABELS:
        raise ValueError(f"Unknown cluster rank: {rank}")

    lengths = np.bincount(np.load(molecule_lookup_npy).astype(int))
    species_by_length = species_by_length or {30: "cag10", 141: "cag47"}
    species = {
        i: species_by_length.get(int(length), f"{int(length)} nt")
        for i, length in enumerate(lengths)
    }
    chosen, edges = _frame_edges(graph_npz, frame, min_contacts)

    if trajectory_gsd is not None and frame == -1:
        import gsd.hoomd
        with gsd.hoomd.open(str(trajectory_gsd), "r") as trajectory:
            last_frame = len(trajectory) - 1
        if chosen != last_frame:
            raise ValueError(
                f"Last cached frame is {chosen}, but the trajectory ends "
                f"at frame {last_frame}. Regenerate the graph with stride=1."
            )

    graph, _, sizes = _component_groups(len(lengths), edges)
    ranked = _rank_members(graph, lengths)
    ranks_by_molecule = {
        molecule: RANK_LABELS[key]
        for key, members in ranked.items() for molecule in members
    }
    if rank != "all":
        if not ranked[rank]:
            raise ValueError(f"No molecules in rank {rank} at frame {chosen}")
        graph = graph.subgraph(ranked[rank]).copy()

    # Compute the layout once; HTML interaction does not run force simulation.
    n = len(graph)
    positions = nx.spring_layout(
        graph, seed=42, iterations=250, scale=850,
        k=1.8 / np.sqrt(max(n, 1)), weight="weight",
    )
    net = Network(
        height="850px", width="100%", notebook=False,
        directed=False, bgcolor="#FFFFFF", font_color="#222222",
        cdn_resources="in_line",
    )
    for molecule in graph.nodes:
        kind = species[molecule]
        color = style.species_colors.get(kind, "#888888")
        degree = graph.degree[molecule]
        x, y = positions[molecule]
        net.add_node(
            int(molecule), label="", x=float(x), y=float(y),
            physics=False, fixed=True,
            shape="dot", size=7 + min(11, 2.5 * np.sqrt(degree)),
            color={"background": color, "border": color},
            title=(f"Molecule {molecule}<br>"
                   f"RNA: {kind}<br>"
                   f"Rank: {ranks_by_molecule[molecule]}<br>"
                   f"Cluster size: {sizes[molecule]} molecules<br>"
                   f"Degree: {degree}"),
        )
    for a, b, data in graph.edges(data=True):
        kind_a, kind_b = species[a], species[b]
        contact_type = _contact_type(kind_a, kind_b)
        color = style.contact_colors.get(contact_type, "#999999")
        weight = int(data["weight"])
        net.add_edge(
            int(a), int(b), color=color,
            width=.7 + min(3.0, np.sqrt(weight) * .45),
            title=f"{kind_a} – {kind_b}: {weight} contacts",
        )
    net.set_options('{"physics":{"enabled":false},'
                    '"interaction":{"hover":true,"dragNodes":false,'
                    '"navigationButtons":true,"zoomView":true},'
                    '"edges":{"smooth":false},'
                    '"nodes":{"borderWidth":0}}')

    output = Path(output_html)
    output.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(output), open_browser=False)
    html = output.read_text(encoding="utf-8")
    title = "All clusters" if rank == "all" else RANK_LABELS[rank]
    total_particles = int(lengths.sum())
    rank_summary = [
        f"{RANK_LABELS[key]}: {len(members)} molecules, "
        f"{sum(int(lengths[i]) for i in members) / total_particles:.0%} particles"
        for key, members in ranked.items() if members
    ] if rank == "all" else []
    html = html.replace(
        "</body>",
        _legend(style, title, chosen, graph.number_of_nodes(),
                graph.number_of_edges(), rank_summary) + "\n</body>",
    )
    output.write_text(html, encoding="utf-8")
    return output
