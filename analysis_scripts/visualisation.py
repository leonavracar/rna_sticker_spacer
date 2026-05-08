import numpy as np
import math
import matplotlib.pyplot as plt
import hoomd
import gsd
import gsd.hoomd
import plotly.graph_objects as go
import hashlib
import freud
from sklearn.decomposition import PCA

color_dict = {"B":"grey",
              "A" :"red",
              "A1":"red",
              "A2":"red",
              "U" :"blue",
              "U1":"blue",
              "U2":"blue",
              "C" :"yellow",
              "C1":"yellow",
              "C2":"yellow",
              "G" :"green",
              "G1":"green",
              "G2":"green",
              "BREAK": "black"
             }


def bond_positions(positions, bonds):
    """
    Calculate the coordinates of bonds based on the positions of particles in a snapshot.
    
    PARAMETERS
    ----------
    positions : np.array(N,3)
        Array containing x,y,z coordinates of N particles.
    bonds : np.array(N,2)
        Array containing the indices of the particles forming each bond.
    
    OUTPUT
    ------
    bond_coordinates : list(N,2,3)
        A list containing the coordinates of the 2 particles in each bond.
    """
    bond_coordinates = []
    for bond in bonds:
        bond_coordinates.append((positions[bond[0]],
                                 positions[bond[1]]))
    return bond_coordinates


def unwrap_coords(snapshot):
    box = freud.box.Box.from_box(snapshot.configuration.box)
    return box.unwrap(snapshot.particles.position, snapshot.particles.image)

# def unwrap_bonds(snapshot):
#     positions = unwrap_coords(snapshot)
#     bond_coordinates = positions[snapshot.bonds.group[:,0]]-positions[snapshot.bonds.group[:,1]]
#     return bond_coordinates

def unwrap_and_recenter(snapshot):
    '''
    Transforms coordinates of a snapshot to center them around (0,0,0) for visualisation.

    PARAMETERS
    ----------
    snapshot : hoomd.Snapshot
        snapshot of a hoomd simulation
    
    OUTPUT
    ------
    wrapped : array with new positions
    '''
    box = freud.box.Box.from_box(snapshot.configuration.box)
    unwrapped = box.unwrap(snapshot.particles.position, snapshot.particles.image)
    
    # Center the system around (0, 0, 0)
    center = unwrapped.mean(axis=0)
    #shifted = unwrapped - center
    
    # Wrap back into box (this keeps the structure inside the box, but now recentered)
    #wrapped = box.wrap(shifted)
    
    return unwrapped - center # added -center

eigenvectors = [[1,0,0],[0,1,0],[0,0,1]]

def align_coords(positions):
    global eigenvectors    
    pca = PCA(n_components=3)
    pca.fit(positions)

    for i in range(3):
        if np.dot(pca.components_[i],eigenvectors[i])<0:
            pca.components_[i] *= -1
    
    eigenvectors = pca.components_    
    aligned = pca.transform(positions)
    return np.array(aligned)

def align_snapshot(snapshot):
    return align_coords(snapshot.particles.position)

def reinsert_periodically(arr,periodicity):
    '''
    Periodically inserts array elements (used to give illusion of correct bonds.)
    e.g. [0,1,2,3,4,5], periodicity 3 --> [0,1,2,0,3,4,5,3]

    PARAMETERS
    ----------
    arr : np.array or list
        the list to transform
    periodicity : int
        which elements should be inserted

    OUTPUT
    ------
    new : np.array with insertions
    '''
    new = []
    for idx,x in enumerate(arr):
        if idx%periodicity==0 and idx!=0:
            new.append(arr[idx-periodicity])
        new.append(x)
    new.append(arr[-periodicity])
    return np.array(new)

def get_molecule_ranges_from_bonds(snapshot):
    """Return list of (start_idx, end_idx) for each molecule based on bond connectivity."""
    N = snapshot.particles.N
    groups = snapshot.bonds.group

    neighbors = [[] for _ in range(N)]
    for i, j in groups:
        i = int(i); j = int(j)
        neighbors[i].append(j)
        neighbors[j].append(i)

    visited = [False] * N
    components = []

    for start in range(N):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        comp = []
        while stack:
            v = stack.pop()
            comp.append(v)
            for w in neighbors[v]:
                if not visited[w]:
                    visited[w] = True
                    stack.append(w)
        components.append(sorted(comp))

    ranges = [(min(c), max(c)) for c in components]
    ranges.sort()
    return ranges


def insert_breaks_into_coords(coords, break_indices):
    """coords: (N, 3), break_indices: indices AFTER which to insert a break row."""
    parts = []
    last = 0
    for br in break_indices:
        br = int(br)
        parts.append(coords[last: br + 1])
        # insert a dummy point to break the line
        parts.append(np.array([[np.nan, np.nan, np.nan]]))
        last = br + 1
    parts.append(coords[last:])
    return np.vstack(parts)


def insert_breaks_into_list(lst, break_indices, break_value):
    """Insert break_value after each break index in a 1D list."""
    new = []
    last = 0
    for br in break_indices:
        br = int(br)
        new.extend(lst[last: br + 1])
        new.append(break_value)
        last = br + 1
    new.extend(lst[last:])
    return new


def minimum_image(delta, box_lengths):
    """
    Apply minimum-image convention to displacement vectors.
    delta: (..., 3)
    box_lengths: array-like of shape (3,)
    """
    box_lengths = np.asarray(box_lengths, dtype=float)
    return delta - box_lengths * np.round(delta / box_lengths)


def unwrap_molecule_from_bonds(pos, bonds, box_lengths):
    """
    Unwrap one connected molecule using its internal bonds.
    
    pos: (N, 3) wrapped coordinates for the molecule
    bonds: list of (i, j) using LOCAL molecule indices
    box_lengths: (3,)
    """
    n = len(pos)
    if n == 0:
        return pos.copy()

    adj = [[] for _ in range(n)]
    for i, j in bonds:
        adj[i].append(j)
        adj[j].append(i)

    unwrapped = np.full_like(pos, np.nan, dtype=float)
    visited = np.zeros(n, dtype=bool)

    stack = [0]
    unwrapped[0] = pos[0]
    visited[0] = True

    while stack:
        i = stack.pop()
        for j in adj[i]:
            if not visited[j]:
                delta = pos[j] - pos[i]
                delta = minimum_image(delta, box_lengths)
                unwrapped[j] = unwrapped[i] + delta
                visited[j] = True
                stack.append(j)

    # if molecule has isolated particles for some reason
    for k in range(n):
        if not visited[k]:
            unwrapped[k] = pos[k]

    return unwrapped


def get_molecule_bond_groups(frame, mol_ranges):
    """
    Returns bonds for each molecule, converted to local indices.
    Assumes bonds do not connect different molecules.
    """
    all_bonds = frame.bonds.group
    mol_bonds = []

    for start, end in mol_ranges:
        local_bonds = []
        for i, j in all_bonds:
            if start <= i <= end and start <= j <= end:
                local_bonds.append((i - start, j - start))
        mol_bonds.append(local_bonds)

    return mol_bonds


def unwrap_and_pack_molecules(frame):
    """
    Better plotting coordinates for multi-molecule systems under PBC.

    Strategy:
    - unwrap each molecule internally using bonds
    - place molecule COMs near each other using minimum-image convention
    - center the whole system
    """
    pos = np.asarray(frame.particles.position, dtype=float)
    box_lengths = np.asarray(frame.configuration.box[:3], dtype=float)

    mol_ranges = get_molecule_ranges_from_bonds(frame)
    mol_bonds = get_molecule_bond_groups(frame, mol_ranges)

    mol_coords = []
    mol_coms_wrapped = []

    # unwrap each molecule internally
    for (start, end), bonds in zip(mol_ranges, mol_bonds):
        mol_pos = pos[start:end+1]
        mol_unwrapped = unwrap_molecule_from_bonds(mol_pos, bonds, box_lengths)
        mol_coords.append(mol_unwrapped)

        # use wrapped COM only to determine relative placement between molecules
        mol_coms_wrapped.append(np.mean(mol_pos, axis=0))

    mol_coms_wrapped = np.asarray(mol_coms_wrapped)

    # place molecule COMs relative to the first molecule using minimum image
    placed_coords = [mol_coords[0].copy()]
    ref_com_unwrapped = np.mean(mol_coords[0], axis=0)
    ref_com_wrapped = mol_coms_wrapped[0]

    for k in range(1, len(mol_coords)):
        delta = mol_coms_wrapped[k] - ref_com_wrapped
        delta = minimum_image(delta, box_lengths)

        target_com = ref_com_unwrapped + delta
        current_com = np.mean(mol_coords[k], axis=0)

        shift = target_com - current_com
        placed_coords.append(mol_coords[k] + shift)

    new_pos = np.concatenate(placed_coords, axis=0)

    # center whole system together
    system_com = np.mean(new_pos, axis=0)
    new_pos = new_pos - system_com

    return new_pos


def AnimateTrajectory(trajectory, structure_type="directional-branch",
                      aligned=True, unwrapped=True, size=6,
                      type_colors=color_dict, axlim=50, show_background=True):
    '''
    Modified version of Ankits PlotPolymer.AnimateTrajectory
    Plot hoomd simulation frames with slider, color based on particle type.
    '''

    print(len(trajectory))

    types_list = [trajectory[0].particles.types[x]
                  for x in trajectory[0].particles.typeid]
    color_list = [type_colors[x] for x in types_list]
    hover_list = [f"{i} - {t}" for i, t in enumerate(types_list)]

    if unwrapped:
        positions = np.array([unwrap_and_pack_molecules(frame)
                             for frame in trajectory])
    else:
        positions = np.array(
            [frame.particles.position for frame in trajectory])

    if aligned:
        positions = np.array([align_coords(frame) for frame in positions])

    mol_ranges = get_molecule_ranges_from_bonds(trajectory[0])
    break_indices = [end for (_, end) in mol_ranges[:-1]]

    if structure_type == "branch":
        periodicity = 2
        positions = np.array(
            [reinsert_periodically(x, periodicity) for x in positions])
        types_list = reinsert_periodically(types_list, periodicity)
        color_list = reinsert_periodically(color_list, periodicity)
        hover_list = reinsert_periodically(hover_list, periodicity)

        if len(break_indices) > 0:
            p_break_indices = [idx + idx //
                               periodicity for idx in break_indices]
            positions = np.array([
                insert_breaks_into_coords(pos, p_break_indices)
                for pos in positions
            ])
            types_list = insert_breaks_into_list(
                types_list, p_break_indices, "BREAK")
            color_list = insert_breaks_into_list(
                color_list, p_break_indices, type_colors["BREAK"])
            hover_list = insert_breaks_into_list(
                hover_list, p_break_indices, "BREAK")

    elif structure_type == "directional-branch":
        periodicity = 3
        positions = np.array(
            [reinsert_periodically(x, periodicity) for x in positions])
        types_list = reinsert_periodically(types_list, periodicity)
        color_list = reinsert_periodically(color_list, periodicity)
        hover_list = reinsert_periodically(hover_list, periodicity)

        if len(break_indices) > 0:
            p_break_indices = [b + 1 + (b // periodicity)
                               for b in break_indices]
            positions = np.array([
                insert_breaks_into_coords(pos, p_break_indices)
                for pos in positions
            ])
            types_list = insert_breaks_into_list(
                types_list, p_break_indices, "BREAK")
            color_list = insert_breaks_into_list(
                color_list, p_break_indices, type_colors["BREAK"])
            hover_list = insert_breaks_into_list(
                hover_list, p_break_indices, "BREAK")

    max_abs = np.nanmax(np.abs(positions))
    axlim = float(max_abs * 1.05)

    fig = go.Figure()
    for step in range(len(trajectory)):
        x, y, z = positions[step, :, 0], positions[step,
                                                   :, 1], positions[step, :, 2]

        fig.add_trace(go.Scatter3d(
            visible=False,
            x=x,
            y=y,
            z=z,
            hovertext=hover_list,
            marker=dict(color=color_list, size=size, opacity=1),
            line=dict(color="gray"),
        ))

    fig.data[0].visible = True

    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)}]
        )
        step["args"][0]["visible"][i] = True
        steps.append(step)

    sliders = [dict(
        transition={"duration": 300, "easing": "cubic-in-out"},
        active=0,
        pad={"t": 5},
        steps=steps
    )]

    fig.update_layout(
        margin=dict(l=10, r=10),
        width=800, height=800,
        sliders=sliders,
        scene=dict(
            xaxis=dict(range=[-axlim, axlim], showbackground=show_background),
            yaxis=dict(range=[-axlim, axlim], showbackground=show_background),
            zaxis=dict(range=[-axlim, axlim], showbackground=show_background)
        ),
        scene_aspectmode='cube'
    )

    print(len(fig.data))
    return fig


# def AnimateTrajectory(trajectory, structure_type="directional-branch",
#                       aligned=True, unwrapped=True,size=6,
#                       type_colors=color_dict, axlim=50, show_background=True):
#     '''
#     Modified version of Ankits PlotPolymer.AnimateTrajectory
#     Plot hoomd simulation frames with slider, color based on particle type.
        
#     PARAMETERS
#     ----------
#     trajectory : gsd.hoomd.HOOMDTrajectory
#         trajectory of a hoomd simulation
#     structure_type : string
#         which structure_type (chain, directional-branch etc.) to use
#     aligned : bool
#         whether to use PCA alignment in animation
#     unwrapped : bool
#         whether to use unwrapped coordinates (periodic boundary conditions)
#     type_colors : dict
#         dictionary mapping colors to particle types
#     axlim : int 
#         x,y,z axis range in plot will be [-axlim,+axlim]
    
#     OUTPUT
#     ------
#     fig : plotly figure object
#     '''

#     print(len(trajectory))

#     types_list = [trajectory[0].particles.types[x] for x in trajectory[0].particles.typeid] # ['B', 'G1', 'G2']
#     color_list = [type_colors[x] for x in types_list] #['grey', 'green', 'green']
#     hover_list = [f"{x[1]} - {x[0]}" for x in enumerate(types_list)] # ['B - 0',...]

#     positions = np.array([frame.particles.position for frame in trajectory])
    
#     if unwrapped:
#         positions = np.array([unwrap_and_recenter(frame) for frame in trajectory])
#     if aligned:
#         positions = np.array([align_coords(frame) for frame in positions])  
        
#     # --- COMPUTE MOLECULE BREAKS once from first frame ---
#     mol_ranges = get_molecule_ranges_from_bonds(trajectory[0])
#     break_indices = [end for (_, end) in mol_ranges[:-1]]  # after each molecule except last

#     # if len(break_indices) > 0:
#     #     positions = np.array([
#     #         insert_breaks_into_coords(pos, break_indices)
#     #         for pos in positions
#     #     ]) # inserts a [nan, nan, nan] after break_indices
#     #     types_list = insert_breaks_into_list(types_list, break_indices, "BREAK")
#     #     color_list = insert_breaks_into_list(color_list, break_indices, type_colors["BREAK"])
#     #     hover_list = insert_breaks_into_list(hover_list, break_indices, "BREAK")
    
#     # ----  branch / directional-branch ----
    
#     if structure_type=="branch":
#         periodicity=2
#         p_break_indices = [idx + idx // periodicity for idx in break_indices] # adjusting break index for periodicity
#         positions = np.array([reinsert_periodically(x,periodicity) for x in positions])
#         types_list = reinsert_periodically(types_list,periodicity)
#         color_list = reinsert_periodically(color_list,periodicity)
#         hover_list = reinsert_periodically(hover_list,periodicity)

#         if len(break_indices) > 0:
#             p_break_indices = [idx + idx // periodicity for idx in break_indices] # adjusting break index for periodicity
#             positions = np.array([
#                 insert_breaks_into_coords(pos, p_break_indices)
#                 for pos in positions
#                 ]) # inserts a [nan, nan, nan] after p_break_indices
#             types_list = insert_breaks_into_list(types_list, p_break_indices, "BREAK")
#             color_list = insert_breaks_into_list(color_list, p_break_indices, type_colors["BREAK"])
#             hover_list = insert_breaks_into_list(hover_list, p_break_indices, "BREAK")
    
#     elif structure_type=="directional-branch":
#         periodicity = 3
#         positions = np.array([reinsert_periodically(x,periodicity) for x in positions])
#         types_list = reinsert_periodically(types_list,periodicity)
#         color_list = reinsert_periodically(color_list,periodicity)        
#         hover_list = reinsert_periodically(hover_list,periodicity)
        
#         if len(break_indices) > 0:
#             print(break_indices)
#             p_break_indices = [b + 1 + (b // periodicity) for b in break_indices]
#             print(p_break_indices)
#             #p_break_indices = [idx + idx // periodicity for idx in break_indices] # adjusting break index for periodicity
#             positions = np.array([
#                 insert_breaks_into_coords(pos, p_break_indices)
#                 for pos in positions
#                 ]) # inserts a [nan, nan, nan] after p_break_indices
#             types_list = insert_breaks_into_list(types_list, p_break_indices, "BREAK")
#             color_list = insert_breaks_into_list(color_list, p_break_indices, type_colors["BREAK"])
#             hover_list = insert_breaks_into_list(hover_list, p_break_indices, "BREAK")    
    
#     # Calculate axlim from the new particle positions in case of multiple molecules
#     max_abs = np.nanmax(np.abs(positions))
#     axlim = float(max_abs * 1.05)  # small padding

#     fig = go.Figure()
#     for step in range(len(trajectory)):
#         x, y, z = positions[step, :, 0], positions[step, :, 1], positions[step, :, 2]
        
#         # First trace
#         fig.add_trace(go.Scatter3d(
#             visible=False,
#             x=x,
#             y=y,
#             z=z,
#             hovertext=[f"{x[1]} - {x[0]}" for x in enumerate(types_list)],
#             marker=dict(color=color_list, size=size, opacity=1),
#             line=dict(color="gray"), 
#         ))

        
#     fig.data[0].visible = True

#     steps = []
#     for i in range(0, len(fig.data)):
#         step = dict(
#             method="update",
#             args=[{"visible": [False] * len(fig.data)}])
#         step["args"][0]["visible"][i] = True
#         steps.append(step)
#     sliders = [dict(
#                 transition={"duration": 300, "easing": "cubic-in-out"},
#                 active=len(trajectory),
#                 pad={"t": 5},
#                 steps=steps)]
    
#     fig.update_layout(margin=dict(l=10, r=10),
#                       width=800, height=800,
#                       sliders=sliders,
#                       scene=dict(xaxis=dict(range=[-axlim, axlim], showbackground=show_background),
#                                  yaxis=dict(range=[-axlim, axlim], showbackground=show_background),
#                                  zaxis=dict(range=[-axlim, axlim], showbackground=show_background)),
#                       scene_aspectmode='cube')
#     print(len(fig.data))
#     return fig


def plot_data_dictionary(data, x_key=None, y_keys=None):
    """
    Plot a data dictionary as stacked subplots with shared x-axis.
    
    Parameters
    ----------
    data : dict
        Dictionary with keys mapping to numpy arrays.
        First key (or x_key if specified) is used as x-axis.
    x_key : str, optional
        Key to use for x-axis. If None, uses first key in data.
    y_keys : list, optional
        Keys to plot as separate subplots. If None, plots all except x_key.
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive plotly figure with stacked subplots.
    """
    from plotly.subplots import make_subplots
    
    if x_key is None:
        x_key = list(data.keys())[0]
    
    if y_keys is None:
        y_keys = [k for k in data.keys() if k != x_key]
    
    x = data[x_key]
    
    # Create subplots with shared x-axis
    fig = make_subplots(
        rows=len(y_keys),
        cols=1,
        shared_xaxes=True,
        subplot_titles=y_keys
    )
    
    # Add traces for each y_key
    for i, key in enumerate(y_keys, start=1):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=data[key],
                mode='lines',
                name=key
            ),
            row=i, col=1
        )
    
    # Update layout to match your original
    fig.update_layout(
        height=250 * len(y_keys),
        showlegend=False
    )
    
    return fig


def plot_scatter(x, y, xlabel="x", ylabel="y", title=None, figsize=(6, 4), s=5, alpha=0.3, ax=None):
    """
    Create a scatter plot with matplotlib.
    
    Parameters
    ----------
    x : array-like
        X-axis data
    y : array-like
        Y-axis data
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    title : str, optional
        Plot title
    figsize : tuple
        Figure size as (width, height). Default is (6, 4).
    s : int or float
        Scatter point size. Default is 5.
    alpha : float
        Point transparency (0-1). Default is 0.3.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, creates new figure.
    
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    ax.scatter(x, y, s=s, alpha=alpha)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if title:
        ax.set_title(title)
    
    plt.tight_layout()
    
    return ax


def SaveInteractiveTrajectory(trajectory,structure_type="directional-branch",
                              aligned=True, unwrapped=True,
                              type_colors=color_dict, axlim=50, show_background=False,
                              filename="interactive_trajectory.html"):
    """
    Saves an interactive trajectory (the result of AnimateTrajectory) as an HTML file.
    
    PARAMETERS
    ----------
    trajectory : gsd.hoomd.HOOMDTrajectory
        trajectory of a hoomd simulation
    filename : str, optional
        name of the file to save the interactive trajectory, default is "interactive_trajectory.html"
    for other parameters refer to AnimateTrajectory

    OUTPUT
    ------
    None
    """
    print(len(trajectory))
    fig = AnimateTrajectory(trajectory, structure_type=structure_type,
                            aligned=aligned, unwrapped=unwrapped,
                            type_colors=type_colors, axlim=axlim, show_background=show_background)
    fig.write_html(filename)
    return 

def InteractiveTopologyViewer(snapshot,type_colors=color_dict,axlim=50):
    """
    Interactive viewer for topology with particles and bonds visualization.
    
    PARAMETERS
    ----------
    snapshot : gsd.hoomd.Frame
        The snapshot to be visualized.
    type_colors : dict, optional
        A dictionary mapping colors to particle types.
    axlim : int, optional
        The axis range in the plot will be [-axlim, +axlim].
    
    OUTPUT
    ------
    fig : plotly.graph_objects.Figure
        A Plotly figure containing the interactive visualization.
    """
    types_list = [snapshot.particles.types[x] for x in snapshot.particles.typeid]
    color_list = [type_colors[x] for x in types_list]

    fig = go.Figure()
    
    # Plot particles
    fig.add_trace(go.Scatter3d(
        visible=True,
        x=snapshot.particles.position[:,0],
        y=snapshot.particles.position[:,1],
        z=snapshot.particles.position[:,2],
        hovertext=[f"{x[1]} - {x[0]}" for x in enumerate(types_list)],
        marker=dict(color=color_list,size=6,opacity=1),
        mode='markers'))

    # Plot bonds 
    bond_x = []
    bond_y = []
    bond_z = []
    bond_color_list = []
    for i,bond in enumerate(snapshot.bonds.group):
        particle1 = snapshot.particles.position[bond[0]]
        particle2 = snapshot.particles.position[bond[1]]
        bond_x += [particle1[0], particle2[0], None]  # None to break the line
        bond_y += [particle1[1], particle2[1], None]
        bond_z += [particle1[2], particle2[2], None]
        
        if "scaffold" in snapshot.bonds.types[snapshot.bonds.typeid[i]]:
            bond_color_list += ['gray','gray','gray']
        else: 
            bond_color_list += ['black','black','black']

    fig.add_trace(go.Scatter3d(
        visible=True,
        x=bond_x,
        y=bond_y,
        z=bond_z,
        mode='lines',
        line=dict(color=bond_color_list, width=2),
        hoverinfo="none"))    

    fig.update_layout(margin=dict(l=10, r=10),
                    width=800,height=800,
                    scene=dict(xaxis=dict(range=[-axlim,axlim],showbackground=False),
                               yaxis=dict(range=[-axlim,axlim],showbackground=False),
                               zaxis=dict(range=[-axlim,axlim],showbackground=False)),
                    scene_aspectmode='cube')

    fig.update_layout(showlegend=False)

    return fig


def AnimateStickerSpacerTrajectory(
    trajectory,
    aligned=True,
    unwrapped=True,
    size=6,
    type_colors=color_dict,
    show_background=True,
    axlim=None,
):
    """
    Animate a sticker-spacer trajectory for the alternating backbone-base model.

    Assumes particle order within each molecule is:
        B, base, B, base, B, base, ...

    Molecules are drawn as connected chains in particle order, with NaN breaks
    inserted between molecules so separate molecules are not connected.

    PARAMETERS
    ----------
    trajectory : gsd.hoomd.HOOMDTrajectory
        Trajectory to visualize.
    aligned : bool
        If True, align each frame.
    unwrapped : bool
        If True, unwrap periodic coordinates and recenter.
    size : int or float
        Marker size.
    type_colors : dict
        Mapping from particle type name to color.
    show_background : bool
        Whether to show 3D axis panes.
    axlim : float or None
        Fixed axis half-width. If None, compute automatically.

    OUTPUT
    ------
    fig : plotly.graph_objects.Figure
        Interactive 3D animation with slider.
    """
    print(len(trajectory))

    first = trajectory[0]

    # Per-particle metadata from first frame
    types_list = [first.particles.types[t] for t in first.particles.typeid]
    color_list = [type_colors.get(t, "black") for t in types_list]
    hover_list = [f"{i} - {t}" for i, t in enumerate(types_list)]

    # Coordinates for all frames
    if unwrapped:
        positions = np.array([unwrap_and_recenter(frame)
                             for frame in trajectory])
    else:
        positions = np.array(
            [frame.particles.position for frame in trajectory])

    if aligned:
        positions = np.array([align_coords(pos) for pos in positions])

    # Determine molecule boundaries from bonds
    mol_ranges = get_molecule_ranges_from_bonds(first)
    break_indices = [end for (_, end) in mol_ranges[:-1]]

    # Insert NaN breaks between molecules so lines do not connect different molecules
    if len(break_indices) > 0:
        positions = np.array([
            insert_breaks_into_coords(pos, break_indices)
            for pos in positions
        ])
        color_list = insert_breaks_into_list(
            color_list, break_indices, "black")
        hover_list = insert_breaks_into_list(
            hover_list, break_indices, "BREAK")

    # Auto axis limit
    if axlim is None:
        max_abs = np.nanmax(np.abs(positions))
        axlim = float(max_abs * 1.05) if max_abs > 0 else 1.0

    fig = go.Figure()

    for step in range(len(trajectory)):
        x = positions[step, :, 0]
        y = positions[step, :, 1]
        z = positions[step, :, 2]

        fig.add_trace(go.Scatter3d(
            visible=False,
            x=x,
            y=y,
            z=z,
            mode="lines+markers",
            hovertext=hover_list,
            hoverinfo="text",
            marker=dict(
                color=color_list,
                size=size,
                opacity=1,
            ),
            line=dict(
                color="gray",
                width=4,
            ),
        ))

    if len(fig.data) > 0:
        fig.data[0].visible = True

    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)}],
        )
        step["args"][0]["visible"][i] = True
        steps.append(step)

    sliders = [dict(
        transition={"duration": 300, "easing": "cubic-in-out"},
        active=0,
        pad={"t": 5},
        steps=steps,
    )]

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        width=800,
        height=800,
        sliders=sliders,
        scene=dict(
            xaxis=dict(range=[-axlim, axlim], showbackground=show_background),
            yaxis=dict(range=[-axlim, axlim], showbackground=show_background),
            zaxis=dict(range=[-axlim, axlim], showbackground=show_background),
        ),
        scene_aspectmode="cube",
        showlegend=False,
    )

    return fig

def AnimateStickerSpacerTrajectoryVideo(
    trajectory,
    aligned=True,
    unwrapped=True,
    size=6,
    type_colors=color_dict,
    show_background=True,
    axlim=None,
    frame_duration=200,  # ms between frames for auto-play
):
    """
    Animate a sticker-spacer trajectory with a Play/Pause button for recording.
    """
    print(len(trajectory))

    first = trajectory[0]

    # Per-particle metadata from first frame
    types_list = [first.particles.types[t] for t in first.particles.typeid]
    color_list = [type_colors.get(t, "black") for t in types_list]
    hover_list = [f"{i} - {t}" for i, t in enumerate(types_list)]

    # Coordinates for all frames
    if unwrapped:
        positions = np.array([unwrap_and_recenter(frame)
                             for frame in trajectory])
    else:
        positions = np.array(
            [frame.particles.position for frame in trajectory])

    if aligned:
        positions = np.array([align_coords(pos) for pos in positions])

    # Determine molecule boundaries from bonds
    mol_ranges = get_molecule_ranges_from_bonds(first)
    break_indices = [end for (_, end) in mol_ranges[:-1]]

    if len(break_indices) > 0:
        positions = np.array([
            insert_breaks_into_coords(pos, break_indices)
            for pos in positions
        ])
        color_list = insert_breaks_into_list(
            color_list, break_indices, "black")
        hover_list = insert_breaks_into_list(
            hover_list, break_indices, "BREAK")

    # Auto axis limit
    if axlim is None:
        max_abs = np.nanmax(np.abs(positions))
        axlim = float(max_abs * 1.05) if max_abs > 0 else 1.0

    # --- Build figure with frames for animation ---
    fig = go.Figure()

    # Initial trace (frame 0)
    fig.add_trace(go.Scatter3d(
        x=positions[0, :, 0],
        y=positions[0, :, 1],
        z=positions[0, :, 2],
        mode="lines+markers",
        hovertext=hover_list,
        hoverinfo="text",
        marker=dict(color=color_list, size=size, opacity=1),
        line=dict(color="gray", width=4),
    ))

    # Create animation frames
    frames = []
    for step in range(len(trajectory)):
        frames.append(go.Frame(
            data=[go.Scatter3d(
                x=positions[step, :, 0],
                y=positions[step, :, 1],
                z=positions[step, :, 2],
                mode="lines+markers",
                hovertext=hover_list,
                hoverinfo="text",
                marker=dict(color=color_list, size=size, opacity=1),
                line=dict(color="gray", width=4),
            )],
            name=str(step),
        ))
    fig.frames = frames

    # Slider steps
    slider_steps = [
        dict(
            args=[[str(i)], dict(
                frame=dict(duration=frame_duration, redraw=True),
                mode="immediate",
                transition=dict(duration=0),
            )],
            label=str(i),
            method="animate",
        )
        for i in range(len(trajectory))
    ]

    fig.update_layout(
        margin=dict(l=10, r=10, t=50, b=10),
        width=800,
        height=800,
        scene=dict(
            xaxis=dict(range=[-axlim, axlim], showbackground=show_background),
            yaxis=dict(range=[-axlim, axlim], showbackground=show_background),
            zaxis=dict(range=[-axlim, axlim], showbackground=show_background),
        ),
        scene_aspectmode="cube",
        showlegend=False,
        # Play / Pause buttons
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=1.05,
            x=0.5,
            xanchor="center",
            buttons=[
                dict(
                    label="▶ Play",
                    method="animate",
                    args=[None, dict(
                        frame=dict(duration=frame_duration, redraw=True),
                        fromcurrent=True,
                        transition=dict(duration=0),
                    )],
                ),
                dict(
                    label="⏸ Pause",
                    method="animate",
                    args=[[None], dict(
                        frame=dict(duration=0, redraw=False),
                        mode="immediate",
                        transition=dict(duration=0),
                    )],
                ),
            ],
        )],
        # Slider
        sliders=[dict(
            active=0,
            pad=dict(t=40),
            transition=dict(duration=0),
            steps=slider_steps,
        )],
    )

    return fig


# def AnimateStickerSpacer(trajectory,
#                          aligned=True,
#                          unwrapped=True,
#                          size=6,
#                          type_colors=color_dict,
#                          show_background=True):
#     """
#     Animate a simple sticker-spacer trajectory as a connected 3D chain.

#     PARAMETERS
#     ----------
#     trajectory : gsd.hoomd.HOOMDTrajectory
#         Trajectory to visualize.
#     aligned : bool
#         If True, align each frame with PCA.
#     unwrapped : bool
#         If True, unwrap periodic coordinates and recenter.
#     size : int or float
#         Marker size for particles.
#     type_colors : dict
#         Mapping from particle type name to color.
#     show_background : bool
#         Whether to show 3D axis background panes.

#     OUTPUT
#     ------
#     fig : plotly.graph_objects.Figure
#         Interactive slider-based 3D animation.
#     """
#     print(len(trajectory))

#     # Particle metadata from first frame
#     first = trajectory[0]
#     types_list = [first.particles.types[type_id]
#                   for type_id in first.particles.typeid]
#     color_list = [type_colors.get(t, "black") for t in types_list]
#     hover_list = [f"{idx} - {t}" for idx, t in enumerate(types_list)]

#     # Coordinates for all frames
#     if unwrapped:
#         positions = np.array([unwrap_and_recenter(frame)
#                              for frame in trajectory])
#     else:
#         positions = np.array(
#             [frame.particles.position for frame in trajectory])

#     if aligned:
#         positions = np.array([align_coords(pos) for pos in positions])

#     # If there are multiple molecules, insert NaN breaks so lines do not connect them
#     mol_ranges = get_molecule_ranges_from_bonds(first)
#     break_indices = [end for (_, end) in mol_ranges[:-1]]

#     if len(break_indices) > 0:
#         positions = np.array([
#             insert_breaks_into_coords(pos, break_indices)
#             for pos in positions
#         ])
#         color_list = insert_breaks_into_list(
#             color_list, break_indices, "black")
#         hover_list = insert_breaks_into_list(
#             hover_list, break_indices, "BREAK")

#     # Auto axis limit
#     max_abs = np.nanmax(np.abs(positions))
#     axlim = float(max_abs * 1.05) if max_abs > 0 else 1.0

#     fig = go.Figure()

#     for step in range(len(trajectory)):
#         x = positions[step, :, 0]
#         y = positions[step, :, 1]
#         z = positions[step, :, 2]

#         fig.add_trace(go.Scatter3d(
#             visible=False,
#             x=x,
#             y=y,
#             z=z,
#             mode="lines+markers",
#             hovertext=hover_list,
#             hoverinfo="text",
#             marker=dict(
#                 color=color_list,
#                 size=size,
#                 opacity=1,
#             ),
#             line=dict(
#                 color="gray",
#                 width=4,
#             ),
#         ))

#     if len(fig.data) > 0:
#         fig.data[0].visible = True

#     steps = []
#     for i in range(len(fig.data)):
#         step = dict(
#             method="update",
#             args=[{"visible": [False] * len(fig.data)}],
#         )
#         step["args"][0]["visible"][i] = True
#         steps.append(step)

#     sliders = [dict(
#         transition={"duration": 300, "easing": "cubic-in-out"},
#         active=0,
#         pad={"t": 5},
#         steps=steps,
#     )]

#     fig.update_layout(
#         margin=dict(l=10, r=10, t=10, b=10),
#         width=800,
#         height=800,
#         sliders=sliders,
#         scene=dict(
#             xaxis=dict(range=[-axlim, axlim], showbackground=show_background),
#             yaxis=dict(range=[-axlim, axlim], showbackground=show_background),
#             zaxis=dict(range=[-axlim, axlim], showbackground=show_background),
#         ),
#         scene_aspectmode="cube",
#         showlegend=False,
#     )

#     return fig
