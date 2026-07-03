from plotly.subplots import make_subplots
import plotly.graph_objects as go
import json
from pathlib import Path
from typing import Any
import os
import difflib

import numpy as np
import pandas as pd
import gsd.hoomd
import freud
import matplotlib.pyplot as plt
import operator

from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


from analysis_scripts.distances import (
    read_param,
    compute_pairing_summary_metrics,
    compute_pairing_frame_metrics,
    get_pair_array_simple,
    EXPECTED_PAIRS_AGUC30,
)

# TO DO

# Get rid of fill_nan_metrics, replace with simple logic
# Add type hints instead of so many comments
# Close trajectories after open
# Trajectory check, use isinstance instead of whatvr is now
# Velociy check (if velocity is 0)
# Make the comments unified format with comments in visualization.py


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def parse_parameters_file(filepath):
    """
    Parse parameters.txt file into DataFrame (one row per parameter set).
    """
    rows = []
    with open(filepath, 'r') as f:
        for line in f:
            tokens = line.strip().split('--')[1:]
            param_dict = {}
            for token in tokens:
                key, *value = token.strip().split()
                val = value[0]
                if val.replace('.', '', 1).replace('-', '', 1).isdigit():
                    val = float(val)
                    if val.is_integer():
                        val = int(val)
                param_dict[key] = val
            rows.append(param_dict)
    return pd.DataFrame(rows)


def _fmt(val):
    if isinstance(val, float) and val.is_integer():
        return str(int(val))
    return str(val)


def row_to_folder_name(row):
    parts = [f"{col}_{_fmt(val)}" for col, val in row.items()]
    return "_".join(parts)


def param_from_folder_name(raw_name, valid_names):
    """
    Map a raw folder name on disk to the parameter folder name.
    If folder name is "0_E_att..." the param name is without the prefix "E_att..."

    - If raw_name is already a param name (in valid_names), return it.
    - If it looks like '<index>_<params>' and the params part is valid,
      return the params part.
    - Otherwise raise KeyError.
    """
    if raw_name in valid_names:
        return raw_name

    parts = raw_name.split("_", 1)
    if len(parts) == 2 and parts[0].isdigit() and parts[1] in valid_names:
        return parts[1]

    raise KeyError(f"No raw match for folder '{raw_name}'")


def unwrap_frame(frame):
    """
    Unwrap particle positions accounting for periodic boundaries.
    Input: gsd.hoomd.Frame
    Returns: np.ndarray -> Unwrapped positions, shape (N, 3)
    """
    pos = frame.particles.position
    img = frame.particles.image
    box = frame.configuration.box

    Lx, Ly, Lz, xy, xz, yz = box
    unp = pos.copy()
    unp[:, 0] += img[:, 0] * Lx
    unp[:, 1] += img[:, 1] * Ly
    unp[:, 2] += img[:, 2] * Lz
    return unp


def fill_nan_metrics(param_row, rg_scales, rg_fraction, ke_fraction):
    """Fill NaN for all expected metrics."""
    cols = []
    for s in [str(x).replace('.', '_') for x in rg_scales]:
        cols += [
            f'rg_{s}_mean', f'rg_{s}_std',
            f'rg_{s}_last_{rg_fraction}_mean',
            f'rg_{s}_last_{rg_fraction}_std',
            f'rg_{s}_std_ptg_of_mean'
        ]
    cols += [
        f'ke_mean{ke_fraction}', f'ke_std{ke_fraction}',
        f'ke_max_{ke_fraction}', 'inversion_score'
    ]
    for c in cols:
        param_row[c] = np.nan
    return param_row


# ============================================================================
# HELIX SCORE (ANGLES, DIHEDRALS)
# ============================================================================


def helix_score_ppp_stripe(ppp_angles):
    A = np.asarray(ppp_angles)
    mu = np.nanmean(A, axis=0)
    stripe = np.nanstd(mu)
    noise = np.nanmean(np.nanstd(A, axis=0)) + 1e-12
    return float(stripe / noise)


def helix_score_dihedral_coherence(dihes):
    D = np.asarray(dihes)
    dphi = D[:, 1:] - D[:, :-1]
    return float(np.nanmean(np.cos(dphi)))


def helix_score(ppp_angles, stacking_dihes):
    s1 = helix_score_ppp_stripe(ppp_angles)
    s2 = helix_score_dihedral_coherence(stacking_dihes)
    s1n = 1.0 - np.exp(-s1)        # 0..1-ish
    s2n = (s2 + 1.0) / 2.0         # [-1,1] -> [0,1]
    return float(0.5 * (s1n + s2n))


def stacking_dihedral_groups(N_particles: int):
    """
    Helper to get stacking dihes
    """
    return [[x, x-1, x+2, x+3] for x in range(1, N_particles-4, 3)]


# ============================================================================
# RADIUS OF GYRATION
# ============================================================================

def rg_calc(traj_file, scale=1):
    """
    Calculate radius of gyration for each frame.
    Parameters: gsd.hoomd.HOOMDTrajectory, scaling factor for unit conversion
    Returns: np.ndarray -> Rg value per frame, shape (n_frames,)
    Note: scale = 1 as we length scale in analyze_single_trajectory, rg_calc can still be used independently with scale
    """
    if isinstance(traj_file, str):
        traj = gsd.hoomd.open(traj_file, "r")
    else:
        traj = traj_file

    n_frames = len(traj)
    rg_values = np.zeros(n_frames)

    for i in range(n_frames):
        frame = traj[i]

        box = freud.box.Box.from_box(frame.configuration.box)

        positions = box.unwrap(frame.particles.position, frame.particles.image )
        
        com = np.mean(positions, axis=0)
        diff = positions - com
        sq_dist = np.sum(diff**2, axis=1)
        rg = np.sqrt(np.mean(sq_dist))

        # length scaling to nm, based on backbone distance
        rg_values[i] = rg * scale

    # rg_mean = np.mean(rg_values)

    return rg_values


def rg_stats(rg_values, fraction_last=0.2):
    """
    Calculate statistics from Rg trajectory.
    Input: Rg values per frame and fraction of last frames.
    """
    n_frames = len(rg_values)
    cutoff = int((1 - fraction_last) * n_frames)
    rg_last = rg_values[cutoff:]

    mean = np.mean(rg_values)
    std = np.std(rg_values)
    median = np.median(rg_values)
    last_mean = np.mean(rg_last)
    last_std = np.std(rg_last)
    last_median = np.median(rg_last)
    rg_std_ptg_of_mean = std / mean * 100 if mean else np.nan

    return mean, median, std, last_mean, last_median, last_std, rg_std_ptg_of_mean


# ============================================================================
# KINETIC ENERGY
# ============================================================================

def calc_kinetic_energy(snapshot, mass=1.0):
    """
    Calculate total kinetic energy for a single snapshot.
    Input: gsd.hoomd.Frame
    Returns: Total kinetic energy
    """
    v = snapshot.particles.velocity
    if not v.any():
        raise ValueError("No velocity data found in snapshot.")
    KE = 0.5 * mass * np.sum(v**2)
    return KE


def compute_ke_stats(traj_file, mass=1.0, fraction_last=0.2):
    """Compute mean, std, max KE per particle over last fraction of frames."""
    if type(traj_file) == str:
        traj = gsd.hoomd.open(traj_file, "r")
    else:
        traj = traj_file

    N_particles = traj[0].particles.N

    total_frames = len(traj)
    start = int(total_frames * (1 - fraction_last))
    ke_vals = []
    for i in range(start, total_frames):
        ke_vals.append(calc_kinetic_energy(traj[i], mass))
    ke_vals = np.array(ke_vals)

    mean = np.mean(ke_vals) / N_particles
    std = np.std(ke_vals) / N_particles
    max_val = np.max(ke_vals) / N_particles
    return mean, std, max_val


def compute_ke_over_time(traj_file, mass=1.0, per_particle=True):
    """
    Compute kinetic energy for each frame in a trajectory.

    Parameters
    ----------
    traj_file : str or gsd.hoomd.HOOMDTrajectory
        Either a path to a .gsd file or an open trajectory object.
    mass : float, optional
        Per-particle mass (identical for all particles).
    per_particle : bool, optional
        If True, return KE per particle. If False, return total KE.

    Returns
    -------
    np.ndarray
        1D array of kinetic energy values, one per frame.
    """
    if isinstance(traj_file, str):
        traj = gsd.hoomd.open(traj_file, "r")
        close_after = True
    else:
        traj = traj_file
        close_after = False

    N_particles = traj[0].particles.N

    ke_values = []

    for snapshot in traj:
        ke_total = calc_kinetic_energy(snapshot, mass=mass)  # total KE
        if per_particle and N_particles:
            ke_values.append(ke_total / N_particles)
        else:
            ke_values.append(ke_total)

    if close_after:
        traj.close()

    return np.array(ke_values)


# ============================================================================
# DIHEDRALS INVERSION AND STACKING SCORE
# ============================================================================

def calculate_dihedral(p1, p2, p3, p4, box=None):
    """
    Calculate dihedral angle from four 3D points.
    
    Input:3D coordinate vectors
    Returns: Dihedral angle in radians
    """
    if box is None:
        u1, u2, u3 = p2 - p1, p3 - p2, p4 - p3
    else:
        # minimum-image displacement vectors
        u1, u2, u3 = box.wrap(p2 - p1), box.wrap(p3 - p2), box.wrap(p4 - p3)

    y = np.dot(np.linalg.norm(u2) * u1, np.cross(u2, u3))
    x = np.dot(np.cross(u1, u2), np.cross(u2, u3))
    return np.arctan2(y, x)


def calc_all_dihedrals_over_time(traj, dihe_group, stride=10):
    """
    Computes dihedrals for a group (list of quadruplets of particle indices) over time.
    Stride controls frame subsampling. Stride = 10 uses every 10th frame.
    """
    n_frames = len(traj)
    sampled_indices = range(0, n_frames, stride)  # set frames to loop over
    n_samples = len(sampled_indices)
    n_dihedrals = len(dihe_group)

    results = np.zeros((n_samples, n_dihedrals))

    for f_idx, i in enumerate(sampled_indices):  # loop over frames
        frame = traj[i]
        pos = frame.particles.position
        box = freud.box.Box.from_box(frame.configuration.box)
        for d_idx, (a, b, c, d) in enumerate(dihe_group):  # loop over all dihedral groups
            results[f_idx, d_idx] = calculate_dihedral(
                pos[a], pos[b], pos[c], pos[d], box=box)

    return results


def inversion_score(traj_file, stride=10):
    """
    Calculates the fraction of positive directional dihedrals, avg over all dihes.
    """

    opened_here = False
    if isinstance(traj_file, str):
        traj = gsd.hoomd.open(traj_file, 'r')
        opened_here = True
    else:
        traj = traj_file
    N_particles = traj[0].particles.N

    dihe_group = [[x, x+3, x+4, x+5] for x in range(0, N_particles - 4, 3)]
    dihes = calc_all_dihedrals_over_time(traj, dihe_group, stride=stride)

    # Fraction of positive dihedrals per frame, averaged over time
    scores = (dihes > 0).mean(axis=0)
    if opened_here:
        traj.close()
    return np.mean(scores)


def stacking_score(traj_file, stride=10):
    """Takes the absolute values of each stacking dihedral and normalizes it by dividing with pi.
    Then calculates mean of those deviations (e.g. score of 0.4 = 40% on avg deviation).
    Interpretation: closer to 0 = more stacked, 1 = maximum deviation. 

    Note: only suitable for a 3-particles per nucleotide architecture.
    """
    opened_here = False
    if isinstance(traj_file, str):
        traj = gsd.hoomd.open(traj_file, 'r')
        opened_here = True
    else:
        traj = traj_file
    N_particles = traj[0].particles.N

    dihe_group_stack = [
        [i + 1, i, i + 3, i + 4]
        for i in range(0, N_particles - 4, 3)
    ]

    dihes = calc_all_dihedrals_over_time(traj, dihe_group_stack, stride=stride)

    if opened_here:
        traj.close()
    return 1 - np.mean(np.abs(dihes)/np.pi)


def stacking_score_simple(traj_file, stride=10):
    """Takes the absolute values of stacking dihedrals and normalizes by pi.
    Then calculates mean of those deviations.
    Interpretation: closer to 0 = more stacked, 1 = maximum deviation.
    
    Note: Assumes 2 particles per nucleotide and uses bead index 1 of each nucleotide.
    """
    opened_here = False
    if isinstance(traj_file, str):
        traj = gsd.hoomd.open(traj_file, 'r')
        opened_here = True
    else:
        traj = traj_file

    N_particles = traj[0].particles.N
    n_nt = N_particles // 2

    # use bead 1 of four consecutive nucleotides
    dihe_group_stack = [
        [2*i + 1, 2*(i+1) + 1, 2*(i+2) + 1, 2*(i+3) + 1]
        for i in range(n_nt - 3)
    ]

    if not dihe_group_stack:
        if opened_here:
            traj.close()
        return np.nan

    dihes = calc_all_dihedrals_over_time(traj, dihe_group_stack, stride=stride)

    if opened_here:
        traj.close()

    return 1 - np.mean(np.abs(dihes) / np.pi)

# ============================================================================
# STACKING AND INVERSION OVER TIME
# ============================================================================


def stacking_score_over_time(traj_file, stride=1):
    """
    Framewise stacking score for the 3-particles-per-nucleotide model.

    Interpretation:
        closer to 0 = more stacked
        closer to 1 = larger deviation from ideal stacking

    Returns:
        np.ndarray of shape (T,) where T is number of sampled frames.
    """
    opened_here = False
    if isinstance(traj_file, str):
        traj = gsd.hoomd.open(traj_file, 'r')
        opened_here = True
    else:
        traj = traj_file

    try:
        N_particles = traj[0].particles.N
        dihe_group_stack = [
            [i + 1, i, i + 3, i + 4]
            for i in range(0, N_particles - 4, 3)
        ]

        if not dihe_group_stack:
            return np.array([], dtype=float)

        dihes = calc_all_dihedrals_over_time(
            traj, dihe_group_stack, stride=stride)
        return np.mean(np.abs(dihes) / np.pi, axis=1)

    finally:
        if opened_here:
            traj.close()


def inversion_score_over_time(traj_file, stride=1):
    """
    Framewise inversion score for the directional dihedrals.

    Returns:
        np.ndarray of shape (T,) with fraction of positive directional
        dihedrals in each sampled frame.
    """
    opened_here = False
    if isinstance(traj_file, str):
        traj = gsd.hoomd.open(traj_file, 'r')
        opened_here = True
    else:
        traj = traj_file

    try:
        N_particles = traj[0].particles.N
        dihe_group = [[x, x+3, x+4, x+5] for x in range(0, N_particles - 4, 3)]

        if not dihe_group:
            return np.array([], dtype=float)

        dihes = calc_all_dihedrals_over_time(traj, dihe_group, stride=stride)
        return np.mean(dihes > 0, axis=1)

    finally:
        if opened_here:
            traj.close()

# ============================================================================
# ANGLES
# ============================================================================


def calculate_angle(p1, p2, p3, box=None):
    """
    Calculate the angle between three points, accounting for PBC.
    Args:
        p1, p2, p3: Coordinates of the three points forming the angle.
        box : A freud.box.Box object.
        
    Returns: Angle in radians in the range [0, pi].
    """
    v1 = p1-p2
    v2 = p3-p2

    # Vector magnitude
    if box is None:
        v1 = p1 - p2
        v2 = p3 - p2
    else:
        # minimum-image displacement vectors to account for PBC
        v1 = box.wrap(p1 - p2)
        v2 = box.wrap(p3 - p2)

    v1_m = np.linalg.norm(v1)
    v2_m = np.linalg.norm(v2)
    # Dot product
    cosang = np.dot(v1, v2) / (v1_m * v2_m)
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.arccos(cosang)


def get_angle_groups_by_type(frame: gsd.hoomd.Frame) -> dict[str, np.ndarray]:
    """Maps the angles by types e.g. "Phosphate-Phosphate-Base" to their values.
    Args:
        frame: gsd.hoomd.Frame object

    Returns:
        by_name: dictionary of angle types: angle values
    """
    groups = np.array(frame.angles.group)      # (M, 3)
    typeids = np.array(frame.angles.typeid)    # (M,)
    types = list(frame.angles.types)           # list of names

    by_name = {}
    for name in types:
        tid = types.index(name)
        by_name[name] = groups[typeids == tid]
    return by_name


def calc_angles_over_time(traj, angle_group, stride=10):
    """
    Returns angles of angle_group type
    :param stride: Every stride-th frame is used for calculations
    :return: all angles of that type for each frame 
    """
    sampled_indices = range(0, len(traj), stride)
    sampled_indices = list(sampled_indices)

    n_samples = len(sampled_indices)
    n_angles = len(angle_group)

    results = np.zeros((n_samples, n_angles))
    for f_idx, i in enumerate(sampled_indices):
        frame = traj[i]
        pos = frame.particles.position

        box = freud.box.Box.from_box(frame.configuration.box)  # PBC
        for a_idx, (a, b, c) in enumerate(angle_group):

            results[f_idx, a_idx] = calculate_angle(
                pos[a], pos[b], pos[c], box=box)
    return results


def angles_by_type_over_time(traj_file, stride=10, bpp=False):
    """
    Returns all angles for each frame.
    Angles can be accessed by type by using out["Phosphate-Phosphate-Phosphate"]
    Args:
        bpp -> Calculate bpp or exclude it
    """
    traj = gsd.hoomd.open(traj_file, 'r') if isinstance(
        traj_file, str) else traj_file

    by_type = get_angle_groups_by_type(traj[0])

    # Removes the BPP angle from the by_type dictionary and further calculations if bpp=False
    if not bpp:
        by_type.pop("Base-Phosphate-Phosphate", None)

    out = {}
    for name, group in by_type.items():
        out[name] = calc_angles_over_time(traj, group, stride=stride)
    return out


# ============================================================================
# VIENNA RNA 
# ============================================================================

def read_dotps_ubox(ps_path: str, zero_based: bool = True, square: bool = True):
    """
    Reads ViennaRNA dot.ps and extracts ubox entries.

    Returns:
      pairs: list of (i, j, p) with i<j
      P: dict[(i, j)] -> p
    Notes:
      In dot.ps, ubox stores sqrt(p). If square=True, we return p = (sqrt(p))^2.
    """
    pairs = []
    P = {}

    with open(ps_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 4:
                continue
            if parts[3] != "ubox":
                continue

            i = int(parts[0])
            j = int(parts[1])
            val = float(parts[2])

            if zero_based:
                i -= 1
                j -= 1
            if i > j:
                i, j = j, i

            # square because dot.ps stores sqrt(p)
            p = val * val if square else val
            pairs.append((i, j, p))
            P[(i, j)] = p

    return pairs, P

# ============================================================================
# MAIN ANALYSIS FUNCTIONS
# ============================================================================


def analyze_single_trajectory(traj_file,
                              rg_scales=[0.6],
                              rg_fraction=0.2,
                              ke_fraction=0.2,
                              params_path=None,
                              pairing_force_recompute=False):
    """
    Run all analyses on a single trajectory file.
    """
    results = {}
    opened_here = False
    if isinstance(traj_file, str):
        traj = gsd.hoomd.open(traj_file, "r")
        opened_here = True
    else:
        traj = traj_file

    # Determine number of particles
    N_particles = traj[0].particles.N

    # Compute Rg base once
    rg_base = rg_calc(traj, scale=1.0)

    for scale in rg_scales:
        rg_vals = rg_base * scale
        mean, median, std, last_mean, last_median, last_std, rg_std_ptg_of_mean = rg_stats(
            rg_vals, fraction_last=rg_fraction
        )
        scale_str = str(scale).replace('.', '_')
        results[f'rg_{scale_str}_mean'] = mean
        results[f'rg_{scale_str}_median'] = median
        results[f'rg_{scale_str}_std'] = std
        results[f'rg_{scale_str}_last_{rg_fraction}_mean'] = last_mean
        results[f'rg_{scale_str}_last_{rg_fraction}_median'] = last_median
        results[f'rg_{scale_str}_last_{rg_fraction}_std'] = last_std
        results[f'rg_{scale_str}_std_ptg_of_mean'] = rg_std_ptg_of_mean

    # KE uses same traj object
    ke_mean, ke_std, ke_max = compute_ke_stats(
        traj,
        fraction_last=ke_fraction,
    )
    results[f'ke_mean_{ke_fraction}'] = ke_mean
    results[f'ke_std_{ke_fraction}'] = ke_std
    results[f'ke_max_{ke_fraction}'] = ke_max

    # Inversion uses same traj object
    inv_score = inversion_score(traj)
    results['inversion_score'] = inv_score

    # Stacking score
    stack_score = stacking_score(traj)
    results['stacking_score'] = stack_score

    # Helix score (PPP stripe + stacking dihe coherence)
    try:
        angles = angles_by_type_over_time(traj, stride=10, bpp=False)
        ppp = angles["Phosphate-Phosphate-Phosphate"]

        dihe_group_stack = stacking_dihedral_groups(N_particles)
        stacking_dihes = calc_all_dihedrals_over_time(traj, dihe_group_stack, stride = 10)
        results["helix_score"] = helix_score(ppp, stacking_dihes)
    except Exception as e:
        print(f"Helix score failed for {traj_file}: {e}")
        results["helix_score"] = np.nan


# Pairing-derived metrics (computed from cached arrays)
    try:
        if isinstance(traj_file, str) and params_path is not None:
            cutoff = read_param(params_path, "r_att_base")

            traj_path = Path(traj_file)
            pairing_cache_dir = traj_path.parent / "pairing_cache"
            pairing_cache_dir.mkdir(exist_ok=True)

            pairing_cache_path = pairing_cache_dir / \
                f"{traj_path.stem}_pairing_cache.npz"

            pair_results = compute_pairing_summary_metrics(
                traj_file=traj_file,
                cache_path=pairing_cache_path,
                cutoff=cutoff,
                exclude_chain=2,
                stride=5,
                expected=EXPECTED_PAIRS_AGUC30,
                force_recompute=pairing_force_recompute
            )
            results.update(pair_results)
        else:
            results["mean_stuckness"] = np.nan
            results["switch_rate_all"] = np.nan
            results["switch_rate_paired"] = np.nan
            results["mean_disagree_frac"] = np.nan
            results["mean_Q"] = np.nan
    except Exception as e:
        print(f"Pairing metrics failed for {traj_file}: {e}")
        results["mean_stuckness"] = np.nan
        results["switch_rate_all"] = np.nan
        results["switch_rate_paired"] = np.nan
        results["mean_disagree_frac"] = np.nan
        results["mean_Q"] = np.nan

    if opened_here:
        traj.close()

    return results


def analyze_trajectories(base_path,
                         sweep_dir,
                         rg_scales=[0.6],
                         rg_fraction=0.2,
                         ke_fraction=0.2,
                         pairing_force_recompute=False,
                         simple_model=False,
                         energy_from_csv=False):
    """
    Analyze all trajectories under base_path/sweep_dir.
 
    - Matches folders to parameter sets from parameters.txt
    - For each trajectory_*.gsd in each folder: computes Rg (for all rg_scales), KE stats, inversion score
    - For simple 2-bead model: simple_model=True
    - energy_from_csv: if True, reads framewise CSVs from framewise_csvs/ and
      adds last-5% energy summary columns to each row
    - Returns:
        df_per_rep : one row per trajectory (replicate)
        df_avg     : averaged metrics per parameter combination
    """
    base_path = Path(base_path)
    sweep_path = base_path / sweep_dir

    # Parse parameters and build a lookup by folder_name
    df_params = parse_parameters_file(base_path / sweep_dir / "parameters.txt")
    df_params["folder_name"] = df_params.apply(row_to_folder_name, axis=1)
    param_lookup = df_params.set_index("folder_name")

    all_rows = []
    valid_names = set(param_lookup.index)

    # Iterate over all parameter folders inside sweep_dir
    for folder in sorted(p for p in sweep_path.iterdir() if p.is_dir()):
        raw_folder_name = folder.name  # e.g. "0_E_att_back_2.5_..."

        try:
            folder_name = param_from_folder_name(raw_folder_name, valid_names)
        except KeyError:
            print(f"No matching params found for {raw_folder_name}")
            closest = difflib.get_close_matches(
                raw_folder_name.split("_", 1)[-1], list(valid_names), n=1)
            print(
                f"No matching params found for {raw_folder_name}. Closest: {closest}")

            continue

        base_params = param_lookup.loc[folder_name].to_dict()
        base_params.pop("folder_name", None)
        # ensure folder name is available in each row
        base_params["folder_name"] = raw_folder_name

        params_path = folder / "params.json"
        # Pick up all trajectory_*.gsd files
        traj_files = sorted(folder.glob("trajectory_*.gsd"))
        if not traj_files:
            print(f"No trajectories found in {folder_name}")
            continue

        for traj_path in traj_files:
            row = base_params.copy()
            row["traj_replicate"] = traj_path.stem  # e.g. "trajectory_0"

            try:
                if simple_model:
                    results = analyze_single_trajectory_simple(
                        str(traj_path),
                        rg_scales=rg_scales,
                        rg_fraction=rg_fraction,
                        ke_fraction=ke_fraction,
                        params_path=str(params_path),
                        pairing_force_recompute=pairing_force_recompute
                    )
                else:
                    results = analyze_single_trajectory(
                        str(traj_path),
                        rg_scales=rg_scales,
                        rg_fraction=rg_fraction,
                        ke_fraction=ke_fraction,
                        params_path=str(params_path),
                        pairing_force_recompute=pairing_force_recompute
                    )
                row.update(results)

            except Exception as e:
                print(f"Skipping {traj_path}: {e}")
                row = fill_nan_metrics(
                    row, rg_scales, rg_fraction, ke_fraction)
                row["folder_name"] = raw_folder_name
                row["traj_replicate"] = traj_path.stem
                # Fill pairing metrics too
                row["mean_stuckness"] = np.nan
                row["switch_rate_all"] = np.nan
                row["switch_rate_paired"] = np.nan
                row["mean_disagree_frac"] = np.nan
                row["mean_Q"] = np.nan

            # ── energy enrichment from framewise CSV ──────────────────────
            if energy_from_csv:
                csv_path = (
                    folder / "framewise_csvs" /
                    f"{traj_path.stem}_framewise.csv"
                )
                if csv_path.exists():
                    energy_summary = extract_energy_summary_from_csv(csv_path)
                else:
                    print(
                        f"  No framewise CSV found for {traj_path.stem}, filling NaN")
                    energy_summary = extract_energy_summary_from_csv(
                        csv_path  # will fail and return NaN dict
                    )
                row.update(energy_summary)

            all_rows.append(row)

    df_per_rep = pd.DataFrame(all_rows)

    if df_per_rep.empty:
        return df_per_rep, pd.DataFrame()

    # group only by the true parameter columns, not by folder name
    param_cols = [c for c in df_params.columns if c !=
                  "folder_name"] + ["folder_name"]

  
    metric_cols = [
        c for c in df_per_rep.columns
        if c not in param_cols + ["traj_replicate"]
    ]

    df_avg = (
        df_per_rep
        .groupby(param_cols, as_index=False)[metric_cols]
        .mean()
    )

    df_avg.rename(
        columns={c: f"{c}_avg" for c in metric_cols},
        inplace=True
    )

    return df_per_rep, df_avg


def analyze_single_trajectory_simple(traj_file,
                                     rg_scales=[0.6],
                                     rg_fraction=0.2,
                                     ke_fraction=0.2,
                                     params_path=None,
                                     pairing_force_recompute=False):
    """
    Run only architecture-safe analyses on a single trajectory file:
    Rg stats, KE stats, stacking score for 2-bead model,
    and simple pairing-derived metrics.
    """
    results = {}
    opened_here = False

    if isinstance(traj_file, str):
        traj = gsd.hoomd.open(traj_file, "r")
        opened_here = True
    else:
        traj = traj_file

    # Compute Rg base once
    rg_base = rg_calc(traj, scale=1.0)

    for scale in rg_scales:
        rg_vals = rg_base * scale
        mean, median, std, last_mean, last_median, last_std, rg_std_ptg_of_mean = rg_stats(
            rg_vals, fraction_last=rg_fraction
        )
        scale_str = str(scale).replace('.', '_')
        results[f'rg_{scale_str}_mean'] = mean
        results[f'rg_{scale_str}_median'] = median
        results[f'rg_{scale_str}_std'] = std
        results[f'rg_{scale_str}_last_{rg_fraction}_mean'] = last_mean
        results[f'rg_{scale_str}_last_{rg_fraction}_median'] = last_median
        results[f'rg_{scale_str}_last_{rg_fraction}_std'] = last_std
        results[f'rg_{scale_str}_std_ptg_of_mean'] = rg_std_ptg_of_mean

    ke_mean, ke_std, ke_max = compute_ke_stats(
        traj,
        fraction_last=ke_fraction,
    )
    results[f'ke_mean_{ke_fraction}'] = ke_mean
    results[f'ke_std_{ke_fraction}'] = ke_std
    results[f'ke_max_{ke_fraction}'] = ke_max

    results['stacking_score'] = stacking_score_simple(traj)

    # Pairing-derived metrics (computed from cached arrays)
    try:
        if isinstance(traj_file, str) and params_path is not None:
            cutoff = read_param(params_path, "r_att_base")

            traj_path = Path(traj_file)
            pairing_cache_dir = traj_path.parent / "pairing_cache"
            pairing_cache_dir.mkdir(exist_ok=True)

            pairing_cache_path = pairing_cache_dir / f"{traj_path.stem}_pairing_cache.npz"

            pair_results = compute_pairing_summary_metrics(
                traj_file=traj_file,
                cache_path=pairing_cache_path,
                cutoff=cutoff,
                exclude_chain=2,
                stride=5,
                expected=EXPECTED_PAIRS_AGUC30,
                force_recompute=pairing_force_recompute,
                compute_closest=False,
                pair_array_func=get_pair_array_simple,
            )
            results.update(pair_results)
        else:
            results["paired_frac_last_0_2_mean"] = np.nan
            results["mean_stuckness"] = np.nan
            results["switch_rate_all"] = np.nan
            results["switch_rate_paired"] = np.nan
            results["mean_disagree_frac"] = np.nan
            results["mean_Q"] = np.nan
    except Exception as e:
        print(f"Pairing metrics failed for {traj_file}: {e}")
        results["paired_frac_last_0_2_mean"] = np.nan
        results["mean_stuckness"] = np.nan
        results["switch_rate_all"] = np.nan
        results["switch_rate_paired"] = np.nan
        results["mean_disagree_frac"] = np.nan
        results["mean_Q"] = np.nan

    if opened_here:
        traj.close()

    return results

# ============================================================================
# CSV GENERATION
# ============================================================================

def save_analysis_csvs(df_per_rep, df_avg, base_path, prefix="analysis"):
    """
    Save per-replicate and averaged DataFrames to CSV files.   
    Input: pd.DataFrames, dir to save results, prefix for naming
    """
    per_rep_path = os.path.join(base_path, f"{prefix}_per_replicate.csv")
    avg_path = os.path.join(base_path, f"{prefix}_averaged.csv")

    df_per_rep.to_csv(per_rep_path, index=False)
    df_avg.to_csv(avg_path, index=False)

    print(f"Saved per-replicate results to: {per_rep_path}")
    print(f"Saved averaged results to: {avg_path}")

    # Open for looking at right away
    df_per_rep_loaded = pd.read_csv(per_rep_path)
    df_avg_loaded = pd.read_csv(avg_path)
    print(
        f"\nReloaded CSVs ({len(df_per_rep_loaded)} and {len(df_avg_loaded)} rows).")

    return df_per_rep_loaded, df_avg_loaded


# ============================================================================
# POTENTIAL ENERGY
# ============================================================================

# Transform an angle/dihe in +/- pi to 0-pi

def angular_distance_from_pi(phi):
    """
    Smallest circular distance from ±pi.
    Returns values in [0, pi].
    0 means ideal (near ±pi), larger means more distorted.
    """

    return np.pi - np.abs(phi)


def calc_pseudo_dihedral(traj, dihe_group=[[1, 4, 2, 5]], stride=1):
    """
    Calculates a dihedral between the planes of the triangles made by A1-A2-U1-U2.
    """
    dihes = calc_all_dihedrals_over_time(
        traj=traj, dihe_group=dihe_group, stride=stride)
    return dihes



def make_epot_dictionary(traj):
    """
    Splits the logged steps and energy terms into a dictionary with simplified names.
        Returns keys with np.arrays as values:
        - step
        - potential_energy
        - bond
        - angle
        - dihedral
        - sqd
    """
    key_map = {
        "Simulation/timestep": "step",
        "md/compute/ThermodynamicQuantities/potential_energy": "potential_energy",
        "md/bond/Harmonic/energy": "bond",
        "md/angle/Harmonic/energy": "angle",
        "md/dihedral/Periodic/energy": "dihedral",
        "sqd_plugin/sqd_potential/SquareDensity/energy": "sqd",
    }

    data = {new_key: [] for new_key in key_map.values()}

    for frame in traj:
        for old_key, new_key in key_map.items():
            arr = np.asarray(frame.log[old_key])
            data[new_key].append(arr.item() if arr.size == 1 else arr.copy())

    return {k: np.array(v) for k, v in data.items()}

# ============================================================================
# Rg, POTENTIAL ENERGY, KE dictionary
# ============================================================================


def make_energy_dictionary(traj):
    """
    Split logged step and energy terms into a dictionary.

    Supports both:
    - troubleshoot7 / old lumped logging
    - troubleshoot8 / split angle+dihedral logging

    Returns keys:
        step
        potential_energy
        sqd
        bond_total
        bond_bp
        bond_pp
        angle_total
        angle_ppp
        angle_ppb
        angle_bpb
        angle_bpp
        dihedral_total
        dihedral_stacking
        dihedral_direction
    """
    first_log = traj[0].log

    data = {
        "step": [],
        "potential_energy": [],
        "sqd": [],
        "bond_total": [],
        "bond_bp": [],
        "bond_pp": [],
        "angle_total": [],
        "angle_ppp": [],
        "angle_ppb": [],
        "angle_bpb": [],
        "angle_bpp": [],
        "dihedral_total": [],
        "dihedral_stacking": [],
        "dihedral_direction": [],
    }

    # key choices
    has_split_angles = "md/angle/Harmonic_1/energy" in first_log
    has_split_dihedrals = "md/dihedral/Periodic_1/energy" in first_log
    has_split_bonds = "md/bond/Harmonic_1/energy" in first_log

    for frame in traj:
        log = frame.log

        data["step"].append(np.asarray(log["Simulation/timestep"]).item())
        data["potential_energy"].append(
            np.asarray(
                log["md/compute/ThermodynamicQuantities/potential_energy"]).item()
        )
        data["sqd"].append(
            np.asarray(
                log["sqd_plugin/sqd_potential/SquareDensity/energy"]).item()
        )

        if has_split_bonds:
            bp = np.asarray(log["md/bond/Harmonic/energy"]).item()
            pp = np.asarray(log["md/bond/Harmonic_1/energy"]).item()
            data["bond_bp"].append(bp)
            data["bond_pp"].append(pp)
            data["bond_total"].append(bp + pp)
        else:
            b = np.asarray(log["md/bond/Harmonic/energy"]).item()
            data["bond_total"].append(b)
            data["bond_bp"].append(np.nan)
            data["bond_pp"].append(np.nan)

        if has_split_angles:
            a0 = np.asarray(log["md/angle/Harmonic/energy"]).item()
            a1 = np.asarray(log["md/angle/Harmonic_1/energy"]).item()
            a2 = np.asarray(log["md/angle/Harmonic_2/energy"]).item()
            a3 = np.asarray(log["md/angle/Harmonic_3/energy"]).item()

            data["angle_ppp"].append(a0)
            data["angle_ppb"].append(a1)
            data["angle_bpb"].append(a2)
            data["angle_bpp"].append(a3)
            data["angle_total"].append(a0 + a1 + a2 + a3)
        else:
            a = np.asarray(log["md/angle/Harmonic/energy"]).item()
            data["angle_ppp"].append(np.nan)
            data["angle_ppb"].append(np.nan)
            data["angle_bpb"].append(np.nan)
            data["angle_bpp"].append(np.nan)
            data["angle_total"].append(a)

        if has_split_dihedrals:
            d0 = np.asarray(log["md/dihedral/Periodic/energy"]).item()
            d1 = np.asarray(log["md/dihedral/Periodic_1/energy"]).item()

            data["dihedral_stacking"].append(d0)
            data["dihedral_direction"].append(d1)
            data["dihedral_total"].append(d0 + d1)
        else:
            d = np.asarray(log["md/dihedral/Periodic/energy"]).item()
            data["dihedral_stacking"].append(np.nan)
            data["dihedral_direction"].append(np.nan)
            data["dihedral_total"].append(d)

    return {k: np.asarray(v) for k, v in data.items()}


def make_framewise_metrics_dataframe(
    traj_file,
    params_path=None,
    cache_path=None,
    cutoff=None,
    exclude_chain=2,
    expected=EXPECTED_PAIRS_AGUC30,
    stride=1,
    rg_scale=0.6,
    pairing_force_recompute=False,
):
    """
    Build a per-frame dataframe for one trajectory.

    Columns:
        step
        paired_strict
        paired_nonstrict
        disagree
        stacking_score
        inversion_score
        potential_energy
        sqd
        bond_total
        bond_bp
        bond_pp
        angle_total
        angle_ppp
        angle_ppb
        angle_bpb
        angle_bpp
        dihedral_total
        dihedral_stacking
        dihedral_direction
        rg
        ke
        Q
    """
    opened_here = False
    if isinstance(traj_file, str):
        traj = gsd.hoomd.open(traj_file, "r")
        opened_here = True
    else:
        traj = traj_file

    try:
        energy = make_energy_dictionary(traj)

        rg = rg_calc(traj, scale=rg_scale)
        ke = compute_ke_over_time(traj, mass=1.0, per_particle=True)

        # framewise stacking/inversion
        stacking = stacking_score_over_time(traj, stride=stride)
        inversion = inversion_score_over_time(traj, stride=stride)

        # pairing metrics
        if cutoff is None:
            if params_path is None:
                raise ValueError("Provide cutoff or params_path.")
            cutoff = read_param(params_path, "r_att_base")

        if cache_path is None:
            if isinstance(traj_file, str):
                cache_path = Path(traj_file).with_name(
                    f"{Path(traj_file).stem}_pairing_cache.npz"
                )
            else:
                raise ValueError(
                    "Provide cache_path when traj_file is not a path string.")

        pairing = compute_pairing_frame_metrics(
            traj_file=traj_file,
            cache_path=cache_path,
            cutoff=cutoff,
            exclude_chain=exclude_chain,
            expected=expected,
            stride=stride,
            force_recompute=pairing_force_recompute,
            compute_closest=True,
            pair_array_func=None,
        )

        # if stride != 1, align all other arrays to sampled frames
        frame_idx = pairing["frame_indices"]

        df = pd.DataFrame({
            "step": energy["step"][frame_idx],
            "paired_strict": pairing["paired_strict"],
            "paired_nonstrict": pairing["paired_nonstrict"],
            "disagree": pairing["disagree"],
            "stacking_score": stacking,
            "inversion_score": inversion,
            "potential_energy": energy["potential_energy"][frame_idx],
            "sqd": energy["sqd"][frame_idx],
            "bond_total": energy["bond_total"][frame_idx],
            "bond_bp": energy["bond_bp"][frame_idx],
            "bond_pp": energy["bond_pp"][frame_idx],
            "angle_total": energy["angle_total"][frame_idx],
            "angle_ppp": energy["angle_ppp"][frame_idx],
            "angle_ppb": energy["angle_ppb"][frame_idx],
            "angle_bpb": energy["angle_bpb"][frame_idx],
            "angle_bpp": energy["angle_bpp"][frame_idx],
            "dihedral_total": energy["dihedral_total"][frame_idx],
            "dihedral_stacking": energy["dihedral_stacking"][frame_idx],
            "dihedral_direction": energy["dihedral_direction"][frame_idx],
            "rg": rg[frame_idx],
            "ke": ke[frame_idx],
            "Q": pairing["Q"],
        })

        return df

    finally:
        if opened_here:
            traj.close()


def extract_energy_summary_from_csv(csv_path, fraction_last=0.05):
    """
    Read a framewise CSV and compute mean of the last fraction_last of frames
    for each energy and structural column. Returns a flat dict of scalars.
 
    Works with both ts7 (no bond_bp/bond_pp, no split angles/dihedrals)
    and ts8 (full split logging). Missing columns are returned as NaN.
 
    Parameters
    ----------
    csv_path : str or Path
    fraction_last : float
        Fraction of frames from the end to average over (default 0.05 = last 5%).
 
    Returns
    -------
    dict with keys like bond_total_last5, Q_last5, rg_last5, etc.
    """
    COLS = [
        # energies
        "bond_total", "bond_bp", "bond_pp",
        "angle_total", "angle_ppp", "angle_ppb", "angle_bpb", "angle_bpp",
        "dihedral_total", "dihedral_stacking", "dihedral_direction",
        # structural
        "paired_strict", "paired_nonstrict", "disagree", "Q",
        "rg", "stacking_score", "inversion_score",
    ]
    suffix = f"_last{int(fraction_last * 100)}"

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  Could not read {csv_path}: {e}")
        return {f"{col}{suffix}": np.nan for col in COLS}

    n = len(df)
    start = int(n * (1 - fraction_last))
    last = df.iloc[start:]

    result = {}
    for col in COLS:
        if col in last.columns:
            vals = pd.to_numeric(last[col], errors="coerce")
            result[f"{col}{suffix}"] = float(
                vals.mean()) if vals.notna().any() else np.nan
        else:
            result[f"{col}{suffix}"] = np.nan

    return result


# ============================================================================
# MAIN PLOTTING FUNCTIONS
# ============================================================================


def plot_ke_time(ke_values,
                 ax=None,
                 smooth_sigma=None,
                 label=None,
                 title=None):
    """
    Plot kinetic energy per frame (usually per particle) vs frame index.
    """
    ke_values = np.asarray(ke_values)
    n_frames = len(ke_values)
    x = np.arange(n_frames)  # frame indices

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(x, ke_values, alpha=0.5, linewidth=1.0, label=label or "KE")

    if smooth_sigma is not None and smooth_sigma > 0:
        ke_smooth = gaussian_filter1d(ke_values, sigma=smooth_sigma)
        ax.plot(x, ke_smooth, linewidth=1.5,
                label=(label + " (smoothed)") if label else "KE (smoothed)")

    ax.set_xlabel("frame")
    ax.set_ylabel("kinetic energy (per particle)")
    if title:
        ax.set_title(title)

    if label is not None or smooth_sigma is not None:
        ax.legend()

    ax.grid(True, alpha=0.3)
    return ax


def plot_rg_time(rg_values, ax=None, smooth_sigma=None, title=None, line_alpha=0.5):
    """
    Plot Rg as a function of frame index, with optional Gaussian smoothing.
    """
    rg_values = np.asarray(rg_values)
    n_frames = len(rg_values)
    x = np.arange(n_frames)

    if ax is None:
        fig, ax = plt.subplots()

    # raw Rg (transparency controlled by line_alpha)
    ax.plot(x, rg_values, alpha=line_alpha, linewidth=1.0, label="Rg")

    # optional smoothed curve
    if smooth_sigma is not None and smooth_sigma > 0:
        rg_smooth = gaussian_filter1d(rg_values, sigma=smooth_sigma)
        ax.plot(x, rg_smooth, linewidth=1.5, label="Rg (smoothed)")

    ax.set_xlabel("Frame")
    ax.set_ylabel(r"$R_g$")
    if title:
        ax.set_title(title)

    if smooth_sigma is not None and smooth_sigma > 0:
        ax.legend()

    # “lined” background
    ax.grid(True, alpha=0.3)

    return ax


def plot_rg_hist_last(rg_values,
                      fraction_last=0.2,
                      bins=30,
                      ax=None,
                      label=None,
                      title=None,
                      show_mean=True):
    """
    Histogram of Rg over the last `fraction_last` of frames.

    Useful to see if the steady state is single- or multi-modal.
    """
    rg_values = np.asarray(rg_values)

    # take last fraction_last of the trajectory
    n = len(rg_values)
    start = int((1.0 - fraction_last) * n)
    rg_ss = rg_values[start:]

    if ax is None:
        fig, ax = plt.subplots()

    # histogram
    counts, edges, patches = ax.hist(
        rg_ss,
        bins=bins,
        density=True,
        alpha=0.7,
        histtype="stepfilled",
        edgecolor="black",
        label=label,
    )

    # optional mean line
    if show_mean:
        m = rg_ss.mean()
        ax.axvline(m, linestyle="--", linewidth=1.5)
        # optional: add to label
        if label is None:
            ax.legend([f"mean = {m:.2f}"])
        else:
            ax.legend()

    elif label is not None:
        ax.legend()

    ax.set_xlabel(r"$R_g$")
    ax.set_ylabel("probability density")

    if title is not None:
        ax.set_title(title)

    ax.grid(True, alpha=0.3)
    return ax

# RG autocorrelation


def autocorr_1d(x, max_lag):
    """
    Compute autocorrelation of a 1D array x up to max_lag.
    Returns acf array of length max_lag+1 with acf[0] = 1.
    """
    x = np.asarray(x)
    x = x - x.mean()
    n = len(x)
    max_lag = min(max_lag, n - 1)

    var = np.dot(x, x)
    acf = np.empty(max_lag + 1)
    for lag in range(max_lag + 1):
        num = np.dot(x[:n-lag], x[lag:])
        acf[lag] = num / var
    return acf


def plot_rg_autocorr(rg_values, max_lag=500, ax=None, title=None):
    """
    Plot autocorrelation of Rg over time.
    """
    acf = autocorr_1d(rg_values, max_lag=max_lag)
    lags = np.arange(len(acf))

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(lags, acf)
    ax.set_xlabel("lag (frames)")
    ax.set_ylabel("autocorrelation of $R_g$")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax


def plot_rg_bimodal(rg_all, bins=80, prominence=100):
    # 1) Get histogram counts and bin edges
    counts, bin_edges = np.histogram(rg_all, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 2) Find peaks in the *counts*
    peak_idx, _ = find_peaks(counts, prominence=prominence)

    # 3) Plot as a histogram
    fig, ax = plt.subplots()
    ax.hist(rg_all, bins=bins, alpha=0.7, histtype="stepfilled",
            edgecolor="black")   # histogram bars

    # 4) Overlay peaks at the bin centers
    peak_x = bin_centers[peak_idx]
    peak_y = counts[peak_idx]
    ax.scatter(peak_x, peak_y, marker="x")  # mark peaks

    # 5) write bin-center values on the plot
    for x, y in zip(peak_x, peak_y):
        ax.text(x, y, f"{x:.2f}", ha="center", va="bottom")

    ax.set_xlabel("Rg")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3)
    return ax


ANGLE_PARAM_KEYS = {
    "Phosphate-Phosphate-Phosphate": "angle_branch_dir_ppp_t0",
    "Phosphate-Phosphate-Base":      "angle_branch_dir_ppb_t0",
    "Base-Phosphate-Base":           "angle_branch_dir_bpb_t0",
}


def plot_angle_heatmap(angles, angle_name, params_path=None, ax=None):
    """
    angles: (n_frames, n_angles) in radians
    angle_name: must match keys in ANGLE_PARAM_KEYS
    params_path: path to params.json (optional); if present, adds expected value to title
    """
    A = np.degrees(np.asarray(angles))
    n_frames, m = A.shape

    # --- PPB special case: 2 angles per backbone position
    if angle_name == "Phosphate-Phosphate-Base":
        if m % 2 != 0:
            raise ValueError(f"PPB expects even number of angles, got {m}")
        npos = m // 2
        A1 = A[:, :npos]      # split in half (blocked ordering)
        A2 = A[:, npos:]
        H = np.empty((2 * npos, n_frames))
        H[0::2, :] = A1.T
        H[1::2, :] = A2.T
        ylab = "backbone position (2 rows each)"
    else:
        H = A.T
        ylab = "angle index"

    # --- expected value from params.json (if available)
    expected_str = ""
    if params_path is not None:
        try:
            with open(params_path, "r", encoding="utf-8") as f:
                params = json.load(f)
            key = ANGLE_PARAM_KEYS.get(angle_name, None)
            if key is not None and params.get(key) is not None:
                expected_deg = np.degrees(float(params[key]))
                expected_str = f" | expected {expected_deg:.1f}°"
        except Exception:
            pass  # keep it simple: no expected shown if anything fails

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    im = ax.imshow(H, aspect="auto", origin="lower")
    fig.colorbar(im, ax=ax, label="angle (deg)")
    ax.set_title(f"{angle_name}{expected_str}")
    ax.set_xlabel("frame")
    ax.set_ylabel(ylab)

    return ax


METRICS = ["rg_0_6_mean", "ke_mean_0.2",
           "inversion_score", "paired_frac_last_0_2_mean"]

def plot_replicate_variability_pct(df_per_rep, folder_name, metrics=METRICS, sort_by="traj_replicate",
                                   eps=1e-12, clip_pct=50, annotate=True, fmt="{:.3g}"):
    sub = df_per_rep[df_per_rep["folder_name"] == folder_name].copy()
    if sub.empty:
        raise ValueError(f"No rows found for folder_name={folder_name!r}")

    if sort_by in sub.columns:
        sub = sub.sort_values(sort_by)

    for m in metrics:
        sub[m] = pd.to_numeric(sub[m], errors="coerce")

    xlabels = sub["traj_replicate"].astype(str).to_list()
    x = np.arange(len(sub))

    fig, ax = plt.subplots(figsize=(10, 4))

    for m in metrics:
        vals = sub[m].to_numpy(dtype=float)
        med = np.nanmedian(vals)
        scale = max(abs(med), eps)
        pct = 100.0 * (vals - med) / scale
        if clip_pct is not None:
            pct = np.clip(pct, -clip_pct, clip_pct)

        line = ax.plot(x, pct, marker="o", linewidth=2,
                       label=f"{m} (median={med:.4g})")[0]

        if annotate:
            # small vertical offset so labels don't sit exactly on the marker
            for xi, yi, raw in zip(x, pct, vals):
                if np.isnan(raw) or np.isnan(yi):
                    continue
                ax.annotate(fmt.format(raw),
                            (xi, yi),
                            textcoords="offset points",
                            xytext=(0, 6),
                            ha="center",
                            fontsize=8,
                            alpha=0.75,
                            color=line.get_color())

    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.set_ylim(-clip_pct-5, clip_pct+5)
    ax.set_ylabel("% deviation from median")
    ax.set_title(f"Replicate variability")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    return ax



# ============================================================================
# FRAMEWISE SUMMARY PLOT
# ============================================================================
# Dependencies: numpy, pandas, plotly, scipy


def compute_transition_score(
    df: pd.DataFrame,
    smooth_window: int = 25,
    score_smooth_window: int = 100,
    features: list[str] | None = None,
) -> tuple[pd.Series, pd.Series]:
    """
    Compute a per-frame transition score as Euclidean speed in normalised
    feature space.

    Parameters
    ----------
    df : pd.DataFrame
        Framewise dataframe (output of make_framewise_metrics_dataframe).
    smooth_window : int
        Rolling window used to smooth raw features before differencing.
    score_smooth_window : int
        Rolling window applied to the raw score for peak detection.
    features : list[str] | None
        Columns to include. Defaults to
        ["paired_strict", "disagree", "rg", "potential_energy"].

    Returns
    -------
    transition_score : pd.Series  (raw)
    score_smooth     : pd.Series  (smoothed, used for peak detection)
    """
    if features is None:
        features = ["paired_strict", "disagree", "rg", "potential_energy"]

    # smooth each feature first
    smoothed = {}
    for col in features:
        if col in df.columns:
            smoothed[col] = (
                df[col].rolling(window=smooth_window,
                                center=True, min_periods=1).mean()
            )

    X = pd.DataFrame(smoothed)
    X = (X - X.mean()) / X.std(ddof=0)
    transition_score = X.diff().pow(2).sum(axis=1).pow(0.5)
    score_smooth = transition_score.rolling(
        window=score_smooth_window, center=True, min_periods=1
    ).mean()

    return transition_score, score_smooth


def detect_transitions(
    steps: np.ndarray,
    score_smooth: pd.Series,
    height_quantile: float = 0.87,
    distance_fraction: float = 1 / 30,
) -> np.ndarray:
    """
    Detect transition events from the smoothed transition score.

    Parameters
    ----------
    steps : np.ndarray
        Step values (x-axis).
    score_smooth : pd.Series
        Smoothed transition score.
    height_quantile : float
        Minimum peak height as a quantile of the score distribution.
    distance_fraction : float
        Minimum distance between peaks as a fraction of total frames.

    Returns
    -------
    transition_steps : np.ndarray
        Step values at detected transitions.
    """
    peaks, _ = find_peaks(
        score_smooth,
        height=np.quantile(score_smooth.dropna(), height_quantile),
        distance=max(1, int(len(score_smooth) * distance_fraction)),
    )
    return steps[peaks], peaks


def make_summary_figure(
    df: pd.DataFrame,
    smooth_window: int = 25,
    score_smooth_window: int = 100,
    height_quantile: float = 0.87,
    title: str | None = None,
) -> go.Figure:
    """
    Build an interactive Plotly summary figure for a single framewise CSV.

    Subplots (7 rows, shared x-axis):
        0 – Potential energy
        1 – SQD
        2 – Bond
        3 – Angles (total + PPP / PPB / BPB / BPP if available)
        4 – Dihedrals (total + stacking / direction if available)
        5 – Structure (paired_strict, Q, disagree)
        6 – Transition score (raw + smoothed + detected peaks)

    Vertical dashed lines mark detected transitions across all subplots.
    Open-circle markers at the last frame show end values.

    Parameters
    ----------
    df : pd.DataFrame
        Output of make_framewise_metrics_dataframe (or loaded from CSV).
    smooth_window : int
        Rolling window for smoothing energy / structure signals.
    score_smooth_window : int
        Rolling window for smoothing the transition score.
    height_quantile : float
        Quantile threshold for transition peak detection.
    title : str | None
        Optional figure title.

    Returns
    -------
    fig : go.Figure
    """
    df = df.copy()
    steps = df["step"].values

    # ── smooth energy / structure columns ────────────────────────────────────
    cols_to_smooth = [
        "potential_energy", "sqd", "bond",
        "angle_total", "angle_ppp", "angle_ppb", "angle_bpb", "angle_bpp",
        "dihedral_total", "dihedral_stacking", "dihedral_direction",
        "paired_strict", "Q", "disagree", "rg",
    ]
    for col in cols_to_smooth:
        if col in df.columns:
            df[f"{col}_smooth"] = (
                df[col].rolling(window=smooth_window,
                                center=True, min_periods=1).mean()
            )

    # ── transition score ──────────────────────────────────────────────────────
    transition_score, score_smooth = compute_transition_score(
        df,
        smooth_window=smooth_window,
        score_smooth_window=score_smooth_window,
    )
    transition_steps, peak_indices = detect_transitions(
        steps, score_smooth, height_quantile=height_quantile
    )

    # ── vline shapes (paper y-coords span all rows) ───────────────────────────
    vline_shapes = [
        dict(
            type="line", x0=s, x1=s, y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(color="gray", width=1, dash="dot"),
        )
        for s in transition_steps
    ]

    # ── build subplots ────────────────────────────────────────────────────────
    subplot_titles = [
        "Potential Energy", "SQD", "Bond",
        "Angles", "Dihedrals", "Structure", "Transition Score",
    ]
    fig = make_subplots(
        rows=7, cols=1,
        shared_xaxes=True,
        subplot_titles=subplot_titles,
        vertical_spacing=0.03,
        row_heights=[1, 1, 0.8, 1, 1, 1, 0.8],
    )

    def _hover(label):
        return f"{label}: %{{y:.4g}}<extra></extra>"

    def _raw(ax, col, color="lightblue"):
        if col not in df.columns:
            return
        fig.add_trace(go.Scatter(
            x=steps, y=df[col],
            mode="lines", line=dict(color=color, width=0.5),
            opacity=0.35, showlegend=False,
            hovertemplate=_hover(col),
        ), row=ax, col=1)

    def _smooth(ax, col, label, color, width=2):
        scol = f"{col}_smooth"
        src = df[scol] if scol in df.columns else (
            df[col] if col in df.columns else None)
        if src is None:
            return
        fig.add_trace(go.Scatter(
            x=steps, y=src,
            mode="lines", line=dict(color=color, width=width),
            name=label, hovertemplate=_hover(label),
        ), row=ax, col=1)

    def _end_marker(ax, col, label, color):
        """Open circle + value label at the last valid frame."""
        scol = f"{col}_smooth"
        src = df[scol] if scol in df.columns else (
            df[col] if col in df.columns else None)
        if src is None:
            return
        valid = src.dropna()
        if valid.empty:
            return
        last_x, last_y = steps[valid.index[-1]], valid.iloc[-1]
        fig.add_trace(go.Scatter(
            x=[last_x], y=[last_y],
            mode="markers+text",
            marker=dict(color=color, size=8, symbol="circle-open",
                        line=dict(width=2)),
            text=[f"{last_y:.3g}"],
            textposition="middle right",
            textfont=dict(size=9, color=color),
            showlegend=False,
            hovertemplate=f"last frame — {label}: %{{y:.4g}}<extra></extra>",
        ), row=ax, col=1)

    # ── row 1: potential energy ───────────────────────────────────────────────
    _raw(1, "potential_energy")
    _smooth(1, "potential_energy", "potential_energy", "darkorange")
    # zoom y away from initial drop
    skip = max(1, len(df) // 50)
    pe = df["potential_energy_smooth"].iloc[skip:] if "potential_energy_smooth" in df.columns \
        else df["potential_energy"].iloc[skip:]
    pad = (pe.max() - pe.min()) * 0.25
    fig.update_yaxes(range=[pe.min() - pad, pe.max() + pad], row=1, col=1)
    _end_marker(1, "potential_energy", "PE", "darkorange")

    # ── row 2: SQD ────────────────────────────────────────────────────────────
    _raw(2, "sqd")
    _smooth(2, "sqd", "SQD", "darkorange")
    _end_marker(2, "sqd", "SQD", "darkorange")

    # ── row 3: bond ───────────────────────────────────────────────────────────
    _smooth(3, "bond", "bond", "steelblue", width=1.0)
    _end_marker(3, "bond", "bond", "steelblue")

    # ── row 4: angles ─────────────────────────────────────────────────────────
    angle_traces = [
        ("angle_total", "angle total", "steelblue"),
        ("angle_ppp",   "PPP",         "darkorange"),
        ("angle_ppb",   "PPB",         "green"),
        ("angle_bpb",   "BPB",         "red"),
        ("angle_bpp",   "BPP",         "purple"),
    ]
    for col, label, color in angle_traces:
        _smooth(4, col, label, color)
    _end_marker(4, "angle_total", "angle total", "steelblue")

    # ── row 5: dihedrals ──────────────────────────────────────────────────────
    dih_traces = [
        ("dihedral_total",     "dihedral total", "steelblue"),
        ("dihedral_stacking",  "stacking",       "darkorange"),
        ("dihedral_direction", "direction",       "green"),
    ]
    for col, label, color in dih_traces:
        _smooth(5, col, label, color)
    _end_marker(5, "dihedral_total", "dihedral total", "steelblue")

    # ── row 6: structure ──────────────────────────────────────────────────────
    struct_traces = [
        ("paired_strict", "paired_strict", "steelblue"),
        ("Q",             "Q",             "darkorange"),
        ("disagree",      "disagree",      "green"),
    ]
    for col, label, color in struct_traces:
        _smooth(6, col, label, color)
    _end_marker(6, "paired_strict", "paired_strict", "steelblue")

    # ── row 7: transition score ───────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=steps, y=transition_score,
        mode="lines", line=dict(color="steelblue", width=0.4),
        opacity=0.4, name="raw score",
        hovertemplate=_hover("raw"),
    ), row=7, col=1)
    fig.add_trace(go.Scatter(
        x=steps, y=score_smooth,
        mode="lines", line=dict(color="black", width=1.0),
        name="transition score (smoothed)",
        hovertemplate=_hover("score"),
    ), row=7, col=1)
    fig.add_trace(go.Scatter(
        x=transition_steps,
        y=score_smooth.values[peak_indices],
        mode="markers",
        marker=dict(color="red", size=8, symbol="circle"),
        name="detected transitions",
        hovertemplate="score: %{y:.4g}<extra></extra>",
    ), row=7, col=1)

    # ── x-axis ticks ──────────────────────────────────────────────────────────
    x_max = steps.max()
    major = np.arange(0, x_max + 1e5, 1e6)
    minor = np.arange(0.5e6, x_max, 1e6)
    all_ticks = np.sort(np.concatenate([major, minor]))
    tick_labels = [f"{t / 1e6:.1f}" for t in all_ticks]
    fig.update_xaxes(
        tickvals=all_ticks, ticktext=tick_labels,
        tickfont=dict(size=9), row=7, col=1,
    )
    fig.update_xaxes(title_text="step (×10⁶)", row=7, col=1)

    # ── layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        height=1600,
        width=1200,
        shapes=vline_shapes,
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="v", x=1.01, y=1),
        margin=dict(r=130),
        title=dict(text=title, x=0.5) if title else None,
    )
    fig.update_annotations(font_size=11)

    return fig
