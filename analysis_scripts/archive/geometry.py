from __future__ import annotations
from typing import Sequence
import numpy as np
import gsd.hoomd
import freud
from numpy.typing import NDArray

from pathlib import Path

from analysis_scripts.helpers import (
    make_inversion_dihe_group_3pt,
    make_stacking_dihe_group_3pt,
    _prepare_geometry_groups,
    save_npz,
    load_npz,
    as_traj,
    get_angle_groups_by_type,
    get_bond_groups_by_type,
    get_dihedral_groups_by_type,
    get_n_particles

)

# ============================================================================
# DIHEDRALS INVERSION AND STACKING SCORE
# ============================================================================


def calculate_dihedral(
    p1: NDArray[np.float64],
    p2: NDArray[np.float64],
    p3: NDArray[np.float64],
    p4: NDArray[np.float64],
    box: freud.box.Box | None = None,
) -> np.float64:
    """
    Calculate the dihedral angle between four points. Returns the angle in rad.
    """
    if box is None:
        u1, u2, u3 = p2 - p1, p3 - p2, p4 - p3
    else:
        # minimum-image displacement vectors
        u1, u2, u3 = box.wrap(p2 - p1), box.wrap(p3 - p2), box.wrap(p4 - p3)

    y = np.dot(np.linalg.norm(u2) * u1, np.cross(u2, u3))
    x = np.dot(np.cross(u1, u2), np.cross(u2, u3))
    return np.float64(np.arctan2(y, x))


def calc_dihedrals_for_frame(
    frame: gsd.hoomd.Frame,
    dihe_group: Sequence[tuple[int, int, int, int]],
) -> NDArray[np.float64]:
    """
    Compute all dihedral angles for a single frame. dihe_group are particle-index quadruplets
    defining the dihedrals. Returns an array of dihedral angles in rad, with one value per
    quadruplet.
    """
    pos = frame.particles.position
    box = freud.box.Box.from_box(frame.configuration.box)
    dihedrals = np.empty(len(dihe_group), dtype=np.float64)

    for i, (a, b, c, d) in enumerate(dihe_group):
        dihedrals[i] = calculate_dihedral(
            pos[a], pos[b], pos[c], pos[d], box=box
        )
    return dihedrals

def calc_all_dihedrals_over_time(
    traj: gsd.hoomd.HOOMDTrajectory,
    dihe_group: Sequence[tuple[int, int, int, int]],
    stride: int = 1,
) -> NDArray[np.float64]:
    """
    Compute dihedrals for a group over time. dihe_group are particle-index quadruplets
    defining the dihedrals. Stride controls frame subsampling; stride = 10 uses every
    10th frame. Returns an array of dihedral angles in rad with shape
    (n_sampled_frames, n_dihedrals).
    """
    sampled_indices = range(0, len(traj), stride)
    n_samples = len(sampled_indices)
    n_dihedrals = len(dihe_group)

    results = np.empty((n_samples, n_dihedrals), dtype=np.float64)

    for f_idx, i in enumerate(sampled_indices):
        results[f_idx] = calc_dihedrals_for_frame(traj[i], dihe_group)

    return results

def stacking_score_from_dihedrals(dihes: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return framewise stacking scores from dihedral angles in radians."""
    return 1.0 - np.mean(np.abs(dihes) / np.pi, axis=1)

def inversion_score_from_dihedrals(dihes: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return framewise inversion scores from directional dihedrals."""
    return np.mean(dihes > 0, axis=1)

def stacking_score_over_time(
    traj: gsd.hoomd.HOOMDTrajectory,
    stride: int = 1,
) -> NDArray[np.float64]:
    """Return framewise stacking scores for the 3-particles-per-nucleotide model."""
    n_particles = int(traj[0].particles.N)
    dihe_group = make_stacking_dihe_group_3pt(n_particles)

    if not dihe_group:
        return np.array([], dtype=np.float64)

    dihes = calc_all_dihedrals_over_time(traj, dihe_group, stride=stride)
    return stacking_score_from_dihedrals(dihes)

def inversion_score_over_time(
    traj: gsd.hoomd.HOOMDTrajectory,
    stride: int = 1,
) -> NDArray[np.float64]:
    """Return framewise inversion scores for the 3-particles-per-nucleotide model."""
    n_particles = int(traj[0].particles.N)
    dihe_group = make_inversion_dihe_group_3pt(n_particles)

    if not dihe_group:
        return np.array([], dtype=np.float64)

    dihes = calc_all_dihedrals_over_time(traj, dihe_group, stride=stride)
    return inversion_score_from_dihedrals(dihes)



# ============================================================================
# ANGLES
# ============================================================================

def calculate_angle(
    p1: NDArray[np.float64],
    p2: NDArray[np.float64],
    p3: NDArray[np.float64],
    box: freud.box.Box | None = None,
) -> np.float64:
    """
    Calculate the angle between three points. Returns the angle in rad in the range [0, pi].
    """
    if box is None:
        v1 = p1 - p2
        v2 = p3 - p2
    else:
        v1 = box.wrap(p1 - p2)
        v2 = box.wrap(p3 - p2)

    cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.float64(np.arccos(cosang))


def calc_angles_for_frame(
    frame: gsd.hoomd.Frame,
    angle_group: Sequence[tuple[int, int, int]],
) -> NDArray[np.float64]:
    """
    Compute all angles for a single frame. angle_group are particle-index triplets defining the angles.
    Returns an array of angles in rad, with one value per triplet.
    """
    pos = frame.particles.position
    box = freud.box.Box.from_box(frame.configuration.box)
    angles = np.empty(len(angle_group), dtype=np.float64)

    for i, (a, b, c) in enumerate(angle_group):
        angles[i] = calculate_angle(pos[a], pos[b], pos[c], box=box)

    return angles





def calc_angles_over_time(
    traj: gsd.hoomd.HOOMDTrajectory,
    angle_group: Sequence[tuple[int, int, int]],
    stride: int = 1,
) -> NDArray[np.float64]:
    """
    Compute angles for one angle group over time. angle_group are particle-index triplets
    defining the angles. Stride controls frame subsampling. Returns an array of angles in rad
    with shape (n_sampled_frames, n_angles).
    """
    sampled_indices = range(0, len(traj), stride)
    results = np.empty(
        (len(sampled_indices), len(angle_group)), dtype=np.float64)

    for f_idx, i in enumerate(sampled_indices):
        results[f_idx] = calc_angles_for_frame(traj[i], angle_group)

    return results


def angles_by_type_over_time(
    traj: gsd.hoomd.HOOMDTrajectory,
    stride: int = 1,
) -> dict[str, NDArray[np.float64]]:
    """
    Compute all angle types over time. Returns a dictionary mapping each angle type to an array
    of angles in rad with shape (n_sampled_frames, n_angles_of_that_type).
    """
    by_type = get_angle_groups_by_type(traj[0])

    return {
        name: calc_angles_over_time(traj, group, stride=stride)
        for name, group in by_type.items()
    }

# ============================================================================
# BONDS
# ============================================================================


def calculate_bond_length(
    p1: NDArray[np.float64],
    p2: NDArray[np.float64],
    box: freud.box.Box | None = None,
) -> np.float64:
    """
    Calculate the bond length between two points. Returns the distance in simulation units.
    """
    if box is None:
        v = p2 - p1
    else:
        v = box.wrap(p2 - p1)

    return np.float64(np.linalg.norm(v))


def calc_bonds_for_frame(
    frame: gsd.hoomd.Frame,
    bond_group: Sequence[tuple[int, int]],
) -> NDArray[np.float64]:
    """
    Compute all bond lengths for a single frame. bond_group are particle-index pairs
    defining the bonds. Returns an array of bond lengths, with one value per pair.
    """
    pos = frame.particles.position
    box = freud.box.Box.from_box(frame.configuration.box)
    bonds = np.empty(len(bond_group), dtype=np.float64)

    for i, (a, b) in enumerate(bond_group):
        bonds[i] = calculate_bond_length(pos[a], pos[b], box=box)

    return bonds


def calc_all_bonds_over_time(
    traj: gsd.hoomd.HOOMDTrajectory,
    bond_group: Sequence[tuple[int, int]],
    stride: int = 1,
) -> NDArray[np.float64]:
    """
    Compute bond lengths for a group over time. bond_group are particle-index pairs
    defining the bonds. Stride controls frame subsampling. Returns an array of bond
    lengths with shape (n_sampled_frames, n_bonds).
    """
    sampled_indices = range(0, len(traj), stride)
    results = np.empty(
        (len(sampled_indices), len(bond_group)), dtype=np.float64)

    for f_idx, i in enumerate(sampled_indices):
        results[f_idx] = calc_bonds_for_frame(traj[i], bond_group)

    return results




# ============================================================================
# MAIN GEOMETRY FUNCTION
# ============================================================================


def compute_geometry(
    traj: gsd.hoomd.HOOMDTrajectory,
    stride: int = 1,
) -> dict[str, NDArray]:
    """
    Compute all geometric quantities for a trajectory in a single pass.
    Returns a dict suitable for saving with np.savez_compressed.
    """
    frame_indices = np.arange(0, len(traj), stride, dtype=np.int64)
    if frame_indices.size == 0:
        raise ValueError("No frames selected for geometry computation.")

    geometry_groups = _prepare_geometry_groups(traj[int(frame_indices[0])])
    n_frames = len(frame_indices)

    data: dict[str, NDArray] = {"frame": frame_indices}
    for key, (_, group) in geometry_groups.items():
        data[key] = np.empty((n_frames, len(group)), dtype=np.float64)

    for f_idx, frame_idx in enumerate(frame_indices):
        frame = traj[int(frame_idx)]

        for key, (kind, group) in geometry_groups.items():
            if kind == "angle":
                data[key][f_idx] = calc_angles_for_frame(frame, group)
            elif kind == "dihedral":
                data[key][f_idx] = calc_dihedrals_for_frame(frame, group)
            elif kind == "bond":
                data[key][f_idx] = calc_bonds_for_frame(frame, group)

    data["stacking_score"] = stacking_score_from_dihedrals(
        data["dihedral_stacking"])
    data["inversion_score"] = inversion_score_from_dihedrals(
        data["dihedral_direction"])

    return data


def compute_and_save_geometry(
    traj_file: str | Path | gsd.hoomd.HOOMDTrajectory,
    out_path: Path,
    stride: int = 1,
    skip_if_exists: bool = True,
) -> dict[str, NDArray]:
    """
    Compute geometry arrays and save them to disk. If skip_if_exists is True and the file
    already exists, the cached file is loaded instead.
    """
    if skip_if_exists and out_path.exists():
        return load_npz(out_path)

    with as_traj(traj_file) as traj:
        data = compute_geometry(traj, stride=stride)

    save_npz(out_path, data)
    return data

