from pathlib import Path
import pandas as pd
import freud
import gsd.hoomd
import numpy as np
from numpy.typing import NDArray
from analysis_scripts.helpers import load_npz, save_npz, get_n_particles
from analysis_scripts.pairing import Q_from_pair_arrays, disagree_from_closest_arrays, paired_nonstrict_from_closest_arrays, paired_strict_from_pair_arrays

# ============================================================================
# RG
# ============================================================================


def calc_rg_for_frame(
    frame: gsd.hoomd.Frame,
    scale: float = 1.0,
) -> np.float64:
    """
    Compute the radius of gyration for one frame. Returns the Rg value scaled by scale.
    """
    box = freud.box.Box.from_box(frame.configuration.box)
    positions = box.unwrap(frame.particles.position, frame.particles.image)

    com = np.mean(positions, axis=0)
    diff = positions - com
    sq_dist = np.sum(diff**2, axis=1)

    return np.float64(np.sqrt(np.mean(sq_dist)) * scale)


def compute_rg_over_time(
    traj: gsd.hoomd.HOOMDTrajectory,
    scale: float = 1.0,
    stride: int = 1,
) -> NDArray[np.float64]:
    """
    Compute the radius of gyration for each sampled frame.
    Returns an array of Rg values with shape (n_sampled_frames,).
    """
    frame_indices = np.arange(0, len(traj), stride, dtype=np.int64)
    rg_values = np.empty(len(frame_indices), dtype=np.float64)

    for i, frame_idx in enumerate(frame_indices):
        rg_values[i] = calc_rg_for_frame(traj[int(frame_idx)], scale=scale)

    return rg_values

# ============================================================================
# KE
# ============================================================================


def calc_kinetic_energy(frame: gsd.hoomd.Frame,
                        mass: float = 1.0,
                        ) -> np.float64:
    """
    Calculate total kinetic energy for a single snapshot.
    Input: gsd.hoomd.Frame
    Returns: Total kinetic energy
    """
    v = frame.particles.velocity


    if v.size == 0:
        raise ValueError("No velocity data found in frame.")
    KE = 0.5 * mass * np.sum(v**2)
    return KE


def compute_ke_over_time(
    traj: gsd.hoomd.HOOMDTrajectory,
    mass: float = 1.0,
    per_particle: bool = True,
    stride: int = 1,
) -> NDArray[np.float64]:
    """
    Compute kinetic energy for each sampled frame.
    Returns an array of KE values with shape (n_sampled_frames,).
    """
    frame_indices = np.arange(0, len(traj), stride, dtype=np.int64)
    ke_values = np.empty(len(frame_indices), dtype=np.float64)

    n_particles = get_n_particles(traj)
    for i, frame_idx in enumerate(frame_indices):
        ke_total = calc_kinetic_energy(traj[int(frame_idx)], mass=mass)
        ke_values[i] = ke_total / n_particles if per_particle else ke_total

    return ke_values

def compute_observables(
    traj: gsd.hoomd.HOOMDTrajectory,
    rg_scale: float = 1.0,
    ke_mass: float = 1.0,
    ke_per_particle: bool = True,
    stride: int = 1,
) -> dict[str, NDArray]:
    """
    Compute framewise KE and Rg for a trajectory.
    Returns a dict suitable for saving with np.savez_compressed.
    """
    frame_indices = np.arange(0, len(traj), stride, dtype=np.int64)

    return {
        "frame": frame_indices,
        "rg": compute_rg_over_time(traj, scale=rg_scale, stride=stride),
        "ke": compute_ke_over_time(
            traj,
            mass=ke_mass,
            per_particle=ke_per_particle,
            stride=stride,
        ),
    }

# ============================================================================
# ENERGIES
# ============================================================================


def compute_energy_terms(
    traj: gsd.hoomd.HOOMDTrajectory,
    stride: int = 1,
) -> dict[str, NDArray]:
    """
    Compute framewise energy terms from trajectory logs.
    Supports both lumped and split bond, angle, and dihedral logging.
    Returns a dict suitable for downstream framewise analysis.
    """
    frame_indices = np.arange(0, len(traj), stride, dtype=np.int64)
    if frame_indices.size == 0:
        raise ValueError("No frames selected for energy computation.")

    first_log = traj[int(frame_indices[0])].log

    has_split_bonds = "md/bond/Harmonic_1/energy" in first_log
    has_split_angles = "md/angle/Harmonic_1/energy" in first_log
    has_split_dihedrals = "md/dihedral/Periodic_1/energy" in first_log

    data = {
        "frame": frame_indices,
        "E_pot": np.empty(len(frame_indices), dtype=np.float64),
        "E_sqd": np.empty(len(frame_indices), dtype=np.float64),
        "E_bond": np.empty(len(frame_indices), dtype=np.float64),
        "E_bond_bp": np.empty(len(frame_indices), dtype=np.float64),
        "E_bond_pp": np.empty(len(frame_indices), dtype=np.float64),
        "E_angle": np.empty(len(frame_indices), dtype=np.float64),
        "E_angle_ppp": np.empty(len(frame_indices), dtype=np.float64),
        "E_angle_ppb": np.empty(len(frame_indices), dtype=np.float64),
        "E_angle_bpb": np.empty(len(frame_indices), dtype=np.float64),
        "E_angle_bpp": np.empty(len(frame_indices), dtype=np.float64),
        "E_dihedral": np.empty(len(frame_indices), dtype=np.float64),
        "E_dihedral_stacking": np.empty(len(frame_indices), dtype=np.float64),
        "E_dihedral_direction": np.empty(len(frame_indices), dtype=np.float64),
    }

    for i, frame_idx in enumerate(frame_indices):
        log = traj[int(frame_idx)].log

        data["E_pot"][i] = np.asarray(
            log["md/compute/ThermodynamicQuantities/potential_energy"]
        ).item()
        data["E_sqd"][i] = np.asarray(
            log["sqd_plugin/sqd_potential/SquareDensity/energy"]
        ).item()

        if has_split_bonds:
            E_bond_bp = np.asarray(log["md/bond/Harmonic/energy"]).item()
            E_bond_pp = np.asarray(log["md/bond/Harmonic_1/energy"]).item()
            data["E_bond_bp"][i] = E_bond_bp
            data["E_bond_pp"][i] = E_bond_pp
            data["E_bond"][i] = E_bond_bp + E_bond_pp
        else:
            data["E_bond"][i] = np.asarray(
                log["md/bond/Harmonic/energy"]).item()
            data["E_bond_bp"][i] = np.nan
            data["E_bond_pp"][i] = np.nan

        if has_split_angles:
            E_angle_ppp = np.asarray(log["md/angle/Harmonic/energy"]).item()
            E_angle_ppb = np.asarray(log["md/angle/Harmonic_1/energy"]).item()
            E_angle_bpb = np.asarray(log["md/angle/Harmonic_2/energy"]).item()
            E_angle_bpp = np.asarray(log["md/angle/Harmonic_3/energy"]).item()

            data["E_angle_ppp"][i] = E_angle_ppp
            data["E_angle_ppb"][i] = E_angle_ppb
            data["E_angle_bpb"][i] = E_angle_bpb
            data["E_angle_bpp"][i] = E_angle_bpp
            data["E_angle"][i] = E_angle_ppp + \
                E_angle_ppb + E_angle_bpb + E_angle_bpp
        else:
            data["E_angle"][i] = np.asarray(
                log["md/angle/Harmonic/energy"]).item()
            data["E_angle_ppp"][i] = np.nan
            data["E_angle_ppb"][i] = np.nan
            data["E_angle_bpb"][i] = np.nan
            data["E_angle_bpp"][i] = np.nan

        if has_split_dihedrals:
            E_dihedral_stacking = np.asarray(
                log["md/dihedral/Periodic/energy"]).item()
            E_dihedral_direction = np.asarray(
                log["md/dihedral/Periodic_1/energy"]).item()
            data["E_dihedral_stacking"][i] = E_dihedral_stacking
            data["E_dihedral_direction"][i] = E_dihedral_direction
            data["E_dihedral"][i] = E_dihedral_stacking + E_dihedral_direction
        else:
            data["E_dihedral"][i] = np.asarray(
                log["md/dihedral/Periodic/energy"]).item()
            data["E_dihedral_stacking"][i] = np.nan
            data["E_dihedral_direction"][i] = np.nan

    return data


def compute_framewise_metrics(
    energy: dict[str, NDArray],
    geometry: dict[str, NDArray],
    pairing: dict[str, NDArray],
    observables: dict[str, NDArray],
    expected: NDArray[np.int_] | None = None,
) -> pd.DataFrame:
    """
    Build a framewise dataframe from cached per-frame arrays.
    """
    pair_arrays = pairing["pair_arrays"]
    closest_partner0 = pairing["closest_partner0"]
    closest_partner1 = pairing["closest_partner1"]
    # check that w ehave same number of frames saved in all arrays
    np.array_equal(energy["frame"], geometry["frame"])
    np.array_equal(energy["frame"], observables["frame"])
    np.array_equal(energy["frame"], pairing["frame"])

    Q = (
        Q_from_pair_arrays(pair_arrays, expected)
        if expected is not None
        else np.full(len(pair_arrays), np.nan, dtype=np.float64)
    )

    return pd.DataFrame({
        "frame": pairing["frame"],
        "E_pot": energy["E_pot"],
        "E_sqd": energy["E_sqd"],
        "E_bond": energy["E_bond"],
        "E_bond_bp": energy["E_bond_bp"],
        "E_bond_pp": energy["E_bond_pp"],
        "E_angle": energy["E_angle"],
        "E_angle_ppp": energy["E_angle_ppp"],
        "E_angle_ppb": energy["E_angle_ppb"],
        "E_angle_bpb": energy["E_angle_bpb"],
        "E_angle_bpp": energy["E_angle_bpp"],
        "E_dihedral": energy["E_dihedral"],
        "E_dihedral_stacking": energy["E_dihedral_stacking"],
        "E_dihedral_direction": energy["E_dihedral_direction"],
        "rg": observables["rg"],
        "ke": observables["ke"],
        "Q": Q,
        "paired_strict": paired_strict_from_pair_arrays(pair_arrays),
        "paired_nonstrict": paired_nonstrict_from_closest_arrays(
            closest_partner0,
            closest_partner1,
        ),
        "disagree": disagree_from_closest_arrays(
            closest_partner0,
            closest_partner1,
        ),
        "stacking_score": geometry["stacking_score"],
        "inversion_score": geometry["inversion_score"],
    })
