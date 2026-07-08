"""Setup utilities for the one-particle A/U/G/C sticker-spacer model.

This module contains:
- molecule JSON parsing,
- SQD pair parameter assignment,
- random-walk initialization,
- bond/angle topology creation.

Model vocabulary such as particle types, bond types, angle types, and complements
is expected to be attached to ``par`` from ``params.base_info()``.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypeAlias

import gsd.hoomd
import numpy as np
from numpy.typing import NDArray


ParticleType: TypeAlias = str
Sequence1P: TypeAlias = list[ParticleType]
Vector3: TypeAlias = tuple[float, float, float]


def parse_molecules(file: str | Path = "molecules.json") -> list[Sequence1P]:
    """Read molecule definitions from JSON and expand counts into sequence lists."""
    path = Path(file)

    with path.open() as handle:
        molecule_defs = json.load(handle)

    molecules: list[Sequence1P] = []

    for mol_idx, mol_def in enumerate(molecule_defs):
        count = int(mol_def["count"])
        sequence = mol_def["sequence"]

        if count < 0:
            raise ValueError(
                f"Molecule definition {mol_idx} has negative count: {count}."
            )

        if isinstance(sequence, str):
            sequence_list = list(sequence)
        else:
            sequence_list = list(sequence)

        if not sequence_list:
            raise ValueError(
                f"Molecule definition {mol_idx} has an empty sequence.")

        molecules.extend(sequence_list.copy() for _ in range(count))

    if not molecules:
        raise ValueError(f"No molecules were loaded from {path}.")

    return molecules


def validate_model_info(par: argparse.Namespace) -> None:
    """Validate fixed model metadata loaded from params.base_info()."""
    required_attrs = [
        "type_names",
        "complements",
        "bond_types",
        "angle_types",
    ]

    missing_attrs = [name for name in required_attrs if not hasattr(par, name)]
    if missing_attrs:
        raise AttributeError(
            "Missing model metadata on par: "
            f"{missing_attrs}. Attach params.base_info() after parsing arguments."
        )

    if list(par.bond_types) != ["Base-Base"]:
        raise ValueError(
            f"Expected one bond type ['Base-Base'], got {par.bond_types}.")

    if list(par.angle_types) != ["Base-Base-Base"]:
        raise ValueError(
            f"Expected one angle type ['Base-Base-Base'], got {par.angle_types}."
        )


def validate_sequences(
    sequences: Sequence[Sequence[ParticleType]],
    type_names: Sequence[ParticleType],
) -> None:
    """Validate that all sequences are non-empty and contain known particle types."""
    allowed_types = set(type_names)

    if not sequences:
        raise ValueError("At least one molecule sequence is required.")

    for mol_idx, sequence in enumerate(sequences):
        if not sequence:
            raise ValueError(f"Sequence {mol_idx} is empty.")

        unknown_types = sorted(set(sequence) - allowed_types)
        if unknown_types:
            raise ValueError(
                f"Sequence {mol_idx} contains unknown particle type(s): {unknown_types}. "
                f"Allowed types are: {list(type_names)}."
            )


def validate_sqd_cutoffs(par: argparse.Namespace) -> None:
    """Fail clearly if the neighbor-list cutoff is too small for SQD density cutoff."""
    if float(par.nlist_rcut) < float(par.density_upper_rcut):
        raise ValueError(
            "nlist_rcut is smaller than density_upper_rcut. "
            f"Got nlist_rcut={float(par.nlist_rcut):g}, "
            f"density_upper_rcut={float(par.density_upper_rcut):g}. "
            "Increase nlist_rcut so the neighbor list covers the SQD density cutoff."
        )


def is_wc_pair(
    t1: ParticleType,
    t2: ParticleType,
    complements: Mapping[ParticleType, ParticleType],
) -> bool:
    """Return True if two particle types are Watson-Crick complements."""
    return complements.get(t1) == t2 or complements.get(t2) == t1


def sqd_param_dict(
    par: argparse.Namespace,
    attraction: float,
    ncut: float,
) -> dict[str, float]:
    """Create one SQD parameter dictionary from an attraction magnitude and valency."""
    return {
        "A": float(par.E_rep),
        "B": -float(attraction),
        "attr_rcut": float(par.r_att),
        "rep_rcut": float(par.r_rep),
        "density_upper_rcut": float(par.density_upper_rcut),
        "density_lower_rcut": float(par.density_lower_rcut),
        "ncut_ij": float(ncut),
        "ncut_ji": float(ncut),
    }


def define_pairwise_sqd_sticker_spacer(sqd: Any, par: argparse.Namespace) -> None:
    """Assign SQD interactions for the A/U/G/C sticker-spacer model.

    Watson-Crick pairs receive attraction ``E_att`` and valency ``ncut_wc``.
    All other pairs are repulsive-only, with ``B = 0``.

    This function modifies ``sqd.params`` in place.
    """
    validate_model_info(par)
    validate_sqd_cutoffs(par)

    complements: Mapping[ParticleType, ParticleType] = par.complements

    print("define_pairwise_sqd_sticker_spacer called")
    print(f"  type_names: {list(par.type_names)}")
    print(f"  complements: {dict(complements)}")
    print(f"  E_rep: {float(par.E_rep):g}")
    print(f"  E_att: {float(par.E_att):g}")
    print(f"  r_rep: {float(par.r_rep):g}")
    print(f"  r_att: {float(par.r_att):g}")
    print(f"  ncut_wc: {float(par.ncut_wc):g}")
    print(f"  nlist_rcut: {float(par.nlist_rcut):g}")
    print(f"  density_upper_rcut: {float(par.density_upper_rcut):g}")
    print(f"  density_lower_rcut: {float(par.density_lower_rcut):g}")

    for i, t1 in enumerate(par.type_names):
        for t2 in par.type_names[i:]:
            if is_wc_pair(t1=t1, t2=t2, complements=complements):
                label = "WC"
                attraction = float(par.E_att)
                ncut = float(par.ncut_wc)
            else:
                label = "default"
                attraction = 0.0
                ncut = float(par.ncut_wc)

            sqd.params[(t1, t2)] = sqd_param_dict(
                par=par,
                attraction=attraction,
                ncut=ncut,
            )

            print(
                f"[{label}] {t1}-{t2}: "
                f"A={float(par.E_rep):g}, B={-attraction:g}, ncut={ncut:g}"
            )


def random_unit_vector_3d(rng: np.random.Generator) -> Vector3:
    """Sample one random unit vector uniformly on the 3D sphere."""
    theta = 2.0 * math.pi * rng.random()
    phi = math.acos(2.0 * rng.random() - 1.0)

    x = math.sin(phi) * math.cos(theta)
    y = math.sin(phi) * math.sin(theta)
    z = math.cos(phi)

    return (x, y, z)


def random_walk_positions_3d(
    n_particles: int,
    step_size: float,
    box_size: float,
    rng: np.random.Generator,
    start_pos: Sequence[float] = (0.0, 0.0, 0.0),
    max_tries: int = 100_000,
) -> NDArray[np.float64]:
    """Generate a bounded 3D random walk inside a cubic simulation box."""
    if n_particles <= 0:
        raise ValueError(f"n_particles must be positive, got {n_particles}.")
    if step_size <= 0:
        raise ValueError(f"step_size must be positive, got {step_size}.")
    if box_size <= 0:
        raise ValueError(f"box_size must be positive, got {box_size}.")
    if max_tries <= 0:
        raise ValueError(f"max_tries must be positive, got {max_tries}.")

    half_box_size = box_size * 0.5
    start = tuple(float(coord) for coord in start_pos)

    if len(start) != 3:
        raise ValueError("start_pos must contain exactly three coordinates.")
    if any(abs(coord) > half_box_size for coord in start):
        raise ValueError(
            f"start_pos {start} lies outside the box with side length {box_size}."
        )

    positions: list[Vector3] = [start]

    for particle_idx in range(1, n_particles):
        for _ in range(max_tries):
            ux, uy, uz = random_unit_vector_3d(rng)
            oldx, oldy, oldz = positions[particle_idx - 1]

            new_pos = (
                oldx + step_size * ux,
                oldy + step_size * uy,
                oldz + step_size * uz,
            )

            if all(-half_box_size <= coord <= half_box_size for coord in new_pos):
                positions.append(new_pos)
                break
        else:
            raise RuntimeError(
                "Failed to place all monomers inside the box. "
                "Try increasing box_size or max_tries."
            )

    return np.asarray(positions, dtype=np.float64)


def snapshot_sticker_spacer(par: argparse.Namespace) -> gsd.hoomd.Frame:
    """Build a HOOMD snapshot for the one-particle A/U/G/C model."""
    validate_model_info(par)

    sequences: list[Sequence1P] = [list(sequence)
                                   for sequence in par.sequences]
    validate_sequences(sequences=sequences, type_names=par.type_names)

    num_mols = len(sequences)
    total_particles = sum(len(sequence) for sequence in sequences)

    density = float(par.density)
    if density <= 0:
        raise ValueError(f"density must be positive, got {density}.")

    bond_r0 = float(par.bond_r0)
    if bond_r0 <= 0:
        raise ValueError(f"bond_r0 must be positive, got {bond_r0}.")

    volume = total_particles / density
    box_size = volume ** (1.0 / 3.0)

    print("snapshot_sticker_spacer called")
    print(f"  num_mols: {num_mols}")
    print(f"  total_particles: {total_particles}")
    print(f"  density: {density:g}")
    print(f"  box_size: {box_size:.3f}")
    print(f"  bond_r0: {bond_r0:g}")
    print(f"  init_seed: {par.init_seed}")
    print(f"  particle types: {list(par.type_names)}")
    print(f"  bond types: {list(par.bond_types)}")
    print(f"  angle types: {list(par.angle_types)}")

    snapshot = gsd.hoomd.Frame()
    snapshot.particles.N = total_particles
    snapshot.configuration.box = [box_size, box_size, box_size, 0.0, 0.0, 0.0]
    snapshot.particles.types = list(par.type_names)

    type_to_id = {name: i for i, name in enumerate(par.type_names)}

    typeids: list[int] = []
    positions: list[Vector3] = []
    velocities: list[Vector3] = []

    bonds: list[list[int]] = []
    bond_typeids: list[int] = []

    angles: list[list[int]] = []
    angle_typeids: list[int] = []

    offset = 0
    rng = np.random.default_rng(par.init_seed)

    for sequence in sequences:
        n_particles = len(sequence)

        typeids.extend(type_to_id[particle_type] for particle_type in sequence)

        center_offset = rng.uniform(-box_size / 2.0, box_size / 2.0, size=3)

        molecule_positions = random_walk_positions_3d(
            n_particles=n_particles,
            step_size=bond_r0,
            box_size=box_size,
            rng=rng,
            start_pos=center_offset,
        )

        positions.extend(tuple(row) for row in molecule_positions.tolist())
        velocities.extend([(0.0, 0.0, 0.0)] * n_particles)

        molecule_bonds = [
            [offset + i, offset + i + 1]
            for i in range(n_particles - 1)
        ]
        bonds.extend(molecule_bonds)
        bond_typeids.extend([0] * len(molecule_bonds))

        molecule_angles = [
            [offset + i, offset + i + 1, offset + i + 2]
            for i in range(n_particles - 2)
        ]
        angles.extend(molecule_angles)
        angle_typeids.extend([0] * len(molecule_angles))

        offset += n_particles

    assert len(typeids) == total_particles
    assert len(positions) == total_particles
    assert len(velocities) == total_particles
    assert len(bonds) == len(bond_typeids)
    assert len(angles) == len(angle_typeids)

    snapshot.particles.typeid = typeids
    snapshot.particles.position = positions
    snapshot.particles.velocity = velocities
    snapshot.particles.validate()

    snapshot.bonds.N = len(bonds)
    snapshot.bonds.types = list(par.bond_types)
    snapshot.bonds.typeid = bond_typeids
    snapshot.bonds.group = bonds
    snapshot.bonds.validate()

    snapshot.angles.N = len(angles)
    snapshot.angles.types = list(par.angle_types)
    snapshot.angles.typeid = angle_typeids
    snapshot.angles.group = angles
    snapshot.angles.validate()

    print(
        f"Snapshot created with {num_mols} molecules in a box of side {box_size:.3f}")
    print(
        f"Total particles: {total_particles}, "
        f"bonds: {len(bonds)}, angles: {len(angles)}"
    )

    return snapshot
