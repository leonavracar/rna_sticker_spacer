"""Simulation utilities for the one-particle A/U/G/C sticker-spacer model.

This module contains:
- HOOMD force initialization,
- integrator setup,
- simulation state setup,
- logging and writing,
- simulation execution with optional kinetic-energy stopping.

The model-specific particle, bond, and angle names are expected to be attached
to ``par`` from ``params.base_info()`` before these functions are called.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypeAlias

import gsd.hoomd
import hoomd
from hoomd.custom import Action
from hoomd.logging import log
from hoomd.sqd_plugin import sqd_potential

from setup import define_pairwise_sqd_sticker_spacer


ForceList: TypeAlias = list[Any]
ForceObjects: TypeAlias = dict[str, Any]

NLIST_EXCLUSIONS: tuple[str, ...] = ("bond", "1-3", "1-4")


class RuntimeClock(Action):
    """Expose elapsed wall time as a HOOMD logger quantity."""

    def __init__(self) -> None:
        """Initialize the wall-time counter."""
        super().__init__()
        self._t0 = time.perf_counter()

    def act(self, timestep: int) -> None:
        """Required HOOMD action hook; no per-step action is needed."""
        return None

    @log(category="scalar")
    def walltime(self) -> float:
        """Return elapsed wall time in seconds."""
        return time.perf_counter() - self._t0


def validate_simulation_model_info(par: argparse.Namespace) -> None:
    """Validate fixed model metadata required by this simulation module."""
    required_attrs = [
        "type_names",
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
            f"Expected bond_types ['Base-Base'], got {par.bond_types}.")

    if list(par.angle_types) != ["Base-Base-Base"]:
        raise ValueError(
            "Expected angle_types ['Base-Base-Base'], "
            f"got {list(par.angle_types)}."
        )


def configure_bond_force(
    bond_force: Any,
    par: argparse.Namespace,
) -> None:
    """Assign harmonic bond parameters for consecutive bases."""
    bond_force.params["Base-Base"] = {
        "k": float(par.bond_k),
        "r0": float(par.bond_r0),
    }

    print("Bond force configured.")
    print(f"  Base-Base: k={float(par.bond_k):g}, r0={float(par.bond_r0):g}")


def configure_angle_force(
    angle_force: Any,
    par: argparse.Namespace,
) -> None:
    """Assign harmonic angle parameters for consecutive base triplets."""
    angle_force.params["Base-Base-Base"] = {
        "k": float(par.angle_k_ppp),
        "t0": float(par.angle_ppp_t0),
    }

    print("Angle force configured.")
    print(
        "  Base-Base-Base: "
        f"k={float(par.angle_k_ppp):g}, t0={float(par.angle_ppp_t0):g}"
    )


def initialize_model(par: argparse.Namespace) -> tuple[ForceList, ForceObjects]:
    """Create SQD, bond, and angle forces for the A/U/G/C model."""
    validate_simulation_model_info(par)

    print("Initializing model forces...")
    print(f"  nlist_rcut: {float(par.nlist_rcut):g}")
    print(f"  buffer: {float(par.buffer):g}")
    print(f"  exclusions: {NLIST_EXCLUSIONS}")

    nlist = hoomd.md.nlist.Tree(
        buffer=float(par.buffer),
        default_r_cut=float(par.nlist_rcut),
        exclusions=NLIST_EXCLUSIONS,
    )

    sqd = sqd_potential.SquareDensity(
        nlist,
        default_r_cut=float(par.nlist_rcut),
    )
    define_pairwise_sqd_sticker_spacer(sqd=sqd, par=par)

    bond_harmonic = hoomd.md.bond.Harmonic()
    configure_bond_force(bond_force=bond_harmonic, par=par)

    angle_harmonic = hoomd.md.angle.Harmonic()
    configure_angle_force(angle_force=angle_harmonic, par=par)

    forces: ForceList = [
        sqd,
        bond_harmonic,
        angle_harmonic,
    ]

    force_objs: ForceObjects = {
        "sqd": sqd,
        "bond_harmonic": bond_harmonic,
        "angle_harmonic": angle_harmonic,
    }

    print("Model forces initialized.")

    return forces, force_objs


def setup_langevin_method(par: argparse.Namespace) -> Any:
    """Create a Langevin constant-volume method with per-type drag coefficients."""
    method = hoomd.md.methods.Langevin(
        filter=hoomd.filter.All(),
        kT=float(par.kT),
    )

    for particle_type in par.type_names:
        method.gamma[particle_type] = float(par.gamma)

    print("Langevin thermostat configured.")
    print(f"  kT: {float(par.kT):g}")
    print(f"  gamma: {float(par.gamma):g}")

    return method


def setup_dpd_force(par: argparse.Namespace) -> Any:
    """Create a zero-conservative-force DPD thermostat force."""
    nlist = hoomd.md.nlist.Cell(buffer=float(par.buffer))

    dpd = hoomd.md.pair.DPD(
        nlist=nlist,
        kT=float(par.kT),
        default_r_cut=float(par.dpd_r_cut),
    )

    for t1 in par.type_names:
        for t2 in par.type_names:
            dpd.params[(t1, t2)] = {
                "A": 0.0,
                "gamma": float(par.gamma),
            }

    print("DPD thermostat force configured.")
    print(f"  kT: {float(par.kT):g}")
    print(f"  gamma: {float(par.gamma):g}")
    print(f"  dpd_r_cut: {float(par.dpd_r_cut):g}")

    return dpd


def setup_integrator(
    par: argparse.Namespace,
    forces: Sequence[Any],
) -> Any:
    """Build the HOOMD integrator with the requested thermostat."""
    integrator_forces = list(forces)
    thermostat = str(par.thermostat).lower()

    if thermostat == "langevin":
        method = setup_langevin_method(par)

    elif thermostat == "dpd":
        method = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
        integrator_forces.append(setup_dpd_force(par))

    else:
        raise ValueError(
            f"Unknown thermostat {par.thermostat!r}. "
            "Supported values are 'langevin' and 'DPD'."
        )

    integrator = hoomd.md.Integrator(
        dt=float(par.dt),
        methods=[method],
        forces=integrator_forces,
    )

    integrator.integrate_rotational_dof = False

    print("Integrator configured.")
    print(f"  thermostat: {par.thermostat}")
    print(f"  dt: {float(par.dt):g}")
    print(f"  number of forces: {len(integrator_forces)}")

    return integrator


def setup_simulation(
    snapshot: gsd.hoomd.Frame | None = None,
    restart_from: str | Path | None = None,
    restart_frame: int = -1,
    seed: int = 4,
) -> hoomd.Simulation:
    """Create a HOOMD simulation from a fresh snapshot or a restart GSD."""
    if snapshot is None and restart_from is None:
        raise ValueError("Provide either snapshot or restart_from.")

    if snapshot is not None and restart_from is not None:
        raise ValueError("Provide snapshot or restart_from, not both.")

    sim = hoomd.Simulation(
        device=hoomd.device.GPU(),
        seed=int(seed),
    )

    if restart_from is not None:
        sim.create_state_from_gsd(
            filename=str(restart_from),
            frame=int(restart_frame),
        )
        print(
            f"Simulation created from restart: {restart_from} "
            f"(frame {restart_frame})"
        )
        return sim

    sim.create_state_from_snapshot(snapshot)
    print("Simulation created from fresh snapshot.")

    return sim


def add_logger(
    sim: hoomd.Simulation,
    par: argparse.Namespace,
    force_objs: ForceObjects,
) -> tuple[Any, Any]:
    """Attach scalar logging for runtime, thermodynamics, and optional energies."""
    logger = hoomd.logging.Logger(categories=["scalar"])
    logger.add(sim, quantities=["timestep", "tps"])

    runtime_clock = RuntimeClock()
    logger.add(runtime_clock, quantities=["walltime"])

    thermo = hoomd.md.compute.ThermodynamicQuantities(
        filter=hoomd.filter.All(),
    )
    sim.operations.computes.append(thermo)

    logger.add(
        thermo,
        quantities=[
            "kinetic_energy",
            "kinetic_temperature",
        ],
    )

    if bool(par.log_energies):
        logger.add(thermo, quantities=["potential_energy"])
        logger.add(force_objs["sqd"], quantities=["energy"])
        logger.add(force_objs["bond_harmonic"], quantities=["energy"])
        logger.add(force_objs["angle_harmonic"], quantities=["energy"])
        print("Logger created with energy logging.")
    else:
        print("Logger created without per-force energy logging.")

    return logger, thermo


def add_writer(
    sim: hoomd.Simulation,
    par: argparse.Namespace,
    logger: Any,
) -> Any:
    """Attach a periodic GSD trajectory writer to the simulation."""
    output_dir = Path(par.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = output_dir / f"trajectory{par.suffix}.gsd"

    gsd_writer = hoomd.write.GSD(
        filename=str(filename),
        trigger=hoomd.trigger.Periodic(int(par.N_write)),
        mode="wb",
        dynamic=["property", "momentum"],
        logger=logger,
    )

    sim.operations.writers.append(gsd_writer)

    print("GSD writer created.")
    print(f"  filename: {filename}")
    print(f"  write period: {int(par.N_write)}")

    return gsd_writer


def add_progress_printer(
    sim: hoomd.Simulation,
    logger: Any,
    period: int = 100_000,
) -> Any:
    """Attach a periodic table writer for console progress output."""
    table = hoomd.write.Table(
        trigger=hoomd.trigger.Periodic(int(period)),
        logger=logger,
    )

    sim.operations.writers.append(table)

    print("Progress table writer created.")
    print(f"  period: {int(period)}")

    return table


def run_simulation(
    sim: hoomd.Simulation,
    par: argparse.Namespace,
    thermo: Any,
) -> None:
    """Run the simulation, optionally stopping on high kinetic energy per particle."""
    n_steps = int(par.N_steps)
    if n_steps < 0:
        raise ValueError(f"N_steps must be non-negative, got {n_steps}.")

    max_ke_per_particle_stop = par.max_ke_per_particle_stop

    print("Starting simulation run.")
    print(f"  N_steps: {n_steps}")
    print(f"  current timestep: {sim.timestep}")

    if max_ke_per_particle_stop is None:
        sim.run(n_steps)
        print(f"Simulation finished at timestep {sim.timestep}.")
        return

    max_ke_per_particle_stop = float(max_ke_per_particle_stop)
    if max_ke_per_particle_stop <= 0:
        raise ValueError(
            "max_ke_per_particle_stop must be positive when enabled, "
            f"got {max_ke_per_particle_stop}."
        )

    start_check_after_steps = 100_000
    check_period = int(par.N_write)

    if check_period <= 0:
        raise ValueError(f"N_write must be positive, got {check_period}.")

    print("Kinetic-energy stopping enabled.")
    print(f"  max KE/N: {max_ke_per_particle_stop:g}")
    print(f"  checking after timestep: {start_check_after_steps}")
    print(f"  check period: {check_period}")

    steps_done = 0

    while steps_done < n_steps:
        steps_to_run = min(check_period, n_steps - steps_done)
        sim.run(steps_to_run)
        steps_done += steps_to_run

        if sim.timestep < start_check_after_steps:
            continue

        max_ke = max_ke_per_particle_stop * sim.state.N_particles

        if thermo.kinetic_energy > max_ke:
            print(
                f"Stopped early at timestep {sim.timestep}: "
                f"KE/N exceeded {max_ke_per_particle_stop:g}. "
                f"Total KE={thermo.kinetic_energy:g}, threshold={max_ke:g}."
            )
            break

    print(f"Simulation run ended at timestep {sim.timestep}.")
