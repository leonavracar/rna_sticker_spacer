"""Run one replicate of the one-particle A/U/G/C sticker-spacer simulation.

This script:
- parses command-line parameters,
- attaches fixed model metadata from params.base_info(),
- loads molecule sequences,
- assigns random seeds when needed,
- creates a fresh or restarted HOOMD simulation,
- FIRE-minimizes fresh snapshots,
- writes trajectory and metadata,
- runs the simulation.

The model itself is defined by:
- params.py for command-line/default parameters,
- setup.py for molecule parsing and initial snapshot creation,
- simulation.py for HOOMD force/integrator/writer setup.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import time
from pathlib import Path
from typing import Any

import hoomd

from params import base_info, get_params
from setup import parse_molecules, snapshot_sticker_spacer
from simulation import (
    add_logger,
    add_progress_printer,
    add_writer,
    initialize_model,
    run_simulation,
    setup_integrator,
    setup_simulation,
)


def attach_base_info(par: argparse.Namespace) -> None:
    """Attach fixed model metadata from params.base_info() to the namespace."""
    par.__dict__.update(base_info())


def assign_missing_seeds(par: argparse.Namespace) -> None:
    """Assign random initialization and HOOMD seeds if they were not provided."""
    if par.init_seed is None:
        par.init_seed = secrets.randbelow(2**32)

    if par.hoomd_seed is None:
        par.hoomd_seed = secrets.randbelow(65_535) + 1

    if not (1 <= int(par.hoomd_seed) <= 65_535):
        raise ValueError(
            f"hoomd_seed must be in [1, 65535], got {par.hoomd_seed}."
        )

    print("Seeds assigned.")
    print(f"  init_seed: {par.init_seed}")
    print(f"  hoomd_seed: {par.hoomd_seed}")


def resolve_molecule_file(par: argparse.Namespace) -> Path:
    """Resolve the molecule JSON path using common single-run and sweep layouts."""
    molecule_file = Path(par.molecule_file)

    if molecule_file.is_absolute():
        candidates = [molecule_file]
    else:
        output_dir = Path(par.output_dir)
        candidates = [
            molecule_file,
            output_dir / molecule_file,
            output_dir.parent / molecule_file,
        ]

    for candidate in candidates:
        if candidate.exists():
            print(f"Molecule file found: {candidate}")
            return candidate

    candidate_list = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "Molecule file was not found. Tried:\n"
        f"{candidate_list}"
    )


def namespace_to_jsonable(par: argparse.Namespace) -> dict[str, Any]:
    """Convert an argparse namespace to a JSON-serializable dictionary."""
    return dict(par.__dict__)


def write_params_once(par: argparse.Namespace) -> None:
    """Write params.json once per output directory using a simple lock."""
    output_dir = Path(par.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    params_path = output_dir / "params.json"
    lock_dir = output_dir / ".params_json.lock"

    if params_path.exists():
        print(f"Parameters already exist: {params_path}")
        return

    got_lock = False

    try:
        lock_dir.mkdir()
        got_lock = True

        if params_path.exists():
            print(f"Parameters already exist: {params_path}")
            return

        tmp_path = output_dir / f".params{par.suffix}.tmp"

        with tmp_path.open("w") as handle:
            json.dump(namespace_to_jsonable(par),
                      handle, indent=2, default=str)

        tmp_path.replace(params_path)
        print(f"Parameters written: {params_path}")

    except FileExistsError:
        print(
            f"Another process is writing params.json in {output_dir}; skipping.")
        return

    finally:
        if got_lock:
            try:
                lock_dir.rmdir()
            except OSError:
                pass


def write_run_metadata(
    par: argparse.Namespace,
    start_time: float,
    walltime_seconds: float | None = None,
) -> None:
    """Write per-replicate metadata including seeds, command, and wall time."""
    metadata_dir = Path(par.output_dir) / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "suffix": par.suffix,
        "param_mode": par.param_mode,
        "init_seed": par.init_seed,
        "hoomd_seed": par.hoomd_seed,
        "molecule_file": par.molecule_file,
        "output_dir": par.output_dir,
        "restart_from": par.restart_from,
        "restart_frame": par.restart_frame,
        "command": "python " + " ".join(sys.argv),
        "start_time_unix": start_time,
        "walltime_seconds": walltime_seconds,
        "walltime_hours": (
            walltime_seconds / 3600.0
            if walltime_seconds is not None
            else None
        ),
    }

    metadata_path = metadata_dir / f"run_metadata{par.suffix}.json"

    with metadata_path.open("w") as handle:
        json.dump(metadata, handle, indent=2, default=str)

    print(f"Run metadata written: {metadata_path}")


def create_simulation_state(par: argparse.Namespace) -> hoomd.Simulation:
    """Create a simulation either from a restart GSD or from a fresh snapshot."""
    if par.restart_from is not None:
        restart_path = Path(par.restart_from)

        if not restart_path.exists():
            raise FileNotFoundError(f"Restart file not found: {restart_path}")

        print("Creating simulation from restart file.")
        print(f"  restart_from: {restart_path}")
        print(f"  restart_frame: {par.restart_frame}")

        return setup_simulation(
            restart_from=restart_path,
            restart_frame=par.restart_frame,
            seed=par.hoomd_seed,
        )

    print("Creating fresh simulation snapshot.")
    snapshot = snapshot_sticker_spacer(par)

    return setup_simulation(
        snapshot=snapshot,
        seed=par.hoomd_seed,
    )


def minimize_and_thermalize(
    sim: hoomd.Simulation,
    par: argparse.Namespace,
    forces: list[Any],
    start_time: float,
) -> None:
    """FIRE-minimize a fresh state and thermalize particle momenta."""
    print("Starting FIRE minimization.")

    fire = hoomd.md.minimize.FIRE(
        dt=1e-3,
        force_tol=5e-1,
        angmom_tol=5e-1,
        energy_tol=5e-1,
        alpha_start=0.1,
        forces=forces,
        methods=[hoomd.md.methods.ConstantVolume(hoomd.filter.All())],
    )

    sim.operations.integrator = fire

    iteration = 0

    while not fire.converged:
        sim.run(100)
        iteration += 1

        if iteration % 100 == 0:
            print(f"FIRE: timestep={sim.timestep}, converged={fire.converged}")
            sys.stdout.flush()

    print(
        f"Minimization completed after {time.time() - start_time:.2f} seconds.")

    sim.state.thermalize_particle_momenta(
        filter=hoomd.filter.All(),
        kT=float(par.kT),
    )

    print("Particle momenta thermalized.")
    print(f"  kT: {float(par.kT):g}")


def print_run_summary(par: argparse.Namespace) -> None:
    """Print a short summary of the run setup."""
    print("Run setup summary:")
    print(f"  param_mode: {par.param_mode}")
    print(f"  output_dir: {par.output_dir}")
    print(f"  suffix: {par.suffix}")
    print(f"  molecule_file: {par.molecule_file}")
    print(f"  N_steps: {par.N_steps}")
    print(f"  N_write: {par.N_write}")
    print(f"  density: {par.density}")
    print(f"  thermostat: {par.thermostat}")
    print(f"  dry_run: {par.dry_run}")
    print(f"  restart_from: {par.restart_from}")


def main() -> None:
    """Parse parameters, initialize HOOMD, and run one simulation replicate."""
    base_dir = Path(__file__).resolve().parent
    os.chdir(base_dir)

    start_time = time.time()

    parser = get_params()
    par = parser.parse_args()

    attach_base_info(par)
    assign_missing_seeds(par)

    molecule_file = resolve_molecule_file(par)
    par.molecule_file = str(molecule_file)
    par.sequences = parse_molecules(file=molecule_file)

    print_run_summary(par)
    write_params_once(par)

    sim = create_simulation_state(par)
    forces, force_objs = initialize_model(par)

    if par.restart_from is None:
        minimize_and_thermalize(
            sim=sim,
            par=par,
            forces=forces,
            start_time=start_time,
        )
        start_time = time.time()
    else:
        print("Restart mode: skipping FIRE minimization and momentum thermalization.")

    logger, thermo = add_logger(
        sim=sim,
        par=par,
        force_objs=force_objs,
    )

    gsd_writer = add_writer(
        sim=sim,
        par=par,
        logger=logger,
    )

    sim.operations.integrator = setup_integrator(
        par=par,
        forces=forces,
    )

    add_progress_printer(
        sim=sim,
        logger=logger,
        period=100_000,
    )

    if par.dry_run:
        print("Dry run requested; setup completed without running dynamics.")
        gsd_writer.flush()
        return

    write_run_metadata(par=par, start_time=start_time)

    print(f"Running {par.N_steps} steps...")

    try:
        run_simulation(
            sim=sim,
            par=par,
            thermo=thermo,
        )
        gsd_writer.flush()

    finally:
        write_run_metadata(
            par=par,
            start_time=start_time,
            walltime_seconds=time.time() - start_time,
        )

    print(f"Simulation finished in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
