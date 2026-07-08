"""Command-line parameter definitions for RNA sticker-spacer simulations.

Notes
-----
restart_from:
    When restarting from a simulation state, the parameter order in parameters.txt
    for the old and new simulation must be the same. Also check n_replicates and
    folder_name naming logic.
"""

import argparse


DEFAULTS = {
    "V1": {
        "sqd": {
            "r_rep": 0.25,
            "r_att": 0.75,
            "ncut_wc": 1.061,  # valency
            "density_upper_rcut": 1.25,
            "density_lower_rcut": 0.75,
            "nlist_rcut": 1.25,
        },
    },
    "V2": {
        "sqd": {
            "r_rep": 0.25,
            "r_att": 0.75,
            "ncut_wc": 1.061,  # valency
            "density_upper_rcut": 0.807,
            "density_lower_rcut": 0.48,
            "nlist_rcut": 0.807,
        },
    },
}


def get_param_mode(argv: list[str] | None = None) -> str:
    """Read only --param_mode before building the full parser."""
    mode_parser = argparse.ArgumentParser(add_help=False)
    mode_parser.add_argument(
        "--param_mode",
        choices=list(DEFAULTS),
        default="V1",
    )

    mode_args, _ = mode_parser.parse_known_args(argv)
    return mode_args.param_mode


def get_params(argv: list[str] | None = None) -> argparse.ArgumentParser:
    """Create and return the command-line argument parser for simulations."""

    param_mode = get_param_mode(argv)
    sqd_defaults = DEFAULTS[param_mode]["sqd"]

    parser = argparse.ArgumentParser(
        description=f"Simulation Parameters. Default parameter mode: {param_mode}"
    )

    parser.add_argument(
        "--param_mode",
        choices=list(DEFAULTS),
        default=param_mode,
        help=(
            "Parameter default mode to use. "
            "V1 uses exact-valency SQD cutoffs. "
            "V2 uses easier-switching SQD cutoffs."
        ),
    )

    thermostat = parser.add_argument_group("Thermostat")
    io_files = parser.add_argument_group("Input/output files")
    simulation = parser.add_argument_group("Simulation setup")
    seeds = parser.add_argument_group("Random seeds")
    restart = parser.add_argument_group("Restart options")
    geometry = parser.add_argument_group("Bond and angle parameters")
    sqd = parser.add_argument_group("SQD parameters")

    # Thermostat
    thermostat.add_argument(
        "--thermostat",
        type=str,
        default="langevin",
        help="Thermostat method to use.",
    )
    thermostat.add_argument(
        "--dpd_r_cut",
        type=float,
        default=2.0,
        help="Distance cutoff for DPD thermostat.",
    )

    # Input/output files
    io_files.add_argument(
        "--param_file",
        type=str,
        help="Path to a .txt parameter file.",
    )
    io_files.add_argument(
        "--molecule_file",
        type=str,
        default="molecules.json",
        help="Path to JSON file defining molecules.",
    )
    io_files.add_argument(
        "--output_dir",
        type=str,
        default="output_dir",
        help=(
            "Path to a directory containing input files such as parameters.txt "
            "and molecules.json."
        ),
    )
    io_files.add_argument(
        "--suffix",
        type=str,
        default="_",
        help="Suffix added to trajectory.gsd filename.",
    )
    io_files.add_argument(
        "--log_energies",
        action="store_true",
        help=(
            "Log per-force energy terms. By default, only timestep, TPS, "
            "and walltime are logged."
        ),
    )

    # Simulation setup
    simulation.add_argument(
        "--kT",
        type=float,
        default=1.0,
        help="Temperature factor.",
    )
    simulation.add_argument(
        "--N_steps",
        type=int,
        default=40_000_000,
        help="Number of simulation steps.",
    )
    simulation.add_argument(
        "--N_write",
        type=int,
        default=10_000,
        help="Write simulation state to the GSD file every N steps.",
    )
    simulation.add_argument(
        "--max_ke_per_particle_stop",
        type=float,
        default=None,
        help=(
            "Stop simulation if kinetic energy per particle exceeds this value "
            "after 100k steps. Disabled if None."
        ),
    )
    simulation.add_argument(
        "--density",
        type=float,
        default=0.001,
        help=(
            "Particle number density. Used to compute cubic simulation box side "
            "length L = (N_particles / density) ** (1/3)."
        ),
    )
    simulation.add_argument(
        "--dt",
        type=float,
        default=0.01,
        help="Simulation timestep.",
    )
    simulation.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help="Friction factor.",
    )
    simulation.add_argument(
        "--dry_run",
        action="store_true",
        help="Prepare inputs and report settings without actually running the simulation.",
    )

    # Random seeds
    seeds.add_argument(
        "--init_seed",
        type=int,
        default=None,
        help=(
            "Seed for random initial configuration generation. If omitted, "
            "a random 32-bit seed is generated."
        ),
    )
    seeds.add_argument(
        "--hoomd_seed",
        type=int,
        default=None,
        help=(
            "Seed for HOOMD stochastic dynamics. Must be in [1, 65535]. "
            "If omitted, a random seed in this range is generated."
        ),
    )

    # Restart options
    restart.add_argument(
        "--restart_from",
        type=str,
        default=None,
        help="Path to a .gsd file to restart the simulation from.",
    )
    restart.add_argument(
        "--restart_frame",
        type=int,
        default=-1,
        help="Frame index for the state to restart from.",
    )

    # SQD / interaction parameters
    sqd.add_argument(
        "--E_att",
        type=float,
        default=4.0,
        help="Watson-Crick attraction strength.",
    )
    sqd.add_argument(
        "--E_rep",
        type=float,
        default=20.0,
        help="Standard repulsion strength between particles.",
    )
    sqd.add_argument(
        "--r_rep",
        type=float,
        default=sqd_defaults["r_rep"],
        help="Repulsive interaction cutoff distance.",
    )
    sqd.add_argument(
        "--r_att",
        type=float,
        default=sqd_defaults["r_att"],
        help="Attractive interaction cutoff distance.",
    )
    sqd.add_argument(
        "--ncut_wc",
        type=float,
        default=sqd_defaults["ncut_wc"],
        help="Valency/contact cutoff for Watson-Crick interactions.",
    )
    sqd.add_argument(
        "--nlist_rcut",
        type=float,
        default=sqd_defaults["nlist_rcut"],
        help=(
            "Neighbor-list and SQD pair-search cutoff. Usually set equal to "
            "density_upper_rcut."
        ),
    )
    sqd.add_argument(
        "--buffer",
        type=float,
        default=0.3,
        help=(
            "Neighbor-list buffer distance added beyond the interaction cutoff. "
            "Larger values reduce rebuild frequency but increase neighbor-list size."
        ),
    )
    sqd.add_argument(
        "--density_upper_rcut",
        type=float,
        default=sqd_defaults["density_upper_rcut"],
        help=(
            "Upper boundary for r_d. In V1, 1.25 gives exact valency of 1 "
            "for r_att=0.75 and r_rep=0.25. In V2, 0.807 makes bond switching "
            "easier but gives valency > 1."
        ),
    )
    sqd.add_argument(
        "--density_lower_rcut",
        type=float,
        default=sqd_defaults["density_lower_rcut"],
        help=(
            "Lower boundary for r_d. In V1, this is set equal to r_att for "
            "exact valency of 1. In V2, this is 0.48."
        ),
    )

    # Bond and angle parameters
    geometry.add_argument(
        "--bond_r0",
        type=float,
        default=1.0,
        help="Equilibrium bond length.",
    )
    geometry.add_argument(
        "--bond_k",
        type=float,
        default=100.0,
        help=(
            "Bond stiffness. Higher values increase resistance to stretching "
            "or compression."
        ),
    )
    geometry.add_argument(
        "--angle_k_ppp",
        type=float,
        default=2.5,
        help="Angle stiffness for Phosphate-Phosphate-Phosphate angles.",
    )
    geometry.add_argument(
        "--angle_ppp_t0",
        type=float,
        default=2.7,
        help="Equilibrium Phosphate-Phosphate-Phosphate angle in radians.",
    )

    return parser


def base_info() -> dict[str, object]:
    """Return fixed particle, bond, angle, and complement definitions for the model."""
    return {
        "type_names": ["A", "U", "G", "C"],
        "complements": {
            "A": "U",
            "U": "A",
            "G": "C",
            "C": "G",
        },
        "bond_types": ["Base-Base"],
        "angle_types": ["Base-Base-Base"],
    }
