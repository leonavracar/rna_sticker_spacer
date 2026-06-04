import argparse
import numpy as np


"""Notes:
* restart_from: when running from simulation state, parameters.txt order of old + new sim has to be the same, n_replicates and folder_name naming logic
"""


def get_params():

    # Parameters
    parser = argparse.ArgumentParser(description="Simulation Parameters")

    # Thermostat type
    parser.add_argument("--thermostat", type=str, default="langevin", help="")
    parser.add_argument(
        "--dpd_r_cut", type=float, default=2, help="Distance cutoff for DPD"
    )
    # Logging of potential energies
    parser.add_argument("--log_energies", action="store_false",
                    help="Log per-force energy terms (default: only timestep + tps + walltime).")
    # Input files
    parser.add_argument("--param_file", type=str, help="Path to a .txt file")
    parser.add_argument(
        "--molecule_file",
        type=str,
        default="AGUC30.json",
        help="Path to json file defining molecules",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_dir",
        help="Path to a directory containing inputs: parameters.txt and molecules.json files.",
    )
    parser.add_argument("--suffix", type=str, default="_", help="Suffix added to trajectory.gsd filename.")

    # Simulation setup
    parser.add_argument(
        "--kT", type=float, default=1, help="Temperature factor."
    )
    parser.add_argument(
        "--N_steps", type=int, default=40_000_000, help="Steps in simulation."
    )
    parser.add_argument(
        "--N_write",
        type=int,
        default=10_000, # 2k default
        help="After how many steps state is documented in gsd file",
    )
    parser.add_argument(
    "--max_ke_per_particle_stop",
    type=float,
    default=None,
    help="Stop simulation if KE per particle exceeds this value after 100k steps. Disabled if None.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1,
        help="Temp dependend scaling factor for energies = 1/kT",
    )
    parser.add_argument(
        "--density", type=float, default=0.001, help="Particle number density, used to compute the cubic simulation box side length L = (N_particles / density) ** (1/3)."
    )
    parser.add_argument(
        "--dt", type=float, default=0.01, help="Timestep"
    ) 
    parser.add_argument(
        "--gamma", type=float, default=0.1, help="Friction factor"
    )  # 0.1
    parser.add_argument(
        "--dry_run", action="store_true", help="Whether to actually run."
    )
    
    parser.add_argument(
        "--structure_type",
        type=str,
        default="sticker_spacer",
        help="Model architecture, see setup.py",
    )
    parser.add_argument(
    "--nlist_rcut",
    type=float,
    default=0.807,
    help=
        "Neighbor-list and SQD pair-search cutoff, set to density_upper_rcut_base",)
    
    parser.add_argument(
        "--buffer",
        type=float,
        default=0.3,
        help="Neighbor-list buffer distance added beyond the interaction cutoff; larger values reduce rebuild frequency but increase neighbor-list size.",
    )
    # Seed for reproduciblity
    parser.add_argument(
        "--init_seed",
        type=int,
        default=None,
        help="Seed for random initial configuration generation. If omitted, a random 32-bit seed is generated.",
    )

    parser.add_argument(
        "--hoomd_seed",
        type=int,
        default=None,
        help="Seed for HOOMD stochastic dynamics. Must be in [1, 65535]. If omitted, a random seed in this range is generated.",
    )

    # Run sim from state
    parser.add_argument(
        "--restart_from",
        type=str,
        default=None,
        help="Path to a .gsd file to run sim from its state",
    )
    parser.add_argument(
        "--restart_frame",
        type=int,
        default=-1,
        help="Frame index for the state to run sim from",
    )

    # Interaction energy params
    parser.add_argument(
        "--E_att_nonwc_min", type=float, default=0, help="non-WC attraction"
    )
    parser.add_argument("--E_att_nonwc_max", type=float, default=4)

    parser.add_argument("--E_att", type=float, default=10, help="WC attraction")
    parser.add_argument(
        "--E_rep", type=float, default=20, help="standard repulsion between particles"
    )  # Repulsive scale

    parser.add_argument(
        "--E_rep_wc_nondir",
        type=float,
        default=20,
        help="Repulsion for duplicate base of wc paired bases e.g. A1-U1",
    )
    parser.add_argument("--E_rep_wc", type=float, default=20)
    parser.add_argument(
        "--E_att_back", type=float, default=0
    )  # no backbone attraction when zero (0mM Mg2+)

    ## Interaction distance cutoffs (standard for particles except WC interaction)
    parser.add_argument("--r_rep", type=float, default=1.0)
    parser.add_argument("--r_att", type=float, default=1.5)
    parser.add_argument(
        "--r_d", type=float, default=2.0
    )  # "density" cutoff (used in SquareDensity)

    ## Interaction distance cutoffs for WC and non-WC interaction
    parser.add_argument("--r_rep_base", type=float, default=0.25)
    parser.add_argument("--r_att_base", type=float, default=0.75)
    parser.add_argument("--r_d_base", type=float, default=1)

    ## Valency and contact params
    valency_low = 1.03
    valency_high = 1.061

    parser.add_argument(
        "--ncut_ij_comp", type=float, default=valency_low, help="valency of WC"
    )

    parser.add_argument("--ncut_ij_nonwc", type=float, default=valency_high)
    # parser.add_argument('--ncut_ji_nonwc', type=int, default=valency)

    parser.add_argument(
        "--ncut_ij_back", type=float, default=12
    )  # 12 arrives from the way spheres can pack

    parser.add_argument(
        "--ncut_ij_wc_nondir",
        type=float,
        default=valency_high,
        help="Cutoff for non-directional WC e.g. A1-U2, avoids valency being satisfied",
    )

    # Bond and angle params
    parser.add_argument(
        "--bond_length", type=float, default=1
    )  # equilibrium bond length
    parser.add_argument(
        "--bond_k", type=float, default=100.0
    )  # Thurmalai, how stiff bond is (higher = more resistance to stretching/compression)

    parser.add_argument("--angle_k_ppp", type=float, default=100)
    parser.add_argument("--angle_k_ppb", type=float, default=75)
    parser.add_argument("--angle_k_bpb", type=float, default=50)
    parser.add_argument("--angle_k_bpp", type=float, default=35)

    # directional-branched
    parser.add_argument(
        "--angle_branch_dir_ppp_t0", type=float, default=2.7
    )  # 2.7 Phosphate-Phosphate-Phosphate, in radians
    parser.add_argument(
        "--angle_branch_dir_ppb_t0", type=float, default=2 * np.pi / 3
    )  # Phosphate-Phosphate-Base, in radians
    parser.add_argument(
        "--angle_branch_dir_bpb_t0", type=float, default=np.pi / 3
    )  # Base-Phosphate-Base, in radians

    parser.add_argument("--bond_bp_branch_dir_r0", type=float, default=0.5)  # in nm
    parser.add_argument("--bond_pp_branch_dir_r0", type=float, default=1)  # in nm

    parser.add_argument("--k_dihe_stacking", type=float, default=2)
    parser.add_argument("--k_dihe_direction", type=float, default=5)

    # SQD V2.0 parameters
    parser.add_argument(
        "--density_upper_rcut",
        type=float,
        default=1.571,
        help="Upper boundary for r_d, fits r_rep=1, r_a=1.5",
    )
    parser.add_argument(
        "--density_lower_rcut",
        type=float,
        default=1.224,
        help="Lower boundary for r_d, fits r_rep=1, r_a=1.5",
    )
    parser.add_argument(
        "--sqd_version",
        type=int,
        default=5, # 5 for sticker spacer
        help="Which sqd version to use. v2 has additional r_d cutoffs",
    )

    parser.add_argument(
        "--density_upper_rcut_base",
        type=float,
        default=0.807,
        help="Upper boundary for r_d, fits r_rep=0.25, r_a=0.75",
    )
    parser.add_argument(
        "--density_lower_rcut_base",
        type=float,
        default=0.48,
        help="Lower boundary for r_d, fits r_rep=0.25, r_a=0.75",
    )

    # SQD V4 parameters (differentiated A1-U1 and A2-U2 interactions)
    parser.add_argument(
        "--E_att_2", type=float, default=0, help="Attraction for brothers A2-U2, C2-G2"
    )
    parser.add_argument("--E_rep_wc_2", type=float, default=20, help="")
    parser.add_argument("--r_att_base_2", type=float, default=1.5, help="")
    parser.add_argument("--r_rep_base_2", type=float, default=1, help="")
    parser.add_argument(
        "--density_upper_rcut_base_2", type=float, default=1.571, help=""
    )
    parser.add_argument(
        "--density_lower_rcut_base_2", type=float, default=1.224, help=""
    )
    parser.add_argument("--ncut_ij_comp_2", type=float, default=valency_high)

    # Align angle plugin
    parser.add_argument(
        "--use_align_angle", action="store_true", help="Enable align-angle force"
    )
    parser.add_argument(
        "--align_k", type=float, default=20.0, help="Strength of align-angle force"
    )
    parser.add_argument(
        "--align_multiplicity", type=int, default=2, help="1 = polar, 2 = nematic"
    )

    return parser


def derived_params(
    args,
):  # to do: see where this is used and if a function call is necessary
    # Derived parameters (not passed via CLI)
    derived = dict()
    derived["E_att_nonwc"] = -args.E_att_nonwc_min * args.beta
    derived["E_att_comp"] = -args.E_att * args.beta  # Attractive scale
    derived["E_rep_scaled"] = args.E_rep * args.beta  # scale repulsion by beta
    derived["E_rep_wc_scaled"] = args.E_rep_wc * args.beta
    derived["E_att_back_scaled"] = -args.E_att_back * args.beta
    derived["E_rep_wc_nondir_scaled"] = args.E_rep_wc_nondir * args.beta
    derived["E_att_comp_2"] = -args.E_att_2 * args.beta
    derived["E_rep_wc_2_scaled"] = args.E_rep_wc_2 * args.beta
    return derived


def base_info():
    base_info = dict()
    base_info["type_names"] = [
        "B",
        "A1",
        "A2",
        "U1",
        "U2",
        "C1",
        "C2",
        "G1",
        "G2",
        "A",
        "U",
        "G",
        "C",
    ]
    base_info["type_names_original"] = ["B", "A", "U", "G", "C"]
    
    base_info["complements"] = {
        "A1": "U1",
        "U1": "A1",
        "G1": "C1",
        "C1": "G1",
        "A2": "U2",
        "U2": "A2",
        "G2": "C2",
        "C2": "G2",
        "A": "U",
        "U": "A",
        "G": "C",
        "C": "G",
    }
    base_info["complements_1"] = {
        "A1": "U1",
        "U1": "A1",
        "G1": "C1",
        "C1": "G1",
    }
    base_info["complements_2"] = {
        "A2": "U2",
        "U2": "A2",
        "G2": "C2",
        "C2": "G2",
    }

    base_info["non_wc_complements"] = {
        "A1": ["C1", "A1", "G1"],
        "U1": ["C1", "U1", "G1"],
        "C1": ["C1", "U1", "A1"],
        "G1": ["G1", "A1", "U1"],
        "A2": ["C2", "A2", "G2"],
        "U2": ["C2", "U2", "G2"],
        "C2": ["C2", "U2", "A2"],
        "G2": ["G2", "A2", "U2"],
        "A": ["C", "G", "A"],
        "U": ["C", "G", "U"],
        "C": ["A", "U", "C"],
        "G": ["A", "U", "G"],
    }

    base_info["type_names_sticker"] = ["A", "U", "G", "C"]
    
    base_info["non_wc_original"] = {
        "A": ["A", "G", "C"],
        "U": ["U", "G", "C"],
        "G": ["A", "G", "U"],
        "C": ["A", "U", "C"],
    }
    base_info["complements_original"] = {"A": "U", "U": "A", "G": "C", "C": "G"}

    base_info["duplicates"] = {
        "A": ["A1", "A2"],
        "U": ["U1", "U2"],
        "C": ["C1", "C2"],
        "G": ["G1", "G2"],
    }

    base_info["wc_nondirectional"] = {
        "A1": "U2",
        "U1": "A2",
        "A2": "U1",
        "U2": "A1",
        "G1": "C2",
        "C1": "G2",
        "G2": "C1",
        "C2": "G1",
    }
    base_info["non_wc_nondirectional"] = {
        "A1": ["C2", "A2", "G2"],
        "U1": ["C2", "U2", "G2"],
        "C1": ["C2", "U2", "A2"],
        "G1": ["G2", "A2", "U2"],
        "A2": ["C1", "A1", "G1"],
        "U2": ["C1", "U1", "G1"],
        "C2": ["C1", "U1", "A1"],
        "G2": ["G1", "A1", "U1"],
    }

    base_info["all_nondirectional"] = {
        "A1": ["U2", "C2", "A2", "G2"],
        "U1": ["A2", "C2", "U2", "G2"],
        "C1": ["G2", "C2", "U2", "A2"],
        "G1": ["C2", "G2", "A2", "U2"],
        "A2": ["U1", "C1", "A1", "G1"],
        "U2": ["A1", "C1", "U1", "G1"],
        "C2": ["G1", "C1", "U1", "A1"],
        "G2": ["C1", "G1", "A1", "U1"],
    }

    return base_info
