import hoomd
import time
import os
import sys
import json
from pathlib import Path
from params import *
from setup import *
from simulation import *
import secrets


### Metadata for each replicate to log seeds
def write_run_metadata(par, start_time, walltime_seconds = None):
    metadata_dir = Path(par.output_dir) / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "suffix": par.suffix,
        "init_seed": par.init_seed,
        "hoomd_seed": par.hoomd_seed,
        "output_dir": par.output_dir,
        "command": "python " + " ".join(sys.argv),
        "walltime_seconds": walltime_seconds,
        "walltime_hours": walltime_seconds / 3600 if walltime_seconds else None,
    }

    metadata_path = metadata_dir / f"run_metadata{par.suffix}.json"

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"Run metadata written: {metadata_path}")


### Start of script ###
base_dir = Path(__file__).resolve().parent
os.chdir(base_dir)

start_time = time.time()

# Get parameters
par = get_params().parse_args()

# Save params.json for this parameter combination if it doesn't exist (replicates)
params_path = Path(par.output_dir) / "params.json"
run_dir = Path(par.output_dir)
sweep_dir = run_dir.parent

mol_file = Path(par.molecule_file)
if not mol_file.is_absolute():
    mol_file = sweep_dir / mol_file

par.sequences = parse_molecules(file=mol_file) # new fparameter added based on molecules.json
par.__dict__.update(base_info())
par.__dict__.update(derived_params(par))

# Set seed
# if par.seed is None:
#     par.seed = secrets.randbelow(65535) + 1

if par.init_seed is None:
    par.init_seed = secrets.randbelow(2**32)

if par.hoomd_seed is None:
    par.hoomd_seed = secrets.randbelow(65535) + 1

if not (1 <= par.hoomd_seed <= 65535):
    raise ValueError(f"hoomd_seed must be in [1, 65535], got {par.hoomd_seed}")


# Different pairing dictionaries for different models
if par.structure_type == "2_particles":
    par.type_names = par.type_names_original
    par.complements = par.complements_original
    par.non_wc_complements = par.non_wc_original
elif par.structure_type == "sticker_spacer":
    par.type_names = par.type_names_sticker
    par.complements = par.complements_original
    par.non_wc_complements = par.non_wc_original

if not params_path.exists():
    with open(params_path, "w") as f:
        json.dump(par.__dict__, f)

# Create simulation state, 
if par.restart_from is not None:
    restart_path=Path(par.restart_from)
    if not restart_path.exists():
        raise FileNotFoundError(f"Restart file not found: {restart_path}")
    sim = setup_simulation(restart_from=par.restart_from, restart_frame=par.restart_frame, seed =par.hoomd_seed)
else:
    if par.structure_type in['troubleshoot7', 'troubleshoot8']:
        snapshot = snapshot_ts7(par)
    elif par.structure_type == '2_particles':
        snapshot = snapshot_2_particles(par)
    elif par.structure_type == 'sticker_spacer':
        snapshot = snapshot_sticker_spacer(par)
    else:
        raise ValueError("Invalid structure_type.")
    sim = setup_simulation(snapshot=snapshot, seed = par.hoomd_seed)

#snapshot = snapshot_multiple_rnas(par)
forces, force_objs = initialize_model(par)

if par.restart_from is None:
    fire = hoomd.md.minimize.FIRE(
            dt=1e-3,
            force_tol=5e-1,
            angmom_tol=5e-1,
            energy_tol=5e-1,
            alpha_start=0.1,
            forces=forces, # change to list of potentials
            methods=[hoomd.md.methods.ConstantVolume(hoomd.filter.All())])

    sim.operations.integrator = fire

    i=0
    while not fire.converged:
        sim.run(100)
        i += 1
        if i % 100 == 0:
            print(f"FIRE: timestep={sim.timestep} converged={fire.converged}")
            sys.stdout.flush()

    print(f"Minimization completed after {time.time()-start_time:.2f}")

    start_time = time.time()

    # Thermalize momenta (after FIRE minimization) 
    sim.state.thermalize_particle_momenta(
        filter=hoomd.filter.All(),
        kT=par.kT
    )
else:
    print("Restart mode: skipping FIRE minimization and momentum thermalization.")

logger, thermo = add_logger(sim, par, force_objs)
gsd_writer = add_writer(sim, par, logger)
sim.operations.integrator = setup_integrator(par, forces)

add_progress_printer(sim, logger, period=100_000)

# Running sim for N_steps

if not par.dry_run:
    write_run_metadata(par, start_time)
    print(f"Running {par.N_steps} steps...")
    try:
        run_simulation(sim, par, thermo)

        gsd_writer.flush()
    finally:
        write_run_metadata(par, start_time, walltime_seconds=time.time()-start_time)

    print(f"Simulation finished in {time.time()-start_time:.2f} seconds.")

