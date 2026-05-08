import numpy as np
import hoomd
from hoomd.sqd_plugin import sqd_potential

from setup import define_pairwise_sqd_v1
from setup import define_pairwise_sqd_v2
from setup import define_pairwise_sqd_2_particles
from setup import define_pairwise_sqd_v2_brothers
from setup import define_pairwise_sqd_sticker_spacer



# this import is clunky just temporarily so it runs on cluster where AlignAngle is not installed
try:
    from hoomd import align_angle
except ImportError:
    align_angle = None

import time
from hoomd.custom import Action
from hoomd.logging import log

class RuntimeClock(Action):
    def __init__(self):
        super().__init__()
        self._t0 = time.perf_counter()

    def act(self, timestep):
        pass  # required, but we don't need it to do anything

    @log(category='scalar')
    def walltime(self):
        return time.perf_counter() - self._t0

def initialize_model(par):

    nlist = hoomd.md.nlist.Tree(
        buffer=par.buffer, default_r_cut=par.nlist_rcut, exclusions=("bond", "1-3", "1-4")
    )

    sqd = sqd_potential.SquareDensity(nlist, default_r_cut=par.nlist_rcut)

    # SQD potential version pick
    if par.sqd_version == 1:
        define_pairwise_sqd_v1(sqd, par)
    elif par.sqd_version == 2:
        define_pairwise_sqd_v2(sqd, par)
    elif par.sqd_version == 3:
        define_pairwise_sqd_2_particles(sqd, par)
    elif par.sqd_version == 4:
        define_pairwise_sqd_v2_brothers(sqd, par)
    elif par.sqd_version == 5:
        define_pairwise_sqd_sticker_spacer(sqd, par)

    # default single-object version (used in ts7)
    bond_harmonic = hoomd.md.bond.Harmonic()
    angle_harmonic = hoomd.md.angle.Harmonic()
    dihedral_per = hoomd.md.dihedral.Periodic()

    # split-object placeholders for troubleshoot8
    angle_ppp = None
    angle_ppb = None
    angle_bpb = None
    angle_bpp = None
    dihedral_stacking = None
    dihedral_direction = None
    bond_bp = None
    bond_pp = None

    align_force = None  # Align angle (ONLY ADDED TO FORCES IN 2_particles)

    if getattr(par, "use_align_angle", False):
        align_force = align_angle.DirectorAlign()

        align_force.params["Phosphate-Phosphate-Phosphate"] = dict(
            k=0.0, multiplicity=par.align_multiplicity
        )
        align_force.params["Phosphate-Phosphate-Base"] = dict(
            k=par.align_k, multiplicity=par.align_multiplicity
        )
        align_force.params["Base-Phosphate-Phosphate"] = dict(
            k=0.0, multiplicity=par.align_multiplicity
        )

    if par.structure_type == "troubleshoot7":
        angle_harmonic.params["Phosphate-Phosphate-Phosphate"] = dict(
            k=par.angle_k_ppp, t0=par.angle_branch_dir_ppp_t0
        )
        angle_harmonic.params["Phosphate-Phosphate-Base"] = dict(
            k=par.angle_k_ppb, t0=par.angle_branch_dir_ppb_t0
        )
        angle_harmonic.params["Base-Phosphate-Base"] = dict(
            k=par.angle_k_bpb, t0=par.angle_branch_dir_bpb_t0
        )
        angle_harmonic.params["Base-Phosphate-Phosphate"] = dict(
            k=par.angle_k_bpp,
            t0=par.angle_branch_dir_ppp_t0 - par.angle_branch_dir_ppb_t0,
        )

        bond_harmonic.params["Base-Phosphate"] = dict(
            k=par.bond_k, r0=par.bond_bp_branch_dir_r0
        )
        bond_harmonic.params["Phosphate-Phosphate"] = dict(
            k=par.bond_k, r0=par.bond_pp_branch_dir_r0
        )

        dihedral_per.params["Stacking"] = dict(
            k=par.k_dihe_stacking, d=1, n=1, phi0=np.pi
        )
        dihedral_per.params["Direction"] = dict(
            k=par.k_dihe_direction, d=1, n=1, phi0=3 * np.pi / 2
        )

        forcelist = [sqd, bond_harmonic, angle_harmonic, dihedral_per]

    elif par.structure_type == "troubleshoot8":
        # ------------------------------------------------------------------
        # Same topology/parameters as troubleshoot7, but split by type
        # for logging individual angle/dihedral energies.
        # ------------------------------------------------------------------

        bond_bp = hoomd.md.bond.Harmonic()
        bond_pp = hoomd.md.bond.Harmonic()

        # zero both types in both objects first
        for t in ["Base-Phosphate", "Phosphate-Phosphate"]:
            bond_bp.params[t] = dict(k=0.0, r0=1.0)
            bond_pp.params[t] = dict(k=0.0, r0=1.0)

        bond_bp.params["Base-Phosphate"] = dict(
            k=par.bond_k, r0=par.bond_bp_branch_dir_r0
        )
        bond_pp.params["Phosphate-Phosphate"] = dict(
            k=par.bond_k, r0=par.bond_pp_branch_dir_r0
        )

        angle_types = [
            "Phosphate-Phosphate-Phosphate",
            "Phosphate-Phosphate-Base",
            "Base-Phosphate-Base",
            "Base-Phosphate-Phosphate",
        ]

        # Separate angle objects
        angle_ppp = hoomd.md.angle.Harmonic()
        angle_ppb = hoomd.md.angle.Harmonic()
        angle_bpb = hoomd.md.angle.Harmonic()
        angle_bpp = hoomd.md.angle.Harmonic()

        # Zero all types in all objects first
        for t in angle_types:
            angle_ppp.params[t] = dict(k=0.0, t0=0.0)
            angle_ppb.params[t] = dict(k=0.0, t0=0.0)
            angle_bpb.params[t] = dict(k=0.0, t0=0.0)
            angle_bpp.params[t] = dict(k=0.0, t0=0.0)

        # Activate only one type per object
        angle_ppp.params["Phosphate-Phosphate-Phosphate"] = dict(
            k=par.angle_k_ppp, t0=par.angle_branch_dir_ppp_t0
        )
        angle_ppb.params["Phosphate-Phosphate-Base"] = dict(
            k=par.angle_k_ppb, t0=par.angle_branch_dir_ppb_t0
        )
        angle_bpb.params["Base-Phosphate-Base"] = dict(
            k=par.angle_k_bpb, t0=par.angle_branch_dir_bpb_t0
        )
        angle_bpp.params["Base-Phosphate-Phosphate"] = dict(
            k=par.angle_k_bpp,
            t0=par.angle_branch_dir_ppp_t0 - par.angle_branch_dir_ppb_t0,
        )

        # Separate dihedral objects
        dihedral_types = ["Stacking", "Direction"]

        dihedral_stacking = hoomd.md.dihedral.Periodic()
        dihedral_direction = hoomd.md.dihedral.Periodic()

        # Zero all types first
        for t in dihedral_types:
            dihedral_stacking.params[t] = dict(k=0.0, d=1, n=1, phi0=0.0)
            dihedral_direction.params[t] = dict(k=0.0, d=1, n=1, phi0=0.0)

        # Activate one type per object
        dihedral_stacking.params["Stacking"] = dict(
            k=par.k_dihe_stacking, d=1, n=1, phi0=np.pi
        )
        dihedral_direction.params["Direction"] = dict(
            k=par.k_dihe_direction, d=1, n=1, phi0=3 * np.pi / 2
        )

        forcelist = [
            sqd,
            bond_bp,
            bond_pp,
            angle_ppp,
            angle_ppb,
            angle_bpb,
            angle_bpp,
            dihedral_stacking,
            dihedral_direction,
        ]

    elif par.structure_type == "2_particles":
        angle_harmonic.params["Phosphate-Phosphate-Phosphate"] = dict(
            k=par.angle_k_ppp, t0=par.angle_branch_dir_ppp_t0
        )
        angle_harmonic.params["Phosphate-Phosphate-Base"] = dict(
            k=par.angle_k_ppb, t0=par.angle_branch_dir_ppb_t0
        )
        angle_harmonic.params["Base-Phosphate-Phosphate"] = dict(
            k=par.angle_k_bpp,
            t0=par.angle_branch_dir_ppp_t0 - par.angle_branch_dir_ppb_t0,
        )

        bond_harmonic.params["Base-Phosphate"] = dict(
            k=par.bond_k, r0=par.bond_bp_branch_dir_r0
        )
        bond_harmonic.params["Phosphate-Phosphate"] = dict(
            k=par.bond_k, r0=par.bond_pp_branch_dir_r0
        )

        dihedral_per.params["Stacking"] = dict(
            k=par.k_dihe_stacking, d=1, n=1, phi0=np.pi
        )

        forcelist = [sqd, bond_harmonic, angle_harmonic, dihedral_per]

        # Align angle addition
        if align_force is not None:
            forcelist.append(align_force)
    
    elif par.structure_type == "sticker_spacer":
        # Make sure these match snapshot_sticker_spacer bond/angle type names
        bond_harmonic.params["Base-Base"] = dict(k=par.bond_k, r0=par.bond_length)
        angle_harmonic.params["Base-Base-Base"] = dict(
            k=par.angle_k_ppp, t0=par.angle_branch_dir_ppp_t0
        )  # uses PPP angle as param

        forcelist = [sqd, bond_harmonic, angle_harmonic]

        # No dihedrals in sticker_spacer; avoid logging a detached dihedral force (energy would be None).
        dihedral_per = None

    else:
        raise ValueError(f"Unsupported structure_type: {par.structure_type}")

    force_objs = {
        "sqd": sqd,
        "bond_harmonic": bond_harmonic,
        "angle_harmonic": angle_harmonic,  # used by ts7 / 2_particles
        "dihedral_per": dihedral_per,  # used by ts7 / 2_particles
        "bond_bp": bond_bp,
        "bond_pp": bond_pp,
        "angle_ppp": angle_ppp,  # used by ts8
        "angle_ppb": angle_ppb,  # used by ts8
        "angle_bpb": angle_bpb,  # used by ts8
        "angle_bpp": angle_bpp,  # used by ts8
        "dihedral_stacking": dihedral_stacking,  # used by ts8
        "dihedral_direction": dihedral_direction,  # used by ts8
        "align_force": align_force,
    }
    return forcelist, force_objs


def setup_integrator(par, forces):
    """
    """

    if par.thermostat == "langevin":
        method = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=par.kT)
        for (
            t
        ) in (
            par.type_names
        ):  # Change possible: assign per-type drag coefficients (gamma)
            method.gamma[t] = par.gamma
    elif par.thermostat == "DPD":
        method = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
        nlist = hoomd.md.nlist.Cell(buffer=par.buffer)
        dpd = hoomd.md.pair.DPD(nlist=nlist, kT=par.kT, default_r_cut=par.dpd_r_cut)
        for a in par.type_names:
            for b in par.type_names:
                dpd.params[(a, b)] = dict(A=0.0, gamma=par.gamma)  # new param for dpd?
        forces.append(dpd)  # in place chane? unsafe
    else:
        raise ValueError(f"Unknown thermostat: {par.thermostat}")

    integrator = hoomd.md.Integrator(  # Build the integrator
        dt=par.dt, methods=[method], forces=forces  # time step
    )
    # Align angle (rotational_dof enabled)
    integrator.integrate_rotational_dof = bool(getattr(par, "use_align_angle", False))

    return integrator


def setup_simulation(snapshot=None, restart_from=None, restart_frame=-1, seed = 4):

    sim = hoomd.Simulation(device=hoomd.device.GPU(), seed=seed)
    if restart_from is not None:
        sim.create_state_from_gsd(filename=str(restart_from), frame=restart_frame)
        print(
            f"Simulation created from restart: {restart_from} (frame {restart_frame})"
        )
    elif snapshot is not None:
        sim.create_state_from_snapshot(snapshot)
        print("Simulation created from fresh snapshot.")
    else:
        raise ValueError("Provide either snapshot or restart_from.")

    return sim


def add_logger(sim, par, force_objs):
    logger = hoomd.logging.Logger(categories=["scalar"])
    logger.add(sim, quantities=["timestep", "tps"])

    runtime_clock = RuntimeClock()
    logger.add(runtime_clock, quantities=["walltime"])
    
    thermo = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
    sim.operations.computes.append(thermo)
    # Kinetic energy
    logger.add(
        thermo,
        quantities=[
            "kinetic_energy",
            "kinetic_temperature",
        ],
    )
    # No energy logging if log_energies = True
    if not par.log_energies:
        print("Logger created.")
        return logger, thermo
    
    # Energy logging if log_energies = True
    logger.add(thermo, quantities=["potential_energy"])
    logger.add(force_objs["sqd"], quantities=["energy"])

    # troubleshoot8: split bond objects
    if force_objs.get("bond_bp") is not None:
        logger.add(force_objs["bond_bp"], quantities=["energy"])
        logger.add(force_objs["bond_pp"], quantities=["energy"])
    else:
        logger.add(force_objs["bond_harmonic"], quantities=["energy"])

    # troubleshoot8: split angle objects
    if force_objs.get("angle_ppb") is not None:
        logger.add(force_objs["angle_ppp"], quantities=["energy"])
        logger.add(force_objs["angle_ppb"], quantities=["energy"])
        logger.add(force_objs["angle_bpb"], quantities=["energy"])
        logger.add(force_objs["angle_bpp"], quantities=["energy"])
    else:
        logger.add(force_objs["angle_harmonic"], quantities=["energy"])  # ts7

    # troubleshoot8: split dihedral objects
    if force_objs.get("dihedral_stacking") is not None:
        logger.add(force_objs["dihedral_stacking"], quantities=["energy"])
        logger.add(force_objs["dihedral_direction"], quantities=["energy"])
    elif force_objs.get("dihedral_per") is not None:
        logger.add(force_objs["dihedral_per"], quantities=["energy"])  # ts7
    if force_objs["align_force"] is not None:
        logger.add(force_objs["align_force"], quantities=["energy"])

    print("Logger created.")
    return logger, thermo


def add_writer(sim, par, logger):

    gsd_writer = hoomd.write.GSD(
        filename=f"{par.output_dir}/trajectory{par.suffix}.gsd",
        trigger=hoomd.trigger.Periodic(par.N_write),
        mode="wb",
        dynamic=["property", "momentum"],  # saves positions, velocities, etc.
        logger=logger,
    )
    sim.operations.writers.append(gsd_writer)
    print("GSD writer created.")
    return gsd_writer


def add_progress_printer(sim, logger, period=100_000):
    table = hoomd.write.Table(
        trigger=hoomd.trigger.Periodic(period),
        logger=logger,
    )
    sim.operations.writers.append(table)
    return table

def run_simulation(sim, par, thermo):
    """Run simulation and optionally stop if KE per particle exceeds threshold."""

    max_ke_per_particle_stop = par.max_ke_per_particle_stop

    if max_ke_per_particle_stop is None:
        sim.run(par.N_steps)
        return
    
    # KE stopping
    start_check_after_steps = 100_000 # initial KE blow ups ignored
    check_period = par.N_write # check KE is not blowing up every N_write steps

    steps_done = 0

    while steps_done < par.N_steps:
        steps_to_run = min(check_period, par.N_steps - steps_done)
        sim.run(steps_to_run)
        steps_done += steps_to_run

        if sim.timestep < start_check_after_steps:
            continue

        max_ke = max_ke_per_particle_stop * sim.state.N_particles

        if thermo.kinetic_energy > max_ke:
            print(
                f"Stopped early at timestep {sim.timestep}: "
                f"KE/N exceeded {max_ke_per_particle_stop}"
            )
            break