"""Physical RNA and nucleotide concentrations from reduced particle density."""

from __future__ import annotations

AVOGADRO_PER_MOL: float = 6.02214076e23


def concentrations_from_density(
    density: float,
    n_particles: int,
    n_molecules: int,
    nm_per_sim_length_unit: float,
) -> dict[str, float]:
    """Convert reduced particle density to RNA-chain and nucleotide molarity."""
    if density <= 0:
        raise ValueError("Particle density must be positive")
    if n_particles <= 0 or n_molecules <= 0 or n_molecules > n_particles:
        raise ValueError("Particle and molecule counts must be positive and consistent")
    if nm_per_sim_length_unit <= 0:
        raise ValueError("Simulation length conversion must be positive")

    volume_sim = n_particles / density
    volume_liters = volume_sim * nm_per_sim_length_unit**3 * 1e-24
    inverse_molar_volume = 1.0 / (AVOGADRO_PER_MOL * volume_liters)

    return {
        "conc_mol_per_L": n_molecules * inverse_molar_volume,
        "conc_nuc_per_L": n_particles * inverse_molar_volume,
    }
