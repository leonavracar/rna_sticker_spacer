"""Optional GSD-only pass for logged energies and kinetic energy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ENERGY_KEYS: dict[str, str] = {
    "E_pot": "md/compute/ThermodynamicQuantities/potential_energy",
    "E_sqd": "sqd_plugin/sqd_potential/SquareDensity/energy",
    "E_bond": "md/bond/Harmonic/energy",
    "E_angle": "md/angle/Harmonic/energy",
}
KE_KEY = "md/compute/ThermodynamicQuantities/kinetic_energy"


def energy_framewise(
    trajectory: Path, frame_indices: np.ndarray, dt: float | None = None
) -> pd.DataFrame:
    """Read only the requested frames; missing log channels become NaN."""
    import gsd.hoomd

    rows: list[dict[str, float | int]] = []
    with gsd.hoomd.open(str(trajectory), "r") as gsd:
        for index in frame_indices:
            index = int(index)
            frame = gsd[index]
            row: dict[str, float | int] = {
                "frame": index,
                "step": int(frame.configuration.step),
            }
            if dt is not None:
                row["time"] = row["step"] * dt
            logs = frame.log or {}
            for column, key in ENERGY_KEYS.items():
                row[column] = float(np.asarray(logs[key]).item()) if key in logs else np.nan
            row["ke_per_particle"] = (
                float(np.asarray(logs[KE_KEY]).item()) / frame.particles.N
                if KE_KEY in logs and frame.particles.N else np.nan
            )
            rows.append(row)
    return pd.DataFrame(rows)
