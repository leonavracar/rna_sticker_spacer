from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from analysis_scripts.pairing import (
    Q_from_pair_arrays,
    disagree_from_closest_arrays,
    paired_nonstrict_from_closest_arrays,
    paired_strict_from_pair_arrays,
)


# ============================================================================
# GENERIC ARRAY HELPERS
# ============================================================================


def _last_fraction(
    values: NDArray[np.float64],
    fraction: float,
) -> NDArray[np.float64]:
    """
    Return the last fraction of a 1D array.
    """
    if len(values) == 0:
        return values

    start = int((1.0 - fraction) * len(values))
    return values[start:]


def _mean(values: NDArray[np.float64]) -> np.float64:
    """
    Return the mean of a 1D array.
    """
    if len(values) == 0:
        return np.float64(np.nan)
    return np.float64(np.mean(values))


def _std(values: NDArray[np.float64]) -> np.float64:
    """
    Return the standard deviation of a 1D array.
    """
    if len(values) == 0:
        return np.float64(np.nan)
    return np.float64(np.std(values))


def _min(values: NDArray[np.float64]) -> np.float64:
    """
    Return the minimum of a 1D array.
    """
    if len(values) == 0:
        return np.float64(np.nan)
    return np.float64(np.min(values))


def _max(values: NDArray[np.float64]) -> np.float64:
    """
    Return the maximum of a 1D array.
    """
    if len(values) == 0:
        return np.float64(np.nan)
    return np.float64(np.max(values))


def _cv(values: NDArray[np.float64]) -> np.float64:
    """
    Return the coefficient of variation of a 1D array as std / abs(mean).
    Returns NaN if the mean is zero or the array is empty.
    """
    if len(values) == 0:
        return np.float64(np.nan)

    mean = np.mean(values)
    if np.isclose(mean, 0.0):
        return np.float64(np.nan)

    return np.float64(np.std(values) / abs(mean))


# ============================================================================
# PAIRING-LEVEL HELPERS
# ============================================================================


def switching_rate_from_pair_arrays(
    pair_arrays: NDArray[np.int_],
    ignore_unpaired: bool = False,
) -> np.float64:
    """
    Compute the average per-residue switching rate from frame to frame.
    """
    if len(pair_arrays) < 2:
        return np.float64(np.nan)

    prev = pair_arrays[:-1]
    curr = pair_arrays[1:]

    if not ignore_unpaired:
        switches = curr != prev
        per_base = np.mean(switches, axis=0)
        return np.float64(np.mean(per_base))

    valid = (curr >= 0) & (prev >= 0)
    switch_mask = (curr != prev) & valid

    switch_counts = np.sum(switch_mask, axis=0)
    denom = np.sum(valid, axis=0)

    per_base = np.divide(
        switch_counts,
        denom,
        out=np.zeros_like(switch_counts, dtype=float),
        where=denom > 0,
    )
    return np.float64(np.mean(per_base))


# ============================================================================
# PER-REPLICATE SUMMARIES
# ============================================================================

def summarize_rg(
    observables: dict[str, NDArray],
    fraction_last20: float = 0.2,
    fraction_last5: float = 0.05,
) -> dict[str, np.float64]:
    """
    Summarize radius-of-gyration values for one replicate.
    """
    rg = np.asarray(observables["rg"], dtype=np.float64)
    rg_last20 = _last_fraction(rg, fraction_last20)
    rg_last5 = _last_fraction(rg, fraction_last5)

    rg_mean = _mean(rg)
    rg_std = _std(rg)

    return {
        "rg_mean": rg_mean,
        "rg_std": rg_std,
        "rg_std_ptg_of_mean": np.float64(100.0 * rg_std / rg_mean) if not np.isclose(rg_mean, 0.0) else np.float64(np.nan),
        "rg_mean_last20": _mean(rg_last20),
        "rg_std_last20": _std(rg_last20),
        "rg_std_ptg_of_mean_last20": (
            np.float64(100.0 * _std(rg_last20) / _mean(rg_last20))
            if not np.isclose(_mean(rg_last20), 0.0)
            else np.float64(np.nan)
        ),
        "rg_last5": _mean(rg_last5),
    }


def summarize_ke(
    observables: dict[str, NDArray],
    fraction_last20: float = 0.2,
) -> dict[str, np.float64]:
    """
    Summarize kinetic-energy values for one replicate.
    """
    ke = np.asarray(observables["ke"], dtype=np.float64)
    ke_last20 = _last_fraction(ke, fraction_last20)

    return {
        "ke_mean_last20": _mean(ke_last20),
        "ke_std_last20": _std(ke_last20),
        "ke_max": _max(ke_last20),
    }


def summarize_pairing(
    pairing: dict[str, NDArray],
    expected: NDArray[np.int_] | None = None,
    fraction_last20: float = 0.2,
    fraction_last5: float = 0.05,
) -> dict[str, np.float64]:
    """
    Summarize pairing-derived metrics for one replicate.
    """
    pair_arrays = np.asarray(pairing["pair_arrays"])
    closest_partner0 = np.asarray(pairing["closest_partner0"])
    closest_partner1 = np.asarray(pairing["closest_partner1"])

    paired_strict = paired_strict_from_pair_arrays(pair_arrays)
    paired_nonstrict = paired_nonstrict_from_closest_arrays(
        closest_partner0,
        closest_partner1,
    )
    disagree = disagree_from_closest_arrays(
        closest_partner0,
        closest_partner1,
    )
    Q = (
        Q_from_pair_arrays(pair_arrays, expected)
        if expected is not None
        else np.full(len(pair_arrays), np.nan, dtype=np.float64)
    )

    return {
        "paired_frac_last20_mean": _mean(_last_fraction(paired_strict, fraction_last20)),
        "paired_strict_last5": _mean(_last_fraction(paired_strict, fraction_last5)),
        "paired_nonstrict_last5": _mean(_last_fraction(paired_nonstrict, fraction_last5)),
        "disagree_last5": _mean(_last_fraction(disagree, fraction_last5)),
        "Q_last5": _mean(_last_fraction(Q, fraction_last5)),
        "mean_Q": _mean(Q),
        "mean_disagree_frac": _mean(disagree),
        "switch_rate_all": switching_rate_from_pair_arrays(
            pair_arrays,
            ignore_unpaired=False,
        ),
        "switch_rate_paired": switching_rate_from_pair_arrays(
            pair_arrays,
            ignore_unpaired=True,
        ),
    }


def summarize_geometry(
    geometry: dict[str, NDArray],
    fraction_last5: float = 0.05,
) -> dict[str, np.float64]:
    """
    Summarize geometry-derived framewise scores for one replicate.
    """
    stacking = np.asarray(geometry["stacking_score"], dtype=np.float64)
    inversion = np.asarray(geometry["inversion_score"], dtype=np.float64)

    return {
        "stacking_score_mean": _mean(stacking),
        "stacking_score_last5": _mean(_last_fraction(stacking, fraction_last5)),
        "inversion_score_mean": _mean(inversion),
        "inversion_score_last5": _mean(_last_fraction(inversion, fraction_last5)),
    }


def summarize_energy(
    energy: dict[str, NDArray],
    fraction_last5: float = 0.05,
) -> dict[str, np.float64]:
    """
    Summarize framewise energy terms for one replicate using last-5% means.
    """
    keys = [
        "E_pot",
        "E_sqd",
        "E_bond",
        "E_bond_bp",
        "E_bond_pp",
        "E_angle",
        "E_angle_ppp",
        "E_angle_ppb",
        "E_angle_bpb",
        "E_angle_bpp",
        "E_dihedral",
        "E_dihedral_stacking",
        "E_dihedral_direction",
    ]

    out: dict[str, np.float64] = {}
    for key in keys:
        values = np.asarray(energy[key], dtype=np.float64)
        out[f"{key}_last5"] = _mean(_last_fraction(values, fraction_last5))

    return out


def compute_replicate_analysis(
    pairing: dict[str, NDArray],
    geometry: dict[str, NDArray],
    observables: dict[str, NDArray],
    energy: dict[str, NDArray] | None = None,
    expected: NDArray[np.int_] | None = None,
    fraction_last20: float = 0.2,
    fraction_last5: float = 0.05,
) -> dict[str, np.float64]:
    """
    Compute one flat summary dictionary for one replicate from cached arrays.
    """
    results: dict[str, np.float64] = {}
    results.update(
        summarize_rg(
            observables,
            fraction_last20=fraction_last20,
            fraction_last5=fraction_last5,
        )
    )
    results.update(
        summarize_ke(
            observables,
            fraction_last20=fraction_last20,
        )
    )
    results.update(
        summarize_pairing(
            pairing,
            expected=expected,
            fraction_last20=fraction_last20,
            fraction_last5=fraction_last5,
        )
    )
    results.update(
        summarize_geometry(
            geometry,
            fraction_last5=fraction_last5,
        )
    )

    if energy is not None:
        results.update(
            summarize_energy(
                energy,
                fraction_last5=fraction_last5,
            )
        )

    return results


# ============================================================================
# GROUPED SUMMARY ACROSS REPLICATES
# ============================================================================


def _group_min_max_sd(
    df: pd.DataFrame,
    column: str,
) -> dict[str, np.float64]:
    """
    Compute min, max, and standard deviation across replicates for one column.
    """
    values = df[column].dropna().to_numpy(dtype=float)
    return {
        f"{column}_min": _min(values),
        f"{column}_max": _max(values),
        f"{column}_sd": _std(values),
    }


def _group_min_max_cv(
    df: pd.DataFrame,
    column: str,
) -> dict[str, np.float64]:
    """
    Compute min, max, and coefficient of variation across replicates for one column.
    """
    values = df[column].dropna().to_numpy(dtype=float)
    return {
        f"{column}_min": _min(values),
        f"{column}_max": _max(values),
        f"{column}_cv": _cv(values),
    }


def compute_group_summary(
    df_per_rep: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    """
    Compute grouped summary metrics across replicates for each parameter combination.
    """
    rows: list[dict[str, object]] = []

    paired_frac_cols = [
        "paired_frac_last20_mean",
    ]

    energy_last5_cols = [
        "E_pot_last5",
        "E_sqd_last5",
        "E_bond_last5",
        "E_bond_bp_last5",
        "E_bond_pp_last5",
        "E_angle_last5",
        "E_angle_ppp_last5",
        "E_angle_ppb_last5",
        "E_angle_bpb_last5",
        "E_angle_bpp_last5",
        "E_dihedral_last5",
        "E_dihedral_stacking_last5",
        "E_dihedral_direction_last5",
    ]

    for group_values, df_group in df_per_rep.groupby(group_cols, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        row = dict(zip(group_cols, group_values))
        row["n_replicates"] = int(len(df_group))

        if "rg_last5" in df_group.columns:
            mask = df_group["rg_last5"] >= 1.8
            row["n_rg_last5_ge_1_8"] = int(mask.sum())
            row["frac_rg_last5_ge_1_8"] = float(
                mask.mean()) if len(mask) else np.nan
        else:
            row["n_rg_last5_ge_1_8"] = np.nan
            row["frac_rg_last5_ge_1_8"] = np.nan

        for col in paired_frac_cols:
            if col in df_group.columns:
                row.update(_group_min_max_sd(df_group, col))

        for col in energy_last5_cols:
            if col in df_group.columns:
                row.update(_group_min_max_cv(df_group, col))

        rows.append(row)

    return pd.DataFrame(rows)
