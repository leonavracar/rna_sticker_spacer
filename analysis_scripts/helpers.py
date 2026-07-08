from __future__ import annotations
import json
import pandas as pd
import difflib
import gsd.hoomd
import numpy as np
from numpy.typing import NDArray
from contextlib import contextmanager
from typing import Iterator
from pathlib import Path
import freud
from typing import Any
# ============================================================================
# GLOBAL
# ============================================================================

def unwrap_frame(frame: gsd.hoomd.Frame) -> NDArray[np.float64]:
    """Return unwrapped particle positions for one HOOMD frame."""
    box = freud.box.Box.from_box(frame.configuration.box)
    positions = box.unwrap(frame.particles.position, frame.particles.image)

    return np.asarray(positions, dtype=np.float64)

def get_n_particles(traj: gsd.hoomd.HOOMDTrajectory) -> int:
    """Return the number of particles in the first trajectory frame."""
    return int(traj[0].particles.N)


def read_param(params_path: Path, key: str) -> float | int | str | None:
    """Read one parameter from a params.json file."""
    with open(params_path, "r", encoding="utf-8") as f:
        content = f.read()
    decoder = json.JSONDecoder()
    data, _ = decoder.raw_decode(content)
    return data.get(key)

@contextmanager
def as_traj(
    traj_file: str | Path | gsd.hoomd.HOOMDTrajectory,
) -> Iterator[gsd.hoomd.HOOMDTrajectory]:
    """
    Yield a trajectory. If a path is given, the trajectory is opened here and closed on exit.
    """
    if isinstance(traj_file, (str, Path)):
        traj = gsd.hoomd.open(str(traj_file), "r")
        try:
            yield traj
        finally:
            traj.close()
    else:
        yield traj_file


def save_npz(out_path: Path, data: dict[str, NDArray]) -> None:
    """
    Save arrays to a compressed .npz file.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **data)


def load_npz(cache_path: Path) -> dict[str, NDArray]:
    """
    Load arrays from a compressed .npz file.
    """
    with np.load(cache_path) as data:
        return {key: data[key] for key in data.files}


def _parse_value(token: str) -> int | float | str:
    """
    Parse one parameter value token as int, float, or string.
    """
    try:
        value = float(token)
    except ValueError:
        return token

    if value.is_integer():
        return int(value)
    return value


def parse_parameters_file(params_path: Path) -> pd.DataFrame:
    """
    Parse a parameters.txt file containing whitespace-separated flag/value pairs
    on each line into a DataFrame.
    """
    rows: list[dict[str, int | float | str]] = []

    with open(params_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            tokens = line.split()
            if len(tokens) % 2 != 0:
                raise ValueError(
                    f"Malformed parameters line with odd number of tokens: {line}"
                )

            row: dict[str, int | float | str] = {}
            for i in range(0, len(tokens), 2):
                key = tokens[i]
                value = tokens[i + 1]

                if not key.startswith("--"):
                    raise ValueError(
                        f"Expected parameter flag starting with '--', got '{key}' in line: {line}"
                    )

                row[key.removeprefix("--")] = _parse_value(value)

            rows.append(row)

    return pd.DataFrame(rows)


def get_swept_param_keys(params_path: Path) -> list[str]:
    """
    Read parameters.txt and return the ordered list of swept parameter keys.
    """
    df_params = parse_parameters_file(params_path)
    return list(df_params.columns)


# def load_params_json(params_path: Path) -> dict[str, object]:
#     """
#     Load one params.json file.
#     """
#     with open(params_path, "r", encoding="utf-8") as f:
#         return json.load(f)

def load_params_json(params_path: Path) -> dict[str, Any]:
    """Load params.json, tolerating extra concatenated JSON after the first object."""
    with open(params_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        raise ValueError(f"Empty params.json: {params_path}")

    decoder = json.JSONDecoder()
    data, end = decoder.raw_decode(content)

    extra = content[end:].strip()
    if extra:
        print(f"WARNING: Ignoring extra JSON data in {params_path} after char {end}")

    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object in {params_path}, got {type(data).__name__}")

    return data

def get_row_params_from_json(
    run_dir: Path,
    swept_param_keys: list[str],
) -> dict[str, object]:
    """
    Read the swept parameter values for one run folder from params.json.
    """
    params = load_params_json(run_dir / "params.json")

    row: dict[str, object] = {}
    for key in swept_param_keys:
        if key not in params:
            raise KeyError(f"Key '{key}' not found in {run_dir / 'params.json'}")
        row[key] = params[key]

    row["folder_name"] = run_dir.name
    return row


def drop_all_nan_columns(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Drop columns that are entirely NaN and return the cleaned DataFrame and dropped column names.
    """
    dropped = [col for col in df.columns if df[col].isna().all()]
    return df.drop(columns=dropped), dropped


# def _parse_value(token: str) -> int | float | str:
#     """
#     Parse one parameter value token as int, float, or string.
#     """
#     try:
#         value = float(token)
#     except ValueError:
#         return token

#     if value.is_integer():
#         return int(value)
#     return value


# def get_param_keys_from_folder_name(
#     folder_name: str,
#     candidate_keys: list[str],
# ) -> list[str]:
#     """
#     Return the parameter keys that appear in one folder-name suffix.
#     Keys are matched greedily from left to right using the provided candidate keys.
#     """
#     suffix = folder_name.split("_", 1)[-1]
#     parts = suffix.split("_")

#     candidate_parts = {
#         key: key.split("_")
#         for key in candidate_keys
#     }
#     sorted_keys = sorted(
#         candidate_keys,
#         key=lambda key: len(candidate_parts[key]),
#         reverse=True,
#     )

#     keys_found: list[str] = []
#     i = 0

#     while i < len(parts):
#         matched_key = None

#         for key in sorted_keys:
#             key_parts = candidate_parts[key]
#             n = len(key_parts)

#             if parts[i:i + n] == key_parts:
#                 matched_key = key
#                 keys_found.append(key)
#                 i += n + 1
#                 break

#         if matched_key is None:
#             raise ValueError(
#                 f"Could not match folder-name tokens near: {parts[i:]} in {folder_name}"
#             )

#     return keys_found

# def parse_parameters_file(params_path: Path) -> pd.DataFrame:
#     """
#     Parse a parameters.txt file containing whitespace-separated flag/value pairs
#     on each line into a DataFrame.
#     """
#     rows: list[dict[str, int | float | str]] = []

#     with open(params_path, "r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue

#             tokens = line.split()
#             if len(tokens) % 2 != 0:
#                 raise ValueError(
#                     f"Malformed parameters line with odd number of tokens: {line}"
#                 )

#             row: dict[str, int | float | str] = {}
#             for i in range(0, len(tokens), 2):
#                 key = tokens[i]
#                 value = tokens[i + 1]

#                 if not key.startswith("--"):
#                     raise ValueError(
#                         f"Expected parameter flag starting with '--', got '{key}' in line: {line}"
#                     )

#                 row[key.removeprefix("--")] = _parse_value(value)

#             rows.append(row)

#     return pd.DataFrame(rows)


# def row_to_folder_name(row: pd.Series) -> str:
#     """
#     Convert one parameter row into the corresponding folder-name suffix.
#     """
#     parts = [f"{col}_{row[col]}" for col in row.index]
#     return "_".join(str(x) for x in parts)


# def param_from_folder_name(
#     raw_folder_name: str,
#     valid_names: set[str],
# ) -> str:
#     """
#     Match one run folder name to a parameter-name suffix from parameters.txt.
#     """
#     suffix = raw_folder_name.split("_", 1)[-1]
#     if suffix in valid_names:
#         return suffix

#     closest = difflib.get_close_matches(suffix, list(valid_names), n=1)
#     if closest:
#         return closest[0]

#     raise KeyError(f"No matching params found for folder {raw_folder_name}.")


# def build_param_lookup(sweep_path: Path) -> pd.DataFrame:
#     """
#     Build a parameter lookup DataFrame from parameters.txt in one sweep directory.
#     """
#     params_path = sweep_path / "parameters.txt"
#     df_params = parse_parameters_file(params_path)
#     df_params["folder_name"] = df_params.apply(row_to_folder_name, axis=1)
#     return df_params


def drop_all_nan_columns(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Drop columns that are entirely NaN and return the cleaned DataFrame and dropped column names.
    """
    dropped = [col for col in df.columns if df[col].isna().all()]
    return df.drop(columns=dropped), dropped


# ============================================================================
# GEOMETRY
# ============================================================================


def make_stacking_dihe_group_3pt(n_particles: int) -> list[tuple[int, int, int, int]]:
    """Return stacking dihedral quadruplets for the 3-particles-per-nucleotide model."""
    return [(i + 1, i, i + 3, i + 4) for i in range(0, n_particles - 4, 3)]


def make_stacking_dihe_group_2pt(n_particles: int) -> list[tuple[int, int, int, int]]:
    """Return stacking dihedral quadruplets for the 2-particles-per-nucleotide model."""
    n_nt = n_particles // 2
    return [
        (2 * i + 1, 2 * (i + 1) + 1, 2 * (i + 2) + 1, 2 * (i + 3) + 1)
        for i in range(n_nt - 3)
    ]


def make_inversion_dihe_group_3pt(n_particles: int) -> list[tuple[int, int, int, int]]:
    """Return directional dihedral quadruplets for the 3-particles-per-nucleotide model."""
    return [(x, x + 3, x + 4, x + 5) for x in range(0, n_particles - 4, 3)]


def get_dihedral_groups_by_type(
    frame: gsd.hoomd.Frame,
) -> dict[str, NDArray[np.int64]]:
    """
    Group dihedral quadruplets by dihedral type name. Returns a dictionary mapping each
    dihedral type to its particle-index quadruplets.
    """
    groups = np.asarray(frame.dihedrals.group, dtype=np.int64)
    typeids = np.asarray(frame.dihedrals.typeid, dtype=np.int64)
    types = list(frame.dihedrals.types)

    return {name: groups[typeids == tid] for tid, name in enumerate(types)}


def get_angle_groups_by_type(
    frame: gsd.hoomd.Frame,
) -> dict[str, NDArray[np.int64]]:
    """
    Group angle triplets by angle type name. Returns a dictionary mapping each angle type
    to its particle-index triplets.
    """
    groups = np.asarray(frame.angles.group, dtype=np.int64)
    typeids = np.asarray(frame.angles.typeid, dtype=np.int64)
    types = list(frame.angles.types)

    return {name: groups[typeids == tid] for tid, name in enumerate(types)}


def get_bond_groups_by_type(
    frame: gsd.hoomd.Frame,
) -> dict[str, NDArray[np.int64]]:
    """
    Group bond pairs by bond type name. Returns a dictionary mapping each bond type
    to its particle-index pairs.
    """
    groups = np.asarray(frame.bonds.group, dtype=np.int64)
    typeids = np.asarray(frame.bonds.typeid, dtype=np.int64)
    types = list(frame.bonds.types)

    return {name: groups[typeids == tid] for tid, name in enumerate(types)}


def _split_group_in_half(
    group: NDArray[np.int64],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """
    Split one geometry group into two equal contiguous halves.
    """
    n = len(group)
    if n % 2 != 0:
        raise ValueError("Expected an even-sized group to split into two halves.")

    mid = n // 2
    return group[:mid], group[mid:]


def _require_group(
    groups_by_type: dict[str, NDArray[np.int64]],
    name: str,
) -> NDArray[np.int64]:
    """
    Return one required geometry group by name.
    """
    if name not in groups_by_type:
        raise KeyError(f"Required group '{name}' not found.")
    return groups_by_type[name]


def _prepare_geometry_groups(
    frame: gsd.hoomd.Frame,
) -> dict[str, tuple[str, NDArray[np.int64]]]:
    """
    Prepare geometry groups for compute_geometry. Each entry maps an output key to
    a geometry kind and the corresponding index group.
    """
    angle_groups = get_angle_groups_by_type(frame)
    bond_groups = get_bond_groups_by_type(frame)
    dihedral_groups = get_dihedral_groups_by_type(frame)

    angle_ppb_b1, angle_ppb_b2 = _split_group_in_half(
        _require_group(angle_groups, "Phosphate-Phosphate-Base")
    )
    angle_bpp_b1, angle_bpp_b2 = _split_group_in_half(
        _require_group(angle_groups, "Base-Phosphate-Phosphate")
    )
    bond_bp_b1, bond_bp_b2 = _split_group_in_half(
        _require_group(bond_groups, "Base-Phosphate")
    )
    dihedral_stacking, _ = _split_group_in_half(
        _require_group(dihedral_groups, "Stacking")
    )
    dihedral_direction, _ = _split_group_in_half(
        _require_group(dihedral_groups, "Direction")
    )

    return {
        "angle_ppp": (
            "angle",
            _require_group(angle_groups, "Phosphate-Phosphate-Phosphate"),
        ),
        "angle_ppb_b1": ("angle", angle_ppb_b1),
        "angle_ppb_b2": ("angle", angle_ppb_b2),
        "angle_bpb": ("angle", _require_group(angle_groups, "Base-Phosphate-Base")),
        "angle_bpp_b1": ("angle", angle_bpp_b1),
        "angle_bpp_b2": ("angle", angle_bpp_b2),
        "dihedral_stacking": ("dihedral", dihedral_stacking),
        "dihedral_direction": ("dihedral", dihedral_direction),
        "bond_bp_b1": ("bond", bond_bp_b1),
        "bond_bp_b2": ("bond", bond_bp_b2),
        "bond_pp": ("bond", _require_group(bond_groups, "Phosphate-Phosphate")),
    }
