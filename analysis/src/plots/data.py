"""Load, filter and reshape analysis tables."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd


def load_sweep(output_root: str | Path, sweep: str, kind: str = "replicates") -> pd.DataFrame:
    """Load the replicate or condition summary CSV for a sweep."""
    if kind not in ("replicates", "summary"):
        raise ValueError("kind must be 'replicates' or 'summary'")
    if Path(sweep).name != sweep or sweep in (".", ".."):
        raise ValueError("sweep must be a directory name, not a path")
    path = Path(output_root) / sweep / f"{kind}.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run analysis/run_pipeline.py --stage all for this sweep."
        )
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"No analyzed records in {path}")
    return df


def filter_rows(df: pd.DataFrame, **conditions: object) -> pd.DataFrame:
    """Filter experimental conditions; each argument is a value or sequence."""
    mask = pd.Series(True, index=df.index)
    for name, condition in conditions.items():
        if name not in df:
            raise KeyError(f"Unknown column {name!r}; available columns: {list(df)}")
        allowed = list(condition) if isinstance(condition, (list, tuple, set)) else [condition]
        mask &= df[name].isin(allowed)
    return df.loc[mask].copy()


def require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [name for name in columns if name not in df.columns]
    if missing:
        raise KeyError("Analysis data are missing required columns: " + ", ".join(missing))


def melt_metrics(
    df: pd.DataFrame, metrics: Sequence[str],
    id_vars: Sequence[str] = ("E_att", "angle_k_ppp"),
) -> pd.DataFrame:
    """Convert selected metric columns to long format."""
    require_columns(df, [*id_vars, *metrics])
    return df.melt(
        id_vars=list(id_vars), value_vars=list(metrics),
        var_name="metric", value_name="value",
    )


def independent_units(df: pd.DataFrame) -> pd.DataFrame:
    """Average trajectories for each scrambled sequence and condition."""
    if "sequence_class" not in df or not df["sequence_class"].eq("scrambled").any():
        return df.copy()
    require_columns(df, ["molecule_file", "folder_name", "traj_replicate"])
    identifiers = [
        col for col in (
            "E_att", "angle_k_ppp", "density", "repeat_count", "n_molecules",
            "n_particles", "sequence_class", "molecule_file", "late_fraction",
            "min_contacts",
        ) if col in df
    ]
    scrambled = df[df["sequence_class"].eq("scrambled")]
    rest = df[~df["sequence_class"].eq("scrambled")]
    numeric = [
        col for col in scrambled.select_dtypes(include="number")
        if col not in identifiers
    ]
    averaged = scrambled.groupby(identifiers, dropna=False, as_index=False)[numeric].mean()
    averaged["traj_replicate"] = "scramble_mean"
    averaged["folder_name"] = scrambled["folder_name"].iloc[0] if len(scrambled) else ""
    return pd.concat([rest, averaged], ignore_index=True, sort=False)


def find_project_root(start: str | Path | None = None) -> Path:
    """Find the repository root from the working directory."""
    start = Path.cwd() if start is None else Path(start)
    for path in (start.resolve(), *start.resolve().parents):
        if (path / "analysis" / "src" / "plots" / "style.py").is_file():
            return path
    raise FileNotFoundError("Open the notebook from inside the rna_sticker_spacer checkout")



def load_rg_frames(
    output_root: str | Path, sweep: str, *,
    late_fraction: float = .20,
    input_sweep: str | Path | None = None,
    density: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize the late-frame Rg distribution stored in pairing archives."""
    if not 0 < late_fraction <= 1:
        raise ValueError("late_fraction must be in (0, 1]")
    trajectories = load_sweep(output_root, sweep, "replicates")
    if density is not None:
        trajectories = filter_rows(trajectories, density=density)
    roots = [Path(output_root) / sweep]
    if input_sweep is not None:
        roots.append(Path(input_sweep))

    records = []
    for row in trajectories.itertuples(index=False):
        relative = (Path(row.folder_name) / "computed_metrics" / "pairing"
                    / f"{row.traj_replicate}_pairing.npz")
        archive = next((root / relative for root in roots
                        if (root / relative).is_file()), None)
        if archive is None:
            raise FileNotFoundError(f"Pairing archive not found: {relative}")
        with np.load(archive) as saved:
            if "Rg_nm" not in saved:
                raise KeyError(f"No Rg_nm data in {archive}")
            radii = np.asarray(saved["Rg_nm"], dtype=float)
        if not len(radii):
            raise ValueError(f"No Rg frames in {archive}")
        radii = radii[-int(np.ceil(len(radii) * late_fraction)):]
        records.append({
            "angle_k_ppp": row.angle_k_ppp,
            "E_att": row.E_att,
            "Rg_nm": float(radii.mean()),
            "Rg_nm_p10": float(np.quantile(radii, .10)),
            "Rg_nm_p90": float(np.quantile(radii, .90)),
        })

    replicates = pd.DataFrame(records)
    summary = (replicates.groupby(["angle_k_ppp", "E_att"], as_index=False)
               .agg(Rg_nm_mean=("Rg_nm", "mean"),
                    Rg_nm_p10=("Rg_nm_p10", "mean"),
                    Rg_nm_p90=("Rg_nm_p90", "mean")))
    return replicates, summary
