"""Small, predictable DataFrame helpers for the wide analysis CSV schema."""
from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping, Sequence

import pandas as pd


def load_sweep(output_root: str | Path, sweep: str, kind: str = "replicates") -> pd.DataFrame:
    """Load CSV from the exact sweep name; never guess research-question folders."""
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


def filter_rows(df: pd.DataFrame, **conditions) -> pd.DataFrame:
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
    """Turn only selected wide columns into tidy data for Seaborn, on demand."""
    require_columns(df, [*id_vars, *metrics])
    return df.melt(
        id_vars=list(id_vars), value_vars=list(metrics),
        var_name="metric", value_name="value",
    )


def independent_units(df: pd.DataFrame) -> pd.DataFrame:
    """Average repeated trajectories of the same scrambled RNA before testing.

    Keeps one row per trajectory for periodic RNA and one row per independently
    scrambled sequence for scrambled RNA, per experimental parameter condition.
    """
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
    # Numeric measurements only: no categorical run/trajectory IDs included.
    numeric = [
        col for col in scrambled.select_dtypes(include="number")
        if col not in identifiers
    ]
    averaged = scrambled.groupby(identifiers, dropna=False, as_index=False)[numeric].mean()
    averaged["traj_replicate"] = "scramble_mean"
    averaged["folder_name"] = scrambled["folder_name"].iloc[0] if len(scrambled) else ""
    return pd.concat([rest, averaged], ignore_index=True, sort=False)


def find_project_root(start: str | Path | None = None) -> Path:
    """Discover the checkout from inside Jupyter, without hardcoded HPC paths."""
    start = Path.cwd() if start is None else Path(start)
    for path in (start.resolve(), *start.resolve().parents):
        if (path / "analysis" / "src" / "plots" / "style.py").is_file():
            return path
    raise FileNotFoundError("Open the notebook from inside the rna_sticker_spacer checkout")
