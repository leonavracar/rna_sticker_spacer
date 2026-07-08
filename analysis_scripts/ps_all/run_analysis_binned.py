#!/usr/bin/env python3
"""
Build analysis_cluster_binned.csv from analysis_clusters.csv.

Produces one wide CSV per sweep with two binnings side-by-side:
  - mol_* : cluster-size bins, weighted by frac_mols
  - pct_* : one-cluster particle-fraction bins, weighted by frac_particles
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

analysis_functions_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(analysis_functions_dir))

from analysis_scripts.plots_ps import CLUSTER_ID_COLS  # reuse the shared id list

BASE_DIR = analysis_functions_dir / "bulk" / "output_dir"

# ---------------------------------------------------------------------------
# Binning definitions (edit here if you want to rerun with different bins)
# ---------------------------------------------------------------------------

# Cluster-size bins (applied to `cluster_size`, aggregate `frac_mols`).
MOL_BIN_EDGES = [0, 1, 2, 3, 9, 19, 49, np.inf]
MOL_BIN_LABELS = ["mono", "di", "tri", "4-9", "10-19", "20-49", "50+"]

# Particle-fraction bins (applied to `one_cluster_particle_frac`,
# aggregate `frac_particles`). "mono" is special-cased by cluster_size == 1.
PCT_BIN_LABELS = ["mono", "1-2", "2-3", "3-9", "10-19", "20-49", "50+"]

# Extra id columns that may be present in the cluster-resolved CSV.
EXTRA_ID_COLS = ["rna", "sequence_class", "molecule_file"]


def assign_pct_bin(cluster_size: pd.Series, frac: pd.Series) -> pd.Series:
    """Assign percentage bins from one_cluster_particle_frac (fraction, not %)."""
    conditions = [
        cluster_size == 1,
        (cluster_size > 1) & (frac < 0.02),
        (frac >= 0.02) & (frac < 0.03),
        (frac >= 0.03) & (frac < 0.10),
        (frac >= 0.10) & (frac < 0.20),
        (frac >= 0.20) & (frac < 0.50),
        frac >= 0.50,
    ]
    return pd.Series(
        np.select(conditions, PCT_BIN_LABELS, default="unassigned"),
        index=cluster_size.index,
    )


def bin_by_size(cdf: pd.DataFrame, id_cols: list[str]) -> pd.DataFrame:
    df = cdf.copy()
    df["mol_bin"] = pd.cut(df["cluster_size"], bins=MOL_BIN_EDGES, labels=MOL_BIN_LABELS)
    if df["mol_bin"].isna().any():
        bad = sorted(df.loc[df["mol_bin"].isna(), "cluster_size"].unique())
        raise ValueError(f"cluster_size values not covered by MOL_BIN_EDGES: {bad}")

    wide = (
        df.groupby([*id_cols, "mol_bin"], observed=True, as_index=False)
          .agg(frac_mols=("frac_mols", "sum"))
          .pivot_table(index=id_cols, columns="mol_bin",
                       values="frac_mols", fill_value=0, observed=True)
          .reset_index()
    )
    for label in MOL_BIN_LABELS:
        if label not in wide.columns:
            wide[label] = 0.0
    wide = wide[[*id_cols, *MOL_BIN_LABELS]]
    wide = wide.rename(columns={lbl: f"mol_{lbl}" for lbl in MOL_BIN_LABELS})
    return wide


def bin_by_pct(cdf: pd.DataFrame, id_cols: list[str]) -> pd.DataFrame:
    df = cdf.copy()
    df["pct_bin"] = assign_pct_bin(df["cluster_size"], df["one_cluster_particle_frac"])
    if (df["pct_bin"] == "unassigned").any():
        n = int((df["pct_bin"] == "unassigned").sum())
        raise ValueError(f"{n} rows fell into 'unassigned' pct bin.")

    wide = (
        df.groupby([*id_cols, "pct_bin"], observed=True, as_index=False)
          .agg(frac_particles=("frac_particles", "sum"))
          .pivot_table(index=id_cols, columns="pct_bin",
                       values="frac_particles", fill_value=0, observed=True)
          .reset_index()
    )
    for label in PCT_BIN_LABELS:
        if label not in wide.columns:
            wide[label] = 0.0
    wide = wide[[*id_cols, *PCT_BIN_LABELS]]
    wide = wide.rename(columns={lbl: f"pct_{lbl}" for lbl in PCT_BIN_LABELS})
    return wide


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build analysis_cluster_binned.csv from analysis_clusters.csv."
    )
    parser.add_argument("sweep_dir", help="Sweep directory name inside BASE_DIR.")
    parser.add_argument(
        "--clusters-csv",
        default=None,
        help="Override input CSV path. Default: <sweep>/analysis_clusters.csv "
             "or analysis_cluster_resolved.csv if the former is missing.",
    )
    args = parser.parse_args()

    sweep_path = BASE_DIR / args.sweep_dir
    if not sweep_path.is_dir():
        raise FileNotFoundError(f"Sweep directory not found: {sweep_path}")

    if args.clusters_csv is not None:
        clusters_csv = Path(args.clusters_csv)
    else:
        candidates = [
            sweep_path / "analysis_cluster_resolved.csv",
            sweep_path / "analysis_clusters.csv",
        ]
        clusters_csv = next((p for p in candidates if p.is_file()), None)
        if clusters_csv is None:
            raise FileNotFoundError(
                f"No cluster CSV found in {sweep_path} "
                f"(tried: {[p.name for p in candidates]})"
            )

    cdf = pd.read_csv(clusters_csv)
    if "molecule_file" in cdf.columns and "rna" not in cdf.columns:
        cdf["rna"] = cdf["molecule_file"].astype(str).str.replace(r"\.json$", "", regex=True)
    
    # Assemble id cols: shared ones + any extras present.
    id_cols = list(CLUSTER_ID_COLS)
    for col in EXTRA_ID_COLS:
        if col in cdf.columns and col not in id_cols:
            id_cols.append(col)
    id_cols = [c for c in id_cols if c in cdf.columns]

    required = {"cluster_size", "frac_mols", "frac_particles", "one_cluster_particle_frac"}
    missing = required - set(cdf.columns)
    if missing:
        raise ValueError(f"Missing required columns in {clusters_csv.name}: {sorted(missing)}")

    mol_df = bin_by_size(cdf, id_cols)
    pct_df = bin_by_pct(cdf, id_cols)

    out = mol_df.merge(pct_df, on=id_cols, how="outer")

    mol_cols = [f"mol_{l}" for l in MOL_BIN_LABELS]
    pct_cols = [f"pct_{l}" for l in PCT_BIN_LABELS]
    out["mol_sum"] = out[mol_cols].sum(axis=1)
    out["pct_sum"] = out[pct_cols].sum(axis=1)

    for name, s in [("mol", out["mol_sum"]), ("pct", out["pct_sum"])]:
        if not np.allclose(s, 1.0, atol=1e-6):
            bad = s.loc[~np.isclose(s, 1.0, atol=1e-6)]
            print(f"WARNING: {name}_sum != 1 for {len(bad)} rows "
                  f"(example: {bad.head().tolist()})")

    out = out[[*id_cols, *mol_cols, *pct_cols, "mol_sum", "pct_sum"]]

    out_path = sweep_path / "analysis_cluster_binned.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved {len(out)} rows to {out_path}")


if __name__ == "__main__":
    main()