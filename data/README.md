# Public, curated analysis data

This directory contains only the small files needed to reproduce the
**reported scientific figures**, plus selected illustrations. No complete
simulation sweeps, SQLite checkpoints, or full-trajectory NPZ archives are
distributed with the repository.

## Structure

- `processed/<sweep>/replicates.csv` — each independent simulation trajectory.
- `processed/<sweep>/summary.csv` — per-condition aggregate statistics.
- `figures/reentrance_xy.png`, `figures/length_xy.png` — representative
  final-frame spatial projections, exported on the HPC system.
- `networks/mixture_Eatt{2,4,8}_network.html` — optional standalone PyVis
  visualizations, exported from graph NPZ files.

The notebooks in `analysis/notebooks/` use the published CSV directory
`data/processed/` and do **not** require raw simulation files for their
default figure cells.

## Copy the completed sweeps

Run from the repository root on the HPC system, **after** full-sweep analysis
has finished:

```bash
mkdir -p data/processed
for sweep in \
  53c_single_mol_duplex \
  50b_reentrant_full \
  51a_repeat_number \
  51b_repeat_number_rand \
  52b_from_frame; do
    mkdir -p "data/processed/$sweep"
    cp "analysis_results/$sweep/replicates.csv" \
       "analysis_results/$sweep/summary.csv" \
       "data/processed/$sweep/"
done
```

Verify that all ten CSVs are complete and correspond to the **original full
sweeps** before committing. In particular, do not replace original research
results with the shortened demonstration trajectory's `summary.csv`: that
example has one trajectory and a different effective averaging window.

The standard pipeline calculates **the final 10% of frames** per trajectory.
For single-chain validation, CSV columns `Rg_nm` (replicates) and
`Rg_nm_mean` / `Rg_nm_std` (summary) describe the mean Rg and its
between-replicate dispersion. This is not the original 20%-window
within-trajectory percentile interval from the earlier plotting notebook.

## Optional exports

The projection cells in `02_reentrance.ipynb` and `03_length.ipynb`
remain opt-in; they read final GSD frames from the original HPC paths. Export
the figures to PNG and embed them in notebook Markdown. The mixture notebook
has opt-in PyVis HTML export from cached molecular graph NPZ files. Standalone
HTML may be downloaded and opened in a local browser; GitHub's notebook
preview will not execute its JavaScript.

The separate `examples/mixture/eatt2/` directory is a **50-frame pipeline
demonstration** and is not input to the published statistical plots.
