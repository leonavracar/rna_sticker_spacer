# Figures and generated exports

The canonical output root is `analysis_results/`. Each sweep has its own
directory, with `summary.csv`, `replicates.csv`, and untracked checkpoint
files. Notebook figure exports belong under that same sweep's `figures/`
subdirectory, e.g.:

```text
analysis_results/
  50b_reentrant_full/
    summary.csv
    replicates.csv
    figures/
      cluster_bins.png
      cluster_bins.pdf
      reentrance_xy.png
```

Set `RNA_ANALYSIS_RESULTS_ROOT` when the results live outside the repository.
Generated files in the HPC directory are **not automatically published**
to GitHub; that requires explicitly staging selected files from a repository
checkout or copying/linking the selected files into it.

The spatial projections read **input GSD trajectories**, not analysis CSVs.
Keep input sweep paths separate from `analysis_results`. Use a representative
final frame as the plotting code currently does and label its provenance.
Figure filenames should be stable, descriptive, and include parameters when
creating more than one figure per family. Prefer PNG for display and PDF
for editable/vector publication figures.

Do not commit every output by default. Review small CSVs and selected figures,
then `git add` them explicitly; keep large GSD, NPZ, and SQLite artifacts out.
