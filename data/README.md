# Analysis data and reproducibility

The analysis pipeline writes results directly to the configured HPC output root,
under a directory named after each input sweep. This is the only canonical
location for generated CSVs and analysis caches.

For example:

```text
/groups/goloborodko/projects/leona/rna_sticker_spacer/analysis_results/
  50b_reentrant_full/
    analysis.sqlite
    replicates.csv
    summary.csv
    <parameter-run>/computed_metrics/
```

The repository does **not** require a second `data/processed/` directory.
Do not copy sweep CSVs merely to run notebooks. Pass the existing
`analysis_results/` root to the notebook data-loading functions, or set
`RNA_ANALYSIS_RESULTS_ROOT` in the notebook's environment. The four curated
notebooks read that variable, defaulting to `analysis_results/` at the
repository root. The plotting loader accepts the root directory as an argument.

Production trajectories, SQLite checkpoints and NPZ archives are not
committed to Git. Small, explicitly curated demonstration data and
figures may be published separately, but must be labeled as snapshots,
not described as outputs of the active pipeline.

The source sweep directory remains separate from the analysis-results
directory; the pipeline must not modify original simulation outputs.

## Optional visual exports

Final-frame GSD projections and PyVis networks are optional exports. Keep
their destinations configurable and do not treat exported images or HTML as
canonical numerical analysis results.
