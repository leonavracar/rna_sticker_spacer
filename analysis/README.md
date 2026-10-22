# RNA simulation analysis

Reproducible analysis of coarse-grained RNA simulations for single-chain
validation, RQ1 (reentrance), RQ2 (length/patterning), and RQ3 (mixtures).

The pipeline runs GSD -> accepted pairing NPZ -> molecular graph NPZ ->
framewise measurements -> **replicate-level 10% means**. It checkpoints each
completed replicate in the input sweep's output folder, then exports **two plotting-ready CSVs per
sweep**. Original simulation input is never modified.

## Run

From the repository root, for example:

~~~bash
python analysis/run_pipeline.py \
  --sweep ../validation/bulk/output_dir/52b_from_frame \
  --output-root ./analysis_results \
  --experiment mixture \
  --stage all
~~~

To reanalyze existing NPZs **without opening GSDs**:

~~~bash
python analysis/run_pipeline.py \
  --sweep ../validation/bulk/output_dir/52b_from_frame \
  --output-root ./analysis_results \
  --experiment mixture \
  --stage analyze
~~~

The available profiles are `single_chain`, `reentrance`,
`length_patterning`, and `mixture`. All profiles use the **final 10%
of sampled frames**. Use `--run-ids 0` to select parameter-run folder
`0_*`; all `trajectory_*.gsd` replicates within it are included.

| Stage | Reads | Produces |
| --- | --- | --- |
| `pairing` | GSD | Accepted-pair NPZ, molecule lookup, exact GSD step indices |
| `graphs` | NPZ | Weighted molecular graph NPZ, plus intramolecular NPZ for mixtures |
| `analyze` | NPZ + `params.json` | SQLite checkpoint and compiled CSVs; **no GSD reads** |
| `all` | GSD, then NPZ | Pairing, graphs, analysis; excludes optional energy |
| `energy` | GSD | Optional logged energies, KE per particle and single-chain Rg, summarized in SQLite |

`--overwrite-npz` only overwrites NPZs beneath the new output root.
Pairing/contact-graph scientific calculations are unchanged by aggregation.
`--save-framewise` optionally saves diagnostic time series per trajectory
under the mirrored run directory, but is off by default.

Single-chain analysis requires a matching ViennaRNA reference CSV. Specify
`--viennarna-reference path/to/reference.csv` to override the configured
reference path; relative paths are resolved against the input sweep.
Single-chain Rg is computed by the optional `energy` stage, since geometry
requires the original GSD frames.

## Output

~~~text
analysis_results/
├── 52b_from_frame/               # automatically named after --sweep
│   ├── analysis.sqlite           # replicate checkpoints for this sweep
│   ├── replicates.csv
│   ├── summary.csv
│   └── <parameter-run>/
│       └── computed_metrics/...  # pairing and graph NPZs
├── 51a_repeat_number/
│   ├── analysis.sqlite
│   ├── replicates.csv
│   └── summary.csv
└── 51b_repeat_number_rand/
    ├── analysis.sqlite
    ├── replicates.csv
    └── summary.csv
~~~

**`replicates.csv`** is a long-format table: each row is one metric for
one independent trajectory. It contains the sweep, parameter-run folder,
trajectory identifier, molecule-file provenance, sequence class
(`repeat`, `scrambled`, `mixed`), CAG repeat count where applicable,
density, E_att, stiffness, fraction of frames, metric group, metric name,
category, species, and value. Concentrations `conc_mol_per_L` and
`conc_nuc_per_L` use the configured length conversion (0.6 nm/unit).

**`summary.csv`** groups observations within that sweep by experimental conditions,
metric group, metric, cluster class or contact type, and species. It contains
`mean`, `std`, `p10`, `p90`, `n` (independent units) and
`n_trajectories`. For periodic sequences, trajectories are the units.
For scrambled controls, trajectories of the *same scramble sequence* are
averaged first, and each distinct scramble has equal weight. We keep the
actual `molecule_file` in replicates.csv for traceability, not as an
experimental grouping condition.

**All RQs use the same 10% window**, including pure-system references
plotted beside mixtures. No old CSVs are read. Legacy per-run CSVs created
by earlier runner versions may remain on disk, but this version neither
creates nor uses them.

If the run is interrupted, previously committed replicates remain in SQLite.
To regenerate all sweep tables from their checkpoint databases (without GSD or
NPZ reads), run:

~~~bash
python analysis/aggregate.py --output-root ./analysis_results
~~~

Or export just one input sweep by adding `--sweep 52b_from_frame`.
The output names are **never based on the research question**; they always
match the input sweep folder. Different sweeps stay independent, including
the periodic and scrambled RQ2 runs, which the notebook can compare.

## Figure mapping

The notebooks do plotting and hypothesis tests; the pipeline produces the
measurements and descriptive summaries:

- **Validation (Figures 6–8):** MFE matches, extra pairs, precision, recall;
  optional GSD-based Rg.
- **RQ1 (Figures 9–12):** `cluster_bins_molecule_count`, base-pair
  intra/inter counts, edge counts/weights, mean molecule recurrence.
- **RQ2 (Figures 13–15):** `cluster_bins_particle_percentage`,
  `ranked_clusters`, mean particle recurrence, edge weights; replicate
  data support permutation tests in the notebook.
- **RQ3 (Figures 16–18):** `ranked_clusters` by species,
  `cluster_contacts` by contact type and `largest_cluster` size,
  molecular graph degree and edge counts. The pure reference panels of
  Figure 16 come from RQ2.

No monomer/oligomer/droplet *threshold classification* is computed.
Monomers remain a legitimate cluster-size bin and ranked material class.
The original thesis's 2D density projections require an explicit optional
GSD visualization step and are not recreated by the compiled tables.
