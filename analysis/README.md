# RNA simulation analysis

Reproducible analysis of coarse-grained RNA simulations for single-chain
validation, RQ1 (reentrance), RQ2 (length, with scrambled controls), and RQ3 (mixtures).

The pipeline runs GSD -> accepted pairing NPZ -> molecular graph NPZ ->
framewise measurements -> **replicate-level 10% means**. It checkpoints each
completed replicate in the input sweep's output folder, then exports **two plotting-ready CSVs per
sweep**. Original simulation input is never modified.

## Environment and automated tests

Use **Python 3.11** for the analysis environment (GitHub Actions also checks
3.12). Install the CPU analysis and testing dependencies from the repository
root:

~~~bash
python -m pip install -r analysis/requirements-test.txt
python -m pytest analysis/tests -q
~~~

The optional `analysis/requirements-trajectory.txt` adds GSD and freud for
recomputing particle pairs or creating GSD projections. This does **not**
install HOOMD-blue or the custom simulation plugin. Details and commands for
notebooks are in [ENVIRONMENT.md](ENVIRONMENT.md).

The [GitHub Actions workflow](../.github/workflows/analysis-tests.yml) runs
**these existing tests** automatically on pushes and pull requests. No
production trajectories, GPU or HPC services are required. The tests cover
scientific calculations, checkpoint exports, the four analysis profiles and
plotting functions.

## Run

From the repository root, for example:

~~~bash
python analysis/run_pipeline.py \
  --sweep ../validation/bulk/output_dir/52b_from_frame \
  --output-root ./analysis_results \
  --experiment mixture \
  --stage all \
  --overwrite-npz
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
| `pairing` | GSD | Accepted-pair NPZ, molecule lookup, exact GSD steps; **single-chain Rg in nm** |
| `graphs` | NPZ | Weighted molecular graph NPZ, plus intramolecular NPZ for mixtures |
| `analyze` | NPZ + `params.json` | SQLite checkpoint and compiled CSVs; **no GSD reads** |
| `all` | GSD, then NPZ | Pairing, graphs, analysis; excludes optional energy |
| `energy` | GSD | Optional logged energies and KE per particle, summarized in SQLite |

Use `--overwrite-npz` with `--stage all` for a genuinely fresh run even when the input sweep already contains cached NPZs. It only overwrites NPZs beneath the new output root; original input archives are unchanged.
Pairing/contact-graph scientific calculations are unchanged by aggregation.
`--save-framewise` optionally saves diagnostic time series per trajectory
under the mirrored run directory, but is off by default.

Single-chain analysis requires a matching ViennaRNA reference CSV. Specify
`--viennarna-reference path/to/reference.csv` to override the configured
reference path; relative paths are resolved against the input sweep.
For **single-chain validation only**, Rg is calculated automatically in nm
from unwrapped particle coordinates while the `pairing` stage is already reading
GSD frames. The values are cached as `Rg_nm` in the pairing NPZ and averaged
over the same final 10% in `analyze`, so `--stage analyze` never reopens GSD.
Other research questions do not calculate Rg.

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

**`replicates.csv`** is **wide**: **one row per trajectory**, one column
per scalar scientific measurement, with parameters and input provenance at the
front. It includes `folder_name`, `traj_replicate`, `molecule_file`, E_att,
stiffness, density, particle/molecule counts, molar concentrations, and the
analysis window. `sequence_class`, `repeat_count`, and `scramble_id` appear
only where meaningful; the sweep name is already part of the directory path.

For example, use `n_base_pairs`, `n_edges`, and `mean_degree` directly.
Cluster-bin measurements use names such as
`cluster_bin_molecule_count_monomer` and
`cluster_bin_particle_percentage_ge_50pct`.
Mixture outputs include `rank_largest_cag10_n_molecules`,
`rank_largest_cag47_fraction_total_particles`,
`largest_contacts_cag10_cag47_fraction_inter_contacts`, and
`largest_mean_degree`. A ranked particle fraction is a fraction of **all
particles in the simulation**, not a fraction of just the ranked cluster.

**`summary.csv`** is also **wide**: **one row per condition**, grouped by RNA
composition/length, attraction, stiffness, density, and analysis parameters.
Each metric has `_mean`, `_std`, `_p10`, and `_p90` columns; for example
`n_edges_mean` and `n_edges_std`. `n` counts statistically independent
units and `n_trajectories` counts raw trajectories. For scrambled controls,
trajectories of each *distinct scramble sequence* are averaged first, then
all scrambles contribute equally. The identity of every scramble is kept in
`replicates.csv`, but does not split the summary into separate conditions.

The pipeline avoids redundant extra columns such as
`inter_particle_contacts_total` (already `n_inter_contacts`) and
`mol_contacts_total` (already `n_edges`).

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

## Validation

From the repository root, run the lightweight regression tests:

~~~bash
python -m pytest analysis/tests -q
~~~

The tests cover the wide replicate and condition exports, correct weighting of
independent scramble sequences, cluster-bin conservation, MFE matching, and
species-resolved contacts in mixtures. The four profile smoke tests operate
entirely on synthetic pairing/graph NPZ data and deliberately have no GSD
trajectory file, verifying that normal analysis never opens GSD.

For real data, validate `replicates.csv` and `summary.csv` for a single
parameter run before processing the full sweep. Compare selected scalar,
ranked-cluster, and largest-cluster contact metrics against previously
verified values using the **same final 10% frame window**.

## Plotting notebooks

The curated research notebooks are in [notebooks/](notebooks/README.md):

- [Single-chain validation](notebooks/01_validation.ipynb)
- [RQ1 — Reentrant clustering](notebooks/02_reentrance.ipynb)
- [RQ2 — RNA length](notebooks/03_length.ipynb)
- [RQ3 — CAG10/CAG47 mixtures](notebooks/04_mixtures.ipynb)

The notebooks load the **wide CSVs** above, use centralized `PlotStyle`
settings and `functools.partial` for reusable axes-first plot functions.
Framewise-only plots are implemented separately and not run in the notebooks.
The RQ1/RQ2 XY projections can optionally read **final GSD frames**;
an optional PyVis network can be exported from an NPZ without GSD.
See the [plotting guide](notebooks/README.md) for details.

## Figure mapping

The notebooks do plotting and hypothesis tests; the pipeline produces the
measurements and descriptive summaries:

- **Validation (Figures 6–8):** MFE matches, extra pairs, precision, recall,
  and automatically cached single-chain Rg.
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
