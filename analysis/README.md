# RNA simulation analysis

This directory contains the maintained analysis pipeline, reusable plotting code, curated research notebooks, and tests for the coarse-grained RNA sticker–spacer simulations in `bulk/`. **This is the working guide**; scientific definitions and the historical thesis-figure inventory live in [docs/metrics.md](docs/metrics.md) and [docs/figures.md](docs/figures.md).

## Repository layout and responsibilities

```text
rna_sticker_spacer/
├── bulk/                      HOOMD-blue simulation source (GPU; custom SQD plugin)
├── analysis/
│   ├── run_pipeline.py        GSD → pairing → graph → measurements → exports
│   ├── aggregate.py           Re-export existing SQLite checkpoints
│   ├── configs/               Analysis profiles and numerical settings
│   ├── src/                   Reusable calculations and plotting functions
│   │   └── plots/             Matplotlib, projections, optional PyVis
│   ├── notebooks/             Four curated research notebooks
│   ├── tests/                 Synthetic tests and opt-in real-GSD integration
│   │   └── fixtures/mixture/
│   │       ├── input/         params.json and trajectory_0.gsd
│   │       └── expected/      replicates.csv and summary.csv
│   └── docs/                  Scientific metric definitions and figure registry
├── analysis_results/<sweep>/  Per-sweep outputs (reviewed CSVs may be tracked)
├── analysis_scripts/          Historical analysis code (preserved for reference)
├── notebooks/                 Legacy research notebooks (preserved for reference)
└── environment.yml           Reviewed research Conda dependencies
```

The pipeline is responsible for **measurements and aggregation**; the curated notebooks select conditions, visualize results, and interpret them. The original simulation and legacy research code are left intact. There is **no second `data/processed/` tree**.

## Environment

The checked-in root [`environment.yml`](../environment.yml) describes the research Conda dependencies and CUDA/build prerequisites. **HOOMD-blue and the custom SQD plugin must each be built from source** and are *not* installed by that YAML. Exact source versions, compiler/CUDA options, and build commands will be documented when the simulation setup is finalized; do not assume that creating the Conda environment alone enables GPU simulations.

For analysis and plots only, from the repository root:

```bash
python -m pip install -r analysis/requirements.txt
```

To reconstruct accepted pairs from GSD, compute geometry from trajectories, or run the real-GSD integration test, also install the trajectory dependencies:

```bash
python -m pip install -r analysis/requirements-trajectory.txt
```

These CPU requirements do not require HOOMD-blue. They are compatible-range specifications, **not exact locks** for the cluster environment. The lightweight GitHub Actions tests run on Python 3.11; the research environment currently uses Python 3.13. Install JupyterLab in the analysis environment when you want to run notebooks.

## Pipeline: from trajectories to CSVs

From the repository root, replace the example sweep path with the location of your actual simulation outputs:

```bash
python analysis/run_pipeline.py \
  --sweep /path/to/validation/bulk/output_dir/52b_from_frame \
  --output-root ./analysis_results \
  --experiment mixture \
  --stage all --overwrite-npz
```

The `--sweep` directory contains parameter-run subdirectories, each with `params.json` and `trajectory_*.gsd`. The **four profiles** are `single_chain`, `reentrance`, `length_patterning`, and `mixture`; corresponding settings live in `analysis/configs/`.

| Stage | Input | Output |
| --- | --- | --- |
| `pairing` | GSD | Filtered complementary particle pairs, molecule lookup, frame indices/steps and optional single-chain Rg |
| `graphs` | Pairing NPZ | Weighted molecule–molecule edges and mixture intramolecular contacts |
| `analyze` | Saved NPZs + `params.json` | Per-replicate measurements, SQLite checkpoints, compiled CSVs; **no GSD read** |
| `all` | GSD | Pairing, graphs, analysis and CSV exports |
| `energy` | GSD logs | Optional energy/kinetic measurements in the checkpoints |

Use `--stage analyze` when the NPZ intermediates already exist. Use `--overwrite-npz` with `--stage all` to recompute instead of reusing archived intermediates. `--save-framewise` opts into per-trajectory diagnostics; they are not written by default. `--run-ids 0` selects matching parameter-run folders, including their trajectory replicates.

Single-chain validation uses a ViennaRNA reference CSV (`--viennarna-reference PATH` can override the configured path). Its Rg values are calculated from unwrapped particle coordinates during the GSD pairing pass, in nm, and reused by NPZ-only analysis. The other profiles do not compute Rg by default.

Every current profile averages the **final 10% of sampled frames** within each replicate. That is the implemented pipeline convention; it differs from the late windows used in some original thesis figures. See the figure registry for that distinction.

### Outputs and statistical units

```text
analysis_results/
  52b_from_frame/
    replicates.csv               one wide row per trajectory
    summary.csv                  one wide row per experimental condition
    analysis.sqlite              resumable replicate checkpoints (not tracked)
    <parameter-run>/
      computed_metrics/          pairing / graph NPZs (not tracked)
    figures/                     optional notebook PNG exports
```

`replicates.csv` contains each trajectory's experimental metadata (`E_att`, `angle_k_ppp`, `density`, composition, replicate ID, window parameters) and scalar measurements such as `n_base_pairs`, `n_edges`, `mean_degree`, cluster-size-bin fractions and mixture-specific contact and ranked-cluster fields.

`summary.csv` groups replicates by condition and exposes per-metric `_mean`, `_std`, `_p10`, and `_p90` columns. `n` counts independent statistical units; `n_trajectories` counts sampled trajectories. For scrambled-sequence controls, trajectories of the **same scramble sequence** are averaged first, then independent scrambles contribute equally.

Cluster-size percentages and ranked material fractions are **particle-weighted relative to all simulated nucleotides**, not just molecules in a cluster. The denominator for largest-cluster *contact-type composition* is instead the number of intermolecular contacts inside that cluster. Consult [metric definitions](docs/metrics.md) before changing these calculations.

To rebuild CSV exports directly from existing checkpoints (no GSD or NPZ reads):

```bash
python analysis/aggregate.py --output-root ./analysis_results
```

Add `--sweep 52b_from_frame` to target one sweep. Input GSDs remain at their original simulation location; outputs are written only under the configured results root. The old `analysis/analysis_output/` folder is **not** part of this workflow and is intentionally left untouched.

## Tests and numerical validation

From the repository root:

```bash
python -m pip install -r analysis/requirements-test.txt
python -m pytest analysis/tests -q
```

This is the fast synthetic unit/regression suite. It tests contact and cluster bookkeeping, CSV/SQLite exports, independent scrambled units and plot contracts, without GPU/HOOMD or real GSD data. The GitHub Actions workflow runs this CPU suite.

The **opt-in integration test** processes the checked-in small mixture trajectory and compares all exported columns against recorded reference CSVs:

```bash
python -m pytest analysis/tests/test_integration_mixture.py -m integration -q
```

Install `analysis/requirements-trajectory.txt` first. The test builds the expected sweep/run directory structure under pytest's temporary directory, reruns GSD → pairing → molecule graphs → analysis, and checks saved CSVs, molecule composition, contact conservation and independently reconstructed late-window contact/edge averages. The baseline in `analysis/tests/fixtures/mixture/expected/` was **generated previously by the analysis code**: it detects regressions, but is not independent proof of all scientific assumptions.

The short test trajectory and its CSVs must **not** be mixed into production scientific summaries. Both suites passed on the research cluster on 2026-10-26 (35 default tests and one GSD integration test); rerun them after changes to numerical behavior or dependencies.

## Notebooks, plots, and figures

The four curated notebooks in `analysis/notebooks/` use `analysis_results/<sweep>/replicates.csv` and `summary.csv`. They call functions from `analysis/src/plots/` rather than duplicating contact analysis.

| Notebook | Scope | Main sweeps |
| --- | --- | --- |
| [01_validation.ipynb](notebooks/01_validation.ipynb) | Single-chain MFE matches, precision/recall, Rg | `53c_single_mol_duplex` |
| [02_reentrance.ipynb](notebooks/02_reentrance.ipynb) | RQ1: reentrant clustering and stiffness | `50b_reentrant_full` |
| [03_length.ipynb](notebooks/03_length.ipynb) | RQ2: repeat lengths and scrambled controls | `51a_repeat_number`, `51b_repeat_number_rand` |
| [04_mixtures.ipynb](notebooks/04_mixtures.ipynb) | RQ3: mixed CAG10/CAG47, pure references | `52b_from_frame`, `51a_repeat_number` |

Start Jupyter from the repository root so the notebooks locate project modules and paths:

```bash
jupyter lab analysis/notebooks/
```

The notebooks default to the repository's `analysis_results/` directory. Set `RNA_ANALYSIS_RESULTS_ROOT` to point them at an existing HPC result tree; **do not copy results into another processing folder**. `SAVE_FIGURES = True` writes selected PNGs under `analysis_results/<sweep>/figures/`; switch it off if you only want interactive plots.

Optional RQ1/RQ2 **final-frame XY projections** read the original GSD files (enable `SHOW_PROJECTIONS`). The mixture notebook can export optional **PyVis networks** from graph NPZs (enable `EXPORT_PYVIS`). Neither feature is required to draw the ordinary summary plots. Historic/advanced XY/XZ/YZ, RDF and 1D diagnostic functions remain in `src/plots/diagnostic.py` and `notebooks/diagnostic_plots.ipynb`.

Reusable plot interfaces:
- `src/plots/style.py`: labels, palettes and `PlotStyle`;
- `src/plots/data.py`: CSV loading, condition filtering, `melt_metrics`, independent units;
- `src/plots/figures.py`, `panels.py`: axes-first plots and multi-panel layouts;
- `src/plots/projections.py`, `networks.py`: optional trajectory projections and contact-network HTMLs.

Only explicitly reviewed `summary.csv`, `replicates.csv` and selected `figures/*.png` or `*.pdf` are eligible for Git tracking under `analysis_results/`. Do not commit large trajectories, caches, NPZs, databases, or every generated figure. For a detailed map of thesis figure inputs and changes in late-window definitions, see [docs/figures.md](docs/figures.md).

## Scope and legacy material

`bulk/` produces simulations and is unchanged by the analysis refactor. `analysis_scripts/` and the root `notebooks/` contain earlier research implementations; they are deliberately preserved and **are not** the maintained pipeline. The old `analysis/analysis_output/` is likewise left untouched.

Scientific methods and design history are retained in Git history; the maintained documentation is this guide plus the two [scientific reference documents](docs/metrics.md).
