# Phase 1 — Analysis audit and scope decisions

**Status:** working draft; metric scope awaiting researcher review.
**Branch:** `dev/analysis-refactor`
**Scope:** analysis only. Keep `bulk/` simulation code unchanged. Do not delete legacy code until test/figure parity is established.

## Goal

Turn the existing thesis analysis into a documented, modular, reproducible analysis package while preserving the scientifically meaningful calculations and visualizations. This document inventories the current implementation, sketches a dependency-oriented pipeline, and records decisions still required.

**Source-of-truth during migration:** `analysis_scripts/` and the research notebooks, *not* the incomplete `analysis/src/` migration. Newer code may contain improvements, but must be compared rather than assumed equivalent.

## Current sources

| Source | Primary responsibility | Migration action |
|---|---|---|
| `analysis_scripts/phase_separation.py` | ~2,279-line collection of pairing, molecule graphs, clusters, dynamics, geometry, ViennaRNA comparison, energetics, statistics and some plotting | Extract by responsibility; validate calculations |
| `analysis_scripts/helpers.py` | GSD/NPZ/JSON readers and metadata, plus earlier geometry helpers | Move general-purpose I/O to `io.py`; preserve historical geometry separately if necessary |
| `analysis_scripts/ps_all/run_compute_ps.py` | GSD → filtered Watson–Crick particle contacts, diagnostics, molecule lookup | Turn into pairing pipeline stage |
| `analysis_scripts/ps_all/run_mol_graph.py` | Particle pairing NPZ → molecule contact-edge NPZ | Turn into contact-graph pipeline stage |
| `analysis_scripts/ps_all/run_time_df_ps.py` | Framewise contacts, clustering, recurrence, energy/time metadata → CSV | Split computational metrics and output assembly |
| `analysis_scripts/ps_all/run_analysis_clusters.py` | Cluster-size distribution and material-normalized quantities | Move to cluster metrics |
| `analysis_scripts/ps_all/run_graph_framewise_summary.py` | Weighted molecule-graph metrics | Move to network metrics |
| `analysis_scripts/ps_all/run_analysis_binned.py` | Binned cluster distributions | Move to cluster binning/data transforms |
| `analysis_scripts/ps_all/run_analysis_ps_summary.py` | Replicate-level summaries | Move to aggregation |
| `analysis_scripts/mix/run_time_df_ps_mix.py` | Mixture-specific composition, contact types and cluster statistics | Share graph core; separate species-aware metrics |
| `analysis_scripts/mix/run_intramol_graph.py` | Per-molecule intramolecular contact NPZ | Move to contact representation |
| `analysis_scripts/mix/run_cluster_resolved_mix.py` | Cluster-resolved mixture summaries and contact saturation | Refactor into generic cluster representation + species metrics |
| `analysis_scripts/mix/run_analysis_ps_mix.py` | Mixture per-replicate summaries | Consolidate with common aggregation |
| `analysis_scripts/single_mol/` | Single-chain geometry, contact recurrence, ViennaRNA/MFE comparison | Optional single-chain analysis stage |
| `analysis_scripts/plots_ps.py` | Plotting, binning, labels, path discovery, plot data transforms | Split plotting from data transformations and filesystem details |
| `analysis_scripts/visualisation.py` | Coordinate unwrapping and static/interactive trajectory viewing | Move reusable visualization functions only after output review |
| `analysis_scripts/plots/cluster_diagnostic_plot_traj.py` | Trajectory-based 2D projections, 1D profiles, RDF, rough clustering diagnostics | Retain 2D/1D/RDF; reconcile its cluster definition with main pairing pipeline |
| `analysis/src/` | Partial migration of functions also in `phase_separation.py` | Compare implementations and preserve one authoritative copy |
| `analysis/notebooks/diagnostic_plots.ipynb` | Density projection and RDF panels across runs/parameters | Convert reusable functions to library; keep curated notebook |
| `analysis_scripts/archive/` | Older computational paths and exploratory methods | Freeze until thesis/figure cross-check; do not silently mix with current outputs |
| Other `notebooks/` and HTML exports | Research explorations and figure outputs | Inventory input notebook sources; curate separately |
| PyVis and other notebooks not yet in this repository | Interactive graph visualizations and new analyses | **Pending:** obtain source notebook(s) before classifying or porting |

## Proposed pipeline and data contracts

```text
GSD + params.json + molecule specification
  |
  +--> particle-level complementary pairing
  |      output: contact arrays, frame indices, distances, diagnostics
  |
  +--> contact representation
  |      output: molecule lookup, inter-/intramolecular edge tables
  |
  +--> metrics (can be requested independently)
  |      +-- contact/graph statistics
  |      +-- cluster components, distributions and fractions
  |      +-- time/recurrence and persistence
  |      +-- mixture species composition/recruitment
  |      +-- single-chain geometry and ViennaRNA comparison
  |      +-- trajectory energy/time series
  |
  +--> framewise tables
  +--> replicate aggregation (e.g. final 10% of frames)
  +--> across-replicate summaries and statistical comparisons
  +--> figures / interactive visualizations / curated notebooks
```

**Core rule:** individual compute functions accept arrays, graphs, trajectory frames, and explicit settings; they must not depend on hard-coded VBC cluster paths or write CSVs/figures. The pipeline/CLI owns run discovery, configuration, caching, logging, and output paths.

A **molecule-level cluster** must be based on the same canonical definition across numerical plots and diagnostics (filtered complementary nucleotide contacts; explicit minimum contacts per molecule pair), unless a diagnostic deliberately uses a different heuristic and labels it clearly.

## Provisional scientific scope — researcher decisions required

These entries are **not deletion instructions**. `REQUIRED` was explicitly requested; `PROPOSED KEEP` and `DECIDE` are awaiting confirmation.

| Area | Candidate metrics or outputs | Draft decision | Notes |
|---|---|---|---|
| Pairing | Complementary contact counts; inter-/intramolecular pairing; filtering/valency ambiguity rate | PROPOSED KEEP | Foundational inputs for other stages |
| Clustering | Cluster-size distributions using absolute number of molecules and relative percentage of the system | **REQUIRED** | Drop separate mono/oligo/drop fractions; retain monomer as a valid distribution bin |
| Graph | Edge count/density, mean degree, contact/edge strength mean and q90 | PROPOSED KEEP | Distinguish raw contacts from edges passing the threshold |
| Mixtures | Species-resolved recruitment, mixed/pure components, homotypic/heterotypic contacts | PROPOSED KEEP | Generic metadata rather than `cag10` and `cag47` hard-coded |
| Mixtures (advanced) | Cluster-resolved species composition, site-normalized contact saturation | DECIDE | Valuable but specialized and currently extensive |
| Single-chain validation | Rg, end-to-end distance, ViennaRNA MFE precision/recall/F1/bp distance | PROPOSED KEEP | Validate comparison definitions and experimental convention |
| Dynamics | Pair contact recurrence/persistence across windows, cluster persistence | DECIDE | Preserve if used to support claims/plots |
| Thermodynamics | Potential, bonded/angle, SQD and kinetic energy traces | DECIDE | At least useful as diagnostics; may not need heavy summary exports |
| Concentration | Density/box-derived concentrations and unit conversion | DECIDE | Useful contextual metadata if conversion assumptions documented |
| Cluster binning | Absolute size bins and relative-percentage bins | **REQUIRED** | Bins are configurable; explicitly distinguish molecule fraction from particle/material fraction in mixtures |
| Transitions | Smoothed feature-change scores and peak detection | DECIDE | Appears exploratory; no demonstrated requirement yet |
| Statistical comparison | Replicate aggregation and permutation tests/adjusted p-values | PROPOSED KEEP | Locate authoritative thesis implementation before migration |
| Geometry extras | Old three-bead-style dihedrals/stacking/inversion | TENTATIVE ARCHIVE | Not obviously part of present one-particle/base model |

### Visualization inventory

| Figure/output | Draft decision | Current implementation / next action |
|---|---|---|
| **2D particle-density projections: XY, XZ, YZ** | **REQUIRED** | `analysis/src/plots/diagnostic.py`, `analysis/notebooks/diagnostic_plots.ipynb`, and legacy diagnostic script |
| Time-averaged 2D projections; parameter-grid comparison panels | **REQUIRED** as projection variants | Ensure precisely defined frame selection, histogram normalization and color scale |
| 1D density profiles | PROPOSED KEEP | Existing diagnostic functions |
| Radial distribution function g(r) | PROPOSED KEEP | Existing diagnostic functions |
| Cluster-size distributions and stacked-bin summaries | PROPOSED KEEP | Existing plot/analysis scripts |
| Time-series, replicate variation, error bars, phase-like/reentrance figures | PROPOSED KEEP | Refactor plotting and transforms in `plots_ps.py` |
| Interactive network/cluster graphs with **PyVis** | **REQUIRED** | Source inspected in user-provided `8_directed_graph_2.ipynb` (not yet committed): two overlapping implementations; preserve both styling presets and extract reusable library function |
| Interactive/animated trajectory visualization (Plotly etc.) | DECIDE | `visualisation.py` is substantial; curate based on actual output |
| Unpublished recent notebook plots and analyses | **CAPTURE SOURCE** | Must inspect before migration; no deletion decision yet |

### Notebook/script policy

1. **Package modules** contain reusable algorithms, reusable plots, and reusable PyVis HTML generation, with inputs/outputs and unit tests.
2. **CLI/scripts** orchestrate repeatable analysis and batch-figure generation, including headless runs on compute clusters.
3. **Curated notebooks** tell the scientific story, perform light exploratory analysis, and assemble figures by importing package functions. A notebook may be experimental before promotion into modules.
4. **Static/HTML outputs** are products; keep a *small curated gallery* of figures, not every generated large export. Do not remove existing notebook outputs until corresponding sources are identified.
5. Advanced visualization dependencies such as `pyvis` can be optional extras; numerical analysis should run without them.
6. The repository currently tracks a notebook of ~53 MB and multiple multi-MB HTML exports. Review before including them in the final public portfolio.

## Specific issues identified during audit

- `analysis/src/pairing.py`, `molecule_graph.py`, `cluster_distribution.py`, and `recurrence.py` duplicate numerous `phase_separation.py` functions; some diverge. Avoid maintaining parallel implementations.
- `analysis/src/recurrence.py` imports from `validation_analysis`, which does not match this repository's import paths.
- `analysis/notebooks/diagnostic_plots.ipynb` passes `frame_idx` to `compute_projection_2d`, but the current `analysis/src/plots/diagnostic.py` function signature has no such argument.
- `analysis_scripts/plots/cluster_diagnostic_plot_traj.py` performs its own molecule-cluster calculation using a raw spatial distance search rather than the filtered complementary contact pipeline. This is acceptable only if intentionally treated as a different, clearly labeled diagnostic; otherwise centralize the definition. Also check whether symmetric neighbor pairs lead to doubled edge counts.
- `run_analysis_binned.py` selects `analysis_cluster_resolved.csv` first when present, but expects `cluster_size`/`frac_mols`/`one_cluster_particle_frac`, while the mixture-resolved schema uses different columns and row semantics.
- `run_analysis_clusters.py` infers equal chain lengths from particle count being divisible by molecule count, which is not a sufficient homogeneity check.
- Analysis paths and imports are repeated in runners; `plots_ps.py` contains hard-coded cluster paths and `sys.path` manipulation.
- `helpers.py` has duplicate `drop_all_nan_columns` implementations and large commented legacy sections.
- `run_cluster_resolved_mix.py` explicitly assumes just CAG10 and CAG47 and codes species by length. Generalize metadata after verifying the original results.
- Stage settings (cutoff radius, nearest partner selection, excluded contour distance, stride, minimum contacts, recurrence window, analysis fraction, bins) must be saved as provenance and affect cache keys/output identity.
- Existing group summaries, combined CSVs, and selective runs need consistent policy to avoid accidentally mixing old and new computations.
- Some plotting code calls `plt.show()` from a save helper; separate plotting, save, and display to enable headless reproducible generation.

## Migration sequence (no numerical implementation changed yet)

1. **Approve metric/visualization scope** and locate the current sources of thesis figures, especially PyVis and newer notebook work.
2. Freeze small representative *legacy* reference outputs: homogeneous system, repeat-length sweep, mixture, and single molecule; record parameters.
3. Define canonical schemas, algorithmic definitions, and metadata/provenance; write tests for tiny synthetic systems.
4. Extract pairing → contact arrays/graphs → clusters and graph statistics. Compare expected/legacy outputs before replacing calls.
5. Add optional mixture, single-chain, recurrence, aggregation and relevant plotting.
6. Introduce one CLI and a minimal quick-start sample. Preserve `analysis_scripts/` until complete end-to-end parity, then archive/delete only reviewed unused material.

## Open research decisions

1. Which metric groups in the scope table should be **core**, **optional**, or **left out**?
2. Which plots in `plots_ps.py` or project notebooks were actually used for the thesis main findings and should be regenerated automatically?
3. Which exact PyVis / newer-notebook implementations and additional figures should be imported into the repository?
4. Is the final public repository intended to support (a) thesis reproducibility only, or (b) a reusable RNA-simulation analysis toolkit, with the thesis as one example?
5. Which representative historical sweep(s) can supply a small baseline for numerical regression tests?

**Until these are answered:** retain all original sources; do not delete analytical functions or change scientific definitions.

## Confirmed decisions and uploaded artifacts — 2026-10-21

**Decisions from researcher**
- **Remove standalone** `mono_frac`, `oligo_frac`, `drop_frac` and their seven-statistic replicate summaries from the *new output specification*. Do not delete the original computations until comparison tests pass.
- Keep cluster size distributions, expressed either as **number of RNA molecules per component** or **size as a percentage of the total molecule population**, making differences in total chain count comparable across repeat-length systems. Still allow the monomer category (`cluster_size == 1`) inside distributions.
- For mixed lengths, distinguish molecule-fraction and particle/material-fraction denominators (they are not interchangeable). Decide which one is used as a plot-axis definition per research question, and clearly label it.
- Rewrite figure-generating notebooks to **import tested calculation and plotting functions**. Notebooks should retain experiment selection, presentation, scientific interpretation, and light exploratory code. No copied complex implementations.
- Retain **XY/XZ/YZ projections**, including parameter-grid plots, and retain **interactive PyVis networks** in the visualization scope.
- Reduce the number of exported metrics and redundant summary statistics; avoid exporting everything automatically.
- Reentrance is the first concrete dataset, not a universal schema for single-chain, repeat-length, or mixture experiments.

### New source material examined (uploaded; not committed to GitHub)

**`8_directed_graph_2.ipynb`:**
- 16 notebook cells; two near-duplicate large PyVis function implementations (`make_pyvis_molecule_graph`), followed by differing settings, including a version showing edge weights and a fixed HTML overlay of mean degree / edge count.
- Uses `networkx.Graph()` (undirected), not `DiGraph`. Molecular contact symmetry suggests retaining an undirected network unless a separate directional meaning is defined.
- Constructs nodes for all molecules, colors by connected-component size, sizes by degree, weights edges by particle contact count, and offers labels, tooltips, interactive physics, and optional physics freeze.
- Notebook currently includes cluster-local paths, wildcard imports, an explicitly selected frame 3999, and a prefilter at `min_contacts=2` followed by a second call at threshold 1. The new graph API should apply a single explicit threshold.
- **Migration:** one library function for graph construction (shared with numerical analyses) and one optional PyVis renderer (`plotting/networks.py`); a short example notebook imports them. Preserve styling presets, optional edge labels, summary overlay, and downloadable HTML.
- HTML rendering has not yet been visually validated in a browser; source code was reviewed.

**`analysis_per_replicate.csv`:**
- 480 rows and 95 columns; 8 attraction strengths × 3 angle stiffness values × 2 densities × 10 replicates. All are `cag47x100.json`.
- 77 columns arise from 11 `inst_*` metrics each expanded to last/mean/std/min/max/late_mean/late_std.
- 21 of those 77 are the now-rejected three-way mono/oligo/drop categories. This is a concrete example of unhelpful width.
- Use **one final-window mean per necessary metric per replicate** as the default analytical output. Use the framewise source to recover dynamics, maxima, etc., and compute between-replicate mean/variance at condition level. Additional statistics belong to explicit, opt-in requests.
- Record the final-window fraction (currently `fraction_last=0.1`) and parameters as metadata rather than repeating every possible derived/constant field in every table.

**`analysis_cluster_binned.csv`:**
- 1,920,000 framewise data rows, 27 columns.
- Contains `mol_*` categories (absolute sizes), `pct_*` categories (currently based on `one_cluster_particle_frac`) and `mol_sum`/`pct_sum` validation columns.
- **Important semantic distinction:** for equal-length chains, molecule count fraction and particle fraction per component coincide; for mixtures they diverge. The proposed relative-length comparison should explicitly define the size measure as `cluster_size_molecules / n_molecules_total`; optionally also export particle material fractions for mixture analyses. Do not silently relabel existing `pct_*` values without checking meaning.
- Store an optional long, detailed framewise table for debugging / advanced analysis, but make late-window **per-replicate binned distributions** the plotting-ready default. A compressed columnar format may be preferable for millions of frame rows, with CSV export as needed.
- `mol_sum` and `pct_sum` should be validation checks in the pipeline/tests, not necessarily permanently exported columns.

**`molecule_lookup.npy`, `trajectory_0_molecule_contact_edges.npz`, `trajectory_0_pairing.npz`:**
- User supplied representative intermediate inputs, suitable for fixture/regression development after checking their contents and source-run metadata. These binary payloads have not yet been decoded or numerically validated in this audit.
- Do not commit real large simulation data to Git; create a small synthetic or sampled fixture later.

### Proposed output tiers

1. **Intermediate / cache:** pairing NPZ, molecule mapping, weighted contact-edge NPZ; provenance attached. No automatic human-facing wide CSV.
2. **Detailed optional:** framewise graph/contact/cluster distributions and time series; useful for diagnostics and reproducing particular figures; may be large.
3. **Plotting-ready default:** per-replicate binned cluster distribution by absolute and relative size; *selected* per-replicate metrics averaged over final 10%; separate condition-level uncertainty across replicates.
4. **Domain extensions:** explicit mixture-specific composition/recruitment, optional detailed cluster resolution, single-chain validation, and separate equilibration diagnostics. Do not force all fields into a universal 95-column CSV.
5. **Visual artifacts:** authored/curated notebook panels, PNG/PDF figure export, optional interactive PyVis HTML.

### Open issues for researcher (prioritized)

- Beyond cluster-size distributions, which **few scalar metrics** must be shown for reentrance: e.g. largest-cluster fraction, contact-graph degree/density, contact strength? This is a scientific decision; default exports should not dictate it.
- Do we keep contact recurrence/persistence, and are kinetic/potential energy outputs required as *diagnostics only*?
- For % cluster size plots, confirm whether the requested denominator is all molecules (length comparison) and whether particle material fraction is additionally desired for mixtures.
- Which other RQ-specific figures deserve first-class generation: homogeneous reentrance, length sweep, scrambled control, single-chain validation, and CAG10+CAG47 mixture?
- Inspect remaining notebook sources and verify generated PyVis HTML visually before settling style/API.
