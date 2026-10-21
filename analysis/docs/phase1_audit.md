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
| Clustering | Cluster-size distributions classified by **absolute molecule count** or by **particle fraction of total system material** | **REQUIRED** | Relative percentages use particles only; remove separate mono/oligo/drop summary metrics, not the monomer histogram category |
| Graph | Edge count/density, mean degree, contact/edge strength mean and q90 | PROPOSED KEEP | Distinguish raw contacts from edges passing the threshold |
| Mixtures | Species-resolved recruitment, mixed/pure components, homotypic/heterotypic contacts | PROPOSED KEEP | Generic metadata rather than `cag10` and `cag47` hard-coded |
| Mixtures (advanced) | Cluster-resolved species composition, site-normalized contact saturation | DECIDE | Valuable but specialized and currently extensive |
| Single-chain validation | Rg, end-to-end distance, ViennaRNA MFE precision/recall/F1/bp distance | PROPOSED KEEP | Validate comparison definitions and experimental convention |
| Dynamics | Pair contact recurrence/persistence across windows, cluster persistence | DECIDE | Preserve if used to support claims/plots |
| Thermodynamics | Potential, bonded/angle, SQD and kinetic energy traces | DECIDE | At least useful as diagnostics; may not need heavy summary exports |
| Concentration | Density/box-derived concentrations and unit conversion | DECIDE | Useful contextual metadata if conversion assumptions documented |
| Cluster binning | Thesis Table 5: molecule-count binning and total-particle-percentage binning | **REQUIRED** | For relative-size bin assignment **and** fraction displayed in each bin, use particle counts / total particle count, including length sweeps and mixtures |
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
- Keep cluster size distributions classified either by **absolute molecule count per component** or by **percentage of all nucleotide particles in the system**, following thesis Methods §5.4 and Table 5. **Never use the total molecule count as the denominator for percentage-based cluster size.** Keep monomers (`cluster_size == 1`) as their own histogram bin.
- For *every* percentage-based cluster distribution, including mixed lengths, compute the **fraction of all system particles** belonging to the cluster/bin. Absolute (non-percentage) cluster size is still the number of molecules. Species contributions and ranked-cluster sizes are also particle-weighted as in the thesis.
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
- **Confirmed from thesis Methods §5.4 and Table 5:** `pct_*` bins classify each cluster by the fraction of **total nucleotide particles** it contains. For homogeneous systems `p_c = s_c * chain_length / N_particles`; for mixtures `p_c = sum(chain_lengths[m] for m in component) / N_particles`. The value in a percentage-bin column is the sum of these particle fractions for all components assigned to that bin. The molecule count is used only for the absolute-size `mol_*` bin identity, *not* as a percentage denominator. For homogeneous systems, molecule and particle weights happen to coincide; compute particle weights explicitly anyway.
- Store an optional long, detailed framewise table for debugging / advanced analysis, but make late-window **per-replicate binned distributions** the plotting-ready default. A compressed columnar format may be preferable for millions of frame rows, with CSV export as needed.
- `mol_sum` and `pct_sum` should be validation checks in the pipeline/tests, not necessarily permanently exported columns.

**`molecule_lookup.npy`, `trajectory_0_molecule_contact_edges.npz`, `trajectory_0_pairing.npz`:**
- User supplied representative intermediate inputs, suitable for fixture/regression development after checking their contents and source-run metadata. These binary payloads have not yet been decoded or numerically validated in this audit.
- Do not commit real large simulation data to Git; create a small synthetic or sampled fixture later.

### Proposed output tiers

1. **Intermediate / cache:** pairing NPZ, molecule mapping, weighted contact-edge NPZ; provenance attached. No automatic human-facing wide CSV.
2. **Detailed optional:** framewise graph/contact/cluster distributions and time series; useful for diagnostics and reproducing particular figures; may be large.
3. **Plotting-ready default:** per-replicate binned cluster distribution by absolute molecule count and relative particle percentage; *selected* per-replicate metrics averaged over an **experiment-specific** final fraction: single-chain 20%, RQ1 reentrance 10%, RQ2 length/patterning 10%, RQ3 mixture 5%, as specified in thesis Methods §5.3; separate condition-level uncertainty across replicates.
4. **Domain extensions:** explicit mixture-specific composition/recruitment, optional detailed cluster resolution, single-chain validation, and separate equilibration diagnostics. Do not force all fields into a universal 95-column CSV.
5. **Visual artifacts:** authored/curated notebook panels, PNG/PDF figure export, optional interactive PyVis HTML.


### Thesis-verified metric contract (confirmed by researcher)

Source: *Biophysical Insights into Repeat-RNA Clustering from Coarse-Grained Molecular Dynamics*, Methods §5.4, pp. 21–24, especially **Table 5** (p. 23); simulation averaging fractions in Methods §5.3, pp. 17–19. The **scientific definitions in the thesis** take precedence over earlier provisional recommendations in this audit.

#### Cluster-size definitions

- Graph: one node per molecule, undirected edges between molecule pairs with **at least two complementary intermolecular base-pair contacts in one frame** (`n_min = 2`). Cluster = connected component including isolated nodes.
- Absolute size: `s_c = number_of_molecules_in_cluster` (integer).
- Percentage size (use *particles exclusively*): `p_c = sum(length[m] for m in cluster) / N_particles`, in `[0,1]`. For homogeneous chain length `l_RNA`, `p_c = s_c * l_RNA / N_particles` (thesis notation).
- Absolute-size bins (by `s_c`): `1` monomer; `2` dimer; `3` trimer; `4–9`; `10–19`; `20–49`; `>=50`.
- Relative-size bins (by `p_c`): monomers (`s_c == 1`) first, then **multimers only**: `0 < p_c < .02`; `.02 <= p_c < .03`; `.03 <= p_c < .10`; `.10 <= p_c < .20`; `.20 <= p_c < .50`; `.50 <= p_c <= 1`.
- **Bar height / per-bin fraction:** `fraction_of_particles_in_bin = sum(number_of_particles_in_component for components_in_bin) / N_particles`. In the homogeneous special case this equals a molecule-weighted fraction; do not switch denominator in code.
- **Invariants:** all clusters partition the molecules; no particle counted twice; sum of the mutually exclusive histogram bin fractions equals 1 (within numerical tolerance).
- **Ranked clusters for mixtures:** sort multimeric clusters by number of particles *within each frame*; report top three separately, remaining multimers together, and monomers together. Each ranked category is weighted by the fraction of all particles. Ranks do not track a persistent cluster identity.

#### Do not confuse the two quantities in plot labels

- `cluster_size_molecules` or `s_c` is a **count** used to assign the left-hand Table 5 bins.
- `cluster_particle_fraction` or `p_c` is a **fraction of the system's particles** used for the right-hand Table 5 bins.
- `particle_fraction_in_bin` is the fraction of total **particles** in components whose size falls into a given category.
- For mixed chains, one molecule's contribution is proportional to its length, not one equal unit. All percentage values concerning cluster material or composition use particle weights.
- Terms `mono_frac`, `oligo_frac`, `drop_frac` from legacy framewise summaries are **not** retained as independent scalar observables; the monomer *distribution bin* remains.

#### Method/provenance settings to test against thesis

- Pair detection: `r_max = 0.7` in simulation length units, complementarity A-U/G-C, intrachain exclusion at separation `|i-j| < 4`, mutual choice of closest candidate to `r_e = 0.50`.
- Contact graph: qualifying molecule pair when `n_min = 2` filtered base pairs; weighted edge count = number of qualifying intermolecular base-pair contacts.
- Recurrence: rolling window of **20** saved frames in the thesis (Methods §5.4, p. 24). Some legacy runners default to 25; ensure replay configuration matches the research question rather than assuming the code default reproduced the thesis.
- Averaging fractions: **single-chain 20%**, **RQ1 reentrance 10%**, **RQ2 repeat length/scrambled 10%**, **RQ3 mixture 5%**. A global 10% default would not reproduce every thesis figure.
- Diagnostic energy traces are described as equilibration diagnostics rather than necessarily required scientific reporting tables.

#### Regression fixtures

Use the uploaded `analysis_cluster_binned.csv` as an example of the existing `mol_*`/`pct_*` output schema. The percentage-bin outputs should reproduce the thesis particle-weighted definitions. When turning actual examples into fixtures, extract only a tiny deterministic subset; avoid committing the multi-million-row CSV as a test dependency.


### Open issues for researcher (prioritized)

- Beyond cluster-size distributions, which **few scalar metrics** must be shown for reentrance: e.g. largest-cluster fraction, contact-graph degree/density, contact strength? This is a scientific decision; default exports should not dictate it.
- Do we keep contact recurrence/persistence, and are kinetic/potential energy outputs required as *diagnostics only*?
- **Resolved:** % cluster sizes and plotted percentage-bin values always use `N_particles` as denominator, including RQ2 length comparisons and RQ3 mixed systems. Verify fidelity to thesis Table 5 with tests.
- Which other RQ-specific figures deserve first-class generation: homogeneous reentrance, length sweep, scrambled control, single-chain validation, and CAG10+CAG47 mixture?
- Inspect remaining notebook sources and verify generated PyVis HTML visually before settling style/API.
