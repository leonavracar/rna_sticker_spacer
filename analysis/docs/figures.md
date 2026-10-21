# Figure registry and visualization architecture

**Status:** Phase 1 draft. This is an **extensible visualization plan**, not a limit to thesis figures. See [metrics.md](metrics.md) for mathematical definitions and [phase1_audit.md](phase1_audit.md) for existing code.

## Scope

The thesis figures form a **baseline reproducibility suite**. New figures, animations, model diagnostics, interactive network viewers, richer data explorations, and future simulation extensions are **first-class outputs**. The plotting API must never assume exactly four experiments or figure IDs 1–18.

Goals:
1. Separate reusable **data transformations**, **drawing/rendering**, and **experiment-specific compositions**.
2. Make each figure reproducible from specified source data, aggregation/window, settings, and code revision.
3. Rebuild curated Jupyter notebooks so they **import library functions** rather than defining hundreds of lines of algorithms/plotting utilities.
4. Permit both notebook display and command-line/headless figure rendering, including HPC runs.
5. Retain important 2D projections and interactive PyVis figures, including new visualizations not found in the thesis.

## 1. Thesis figure reproducibility registry

The actual figures are documented in the thesis Results `6.1–6.4 and Methods `5.1–5.4. This table names their scientific data needs; it does not pretend the existing scripts can currently recreate them all without repair.

| Thesis figure | Scientific figure content | Source dataset/metrics | Presentation and test notes |
|---|---|---|---|
| **2** | Bond-vector orientational correlation vs separation to calibrate persistence length | Single long-chain frames, backbone bond vectors, stiffness; reference e^-1 | Different calibration workflow from MFE validation; preserve physical conversion |
| **3** | Potential, density kernel and valency switching curves | Analytic model-potential parameters from bulk configuration or explicit formula | Pure deterministic plot; validate correspondence with simulated potential |
| **4** | Periodic CAG vs composition-matched scrambled sequence example | Reference sequence and scrambled input configuration | Schematic; design asset, not necessarily a simulation-data figure |
| **5** | ViennaRNA reference stem-loop and base-pair probability dot plot | ViennaRNA MFE prediction, pair probabilities | Reference-generation reproducibility; optional independent RNA visualization dependency |
| **6** | Native/MFE-matching vs extra base pairs, by E_att and stiffness | Per-replicate 20%-window matching/extra counts | One point per replicate plus condition mean |
| **7** | MFE precision–recall quadrants | Per-replicate 20%-window precision/recall | Replicate scatter, stiffness facets, attraction colors, 0.8 guidelines |
| **8** | Rg vs attraction for each stiffness | Per-replicate Rg means from final 20%; reference Rg | Replicate-mean marker, 10th–90th percentile interval, physical nm units |
| **9a** | RQ1 absolute molecule-count-binned cluster-size composition | Molecule connected components; thesis Table 5 size classes | 10%-window per-replicate fractions, average 10 replicates; monomer bin retained |
| **9b** | RQ1 2D XY density-projection panel across E_att and stiffness | Representative final GSD frames, positions, box dimensions | Match panel ordering, consistent projection scale and PBC; representative is not an average |
| **10** | Inter-, intra-, and total base-pair counts vs attraction | Per-replicate 10%-window N_inter, N_intra | Mean across replicates, SD bands, separate stiffness panels |
| **11** | Mean qualifying edge weight and edge count vs attraction | Per-replicate 10%-window weighted contact graphs | Boxplots + points by stiffness; edge threshold n_min = 2 |
| **12** | Molecule-contact recurrence vs attraction | 20-frame rolling recurrence; late 10%, k=2.5 | Replicate points, condition mean and SD band |
| **13a** | RQ2 particle-fraction-size-binned cluster composition | Connected components, particle fraction p_c, Table 5 relative bins | 10% final window; bin heights particle fractions, monomers separate |
| **13b** | RQ2 2D density projections by length and attraction | Representative final GSD frames, particle coordinates | Reusable projection code and configurable panel-grid layout |
| **14** | Particle-level base-pair recurrence over time across repeat lengths | Individual base-pair identities, w=20 | Show time traces, individual replicates + mean, final 10% |
| **15a** | Ranked cluster particle composition: periodic vs scrambled | Per-frame top 3 multimeric particle sizes, other multimers, monomers | Matching lengths, classes and parameter values |
| **15b** | Mean qualifying molecule-pair edge weight: periodic vs scrambled | Replicate-level 10%-window edge weights | Explicit handling of frames with no qualifying edges |
| **15c** | Holm-adjusted p-values for paired-condition comparisons of sequence classes | Two-sided permutation tests on replicate-level mean edge weight; 12 comparisons | 5 periodic and 10 scrambled replicates per condition; preserve family correction |
| **16** | Ranked cluster material for pure CAG10, pure CAG47, and mixture | Top 3 multimers + other multimers + monomers, particle-weighted species contributions | RQ3 mixed frames: final 5%; pure references sourced from RQ2 |
| **17** | Fraction of intermolecular base-pair contact types *within largest component* | CAG10–CAG10, CAG10–CAG47, CAG47–CAG47 accepted contacts, largest cluster | Stacked contact-composition fractions; annotation = absolute mean contact count |
| **18** | Largest-cluster material fraction and connectivity | Largest-cluster total particle fraction, molecule count, edge count, mean degree | 5% final RQ3 window; clearly distinguish contact edge vs particle pair |

**Figures 1** (model representation) and related illustrative schematics may remain curated static scientific assets. A future programmatic schematic generator is optional.

**Important denominator distinction:** All percent *cluster-size* measures and material fractions use **total nucleotide particle count**, including RQ2 and RQ3. In Fig. 17, the plotted contact-type fraction is instead normalized to total **intermolecular contacts inside the largest cluster**; it is not a cluster-size percentage. Absolute cluster-size bins are defined by molecule count, but bin **heights** use particles (numerically identical to molecule fractions for homogeneous equal-length samples).

**Window selection is experiment-specific:** single-chain 20%, reentrance 10%, repeat-length/scrambled 10%, mixtures 5%; source references for mixture comparison must preserve their own original window. Figure 14 uses trajectory/time-series data rather than a scalar late-window summary.

## 2. More than thesis figures: an open figure catalog

Every new visualization gets an independent human-readable ID and a small configuration/recipe. No requirement to publish every exploratory figure; make the best ones reproducible and keep the rest exploratory until useful.

| Category | Examples / design possibilities | Proposed module |
|---|---|---|
| **Trajectory morphology** | XY/XZ/YZ density projections; final-frame and time-averaged; 1D density profiles; consistent cross-run panels; RDF g(r) | `visualization/projections.py` and `plotting/diagnostics.py` |
| **Network visualization (interactive)** | PyVis contact networks with node size by degree, color by connected component / length/species, weighted edges, hover metadata, optional labels, adjustable physics and exportable standalone HTML | `visualization/networks.py` |
| **Network visualization (static)** | Network diagrams for publications, edge-weight histograms, distributions of degree/partners, q90 connectivity if needed | `plotting/networks.py` |
| **Assemblies through time** | Network views at selected frames, side-by-side snapshots, tracked component summaries (tracking itself needs its own defined algorithm), recurrence time traces | `plotting/dynamics.py` |
| **Phase-like sweep explorations** | Heatmaps of cluster size, connected material, contact balance, and/or degree over E_att, stiffness and length | `plotting/sweeps.py` |
| **Species-specific interaction diagnostics** | Per-species recruited material; contact-type diagrams, bipartite-like layouts, species-specific material profiles and mixture networks | `plotting/mixtures.py` |
| **Single-chain structures** | Fold similarity, Rg, MFE contacts, pair maps, persistence length; novel validation visualizations | `plotting/single_chain.py` |
| **Model evaluation and checks** | Pairing filter diagnostics, valency ambiguity, energy equilibration, simulation overlays, invariants/QC | `plotting/diagnostics.py` |
| **Interactive trajectory viewers/animations** | Existing Plotly/trajectory animations to be curated after source/output review | `visualization/trajectories.py` |

The examples above are **candidates, not a promise** that all have been implemented or are needed. We should treat user-supplied future notebooks as source material and add their strongest techniques to the library only after inspecting them.

## 3. Existing 2D-projection code: REQUIRED

Current sources:
- `analysis/src/plots/diagnostic.py`: projection and 1D profile computation/drawing; averaged versions; RDF.
- `analysis/notebooks/diagnostic_plots.ipynb`: XY/XZ/YZ and parameter-grid panels.
- `analysis_scripts/plots/cluster_diagnostic_plot_traj.py`: self-contained 2D/1D/RDF/rough-clustering multipanel diagnostics.

Reusable API should distinguish:
- `project_density(frame_or_positions, plane, bins, box, smoothing=...)` → histogram values + bin edges; pure computation.
- `average_projection(trajectory, frame_selection, ...)` → mean of specified frame histograms with explicit normalization.
- `draw_density_projection(ax, histogram, edges, normalization=..., color_scale=...)` → no trajectory reading, no implicit showing/saving.
- `plot_projection_grid(...)` → high-level panel composition with shared scales and subplot labels.
- `save_figure(fig, output_path, format=...)` → explicit saving; no `plt.show()` side effects inside low-level functions.

Frame selection (final vs averaged), spatial box range, periodic boundaries, colormap, smoothing, figure size, normalization (density vs counts), and shared color scales must be explicit in recipes. For thesis Fig. 9b and 13b, reproduce **final-frame representative projections**, not an inadvertent time average.

A pre-existing notebook sends `frame_idx` to a projection helper not accepting that argument; resolve as part of migration rather than preserving a broken interface.

## 4. PyVis: keep both design variants, one implementation

The uploaded **8_directed_graph_2.ipynb** has 16 cells, including two large alternative copies of the PyVis helper. The underlying molecular graph is **undirected** (`networkx.Graph`) despite the notebook filename — this is appropriate for symmetric pair contacts. Do not introduce directed edges unless a specific physical directionality is defined.

Preserve and clean up:
- Adjustable node size by degree and color by connected-component class or species (component classes are **visual styling**, not resurrected `mono_frac/oligo_frac/drop_frac` reporting metrics).
- Node IDs/degree labels optional; hover title with node/cluster/partner/contact statistics.
- Edge thickness by number of base-pair contacts; optional weight labels.
- Network layout physics presets, background/color themes, and stabilization controls.
- Summary annotation (edge count, mean degree) and portable standalone HTML.
- A second visual preset with labels and weights, rather than duplicated library implementations.

Proposed eventual interface (illustrative only):

~~~python
from rna_analysis.visualization.networks import make_pyvis_graph

graph = load_molecule_graph(edge_file, frame=3999, min_contacts=2)  # proposed API
make_pyvis_graph(graph, output_html="results/network.html", style="weighted")
~~~

Canonical graph construction belongs in shared scientific code; the PyVis renderer **receives the graph** and never recomputes clusters using a conflicting cutoff. Use `pyvis` as an optional dependency so core numerical workflows work without it.

Actual browser rendering/style remains to be tested with output from real uploaded contact data, not assumed.

## 5. Notebook policy: thin curated stories, not source of truth

**Code modules:** general algorithms, transforms, figure drawing, output/writer functions with tests, stable APIs, and docstrings.

**Notebooks:** choose experiments and filter conditions, pass a small settings/config object, import functions, compose figures, explain interpretation, display selected interactive visualizations. It is appropriate to write a few short exploratory transformations; after they prove useful, promote them into a tested module.

**Batch/CLI commands:** load data/configuration, compute requested metrics, render/export figures headlessly for parameter sweeps, and write manifest including code version + figure recipe.

Proposed curated notebooks (names can change):
- `01_single_chain_validation.ipynb`
- `02_reentrance.ipynb`
- `03_length_and_patterning.ipynb`
- `04_mixed_rna_systems.ipynb`
- `05_network_explorer.ipynb`
- `06_trajectory_diagnostics.ipynb`
- `07_beyond_thesis.ipynb` (new analyses, live experiments)

**Never embed machine-specific VBC filesystem paths, wildcard imports, large duplicated functions, or millions of data rows in the curated versions.** A notebook should execute top-to-bottom in a documented environment; optional browser-based network graphics should degrade clearly if renderer unavailable.

Large autogenerated notebook outputs and standalone HTML belong outside the core tracked source (or as a carefully curated small gallery). Keep enough representative media in the repository for a strong GitHub README.

## 6. Reproducibility contract for each figure

A figure recipe/manifest should record:

~~~yaml
id: rq2.length_cluster_composition
experiment: length_patterning
input_dataset: <sweep_id-or-data-location>
selection:
  attraction: [2, 4, 8]
  stiffness: 2.5
aggregation:
  final_fraction: 0.10
  unit: replicate
metric:
  source: cluster_bins
  binning: particle_percentage
  weight: total_particles
renderer: stacked_bars
output:
  formats: [png, pdf]
~~~

Figure IDs are **examples**, not an exhaustive enumeration. Exact config keys are provisional; a final schema should be introduced only once we have working implementations. A plot is reproducible when the inputs and exact recipe can regenerate it with no manual notebook surgery.

Suggested display function API:
- Drawing functions accept an explicit Matplotlib axes object and processed, validated arrays/dataframes; return the artist(s) or figure/axes.
- Composition functions manage panels, legends, shared color normalization and title metadata.
- Export belongs to a distinct pipeline call, with output location and overwrite behavior explicit.
- Plotly and PyVis are optional rendering backends. Do not force headless analysis to install JS/HTML stacks.

## 7. New-figure intake checklist

When a newer notebook or visualization arrives, record:
1. **Question/insight:** what it communicates that other figures do not.
2. **Input:** GSD, contacts, graphs, cluster distributions, metadata, or per-replicate summaries.
3. **Data transformations:** whether already implemented, need generalization, or require new mathematics.
4. **Renderer:** Matplotlib static, Plotly animation, PyVis HTML, or other.
5. **View parameters:** time/frame, parameter selection, binning, normalization, color, interactive controls.
6. **Portfolio value:** which few visuals deserve a README thumbnail or linked interactive example.
7. **Code decision:** extract reusable part → small example notebook; retain temporary exploration; or archive dead experiment.

**Do not treat the thesis as the final limit of figures.** The scientific definitions give us trustworthy inputs for a much larger visualization library.

## 8. Next implementation sequence

1. Build small reference fixtures and unit tests for particle pairs → molecule edges → component sizes → **particle-weighted** cluster bins, including mixed lengths and bin boundaries.
2. Extract reusable geometry-free computations and standardize I/O. Confirm regression against legacy NPZ/CSV data and the thesis' table semantics.
3. Migrate XY/XZ/YZ projection computation + panel layout, verifying frame selection.
4. Extract the PyVis renderer into one implementation with multiple presets; render and visually inspect an HTML example.
5. Refactor only the most useful result plots and notebook sources; add new figure types as needs arise.
6. Preserve original notebooks/scripts until a figure-by-figure parity check succeeds.

**No existing source code is modified by this document.**
