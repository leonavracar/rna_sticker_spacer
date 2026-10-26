# Analysis metrics and output contracts

> **Scientific reference and historical design notes.** Sections 1–2 record
> important definitions and original thesis methods; sections 3–6 contain
> earlier implementation proposals and open questions, **not the current CSV
> schema**. The maintained pipeline, fixed final-10%-of-sampled-frames
> convention, and current exported columns are described in the
> [analysis guide](../README.md). The original thesis windows (20% for
> single-chain and 5% for mixtures) are retained below as historical context.


**Status:** Phase 1 specification, v0.1 — definitions grounded in the MSc thesis; proposed file schemas pending implementation.
**Sources:** *Biophysical Insights into Repeat-RNA Clustering from Coarse-Grained Molecular Dynamics* (2026), especially Methods `5.3–5.4 (thesis pp. 17–24), results `6.1–6.4 (pp. 25–44), and figures 5–18. Consult `5.1–5.2 for the physical model. See [figures.md](figures.md) for visualization requirements and [phase1_audit.md](phase1_audit.md) for code inventory.

> The thesis is the scientific specification. Existing `analysis_scripts/` functions are candidates to migrate, not necessarily canonical definitions. `bulk/` is out of scope and must remain unchanged. Never discard original analysis artifacts before numerical regression tests pass.

## 1. Guiding decisions

- **Particle fractions exclusively for all percentage-based cluster-size comparisons.** The denominator is the total number of nucleotide beads, `N_particles`. This applies to homogeneous, different-length, and mixed systems.
- **Absolute cluster size remains a count of molecules**, `s_c`. Do not convert it to a percentage of molecules.
- **No standalone `mono_frac`, `oligo_frac`, `drop_frac` metrics** in the new standard summaries. The *monomer histogram bin* remains, as does the ranked-cluster `monomers` category.
- **Retain two thesis binning schemes:** bins by number of molecules in a cluster, and bins by percentage of all system particles in a cluster. In both schemes the *height* of a histogram bin is the fraction of **all system particles** belonging to its clusters. For the homogeneous RQ1 special case this numerically equals the thesis's molecule-fraction heights.
- **Common computation, experiment-specific reporting.** RQ1, RQ2, RQ3, and single-chain analysis should not share an indiscriminate wide 95-column report.
- **Derived observables on demand.** Keep detailed intermediate files where necessary for reproducibility but only emit a compact, chosen set of analysis/reporting columns.
- **Replicates are independent units of statistical analysis.** Calculate late-time means *within each replicate* and aggregate across replicate means; do not treat every saved frame as an independent replicate.
- **All threshold and time-window parameters are explicit and recorded.**

## 2. Foundational definitions

### 2.1 Particle contacts (thesis Methods `5.4, pp. 20–21)

Per saved frame, detect candidate pairs at spatial distance `r <= 0.7` (simulation units). Accept only A–U or G–C complements, exclude pairs on the same chain with sequence separation `|i - j| < 4`, and enforce approximate one-to-one contacts by retaining only **mutually preferred partners** (partner closest to potential minimum `r_e = 0.50`). Unique, undirected particle pairs are counted once.

- `N_bp`: accepted particle contacts (base pairs) in a frame.
- `N_inter`: accepted pairs between distinct molecules.
- `N_intra`: accepted pairs within one molecule.
- Invariant: `N_bp = N_inter + N_intra`.
- `f_inter = N_inter / N_bp` and `f_intra = N_intra / N_bp`, when `N_bp > 0`.
- When `N_bp = 0`, preserve raw counts at zero and document the undefined fractions using `NaN`/nullable values (not a fabricated physical ratio). Legacy code sometimes sets zero; record this difference in parity tests.
- The fraction of candidate pairs discarded by the mutual-partner filter is an optional **quality-control diagnostic**, not a required reporting metric.

**Important:** The simulation's density-dependent *energetic* valency constraint and the analysis's discrete partner-matching rule are related but not identical; describe them separately.

### 2.2 Molecular contact graph (Methods `5.4, pp. 21–22)

For each analyzed frame construct one **undirected weighted** graph `G=(V,E)`, one node for every molecule, including isolated molecules. For distinct molecules `i,j` let `w_ij` be the number of accepted particle contacts connecting them in the frame. Create a qualifying graph edge iff `w_ij >= n_min`, where the thesis uses `n_min = 2`.

Core observables:
- `N_edges = |E|`: number of qualifying edges, **not** number of base-pair contacts.
- `mean_edge_weight = (sum_(i,j) in E w_ij) / N_edges`; undefined/nullable if no qualifying edges.
- `mean_degree = 2*N_edges / N_molecules`; isolated molecules contribute degree 0.
- The qualifying edge list stores `molecule_i`, `molecule_j`, `contact_count`, and analyzed frame identifier. Keep the underlying particle-pair data to support recurrence and other investigations.

**Not initially required as primary thesis metrics:** edge-density normalization, degree quantiles, max edge weight, q90 contact strength, clustering coefficient, graph centrality. These may be opt-in exploratory metrics and must not inflate the standard report.

### 2.3 Connected components and cluster size (Methods `5.4, pp. 22–23)

A cluster is a **connected component of the qualifying graph**, including singleton components. For component `C`:
- `cluster_size_molecules = s_c = number of molecules in C`.
- `cluster_size_particles = sum_(m in C) chain_length[m]`.
- `cluster_particle_fraction = p_c = cluster_size_particles / N_particles`.
- Homogeneous reduction: `p_c = s_c * l_RNA / N_particles`.
- `N_particles = sum_m chain_length[m]` for *all* molecules, not just clustered molecules.

**Two separate binning schemes from thesis Table 5:**

| Scheme | Classifying quantity | Seven mutually exclusive bins |
|---|---|---|
| `molecule_count` | `s_c` | `1` (monomer), `2` (dimer), `3` (trimer), `4–9`, `10–19`, `20–49`, `>=50` |
| `particle_percentage` | `p_c` | `monomer` where `s_c=1`; for multimers only: `<2%`, `2–<3%`, `3–<10%`, `10–<20%`, `20–<50%`, `>=50%` |

For either scheme, the per-frame histogram **bar height** for bin `B` is:

`fraction_particles_in_bin(B) = sum_(C assigned to B) cluster_size_particles(C) / N_particles`.

Each particle belongs to exactly one component and one bin, so the heights sum to 1. In homogeneous RQ1 this also reproduces the fraction of molecules in each size bin shown in thesis Fig. 9; for RQ2/RQ3 particle weighting is indispensable.

Monomers *always* belong to the separate monomer bin under `particle_percentage`, even if a single very long RNA represents at least 2% of total particles. Do not classify monomers by `p_c` thresholds.

A complete, unbinned connected-component table can be stored as optional analysis detail. The histogram/binning representation is a downstream *view*, not the only saved scientific information.

### 2.4 Ranked cluster decomposition (Methods `5.4, p. 23)

Each frame:
1. Rank **multimeric** connected components by `cluster_size_particles` in descending order. A deterministic tie-breaker can be used for reproducibility but must not be interpreted as identity.
2. Report **largest**, **second**, **third**, **other multimers**, and **monomers**; absent ranks have zero particle contribution.
3. Each category's contribution is `N_particles_in_category / N_particles`.
4. For species-resolved mixture summaries, decompose every category by its RNA species' particle count; these contributions sum to the category's total.
5. Ranking is recomputed each frame; ranks **do not track persistent cluster identity**.

The number of molecules in a ranked cluster may be reported as an additional annotation; it is **not** the denominator of a percentage metric.

### 2.5 Mixture contacts and connectivity (Methods `5.4, p. 23; Results `6.4)

Support arbitrary species labels/molecule lengths at the core level, with a CAG10/CAG47 example configuration:
- `N_particles_in_cluster_by_species` and `N_molecules_in_cluster_by_species`.
- Per ranked-cluster category: `fraction_total_particles_by_species`.
- Within the **largest connected component**, split accepted *intermolecular particle contacts* by molecule-type pair: CAG10–CAG10, CAG10–CAG47, CAG47–CAG47. Count contacts, and divide by **total largest-cluster intermolecular contacts** to obtain contact-type composition. This contact-type percentage is explicitly **not** a cluster-size percentage; label its denominator.
- For largest-cluster connectivity: `N_edges_largest` and `mean_degree_largest = 2*N_edges_largest / N_molecules_largest`. `largest_cluster_particle_fraction` uses `N_particles` for the whole simulation.
- Advanced site-normalized binding saturation and per-component details from `mix/run_cluster_resolved_mix.py` are **optional**, pending a scientific use case. They are not necessary to reproduce thesis figures 16–18.

### 2.6 Contact recurrence (Methods `5.4, p. 24)

For a rolling window containing `w` sampled frames, define a contact identity as either:
- **Particle recurrence:** unordered accepted nucleotide-particle pair (`i,j`).
- **Molecule recurrence:** unordered **qualifying** molecule-graph edge (`m,n`).

Let `N_unique` be the number of distinct identities seen across the window and `N_c(t)` the number of identities present in each sampled frame.

`recurrence = 1 - N_unique / (sum_(t in window) N_c(t))`.

This is **recurrence**, not a continuous lifetime/survival estimate. Its maximum for a contact repeated in every frame is `1-1/w`, not exactly 1. Record window size and sampling stride; handle windows with no contact observations as undefined/null rather than silently implying a physical score.

**Thesis uses `w = 20` saved frames**, even though some legacy scripts use a different default. RQ1 reports molecule-level recurrence; RQ2 reports particle-level recurrence. Include both as available, but only calculate/emit each when requested.

### 2.7 Single-chain validation (Methods `5.4, p. 21; Results `6.1)

- `radius_of_gyration`: RMS particle displacement from the chain center of mass; properly unwrap periodic coordinates before geometric analysis. Report in simulation units or convert to nm explicitly using the thesis's physical length mapping. Do not quietly mix unit systems.
- `MFE_pairs_matched`: detected base pairs also present in the ViennaRNA MFE reference.
- `extra_pairs`: detected base pairs not present in the reference.
- `MFE_precision = MFE_pairs_matched / (MFE_pairs_matched + extra_pairs)`.
- `MFE_recall = MFE_pairs_matched / number_of_MFE_reference_pairs`.
- Empty-pair/reference edge cases require explicit policy and tests.
- MFE reference sequence, indexing convention, and ViennaRNA reference output are versioned/provenance inputs.
- `end_to_end_distance`, F1, and base-pair distance occur in legacy implementations but are **optional additional metrics**; thesis figures 5–8 require primarily MFE matching/extra pairs, precision/recall, and Rg.

Persistence-length calibration (`Methods `5.1, pp. 12–13`) is a separate model-validation figure/workflow. It calculates mean bond-vector orientation correlations against bond separation and their exponential decay; do not conflate this with the Rg hairpin analysis.

### 2.8 Energy and physical units

Logged simulation energies (total potential, WC/SQD, bond, angle) and kinetic energy per particle are **equilibration diagnostics** (Methods Table 4, p. 20), not default columns in every scientific output. Preserve access as optional framewise records.

Reduced particle density `rho = N_particles / V`, physical length mapping `b_0 ~= 0.6 nm`, and molar strand concentration are contextual metadata; conversions must state units and physical assumptions rather than become universal derived columns.

### 2.9 Statistics and averaging

**Select saved frames belonging to each experiment's thesis late-time window; average over frames within each replicate; then aggregate across replicate-level summaries.** Methodology used for thesis figures:

| Experiment | Late-time fraction | Replicate counts in thesis | Primary figures |
|---|---:|---|---|
| Single chain | Final **20%** | 5 per condition | 5–8 |
| RQ1 — reentrant CAG47 | Final **10%** | 10 per condition | 9–12 |
| RQ2 — repeat length | Final **10%** | 5 per periodic-sequence condition | 13–15 |
| RQ2 — scrambled controls | Final **10%** | 10 per scrambled condition | 15 |
| RQ3 — mixture | Final **5%** | 10 per mixed condition | 16–18 |

The pure CAG10/CAG47 reference conditions shown alongside mixtures originate from RQ2; **do not silently apply the RQ3 5% window to references that were originally summarized using the RQ2 10% window**. Expose the actual reference aggregation and provenance to the figure.

Specific statistics:
- Fig. 8: Rg replicate means with 10th–90th percentile range across replicates.
- Fig. 10: mean and standard deviation across replicate-level late-window contact means.
- Fig. 11: boxplots of replicate means, with points.
- Fig. 12: mean and SD across replicate-level molecule recurrence.
- Fig. 15: repeat vs scrambled **two-sided permutation tests** on replicate-level late-time mean edge weight, with **Holm adjustment** across 12 tests (4 lengths × 3 attraction strengths).
- Every plotted uncertainty and statistical test must document **which unit is being aggregated** (frames, replicates, or conditions).

## 3. Proposed minimal output tables

These are *new output contracts*, not claims about the schema of existing CSV files. Keep storage format pluggable (CSV for small tables; compressed columnar files when framewise data are large). Use stable IDs: `sweep_id`, `run_id`, `replicate_id`, `frame`, and `step` as applicable; store parameter values/provenance once per run where possible.

| Output | Granularity | Minimal fields | Required for |
|---|---|---|---|
| `run_metadata` | Run/replicate | IDs, sequence/species composition, `N_particles`, `N_molecules`, `E_att`, `angle_k_ppp`, `rho`, frame stride, pairing/edge thresholds, late-window settings | Every workflow |
| `pairing` / `molecule_edges` | Analyzed frame (compact arrays) | Accepted pair indices/distances; particle→molecule lookup; weighted molecule edges | Core computations and regression |
| `frame_contacts` | Frame × replicate | Frame/step, `N_bp`, `N_inter`, `N_intra` (optional energy, recurrence) | RQ1, diagnostics |
| `frame_graph` | Frame × replicate | Frame/step, `N_edges`, `mean_edge_weight`, `mean_degree` | RQ1/RQ2/RQ3 |
| `components` | Component × frame × replicate | Component ID within frame, molecule count, particle count, particle fraction, optional species composition / member IDs | Basis for both distributions, ranks, networks |
| `cluster_bins` | Bin × scheme × frame × replicate **or** late-window replicate summary | `binning_scheme`, `bin_label`, `fraction_particles`; IDs and time/window | RQ1, RQ2 |
| `ranked_clusters` | Rank × frame/replicate | Rank category, `fraction_total_particles`, species contributions and optional molecule count | RQ2 control comparisons, RQ3 |
| `replicate_metrics` | Replicate × condition | Only selected late-window means (contacts, graph/recurrence, Rg, MFE, largest-cluster info) | Plotting and tests |
| `condition_summary` | Condition × selected metric | Mean across replicate values, SD or requested interval, `n_replicates` | Presentation, plots |
| `statistical_tests` | Contrast × metric | Conditions, test, statistic/difference, raw and adjusted p, adjustment family | RQ2 repeat vs scrambled |

**One metric belongs in one authoritative representation.** The default `replicate_metrics` table should not mechanically generate `last`, `mean`, `std`, `min`, `max`, `late_mean`, and `late_std` for every observable. If a plot needs a different statistic, compute it explicitly from the relevant framewise table.

Avoid storing percentage-bin checks `pct_sum` and `mol_sum` as recurring research columns: enforce them as validation invariants in tests and optionally in a QC report.

### Suggested compact per-replicate report profiles

- `reentrance`: selected late-window `N_inter`, `N_intra`, `N_edges`, `mean_edge_weight`, `molecule_recurrence` plus the long-format cluster-bin distribution. `mean_degree` is a derived view of `N_edges` and `N_molecules`, so export only when plotted.
- `length_patterning`: particle-weighted bin distribution, ranked particle contributions, `particle_recurrence`, `mean_edge_weight`, sequence class/length and permutation-test inputs.
- `mixture`: ranked-cluster particle contributions by RNA type; largest-component particle fraction, molecular membership, qualifying edges, mean degree; inter-contact composition by type. May reference `reentrance`/`length_patterning` run IDs for pure controls.
- `single_chain`: late-window `Rg`, MFE-matched/extra pairs, precision and recall; separate reference metadata.
- `equilibration`: optional energies, kinetic traces and structural diagnostics without imposing them on all profiles.

## 4. Provenance and validation requirements

Record in a machine-readable analysis manifest:
- Input trajectory path or stable ID and checksum where practical; simulation parameters; chain topology/species assignment and sequence sources.
- Pair criteria (`r_max`, `r_opt`, intrachain exclusion, valency rule); `n_min`; analyzed frame indices/stride.
- The late-window fraction and **exact selected frames/step range**; recurrence window and zero-denominator handling.
- Program commit, package/dependency versions (GSD, Freud, NumPy, pandas, NetworkX, PyVis if used), run timestamp.
- Any data normalization, physical unit conversion, and figure configuration.

Tests must cover: zero contacts/empty edge sets; singleton graph; disconnected clusters; exactly-on-bin-edge fractions; relative bin monomer override; mixed-length component weights; all particles in one component; ranking ties; particle-count conservation; repeated vs changing contact identities; and late-frame selection with stride.

Use uploaded `molecule_lookup.npy`, pairing NPZ, molecule-contact-edge NPZ, and selected CSV rows as *reference material* for future regression tests, but do not commit large binary/full-sweep files or claim numerical parity before these datasets have been checked.

## 5. Existing implementation mapping (entry points, not endorsement)

| Mathematical domain | Source functions/files to inspect first |
|---|---|
| Particle pairs and mutual partner filter | `analysis_scripts/phase_separation.py`: `get_filtered_nlist_base_pairs`, `compute_filtered_nlist_base_pairs_traj`; `ps_all/run_compute_ps.py` |
| Molecule edge list | `build_molecule_contact_edge_dict_from_pairing_npz`; `ps_all/run_mol_graph.py` |
| Cluster components and bins | `component_size_distribution_rows_from_edge_npz`, `ps_all/run_analysis_clusters.py`, `ps_all/run_analysis_binned.py` |
| Graph measures | `graph_framewise_summary_rows_from_edge_npz`, `ps_all/run_graph_framewise_summary.py` |
| Single-chain and MFE | `calc_rg_from_positions`, `calc_framewise_mfe_metrics`, `single_mol/` |
| Recurrence | `recurrence_from_contact_sets`, `framewise_molecule_recurrence_rows_from_nlist_dict` |
| Mixture material and contacts | `mix/run_cluster_resolved_mix.py`, `mix/run_time_df_ps_mix.py` |
| Statistical aggregation | `add_metric_summary_stats`, `ps_all/run_analysis_ps_summary.py`, `mix/run_analysis_ps_mix.py` |
| Interactive graph network | Uploaded `8_directed_graph_2.ipynb`; not yet in repository |

## 6. Open questions before implementation

- Which **additional beyond-thesis** observables will be needed for exploratory plots? Add them only when requested; treat q90/edge-density/etc. as optional extras until then.
- Define consistent handling for undefined fractions/means, especially empty-graph frames; document legacy differences rather than silently changing old results.
- Confirm the exact thesis run/source data paths and input availability for representative numerical regression fixtures.
- Identify and ingest the authoritative notebook sources for all desired figures (including newer visualizations not yet in Git).
