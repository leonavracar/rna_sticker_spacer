TODO:
- document the simulation pipeline separately, maybe in `validation/simulation/docs/simulation_pipeline.md`. This file is only the analysis-side overview.
- add a "metadata" dictionary that is added to every csv
- MIXTURES (which extra columns)
- SINGLE MOL (is kinda separate ig except Rg myb we want for everything)
- Change name distinct/qualifying in fdf to n_mol_pairs_raw, n_mol_pairs_qualifying
- TODO: dynamics (Rg for mols in multi-mol systems, com movement etc.)
- TODO: cluster binning thing for fracions and mol-counts
- Logging instead of print
- Write formulas in documentation and all
- Write tests
# Analysis Pipeline Design

Main pipeline:

`GSD trajectory → pairing NPZ → molecule graph NPZ → cdf → fdf → adf`

Base input location:

`validation/bulk/output_dir/<sweep_dir>/`

Per-run location:

`validation/bulk/output_dir/<sweep_dir>/<run_folder>/`

Per-run computed metrics:

`validation/bulk/output_dir/<sweep_dir>/<run_folder>/computed_metrics/`

Combined sweep-level outputs:

`validation/bulk/output_dir/<sweep_dir>/analysis_combined/`

---

## Level 0: raw trajectory / GSD

Input: `.gsd` trajectory  
Output: none  
Purpose: raw simulation output used as input for the analysis pipeline.  


---

# Raw data

## Level 1: pairing NPZ

Input: trajectory .gsd file
Output: particle-particle contact arrays  
Purpose: identify valid base-base pairs and map molecule id to particle index

Structure:

- `frame_indices`: `np.ndarray`, trajectory frame indices included in the analysis
- `qi_<frame_idx>`: particle indices for one side of each detected contact
- `pi_<frame_idx>`: particle indices for the other side of each detected contact
- `distances_<frame_idx>`: contact distance for each detected particle pair
- `error_rates`: `np.ndarray`, per-frame ambiguity/error rate after valency filtering
- `r_opt`: float, optimal contact distance used for valency filtering

Additional file:

- `molecule_lookup.npy`: maps `particle_index → molecule_id`

Config:

- `r_min`
- `r_max`
- `r_opt`
- `exclude_chain`
- `stride`

Saved in: `<sweep_dir>/<run_dir>/computed_metrics/pairing/trajectory_<replicate>_pairing.npz`

Molecule lookup saved in:

`<sweep_dir>/<run_dir>/computed_metrics/pairing/molecule_lookup.npy`

---

## Level 2: molecule graph NPZ

Input: pairing NPZ + `molecule_lookup.npy`
Output: molecule-molecule contact graph  
Purpose: count how many particle-level contacts occur between each molecule pair in each frame.

Structure:

- `frame`: frame index for each molecule-molecule edge row
- `mol_a`: first molecule ID
- `mol_b`: second molecule ID
- `n_contacts`: number of particle-level contacts between `mol_a` and `mol_b` in that frame
- `frame_indices`: array of all frame indices included in the analysis
- `n_molecules`: total number of molecules in the system

Row meaning: one row = one molecule pair in one frame with at least one particle-level contact.

Config: none for the raw molecule graph (`min_contacts` is applied later when defining cluster edges)

Saved in:`<sweep_dir>/<run_folder>/computed_metrics/mol_graph/trajectory_<replicate>_molecule_contact_edges.npz`

Example:

`validation/bulk/output_dir/51a_repeat_number/0_molecule_file_cag10x470.json_E_att_2_density_0.005_angle_k_ppp_2.5/computed_metrics/mol_graph/trajectory_0_molecule_contact_edges.npz`

---

# Analysis CSVs

## Level 3: cdf — cluster-size distribution dataframe

Input: molecule graph NPZ  
Output: cluster-size distribution CSV  
Purpose: describe the distribution of connected-component sizes per frame after applying a molecule-pair contact threshold.

Row meaning:

One row = all connected components of one specific `cluster_size` in one frame, for one run/trajectory/min_contacts setting.

Example:

If frame 10 has 450 monomers and 10 dimers, then cdf has:

- one row with `cluster_size = 1`, `n_clusters = 450`
- one row with `cluster_size = 2`, `n_clusters = 10`

Config:

- `min_contacts`: minimum number of particle-level contacts required for a molecule pair to count as an edge in the molecule contact graph

Saved in combined sweep output:`<sweep_dir>/analysis_combined/analysis_clusters.csv`

Current columns:
### Run data
- `run_id`
- `folder_name`
- `traj_replicate`
- `frame`
- `step`
- `E_att`
- `angle_k_ppp`
- `density`
- `molecule_file`
### Metadata
- `min_contacts`
### Actual per-frame calculations
- `cluster_size`
- `n_clusters`
- `n_mols_in_clusters`
- `frac_mols`

New columns to add:

### Metadata
- `mol_len`
- `n_molecules_total`
- `n_particles_total`

### Actual per-frame calculations
- `one_cluster_mol_frac`
- `one_cluster_particle_count`
- `one_cluster_particle_frac`
- `n_particles_in_clusters`
- `frac_particles`
- `mass_weighted_cluster_score_contrib`
- `mass_weighted_size_contrib_mol`
- `mass_weighted_size_contrib_particle`

Column naming notes:

- `one_cluster_*` = value for one cluster of this size
- `*_in_clusters` = total over all clusters represented by this row
- `frac_*` = fraction represented by all clusters in this row
- `*_contrib` = row contribution to a frame-level summary metric

Important formulas:

- `n_mols_in_clusters = cluster_size * n_clusters`
- `frac_mols = n_mols_in_clusters / n_molecules_total`
- `one_cluster_mol_frac = cluster_size / n_molecules_total`
- `one_cluster_particle_count = cluster_size * mol_len`
- `one_cluster_particle_frac = one_cluster_particle_count / n_particles_total`
- `n_particles_in_clusters = n_mols_in_clusters * mol_len`
- `frac_particles = n_particles_in_clusters / n_particles_total`
- `mass_weighted_cluster_score_contrib = n_clusters * one_cluster_particle_frac^2`
- `mass_weighted_size_contrib_mol = frac_mols * cluster_size`
- `mass_weighted_size_contrib_particle = frac_particles * one_cluster_particle_count`

Notes:

- For homogeneous systems, `frac_mols` and `frac_particles` are numerically equal.
- For mixed systems, `frac_particles` is the more meaningful material/mass fraction.
- This table is a distribution table, not a component-level table. It does not know which exact molecules belong to which exact component.

---

## Level 4: fdf — framewise dataframe

Input: trajectory .gsd files for energies/KE + pairing npz + cdf
Output: one row per frame  
Purpose: framewise summaries, time-series plotting, and later replicate-level aggregation.

Row meaning: One row = one analyzed frame from one trajectory replicate.

Saved per run/trajectory: `<sweep_dir>/<run_folder>/framewise/trajectory_<replicate>_pairing_framewise_ps.csv`

Saved combined: `validation/bulk/output_dir/<sweep_dir>/analysis_combined/analysis_framewise.csv`

Current columns:

- `frame`
- `mono_frac`
- `oligo_frac`
- `drop_frac`
- `largest_cluster`
- `n_clusters`
- `particle_contacts`
- `distinct_molecule_pairs`
- `qualifying_molecule_pairs`
- `inter_frac`
- `intra_frac`
- `molecule_contact_recurrence_window`
- `E_pot`
- `E_sqd`
- `E_bond`
- `E_angle`
- `step`
- `time`
- `ke_per_particle`
- `run_id`
- `folder_name`
- `traj_replicate`
- `molecule_file`
- `dt`
- `E_att`
- `angle_k_ppp`
- `density`

Columns to add/update:

### Metadata
- `min_contacts`
- `mol_len`
- `n_molecules_total`
- `n_particles_total`
### Actual calculated things
- `largest_cluster_mol_frac`
- `largest_cluster_particle_frac`
- `mass_weighted_cluster_score`
- `mass_weighted_cluster_score_min`
- `mass_weighted_cluster_score_norm`
- `effective_n_clusters`
- `mass_weighted_mean_cluster_size_mol`
- `mass_weighted_mean_cluster_size_particle`
- `frac_particles_in_monomers`
- `frac_particles_in_multimol_clusters`
- `pairing_error_rate`


Column groups:

Metadata:

- `run_id`
- `folder_name`
- `traj_replicate`
- `molecule_file`
- `dt`
- `E_att`
- `angle_k_ppp`
- `density`
- `min_contacts`
- `mol_len`
- `n_molecules_total`
- `n_particles_total`

Frame/time:

- `frame`
- `step`
- `time`

Energy/thermodynamics:

- `E_pot`
- `E_sqd`
- `E_bond`
- `E_angle`
- `ke_per_particle`

Contact counts:

- `particle_contacts`
- `distinct_molecule_pairs`
- `qualifying_molecule_pairs`
- `inter_frac`
- `intra_frac`
- `pairing_error_rate`

Old cluster metrics:

- `mono_frac`
- `oligo_frac`
- `drop_frac`
- `largest_cluster`
- `n_clusters`

New cluster metrics:

- `largest_cluster_mol_frac`
- `largest_cluster_particle_frac`
- `mass_weighted_cluster_score`
- `mass_weighted_cluster_score_min`
- `mass_weighted_cluster_score_norm`
- `effective_n_clusters`
- `mass_weighted_mean_cluster_size_mol`
- `mass_weighted_mean_cluster_size_particle`
- `frac_particles_in_monomers`
- `frac_particles_in_multimol_clusters`

Dynamic/contact recurrence metrics:

- `molecule_contact_recurrence_window`

Notes:

- `mono_frac`, `oligo_frac`, and `drop_frac` are legacy-style raw-size categories and are not ideal for comparing systems with different molecule numbers or molecule lengths.
- New interpretation should prefer particle/material-normalized metrics such as `largest_cluster_particle_frac`, `mass_weighted_cluster_score_norm`, and `frac_particles_in_multimol_clusters`.
- Exact column definitions for fdf should probably go into a separate data dictionary: `validation/analysis/docs/data_dictionary_fdf.md`.

---
## Level 5: bcdf - binned_cluster_distribution.csv

Depending on 


---
## Level 6: gfdf - graph_framewise_summary.csv

run_id
folder_name
traj_replicate
frame
step
E_att
angle_k_ppp
density
molecule_file

n_molecules_total
n_particles_total
min_contacts

edge_density_raw = n_molecule_pairs_raw / possible_molecule_pairs
edge_density_qualifying

mean_degree_raw
mean_degree_qualifying

edge_strength_mean_raw
edge_strength_q90_raw
edge_strength_mean_qualifying
edge_strength_q90_qualifying
edge_strength_max_raw
edge_strength_max_qualifying

total_particle_contacts_raw
total_particle_contacts_qualifying
particle_contacts_per_particle_raw
particle_contacts_per_particle_qualifying

---
## Level 7: adf — per-replicate analysis dataframe

Input: fdf  
Output: one row per replicate/run/condition  
Purpose: aggregate framewise metrics into replicate-level summaries such as mean, standard deviation, minimum, maximum, last value, and late-window statistics.

Saved combined:

`validation/bulk/output_dir/<sweep_dir>/analysis_combined/analysis_per_replicate.csv`

Expected structure:

- metadata columns identifying run/condition/replicate
- summary statistics for selected fdf metrics

Example summary statistics:

- `<metric>_last`
- `<metric>_mean`
- `<metric>_std`
- `<metric>_min`
- `<metric>_max`
- `<metric>_late_mean`
- `<metric>_late_std`

Notes:

- This file is downstream of fdf and can be regenerated quickly.
- Exact metric list can be decided later after fdf is stabilized.
- The most important first step is to clean and stabilize cdf and fdf.