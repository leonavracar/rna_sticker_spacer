# RNA analysis pipeline

The analysis reads simulation runs from an input sweep without modifying them.
Generated NPZ files, CSV summaries and optional framewise outputs are saved under
an independent output root. Each parameter-run directory contains independent
replicate trajectories named trajectory_*.gsd.

## Run the analysis

From the repository root:

~~~bash
python analysis/run_pipeline.py \
  --sweep /path/to/simulation/bulk/output_dir/53a_single_mol \
  --output-root ./analysis_results \
  --experiment single_chain \
  --stage analyze \
  --run-ids 0 \
  --viennarna-reference duplex_mfe.csv
~~~

The --viennarna-reference path is relative to the input sweep unless absolute.
The default is set in configs/single_chain.yaml and must correspond to the
sequence simulated in the GSD.

Experiment profiles: single_chain, reentrance, length_patterning, mixture.
The YAML files contain contact settings and late-time averaging windows.

| Stage | Operation |
| --- | --- |
| pairing | GSD to accepted base-pair NPZ and molecule lookup |
| graphs | Pairing NPZ to weighted molecule-edge NPZ; intramolecular counts for mixtures |
| analyze | NPZs and params.json to per-replicate metrics and CSVs; no GSD reads |
| all | Run pairing, graphs, analyze in order (excludes optional energy) |
| energy | Optional GSD pass for logged energies, kinetic energy and single-chain Rg |

Existing NPZs are reused (output location first, input location second). New
NPZs are written to the output tree only. --overwrite-npz regenerates output
NPZs; it never overwrites the original simulation data.

--run-ids 0 selects the parameter-run directory 0_* and analyzes *every*
trajectory_*.gsd replicate inside it. --save-framewise saves one time-series CSV
per replicate; it is not required for summary tables. Each replicate's small
analysis CSVs are saved immediately under `<run>/analysis/replicates/trajectory_N/`,
and the framewise DataFrame is released before the next replicate. If a later
replicate fails, earlier per-replicate CSVs remain available. Once all selected
replicates finish, the small tables are combined: under each run folder when
--run-ids is set, or at the sweep level for a complete sweep. Every aggregated
CSV records late_fraction and, where relevant, the graph-contact threshold.

`--stage analyze` never opens GSDs, including for single-chain analysis.
The pairing NPZ supplies frame IDs. Pairing NPZs generated with this version
also store an exact `steps` array copied while first generating the pairings.
Older NPZs may not contain `steps`; in that case analyze omits the step column
rather than guessing the simulation's starting step. Existing NPZs need not be
regenerated for the other metrics.

Logged energy, kinetic energy and single-chain Rg need GSD snapshots. Extract
them only when needed by running `--stage energy`. This writes a separate
`<run>/energy/trajectory_N.csv`, aligned by NPZ frame indices. It never uses
old analysis CSVs and is excluded from `--stage all`.

For the mixture profile, the NPZ-only analysis also summarizes species-specific
monomer/oligomer/droplet fractions, recruitment into mixed components,
homotypic/heterotypic degrees, molecule-edge enrichment against the
molecule-count null model, and nucleotide contact saturation. Its YAML specifies
a droplet threshold of 5 molecules and 20/94 pairable sites per CAG10/CAG47 RNA.

## Output layout

~~~text
analysis_results/
└── 53a_single_mol/
    ├── 0_E_att_.../
    │   ├── computed_metrics/      # if NPZs were generated
    │   │   ├── pairing/
    │   │   └── mol_graph/
    │   ├── framewise/            # only with --save-framewise
    │   │   ├── trajectory_0.csv
    │   │   └── trajectory_1.csv
    │   ├── energy/               # only with --stage energy
    │   │   └── trajectory_0.csv
    │   └── analysis/
    │       ├── replicates/        # each replicate saved immediately
    │       │   ├── trajectory_0/
    │       │   │   └── replicate_summary.csv
    │       │   └── trajectory_1/
    │       │       └── replicate_summary.csv
    │       ├── replicate_summary.csv   # combined selected-run outputs
    │       └── cluster_distribution.csv
    └── analysis/                 # when analyzing the entire sweep
        ├── replicate_summary.csv
        ├── cluster_distribution.csv
        └── ranked_clusters.csv   # length/patterning or mixture
~~~

The replicate summary has one row per independent trajectory. Cluster fractions
are particle-weighted. Late-window averages are calculated separately per
replicate, rather than pooling trajectories across a parameter combination.

## Physical concentrations

The original molarity calculations are included in each per-replicate summary
and cluster-distribution table:

- `conc_mol_per_L`: RNA molecule concentration in mol/L.
- `conc_nuc_per_L`: nucleotide/bead concentration in mol/L.

The simulation `density` is bead number density. The box volume is
`n_particles / density` in simulation volume units, converted to liters
using `nm_per_sim_length_unit**3 * 1e-24`. Both concentrations use Avogadro's
constant and the actual number of molecules/particles from the saved lookup.
The `nm_per_sim_length_unit` value is configured for each experiment and is
also used to convert single-chain radius of gyration to nm during the
optional GSD energy/geometry pass.
