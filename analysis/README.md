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
| analyze | Existing NPZs to per-replicate CSV summaries |
| all | Run the stages in order |

Existing NPZs are reused (output location first, input location second). New
NPZs are written to the output tree only. --overwrite-npz regenerates output
NPZs; it never overwrites the original simulation data.

--run-ids 0 selects the parameter-run directory 0_* and analyzes *every*
trajectory_*.gsd replicate inside it. --save-framewise saves one time-series CSV
per replicate; it is not required for summary tables. When --run-ids is set,
summary CSVs are written inside each selected parameter-run folder instead of
replacing the full-sweep files. Every aggregated CSV records the late_fraction
and, where relevant, the minimum graph-contact threshold.

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
    │   └── analysis/             # when selecting this run with --run-ids
    │       ├── replicate_summary.csv
    │       └── cluster_distribution.csv
    └── analysis/                 # when analyzing the entire sweep
        ├── replicate_summary.csv
        ├── cluster_distribution.csv
        └── ranked_clusters.csv   # length/patterning or mixture
~~~

The replicate summary has one row per independent trajectory. Cluster fractions
are particle-weighted. Late-window averages are calculated separately per
replicate, rather than pooling trajectories across a parameter combination.
