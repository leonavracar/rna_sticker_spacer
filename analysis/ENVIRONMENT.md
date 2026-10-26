# Analysis environment

The analysis pipeline is Python-based. **Python 3.11** is the primary
development version; CI also tests Python 3.12. HOOMD-blue and the custom
square-density plugin are needed to **run new simulations**, but not to
analyze cached NPZ files or reproduce the CSV figures.

## Setup

Run these commands from the repository root.

### Analysis, CSV plots and network HTMLs

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r analysis/requirements.txt
```

This environment includes NumPy, pandas, SciPy, Matplotlib, Seaborn,
NetworkX, PyYAML, statsmodels and PyVis. The standard analysis stage reads
cached NPZ files and `params.json`; it does not process GSD contents.

### Existing tests (same dependencies as GitHub Actions)

```bash
python -m pip install -r analysis/requirements-test.txt
python -m pytest analysis/tests -q
```

The tests use small synthetic inputs, including temporary NPZ and SQLite
files. They require neither production trajectory files nor HOOMD/GPU.

### Recompute pairings or make GSD projections

```bash
python -m pip install -r analysis/requirements-trajectory.txt
```

This adds **GSD** and **freud-analysis**, required for pairing calculations,
single-chain Rg extraction from trajectories and trajectory projections.
You can run the analysis code on CPU, including on a workstation, although
large GSD reprocessing is better done on your research computing resources.

### Notebooks

```bash
python -m pip install jupyterlab
jupyter lab analysis/notebooks/
```

Start Jupyter from the repository root. The notebooks plot compiled
`replicates.csv` and `summary.csv` files. Optional PyVis network
visualizations use cached molecular-graph NPZs and do not require GSD.

## Simulation environment

The HOOMD-blue simulation engine and custom SQD potential/plugin are separate
from this **analysis** environment. Their versions, compilation options and
GPU/CUDA dependencies should be documented and locked when the simulation
setup is packaged for a public release. Installing the analysis requirements
does **not** install the simulation engine.

## Dependency policy

`requirements.txt` declares compatible **version ranges**, not an exact
environment lock. `requirements-test.txt` adds pytest; the
`requirements-trajectory.txt` file adds GSD/freud.

Before publishing a final reproducible release, record the **exact tested
package versions** and, separately, the HPC simulation environment.
One simple snapshot, from the validated environment, is:

```bash
python -m pip freeze > analysis/requirements-locked.txt
```

Do this only **after** the tested environment is known to work; do not use
a machine-specific lock snapshot as a substitute for the readable direct
dependency lists.
