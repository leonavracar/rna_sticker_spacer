# Analysis environment

The analysis pipeline is Python-based. Automated tests run on **Python 3.11**.
The HPC simulation environment is maintained separately in Conda and may
include compiled packages and cluster-specific dependencies. HOOMD-blue and
the custom square-density plugin are needed to run new simulations, but not
to analyze cached NPZ files or reproduce the CSV figures.

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

## Sharing the research Conda environment

The actual HPC environment should be documented in a **reviewed**
`environment.yml` at the repository root. It is not yet committed because
its contents must come from the working cluster environment.

Activate the Conda environment you use for these simulations and analyses:

```bash
conda activate YOUR_ENV
conda env export --from-history > environment.yml
```

Review the file before committing it. `--from-history` captures packages
explicitly requested through Conda, not necessarily packages installed with
`pip`. Include required pip-installed packages under `pip:` in the final
YAML, and remove any machine-specific `prefix:` entry. If your Conda
version supports it, the modern equivalent is
`conda export --from-history --format=environment-yaml > environment.yml`.

To compare the portable export with everything currently installed, inspect:

```bash
conda list
python -m pip list
```

After reviewing and committing the file, users can create the environment
with:

```bash
conda env create -f environment.yml
conda activate YOUR_ENV
```

The environment name will be taken from the YAML file; replace `YOUR_ENV`
accordingly. A portable YAML is a list of dependencies, **not** a guarantee
that custom HPC builds or GPU libraries can be installed everywhere.

Record separately any cluster modules, CUDA/compiler toolchains, HOOMD-blue
build details and custom SQD plugin installation instructions. A full
`conda env export` may be kept as a private environment snapshot for exact
version comparison, but it often includes platform-specific packages and
absolute local paths, so do not commit it unreviewed.

GitHub Actions uses the small `requirements-test.txt` environment rather
than reproducing the full HPC setup. Both serve different purposes.

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

Before publishing a reproducible release, record the **exact tested
package versions** alongside the reviewed research Conda environment.
A pip-only snapshot, from the validated analysis environment, can be made with:

```bash
python -m pip freeze > analysis/requirements-locked.txt
```

Do this only **after** the tested environment is known to work; do not use
a machine-specific lock snapshot as a substitute for the readable direct
dependency lists.
