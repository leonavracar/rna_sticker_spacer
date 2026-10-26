# 1P RNA Sticker-Spacer Model

A simplified one-particle-per-base RNA sticker-spacer model for HOOMD-blue molecular dynamics simulations. This model represents RNA molecules using individual particles for A, U, G, and C bases with Watson-Crick pairing interactions.

## Reproducible RNA analysis

The [analysis pipeline](analysis/README.md) calculates nucleotide pairing,
molecular contact graphs, cluster statistics and per-condition summaries for
single-chain validation, reentrance, length/patterning and mixed-RNA systems.
It stores trajectory checkpoints in SQLite, exports plotting-ready CSVs and
produces scientific figures through Jupyter notebooks.

- [Environment setup](analysis/ENVIRONMENT.md) — CPU analysis and tests,
  optional GSD processing, and how to export the research Conda environment.
- [Analysis notebooks](analysis/notebooks/README.md) — the four research
  questions and figures.
- [Automated tests](.github/workflows/analysis-tests.yml) — GitHub Actions
  runs the existing scientific, pipeline and plotting tests on Python 3.11.
  The CI suite requires no GPU or HOOMD-blue installation.

## Overview

This simulation framework implements a coarse-grained RNA model where:
- Each RNA base (A, U, G, C) is represented by a single particle
- Watson-Crick base pairing (A-U and G-C) is modeled with attractive interactions

## Model Description

### Base Types
- **A (Adenine)**: Type 0
- **U (Uracil)**: Type 1
- **G (Guanine)**: Type 2
- **C (Cytosine)**: Type 3

### Interactions
- **Watson-Crick Pairing**: 
  - A-U (types 0-1): Attractive square-density potential
  - G-C (types 2-3): Attractive square-density potential
- **Excluded Volume**: All particles have soft repulsive interactions
- **Connectivity**: Harmonic bonds between consecutive bases
- **Chain Stiffness**: Harmonic angle potentials


## File Structure

```
.
├── params.py           # Parameter definitions and V1/V2 mode switching
├── setup.py            # System initialization and molecule loading
├── simulation.py       # Simulation execution and output handling
├── run_sim.py          # Single simulation runner (command-line interface)
├── sweep.py            # SLURM array job parameter sweep script
└── README.md           # This file
```

### File Descriptions

- **`params.py`**: Defines all simulation parameters including potential strengths, LJ parameters, bond/angle force constants, and temperature. Includes `get_params_mode(mode)` function to switch between V1 and V2 parameter sets.

- **`setup.py`**: Handles system initialization from `molecules.json`, creates HOOMD snapshots, applies bonded interactions (harmonic bonds and angles), and sets up the simulation box.

- **`simulation.py`**: Core simulation engine that sets up pair potentials, integrators, and logging. Manages GSD trajectory output and metadata storage.

- **`run_sim.py`**: Command-line interface for running single simulations with customizable parameters, restart capabilities, and sweep compatibility.

- **`sweep.py`**: SLURM array job script for high-throughput parameter exploration. Automatically generates parameter combinations and submits batch jobs.

## Requirements

### Core Dependencies
- **HOOMD-blue** (v3.x or later): Molecular dynamics simulation engine
- **SQD Plugin**: Custom square-density potential plugin for HOOMD-blue
- **Python** 3.8+

### Python Packages
```bash
numpy
gsd
json
argparse
subprocess
```

## Installation

### 1. Install HOOMD-blue

In order to use the custom SQD plugin, HOOMD-blue has to be built from source.

```bash
# Build from source
git clone https://github.com/glotzerlab/hoomd-blue.git
cd hoomd-blue
mkdir build && cd build
cmake ../ -DENABLE_GPU=ON
make -j4
make install
```

### 2. Install SQD Plugin

```bash
# Clone and build the SQD plugin
git clone <sqd-plugin-repository>
cd sqd-plugin
mkdir build && cd build
cmake ../ -DHOOMD_ROOT=/path/to/hoomd
make -j4
make install
```

## Usage

### Single Simulation Runs

#### Basic Run
```bash
python run_sim.py --output_dir my_simulation --steps 1000000
```

#### With Custom Parameters
```bash
python run_sim.py \
    --output_dir my_sim \
    --steps 5000000 \
    --mode V2
```

### Parameter Sweeps with SLURM

#### 1. Configure Sweep Parameters

Add a parameters.txt file and molecules.json to your output directory <sweep_results> to define the parameter grid. Example for parameters.txt.

```
--E_att 2.0 --angle_k_ppp --2.5
```

#### 2. Submit Sweep

```bash
sbatch sweep.py --output output_dir/sweep_results --n_replicates 1
```


### V1 vs V2 Modes

Two parameter sets that control effective valency through the placement of density cutoffs relative to the attractive interaction range.

**V1 Mode**: Valency ≈ 1 (Specific Pairing)
- Upper density cutoff (1.25) extends beyond the attractive cutoff (r_att = 0.75)
- Lower density cutoff (0.75) sits at the attractive cutoff

Once a Watson-Crick pair forms (e.g., A-U), the density-dependent repulsion activates at 1.25 σ, preventing additional complementary particles from entering the attractive well. This enforces monovalent, one-to-one pairing between complementary bases.

**V2 Mode**: Valency > 1

Upper density cutoff (0.807) sits just above the attractive cutoff (r_att = 0.75)
Lower density cutoff (0.48) sits within the potential well

The narrow density window allows transient multivalent binding, a sticker can coordinate with two or more complementary partners before density-dependent repulsion activates. This enables bond switching and effective valency between 1 and 2.

## Input Files

### molecule_file .json Format

The `molecule_file` file defines the RNA molecules in the system:

```json
{
    {
      "sequence": ["A", "U", "C", "G"],
      "count": 100,
    },
}
```

