
# Analysis metrics

Simulation trajectories are analyzed using particle-level contacts, graph-based clustering metrics, and structural measurements. Detailed definitions and methodology are available in the [MSc thesis](assets/thesis.pdf), particularly **Methods §5.4 (pp. 20–24)**.

## 1. Particle contacts and base pairing

Base-pair contacts are identified between complementary nucleotides (A–U and G–C) within a distance cutoff of **0.7 simulation units**. Intramolecular contacts between nucleotides separated by fewer than four sequence positions are excluded. Approximate one-to-one pairing is enforced by retaining mutually preferred partners based on proximity to the interaction potential minimum ($r_e = 0.50$).

Contacts are classified as:

- **Intermolecular contacts:** Base pairs between different RNA molecules.
- **Intramolecular contacts:** Base pairs within the same RNA molecule.
- **Contact fractions:** Proportion of accepted base pairs belonging to each category.

The analysis-level contact assignment is distinct from the energetic valency constraint used during simulations.

## 2. Molecular connectivity and clustering

RNA interactions are represented as an **undirected, weighted graph**, where each RNA molecule is a node and edge weights correspond to the number of intermolecular base-pair contacts.

An edge is established when two molecules share at least **two accepted contacts**.

The main connectivity metrics are:

- **Number of edges:** Total qualifying intermolecular connections.
- **Mean degree:** Average number of connected neighboring molecules, including isolated molecules.
- **Mean edge weight:** Average number of base-pair contacts per qualifying edge.

A **cluster** is defined as a connected component of this graph, with isolated molecules forming singleton components (monomers).

## 3. Cluster size distributions

Cluster sizes are quantified by both the number of RNA molecules and the total number of nucleotide particles within each component.

Two classification schemes are used:

| Classification | Cluster size bins |
|---|---|
| Molecule count | 1, 2, 3, 4–9, 10–19, 20–49, ≥50 |
| Particle fraction | Monomers, <2%, 2–<3%, 3–<10%, 10–<20%, 20–<50%, ≥50% |

For each bin, the reported value is the **fraction of all simulation particles** belonging to clusters in that category. This particle-weighted representation enables comparisons between systems containing RNA molecules of different lengths.

## 4. Largest clusters and RNA mixtures

Multimeric components are ranked by their total particle count in each analyzed frame.

The particle population is decomposed into:

- Largest, second-largest, and third-largest clusters.
- Remaining multimeric clusters.
- Monomers.

Each category is expressed as a fraction of the total particle population. Cluster rankings are recalculated independently for each frame.

For mixed RNA systems, the analysis additionally quantifies **species composition**, **intermolecular contact composition**, and **connectivity within the largest cluster**.

For example, in CAG10/CAG47 mixtures, contacts are classified as CAG10–CAG10, CAG10–CAG47, or CAG47–CAG47. Their relative contributions are normalized by the total number of intermolecular contacts within the largest cluster.

## 5. Contact recurrence

Contact recurrence measures how frequently the same particle pairs or molecular connections reappear across a rolling window of saved simulation frames.

It is calculated as:

$$
R = 1 - \frac{N_{\mathrm{unique}}}{\sum_{t=1}^{w} N_c(t)}
$$

where $N_{\mathrm{unique}}$ is the number of distinct contacts observed over the window, $N_c(t)$ is the number of contacts in frame $t$, and $w$ is the window size (**20 saved frames**).

Recurrence can be calculated at either the particle-contact or molecular-edge level. It characterizes repeated contact formation rather than continuous contact lifetimes.

## 6. Single-chain structural validation

The RNA model is validated using structural measurements and comparisons with **ViennaRNA** minimum-free-energy (MFE) secondary structures.

The primary metrics include:

- **Radius of gyration ($R_g$):** Measures the spatial extent of an RNA chain around its center of mass.
- **Matched MFE base pairs:** Detected contacts that agree with the ViennaRNA reference structure.
- **Extra base pairs:** Detected contacts absent from the MFE reference.
- **Precision:** Fraction of detected base pairs matching the MFE reference.
- **Recall:** Fraction of MFE reference base pairs recovered in the simulation.

Backbone stiffness is additionally calibrated through **persistence-length analysis**, using bond-vector orientation correlations as a function of sequence separation (Methods §5.1, pp. 12–13).

These metrics provide complementary descriptions of RNA structure, intermolecular connectivity, cluster organization, and interaction dynamics across simulation conditions.
