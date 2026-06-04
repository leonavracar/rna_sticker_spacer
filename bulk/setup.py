import math 
import gsd.hoomd
import numpy as np
import json
import argparse

# Counts how many times define_pairwise_sqd was called, used to print parameters only once
_sqd_init_count = 0

DEFAULT_NONSPECIFIC_NCUT = 12
#### SETUP BASE SEQUENCE ####


def parse_molecules(file="molecules.json"):
    with open(file) as f:
        d = json.load(f)

    mlc_list = []
    for mlc_type in d:
        for _ in range(mlc_type["count"]):
            mlc_list.append(list(mlc_type["sequence"]))

    return mlc_list

def invert_dict(d):
    inv_d = {}
    for k in d.keys():
        inv_d[d[k]] = k
        print(k)
    return inv_d

def reverse_complement(base_seq, par):
    """
    Given a list of bases, return its reverse complement.
    For example, ['A','T','G'] -> ['C','A','T'].
    """
    return [par.complements_original[b] for b in reversed(base_seq)]

def insert_backbone_duplicate_types(sequence,par):
    """
    Given a list of bases, return a new list where
    a backbone 'B' is placed before each base.
    Example: ['A','T'] -> ['B','A','B','T'].
    """
    seq_with_backbone = []
    for b in sequence:
        seq_with_backbone.append("B")
        seq_with_backbone += par.duplicates[b]
    return seq_with_backbone

def insert_backbone(sequence):
    """
    Given a list of bases, return a new list where
    a backbone 'B' is placed before each base.
    Example: ['A','T'] -> ['B','A', 'B', 'T'].
    """
    seq_with_backbone = []
    for b in sequence:
        seq_with_backbone.append("B")
        seq_with_backbone.append(b)
    return seq_with_backbone

def define_pairwise_sqd_v2(sqd, par):
    """SQD v2 = higher/lower distance cutoff for r_d, overlapping with attraction radius partially.
    Interaction setup (intended):
      - Base-base:
          * WC directional (complements): A1-U1, A2-U2 ... (AGCU all have same attraction E_att)
          * non-WC directional (non_wc_complements): directional A1-G1/C1/A1
          * WC nondirectional (wc_nondirectional): A1-U2, A2-U1...B=0, repulsion (A) can be tuned
          * ALL other nondirectional base-base (repulsion-only, base cutoffs, high ncut so it doesn't compete with pairing
      - Backbone-base:
          repulsion-only, standard cutoffs, high ncut
      - Backbone-backbone:
          weak attraction and repulsion
    """
    print("define_pairwise_sqd_v2 called")
    #global _sqd_init_count
    #_sqd_init_count += 1
    #verbose = (_sqd_init_count == 1)

    # Get pairs from dictionaries
    complements = par.complements  # WC directional pairs A1-U1...
    non_wc = par.non_wc_complements  # nonwc directional pairs A1-A1, A1-C1...
    # all_nondir = par.all_nondirectional # all remaining nondirectional interactions A1-A2 (nonwc), A1-U2 (wc)...

    for i, t1 in enumerate(par.type_names):
        for t2 in par.type_names[i:]:

            t1_is_base = (t1 in complements)
            t2_is_base = (t2 in complements)

            # -----------------------
            # Base–Base interactions
            # -----------------------
            if t1_is_base and t2_is_base:

                # 1) WC directional
                if complements.get(t1) == t2:
                    sqd.params[(t1, t2)] = dict(
                        A=par.E_rep_wc_scaled,          # scaled with beta
                        B=par.E_att_comp,               # negative scaled
                        attr_rcut=par.r_att_base,
                        rep_rcut=par.r_rep_base,
                        density_upper_rcut=par.density_upper_rcut_base,  # 0.807
                        density_lower_rcut=par.density_lower_rcut_base,  # 0.48
                        ncut_ij=par.ncut_ij_comp,  
                        ncut_ji=par.ncut_ij_comp, 
                    )
                    print(
                            f"[WC_dir] {t1}-{t2}: A={par.E_rep_wc_scaled}, B={par.E_att_comp}, ncut={par.ncut_ij_comp}")

                # 2) non-WC directional
                elif (t2 in non_wc.get(t1, [])) or (t1 in non_wc.get(t2, [])):
                    sqd.params[(t1, t2)] = dict(
                        A=par.E_rep_wc_scaled,
                        B=par.E_att_nonwc,
                        attr_rcut=par.r_att_base,
                        rep_rcut=par.r_rep_base,
                        density_upper_rcut=par.density_upper_rcut_base,
                        density_lower_rcut=par.density_lower_rcut_base,
                        ncut_ij=par.ncut_ij_nonwc,  # 1
                        ncut_ji=par.ncut_ij_nonwc,  # 1
                    )
                    print(
                            f"[nonWC_dir] {t1}-{t2}: A={par.E_rep_wc_scaled}, B={par.E_att_nonwc}, ncut={par.ncut_ij_nonwc}")

                # WC non-directional interaction (A1-U2, U1-A2, etc.)
                elif (
                    (t1 in par.wc_nondirectional and par.wc_nondirectional[t1] == t2) or
                    (t2 in par.wc_nondirectional and par.wc_nondirectional[t2] == t1)
                ):
                    sqd.params[(t1, t2)] = dict(
                        A=par.E_rep_wc_nondir_scaled,  # <- different rep hrere
                        B=0.0,
                        attr_rcut=par.r_att_base,
                        rep_rcut=par.r_rep_base,
                        density_upper_rcut=par.density_upper_rcut_base,
                        density_lower_rcut=par.density_lower_rcut_base,
                        ncut_ij=par.ncut_ij_wc_nondir,
                        ncut_ji=par.ncut_ij_wc_nondir,
                    )
                    print(
                            f"[wc_nondir] {t1}-{t2}: A={par.E_rep_wc_nondir_scaled}, B=0, ncut={par.ncut_ij_wc_nondir}")
                        # 3) default base-base (repulsion-only, base cutoffs)
                else:
                    sqd.params[(t1, t2)] = dict(
                        A=par.E_rep_wc_scaled,
                        B=0.0,
                        attr_rcut=par.r_att_base,
                        rep_rcut=par.r_rep_base,
                        density_upper_rcut=par.density_upper_rcut_base,
                        density_lower_rcut=par.density_lower_rcut_base,
                        ncut_ij=12,
                        ncut_ji=12,
                    )
                    print(
                            f"[base_default] {t1}-{t2}: A={par.E_rep_wc_scaled}, B=0, ncut=12")

            # -----------------------
            # Backbone–Backbone
            # -----------------------
            elif t1 == t2:
                sqd.params[(t1, t2)] = dict(
                    A=par.E_rep_scaled,
                    B=par.E_att_back_scaled,
                    attr_rcut=par.r_att,
                    rep_rcut=par.r_rep,
                    density_upper_rcut=par.density_upper_rcut,
                    density_lower_rcut=par.density_lower_rcut,
                    ncut_ij=par.ncut_ij_back,  # 12
                    ncut_ji=par.ncut_ij_back,  # 12
                )
                print(
                        f"[back-back] {t1}-{t2}: A={par.E_rep_scaled}, B={par.E_att_back_scaled}, ncut={par.ncut_ij_back}")

            # -----------------------
            # Backbone–Base
            # -----------------------
            else:
                sqd.params[(t1, t2)] = dict(
                    A=par.E_rep_scaled,        # to not have overlaps
                    B=0.0,              # NO attraction
                    attr_rcut=par.r_att,
                    rep_rcut=par.r_rep,
                    density_upper_rcut=par.density_upper_rcut,
                    density_lower_rcut=par.density_lower_rcut,
                    ncut_ij=DEFAULT_NONSPECIFIC_NCUT,
                    ncut_ji=DEFAULT_NONSPECIFIC_NCUT,
                )
                print(
                        f"[back-base] {t1}-{t2}: A={par.E_rep_scaled}, B=0, ncut=12")


def define_pairwise_sqd_v1(sqd, par):
    """SQD v1 = one r_d for the distance cutoff. SQD v2 has higher/lower cutoff for density.
    Interaction setup(intended):
      - Base-base:
          * WC directional(complements): A1-U1, A2-U2 ... (AGCU all have same attraction E_att)
          * non-WC directional(non_wc_complements): directional A1-G1/C1/A1
          * WC nondirectional(wc_nondirectional): A1-U2, A2-U1...B = 0, repulsion(A) can be tuned
          * ALL other nondirectional base-base(repulsion-only, base cutoffs, high ncut so it doesn't compete with pairing
      - Backbone-base:
          repulsion-only, standard cutoffs, high ncut
      - Backbone-backbone:
          weak attraction and repulsion
    """
    print("define_pairwise_sqd_v1 called")
    global _sqd_init_count
    _sqd_init_count += 1
    verbose = (_sqd_init_count == 1)

    # Get pairs from dictionaries
    complements = par.complements # WC directional pairs A1-U1...
    non_wc = par.non_wc_complements # nonwc directional pairs A1-A1, A1-C1...
    #all_nondir = par.all_nondirectional # all remaining nondirectional interactions A1-A2 (nonwc), A1-U2 (wc)...

    for i, t1 in enumerate(par.type_names):
        for t2 in par.type_names[i:]:

            t1_is_base = (t1 in complements)
            t2_is_base = (t2 in complements)

            # -----------------------
            # Base–Base interactions
            # -----------------------
            if t1_is_base and t2_is_base:

                # 1) WC directional
                if complements.get(t1) == t2:
                    sqd.params[(t1, t2)] = dict(
                        A=par.E_rep_wc_scaled,          # scaled with beta
                        B=par.E_att_comp,               # negative scaled
                        attr_rcut=par.r_att_base,
                        rep_rcut=par.r_rep_base,
                        density_rcut=par.r_d_base,
                        ncut_ij=par.ncut_ij_comp, # 1
                        ncut_ji=par.ncut_ij_comp, # 1
                    )
                    if verbose:
                        print(
                            f"[WC_dir] {t1}-{t2}: A={par.E_rep_wc_scaled}, B={par.E_att_comp}, ncut={par.ncut_ij_comp}")


                # 2) non-WC directional
                elif (t2 in non_wc.get(t1, [])) or (t1 in non_wc.get(t2, [])):
                    sqd.params[(t1, t2)] = dict(
                        A=par.E_rep_wc_scaled,
                        B=par.E_att_nonwc,
                        attr_rcut=par.r_att_base,
                        rep_rcut=par.r_rep_base,
                        density_rcut=par.r_d_base,
                        ncut_ij=par.ncut_ij_nonwc, # 1
                        ncut_ji=par.ncut_ij_nonwc, # 1
                    )
                    if verbose:
                        print(
                            f"[nonWC_dir] {t1}-{t2}: A={par.E_rep_wc_scaled}, B={par.E_att_nonwc}, ncut={par.ncut_ij_nonwc}")

                # WC non-directional interaction (A1-U2, U1-A2, etc.)
                elif (
                    (t1 in par.wc_nondirectional and par.wc_nondirectional[t1] == t2) or
                    (t2 in par.wc_nondirectional and par.wc_nondirectional[t2] == t1)
                ):
                    sqd.params[(t1, t2)] = dict(
                        A=par.E_rep_wc_nondir_scaled,  # <- different rep hrere
                        B=0.0,
                        attr_rcut=par.r_att_base,
                        rep_rcut=par.r_rep_base,
                        density_rcut=par.r_d_base,
                        ncut_ij=par.ncut_ij_wc_nondir,
                        ncut_ji=par.ncut_ij_wc_nondir,
                    )
                    if verbose:
                        print(
                            f"[wc_nondir] {t1}-{t2}: A={par.E_rep_wc_nondir_scaled}, B=0, ncut={par.ncut_ij_wc_nondir}")
                                # 3) default base-base (repulsion-only, base cutoffs)
                else:
                    sqd.params[(t1, t2)] = dict(
                        A=par.E_rep_wc_scaled,
                        B=0.0,
                        attr_rcut=par.r_att_base,
                        rep_rcut=par.r_rep_base,
                        density_rcut=par.r_d_base,
                        ncut_ij=DEFAULT_NONSPECIFIC_NCUT,
                        ncut_ji=DEFAULT_NONSPECIFIC_NCUT,
                    )
                    if verbose:
                        print(
                            f"[base_default] {t1}-{t2}: A={par.E_rep_wc_scaled}, B=0, ncut=12")

            # -----------------------
            # Backbone–Backbone
            # -----------------------
            elif t1 == t2:
                sqd.params[(t1, t2)] = dict(
                    A=par.E_rep_scaled,
                    B=par.E_att_back_scaled,
                    attr_rcut=par.r_att,
                    rep_rcut=par.r_rep,
                    density_rcut=par.r_d,
                    ncut_ij=par.ncut_ij_back, # 12
                    ncut_ji=par.ncut_ij_back, # 12
                )
                if verbose:
                    print(
                        f"[back-back] {t1}-{t2}: A={par.E_rep_scaled}, B={par.E_att_back_scaled}, ncut={par.ncut_ij_back}")

            # -----------------------
            # Backbone–Base
            # -----------------------
            else:
                sqd.params[(t1, t2)] = dict(
                    A=par.E_rep_scaled,        # to not have overlaps
                    B=0.0,              # NO attraction
                    attr_rcut=par.r_att,
                    rep_rcut=par.r_rep,
                    density_rcut=par.r_d,
                    ncut_ij=DEFAULT_NONSPECIFIC_NCUT,
                    ncut_ji=DEFAULT_NONSPECIFIC_NCUT,
                )
                if verbose:
                    print(
                        f"[back-base] {t1}-{t2}: A={par.E_rep_scaled}, B=0, ncut=12")


def define_pairwise_sqd_2_particles(sqd, par):
    """SQD (used to be v3) for the alternating 2-bead-per-nucleotide model.

    Interaction setup:
      - Base-base:
          * WC 
          * non-WC
          * all other base-base pairs are repulsion-only
      - Backbone-backbone:
          weak attraction + repulsion
      - Backbone-base:
          repulsion-only
    """
    print("define_pairwise_sqd_2_particles called")


    complements = par.complements
    non_wc = par.non_wc_complements

    for i, t1 in enumerate(par.type_names):
        for t2 in par.type_names[i:]:

            t1_is_base = (t1 in complements)
            t2_is_base = (t2 in complements)

            # -----------------------
            # Base-Base interactions
            # -----------------------
            if t1_is_base and t2_is_base:

                # 1) WC 
                if complements.get(t1) == t2:
                    sqd.params[(t1, t2)] = dict(
                        A=par.E_rep_wc_scaled,
                        B=par.E_att_comp,
                        attr_rcut=par.r_att_base,
                        rep_rcut=par.r_rep_base,
                        density_upper_rcut=par.density_upper_rcut_base,
                        density_lower_rcut=par.density_lower_rcut_base,
                        ncut_ij=par.ncut_ij_comp,
                        ncut_ji=par.ncut_ij_comp,
                    )
                    print(
                            f"[WC] {t1}-{t2}: "
                            f"A={par.E_rep_wc_scaled}, B={par.E_att_comp}, "
                            f"ncut={par.ncut_ij_comp}"
                        )

                # 2) non-WC 
                elif (t2 in non_wc.get(t1, [])) or (t1 in non_wc.get(t2, [])):
                    sqd.params[(t1, t2)] = dict(
                        A=par.E_rep_wc_scaled,
                        B=par.E_att_nonwc,
                        attr_rcut=par.r_att_base,
                        rep_rcut=par.r_rep_base,
                        density_upper_rcut=par.density_upper_rcut_base,
                        density_lower_rcut=par.density_lower_rcut_base,
                        ncut_ij=par.ncut_ij_nonwc,
                        ncut_ji=par.ncut_ij_nonwc,
                    )
                    print(
                            f"[nonWC] {t1}-{t2}: "
                            f"A={par.E_rep_wc_scaled}, B={par.E_att_nonwc}, "
                            f"ncut={par.ncut_ij_nonwc}"
                        )

                # 3) default base-base
                else:
                    sqd.params[(t1, t2)] = dict(
                        A=par.E_rep_wc_scaled,
                        B=0.0,
                        attr_rcut=par.r_att_base,
                        rep_rcut=par.r_rep_base,
                        density_upper_rcut=par.density_upper_rcut_base,
                        density_lower_rcut=par.density_lower_rcut_base,
                        ncut_ij=12,
                        ncut_ji=12,
                    )
                    print(
                            f"[base_default] {t1}-{t2}: "
                            f"A={par.E_rep_wc_scaled}, B=0, ncut=12"
                        )

            # -----------------------
            # Backbone-Backbone
            # -----------------------
            elif t1 == t2:
                sqd.params[(t1, t2)] = dict(
                    A=par.E_rep_scaled,
                    B=par.E_att_back_scaled,
                    attr_rcut=par.r_att,
                    rep_rcut=par.r_rep,
                    density_upper_rcut=par.density_upper_rcut,
                    density_lower_rcut=par.density_lower_rcut,
                    ncut_ij=par.ncut_ij_back,
                    ncut_ji=par.ncut_ij_back,
                )
                print(
                        f"[back-back] {t1}-{t2}: "
                        f"A={par.E_rep_scaled}, B={par.E_att_back_scaled}, "
                        f"ncut={par.ncut_ij_back}"
                    )

            # -----------------------
            # Backbone-Base
            # -----------------------
            else:
                sqd.params[(t1, t2)] = dict(
                    A=par.E_rep_scaled,
                    B=0.0,
                    attr_rcut=par.r_att,
                    rep_rcut=par.r_rep,
                    density_upper_rcut=par.density_upper_rcut,
                    density_lower_rcut=par.density_lower_rcut,
                    ncut_ij=DEFAULT_NONSPECIFIC_NCUT,
                    ncut_ji=DEFAULT_NONSPECIFIC_NCUT,
                )
                print(
                        f"[back-base] {t1}-{t2}: "
                        f"A={par.E_rep_scaled}, B=0, "
                        f"ncut={DEFAULT_NONSPECIFIC_NCUT}"
                    )


def define_pairwise_sqd_v2_brothers(sqd, par):
    """
    Compared to sqd_v2, v2_brothers separates the two 'brother' families.

    Intended behavior:
      - complements_1: main attractive WC patch (e.g. A1-U1, G1-C1)
      - complements_2: second brother family, independently tunable
                       (can be non-attractive, weakly attractive, different cutoffs, etc.)
      - non_wc_complements: directional non-WC interactions
      - wc_nondirectional: cross-family WC-like contacts (e.g. A1-U2, A2-U1), usually repulsive-only
      - other base-base: repulsive-only
      - backbone-backbone: weak attraction / repulsion
      - backbone-base: repulsive-only
    """
    print("define_pairwise_sqd_v2_brothers called")

    global _sqd_init_count
    _sqd_init_count += 1
    verbose = (_sqd_init_count == 1)

    complements_1 = par.complements_1
    complements_2 = par.complements_2
    non_wc = par.non_wc_complements
    wc_nondir = par.wc_nondirectional

    # Any type that appears in either brother complement map is treated as a base-like type
    base_types = set(complements_1) | set(complements_2)

    for i, t1 in enumerate(par.type_names):
        for t2 in par.type_names[i:]:

            t1_is_base = t1 in base_types
            t2_is_base = t2 in base_types

            # -------------------------------------------------
            # Base–Base interactions
            # -------------------------------------------------
            if t1_is_base and t2_is_base:

                # 1) Main attractive WC patch family (brother 1)
                if complements_1.get(t1) == t2:
                    sqd.params[(t1, t2)] = dict(
                        A=par.E_rep_wc_scaled, 
                        B=par.E_att_comp,
                        attr_rcut=par.r_att_base,
                        rep_rcut=par.r_rep_base,
                        density_upper_rcut=par.density_upper_rcut_base,
                        density_lower_rcut=par.density_lower_rcut_base,
                        ncut_ij=par.ncut_ij_comp,
                        ncut_ji=par.ncut_ij_comp,
                    )
                    if verbose:
                        print(
                            f"[WC_dir_1] {t1}-{t2}: "
                            f"A={par.E_rep_wc_scaled}, "
                            f"B={par.E_att_comp}, "
                            f"ncut={par.ncut_ij_comp}"
                        )

                # 2) Secondary brother family (brother 2), independently tunable
                elif complements_2.get(t1) == t2:
                    sqd.params[(t1, t2)] = dict(
                        A=par.E_rep_wc_2_scaled,
                        B=par.E_att_comp_2,
                        attr_rcut=par.r_att_base_2,
                        rep_rcut=par.r_rep_base_2,
                        density_upper_rcut=par.density_upper_rcut_base_2,
                        density_lower_rcut=par.density_lower_rcut_base_2,
                        ncut_ij=par.ncut_ij_comp, # same for both brothers
                        ncut_ji=par.ncut_ij_comp, # same for both brothers
                    )
                    if verbose:
                        print(
                            f"[WC_dir_2] {t1}-{t2}: "
                            f"A={par.E_rep_wc_2_scaled}, "
                            f"B={par.E_att_comp_2}, "
                            f"ncut={par.ncut_ij_comp}"
                        )

                # 3) Directional non-WC
                elif (t2 in non_wc.get(t1, [])) or (t1 in non_wc.get(t2, [])):
                    sqd.params[(t1, t2)] = dict(
                        A=par.E_rep_wc_scaled,
                        B=par.E_att_nonwc,
                        attr_rcut=par.r_att_base,
                        rep_rcut=par.r_rep_base,
                        density_upper_rcut=par.density_upper_rcut_base,
                        density_lower_rcut=par.density_lower_rcut_base,
                        ncut_ij=par.ncut_ij_nonwc,
                        ncut_ji=par.ncut_ij_nonwc,
                    )
                    if verbose:
                        print(
                            f"[nonWC_dir] {t1}-{t2}: "
                            f"A={par.E_rep_wc_scaled}, "
                            f"B={par.E_att_nonwc}, "
                            f"ncut={par.ncut_ij_nonwc}"
                        )

                # 4) WC nondirectional cross-family contacts (e.g. A1-U2)
                elif (
                    (t1 in wc_nondir and wc_nondir[t1] == t2) or
                    (t2 in wc_nondir and wc_nondir[t2] == t1)
                ):
                    sqd.params[(t1, t2)] = dict(
                        A=par.E_rep_wc_nondir_scaled,
                        B=0.0,
                        attr_rcut=par.r_att_base,
                        rep_rcut=par.r_rep_base,
                        density_upper_rcut=par.density_upper_rcut_base,
                        density_lower_rcut=par.density_lower_rcut_base,
                        ncut_ij=DEFAULT_NONSPECIFIC_NCUT,
                        ncut_ji=DEFAULT_NONSPECIFIC_NCUT,
                    )
                    if verbose:
                        print(
                            f"[wc_nondir] {t1}-{t2}: "
                            f"A={par.E_rep_wc_nondir_scaled}, "
                            f"B=0.0, "
                            f"ncut={DEFAULT_NONSPECIFIC_NCUT}"
                        )

                # 5) All other base-base contacts: repulsive only
                else:
                    sqd.params[(t1, t2)] = dict(
                        A=par.E_rep_wc_scaled,
                        B=0.0,
                        attr_rcut=par.r_att_base,
                        rep_rcut=par.r_rep_base,
                        density_upper_rcut=par.density_upper_rcut_base,
                        density_lower_rcut=par.density_lower_rcut_base,
                        ncut_ij=DEFAULT_NONSPECIFIC_NCUT,
                        ncut_ji=DEFAULT_NONSPECIFIC_NCUT,
                    )
                    if verbose:
                        print(
                            f"[base_default] {t1}-{t2}: "
                            f"A={par.E_rep_wc_scaled}, "
                            f"B=0.0, "
                            f"ncut={DEFAULT_NONSPECIFIC_NCUT}"
                        )

            # -------------------------------------------------
            # Backbone–Backbone
            # -------------------------------------------------
            elif t1 == t2:
                sqd.params[(t1, t2)] = dict(
                    A=par.E_rep_scaled,
                    B=par.E_att_back_scaled,
                    attr_rcut=par.r_att,
                    rep_rcut=par.r_rep,
                    density_upper_rcut=par.density_upper_rcut,
                    density_lower_rcut=par.density_lower_rcut,
                    ncut_ij=par.ncut_ij_back,
                    ncut_ji=par.ncut_ij_back,
                )
                if verbose:
                    print(
                        f"[back-back] {t1}-{t2}: "
                        f"A={par.E_rep_scaled}, "
                        f"B={par.E_att_back_scaled}, "
                        f"ncut={par.ncut_ij_back}"
                    )

            # -------------------------------------------------
            # Backbone–Base
            # -------------------------------------------------
            else:
                sqd.params[(t1, t2)] = dict(
                    A=par.E_rep_scaled,
                    B=0.0,
                    attr_rcut=par.r_att,
                    rep_rcut=par.r_rep,
                    density_upper_rcut=par.density_upper_rcut,
                    density_lower_rcut=par.density_lower_rcut,
                    ncut_ij=DEFAULT_NONSPECIFIC_NCUT,
                    ncut_ji=DEFAULT_NONSPECIFIC_NCUT,
                )
                if verbose:
                    print(
                        f"[back-base] {t1}-{t2}: "
                        f"A={par.E_rep_scaled}, "
                        f"B=0.0, "
                        f"ncut={DEFAULT_NONSPECIFIC_NCUT}"
                    )


def define_pairwise_sqd_sticker_spacer(sqd, par):
    """SQD interactions for a 1-bead-per-nucleotide model."""
    print("define_pairwise_sqd_sticker_spacer called")

    complements = par.complements
    non_wc = par.non_wc_complements

    for i, t1 in enumerate(par.type_names):
        for t2 in par.type_names[i:]:

            # WC interaction
            if complements.get(t1) == t2:
                sqd.params[(t1, t2)] = dict(
                    A=par.E_rep_wc_scaled,
                    B=par.E_att_comp,
                    attr_rcut=par.r_att_base,
                    rep_rcut=par.r_rep_base,
                    density_upper_rcut=par.density_upper_rcut_base,
                    density_lower_rcut=par.density_lower_rcut_base,
                    ncut_ij=par.ncut_ij_comp,
                    ncut_ji=par.ncut_ij_comp,
                )
                print(
                    f"[WC] {t1}-{t2}: A={par.E_rep_wc_scaled}, "
                    f"B={par.E_att_comp}, ncut={par.ncut_ij_comp}"
                )

            # non-WC interaction
            elif (t2 in non_wc.get(t1, [])) or (t1 in non_wc.get(t2, [])):
                sqd.params[(t1, t2)] = dict(
                    A=par.E_rep_wc_scaled,
                    B=par.E_att_nonwc,
                    attr_rcut=par.r_att_base,
                    rep_rcut=par.r_rep_base,
                    density_upper_rcut=par.density_upper_rcut_base,
                    density_lower_rcut=par.density_lower_rcut_base,
                    ncut_ij=par.ncut_ij_nonwc,
                    ncut_ji=par.ncut_ij_nonwc,
                )
                print(
                    f"[nonWC] {t1}-{t2}: A={par.E_rep_wc_scaled}, "
                    f"B={par.E_att_nonwc}, ncut={par.ncut_ij_nonwc}"
                )

            # default interaction
            else:
                sqd.params[(t1, t2)] = dict(
                    A=par.E_rep_wc_scaled,
                    B=0.0,
                    attr_rcut=par.r_att_base,
                    rep_rcut=par.r_rep_base,
                    density_upper_rcut=par.density_upper_rcut_base,
                    density_lower_rcut=par.density_lower_rcut_base,
                    ncut_ij=12,
                    ncut_ji=12,
                )
                print(
                    f"[default] {t1}-{t2}: A={par.E_rep_wc_scaled}, "
                    f"B=0.0, ncut=12"
                )


##### Box creation and random vector initialization

def random_unit_vector_3D(rng):
    """
    Generate a random unit vector in 3D by sampling angles uniformly on the sphere.
    """
    theta = 2.0 * math.pi * rng.random()
    phi = math.acos(2.0 * rng.random() - 1.0)

    x = math.sin(phi) * math.cos(theta)
    y = math.sin(phi) * math.sin(theta)
    z = math.cos(phi)

    return (x, y, z)

def random_walk_positions_3D(
    N,
    step_size,
    box_size,
    rng,
    start_pos=(0, 0, 0),
    max_tries=100000,
):
    """
    Place N monomers in 3D via a random walk inside a cubic box centered at the origin.
    First monomer is placed at start_pos.
    Each subsequent monomer is step_size away in a random direction,
    re-rolling if it exits the box boundaries.
    """
    half_box_size = box_size * 0.5

    positions = [start_pos]

    for i in range(1, N):
        success = False
        tries = 0

        while not success and tries < max_tries:
            tries += 1

            ux, uy, uz = random_unit_vector_3D(rng)
            oldx, oldy, oldz = positions[i - 1]

            newx = oldx + step_size * ux
            newy = oldy + step_size * uy
            newz = oldz + step_size * uz

            if (
                -half_box_size <= newx <= half_box_size
                and -half_box_size <= newy <= half_box_size
                and -half_box_size <= newz <= half_box_size
            ):
                positions.append((newx, newy, newz))
                success = True

        if not success:
            raise RuntimeError(
                "Failed to place all monomers inside the box. "
                "Try increasing box_size or max_tries."
            )

    return np.array(positions)


#### Build the Snapshot

def snapshot_ts7(par):
    '''
    Like troubleshoot6 but without the phantom bonds
    '''

    sequences = [insert_backbone_duplicate_types(seq,par) for seq in par.sequences]  # list of lists
    num_mols = len(sequences)
    lengths = [len(seq) for seq in sequences]
    total_particles = sum(lengths)

    snapshot = gsd.hoomd.Frame()

    # Estimate total box size
    volume = total_particles / par.density
    box_size = volume ** (1/3)

    snapshot.particles.N = total_particles
    snapshot.configuration.box = [box_size, box_size, box_size, 0, 0, 0]
    snapshot.particles.types = par.type_names

    typeids = []
    positions = []
    velocities = []
    bonds = []
    angles = []
    dihedrals = []

    bond_types = ['Base-Phosphate', 'Phosphate-Phosphate']
    angle_types = ['Phosphate-Phosphate-Phosphate', 'Phosphate-Phosphate-Base', 'Base-Phosphate-Base','Base-Phosphate-Phosphate']
    dihedral_types = ['Stacking','Direction']

    bond_typeid = []
    angle_typeid = []
    dihedral_typeid = []

    offset = 0
    rng = np.random.default_rng(par.init_seed)
    for mol_idx, sequence in enumerate(sequences):
        N = len(sequence)
        typeids.extend([par.type_names.index(x) for x in sequence])

        # Assign a center offset for this molecule to avoid overlaps
        center_offset = rng.uniform(-box_size / 2, box_size / 2, size=3)
        pos = random_walk_positions_3D(
            N=N, step_size=par.bond_length, box_size=box_size,rng=rng, start_pos=center_offset
        )
        positions.extend(pos)
        velocities.extend([[0.0, 0.0, 0.0]] * N)

        # Bonds
        # B-A1
        bonds.extend([
            [offset + i, offset + i + 1]
            for i in range(0, N, 3)
            if i + 1 < N
        ])
        # B-A2
        bonds.extend([
            [offset + i, offset + i + 2]
            for i in range(0, N, 3)
            if i + 2 < N
        ])
        # B-B
        bonds.extend([
            [offset + i, offset + i + 3]
            for i in range(0, N-3, 3)
        ])
        bond_typeid.extend(
        [0] * (N // 3 * 2) + # Base - Phosphate
        [1] * (N // 3-1) # Phosphate - Phosphate
        ) 

        # Angles
        # Phosphate-Phosphate-Phosphate
        angles.extend([
            [offset + i, offset + i + 3, offset + i + 6]
            for i in range(0, N - 6, 3)
        ])
        # Phosphate-Phosphate-Base
        angles.extend([
            [offset + i, offset + i + 3, offset + i + 5]
            for i in range(0, N - 3, 3)
        ])
        angles.extend([
            [offset + i, offset + i + 3, offset + i + 4]
            for i in range(0, N - 3, 3)
        ])
        
        # Base-Phosphate-Base
        angles.extend([
            [offset + i + 1, offset + i, offset + i + 2]
            for i in range(0, N, 3)
            if i + 2 < N
        ])
        # Base-Phosphate-Phosphate 
        angles.extend([
            [offset + i + 1, offset + i, offset + i + 3]
            for i in range(0, N - 5, 3)
        ])
        angles.extend([
            [offset + i + 2, offset + i, offset + i + 3]
            for i in range(0, N - 5, 3)
        ])

        angle_typeid.extend(
        [0] * (N // 3 - 2) + # PPP
        [1] * (2 * N // 3 - 2) + # PPB
        [2] * (N // 3 )+ # BPB
        [3] * (2 * N // 3 -2)) # BPP

        # Dihedrals
        # Stacking
        dihedrals.extend([
            [offset + i + 1, offset + i, offset + i + 3, offset + i + 4]
            for i in range(0, N - 4, 3)
        ])
        dihedrals.extend([
            [offset + i + 2, offset + i, offset + i + 3, offset + i + 5]
            for i in range(0, N - 5, 3)
        ])
        # Direction
        dihedrals.extend([
            [offset + i, offset + i +3, offset + i + 4, offset + i + 5]
            for i in range(0, N - 5, 3)
        ])
        dihedrals.extend([
            [offset + i + 3, offset + i, offset + i + 2, offset + i + 1]
            for i in range(0, N - 5, 3)
        ])

        dihedral_typeid.extend(        
        [0] * (2 * N // 3 -2) + 
        [1] * (2 * N // 3-2))

        offset += N  # update offset for next molecule

    # Checks for debugging
    assert len(bonds) == len(bond_typeid), (
        f"Bond mismatch: {len(bonds)} groups but {len(bond_typeid)} type IDs"
    )
    assert len(angles) == len(angle_typeid), (
        f"Angle mismatch: {len(angles)} groups but {len(angle_typeid)} type IDs"
    )
    assert len(dihedrals) == len(dihedral_typeid), (
        f"Dihedral mismatch: {len(dihedrals)} groups but {len(dihedral_typeid)} type IDs"
    )

    assert len(typeids) == total_particles, (
        f"Particle type mismatch: expected {total_particles}, got {len(typeids)}"
    )
    assert len(positions) == total_particles, (
        f"Position mismatch: expected {total_particles}, got {len(positions)}"
    )
    assert len(velocities) == total_particles, (
        f"Velocity mismatch: expected {total_particles}, got {len(velocities)}"
    )

    # Finalize snapshot
    snapshot.particles.typeid = typeids
    snapshot.particles.position = positions
    snapshot.particles.velocity = velocities
    snapshot.particles.validate()

    snapshot.bonds.N = len(bonds)
    snapshot.bonds.types = bond_types

    # type 0 (base-phosphate), type 1 (phosphate-phosphate)
    snapshot.bonds.typeid = bond_typeid

    # Bonds
    snapshot.bonds.group = bonds
    snapshot.bonds.validate()

    # Angles
    snapshot.angles.N = len(angles)
    snapshot.angles.types = angle_types
    snapshot.angles.typeid = angle_typeid
    snapshot.angles.group = angles
    snapshot.angles.validate()

    # Dihedrals
    snapshot.dihedrals.N = len(dihedrals)
    snapshot.dihedrals.types = dihedral_types
    snapshot.dihedrals.typeid = dihedral_typeid
    snapshot.dihedrals.group = dihedrals
    snapshot.dihedrals.validate()

    print(f"Snapshot created with {num_mols} RNA molecules in a box of side {box_size:.3f}")
    print(f"Total particles: {total_particles}, bonds: {len(bonds)}, angles: {len(angles)}, dihedrals: {len(dihedrals)}")

    return snapshot


def snapshot_2_particles(par: argparse.Namespace) -> gsd.hoomd.Frame:
    """Simplified 2-particles (B-A/G/C/T) model.

    Args:
        par: Parsed command-line parameters. Expected to contain the
            attributes needed to define particle types and simulation setup.

    Returns:
        A HOOMD snapshot representing the simplified two-particle model.
    """
    sequences = [insert_backbone(seq) for seq in par.sequences]
    num_mols = len(sequences)
    lengths = [len(seq) for seq in sequences]
    total_particles = sum(lengths)

    snapshot = gsd.hoomd.Frame()

    # Estimate total box size
    volume = total_particles / par.density
    box_size = volume ** (1/3)

    snapshot.particles.N = total_particles
    snapshot.configuration.box = [box_size, box_size, box_size, 0, 0, 0]
    snapshot.particles.types = par.type_names
    
    type_to_id = {name: i for i, name in enumerate(par.type_names)}

    typeids = []
    positions = []
    velocities = []
    bonds = []
    angles = []
    dihedrals = []
    

    bond_types = ['Base-Phosphate', 'Phosphate-Phosphate']
    angle_types = ['Phosphate-Phosphate-Phosphate', 'Phosphate-Phosphate-Base', 'Base-Phosphate-Phosphate']
    dihedral_types =  ['Stacking']

    bond_typeid = []
    angle_typeid = []
    dihedral_typeid = []


    offset = 0
    rng = np.random.default_rng(par.init_seed)
    for sequence in sequences:
        N = len(sequence)
        typeids.extend(type_to_id[x] for x in sequence)

        # Assign a center offset for this molecule to avoid overlaps
        center_offset = rng.uniform(-box_size / 2, box_size / 2, size=3)
        pos = random_walk_positions_3D(
            N=N, step_size=par.bond_length, box_size=box_size,rng=rng, start_pos=center_offset
        )
        positions.extend(pos)
        velocities.extend([[0.0, 0.0, 0.0]] * N)

        # Bonds
        # P-B (0,1), (2,3)..
        bonds.extend(
            [[offset + i, offset + i + 1] for i in range(0, N - 1, 2)]
        )
        # P-P (0,2), (2,4)...
        bonds.extend([
            [offset + i, offset + i + 2]
            for i in range(0, N-2, 2)
        ])

        bond_typeid.extend([0] * len(range(0, N - 1, 2)))
        bond_typeid.extend([1] * len(range(0, N - 2, 2)))
        
        # Angles
        # Phosphate-Phosphate-Phosphate: (0,2,4), (2,4,6), ...
        ppp_angles = [
            [offset + i, offset + i + 2, offset + i + 4]
            for i in range(0, N - 4, 2)
        ]

        # Phosphate-Phosphate-Base: (0,2,3), (2,4,5), ...
        # base attached to the second phosphate
        ppb_angles = [
            [offset + i, offset + i + 2, offset + i + 3]
            for i in range(0, N - 3, 2)
        ]

        # Base-Phosphate-Phosphate: (1,2,4), (3,4,6), ...
        # base attached to the first phosphate
        bpp_angles = [
            [offset + i + 1, offset + i + 2, offset + i + 4]
            for i in range(0, N - 4, 2)
        ]

        angles.extend(ppp_angles)
        angles.extend(ppb_angles)
        angles.extend(bpp_angles)

        angle_typeid.extend([0] * len(ppp_angles))  # PPP
        angle_typeid.extend([1] * len(ppb_angles))  # PPB
        angle_typeid.extend([2] * len(bpp_angles))  # BPP
        
        # Stacking dihedrals
        stacking_dihedrals = [
            [offset + i + 1, offset + i, offset + i + 2, offset + i + 3]
            for i in range(0, N - 3, 2)
        ]

        dihedrals.extend(stacking_dihedrals)
        dihedral_typeid.extend([0] * len(stacking_dihedrals))

        offset += N
    
    assert len(typeids) == total_particles
    assert len(positions) == total_particles
    assert len(velocities) == total_particles
    assert len(bonds) == len(bond_typeid)
    assert len(angles) == len(angle_typeid)
    assert len(dihedrals) == len(dihedral_typeid)
    # Align angle
    
    # moments = []
    # for t_id in typeids:
    #     t_name = snapshot.particles.types[t_id]
    #     if t_name == "B":
    #         moments.append([0.0, 0.0, 0.0])
    #     else:
    #         moments.append([1.0, 1.0, 1.0])

    snapshot.particles.typeid = typeids
    snapshot.particles.position = positions
    snapshot.particles.velocity = velocities
    #snapshot.particles.moment_inertia = moments # Align angle
    snapshot.particles.moment_inertia = [[1.0, 1.0, 1.0]] * total_particles # Align angle on all particles

    snapshot.particles.validate()

    snapshot.bonds.N = len(bonds)
    snapshot.bonds.types = bond_types
    snapshot.bonds.typeid = bond_typeid
    snapshot.bonds.group = bonds
    snapshot.bonds.validate()

    snapshot.angles.N = len(angles)
    snapshot.angles.types = angle_types
    snapshot.angles.typeid = angle_typeid
    snapshot.angles.group = angles
    snapshot.angles.validate()

    # Dihedrals
    snapshot.dihedrals.N = len(dihedrals)
    snapshot.dihedrals.types = dihedral_types
    snapshot.dihedrals.typeid = dihedral_typeid
    snapshot.dihedrals.group = dihedrals
    snapshot.dihedrals.validate()

    print(
        f"Snapshot created with {num_mols} molecules in a box of side {box_size:.3f}")
    print(
        f"Total particles: {total_particles}, bonds: {len(bonds)}, angles: {len(angles)}, dihedrals: {len(dihedrals)}")

    return snapshot


def snapshot_sticker_spacer(par):
    sequences = par.sequences
    num_mols = len(sequences)
    total_particles = sum(len(seq) for seq in sequences)

    snapshot = gsd.hoomd.Frame()

    volume = total_particles / par.density
    box_size = volume ** (1 / 3)

    snapshot.particles.N = total_particles
    snapshot.configuration.box = [box_size, box_size, box_size, 0, 0, 0]
    snapshot.particles.types = par.type_names

    type_to_id = {name: i for i, name in enumerate(par.type_names)}

    typeids = []
    positions = []
    velocities = []
    bonds = []
    angles = []

    bond_types = ['Base-Base']
    angle_types = ['Base-Base-Base']

    bond_typeid = []
    angle_typeid = []

    offset = 0
    rng = np.random.default_rng(par.init_seed)

    for sequence in sequences:
        N = len(sequence)
        typeids.extend(type_to_id[x] for x in sequence)

        center_offset = rng.uniform(-box_size / 2, box_size / 2, size=3)
        pos = random_walk_positions_3D(
            N=N, step_size=par.bond_length, box_size=box_size,rng=rng, start_pos=center_offset
        )
        positions.extend(pos)
        velocities.extend([[0.0, 0.0, 0.0]] * N)

        bonds.extend(
            [[offset + i, offset + i + 1] for i in range(N - 1)]
        )
        bond_typeid.extend([0] * (N - 1))

        angles.extend(
            [[offset + i, offset + i + 1, offset + i + 2]
                for i in range(N - 2)]
        )
        angle_typeid.extend([0] * (N - 2))

        offset += N

    assert len(typeids) == total_particles
    assert len(positions) == total_particles
    assert len(velocities) == total_particles
    assert len(bonds) == len(bond_typeid)
    assert len(angles) == len(angle_typeid)

    snapshot.particles.typeid = typeids
    snapshot.particles.position = positions
    snapshot.particles.velocity = velocities
    snapshot.particles.validate()

    snapshot.bonds.N = len(bonds)
    snapshot.bonds.types = bond_types
    snapshot.bonds.typeid = bond_typeid
    snapshot.bonds.group = bonds
    snapshot.bonds.validate()

    snapshot.angles.N = len(angles)
    snapshot.angles.types = angle_types
    snapshot.angles.typeid = angle_typeid
    snapshot.angles.group = angles
    snapshot.angles.validate()

    print(
        f"Snapshot created with {num_mols} molecules in a box of side {box_size:.3f}")
    print(
        f"Total particles: {total_particles}, bonds: {len(bonds)}, angles: {len(angles)}")

    return snapshot
