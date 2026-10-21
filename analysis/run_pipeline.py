#!/usr/bin/env python3
"""Run nucleotide-pairing, molecule-contact and experiment analyses.

Examples:
    python analysis/run_pipeline.py --sweep bulk/output_dir/50b_reentrant_full --experiment reentrance --stage analyze
    python analysis/run_pipeline.py --sweep bulk/output_dir/52b_mixed_repeat_length_full --experiment mixture --stage all
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ANALYSIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ANALYSIS_DIR))
from src.utils.io import load_params_json
from src.framewise import (edge_frames, molecule_lengths, component_frame_rows,
                           contact_frame_rows, summarize_numeric, summarize_bin_frames,
                           late_window)
from src.mixtures import identify_species, mixture_cluster_rows
from src.single_chain import single_chain_framewise, add_gsd_geometry, mfe_pairs_from_csv


def run_directories(sweep: Path, run_ids: list[str] | None):
    for path in sorted(sweep.iterdir()):
        if not path.is_dir() or not (path / "params.json").is_file():
            continue
        if run_ids and path.name.split("_", 1)[0] not in run_ids:
            continue
        yield path


def make_missing_npzs(run_dir: Path, traj: Path, stages: set[str], overwrite: bool, config: dict):
    """Generate requested molecular-contact archives."""
    from src.pairing import compute_filtered_nlist_base_pairs_traj, make_molecule_lookup_table
    from src.molecule_graph import build_molecule_contact_edge_dict_from_pairing_npz
    import gsd.hoomd

    stem = traj.stem
    pairing_dir = run_dir / "computed_metrics" / "pairing"
    graph_dir = run_dir / "computed_metrics" / "mol_graph"
    intra_dir = run_dir / "computed_metrics" / "intramol_graph"
    pairing_path = pairing_dir / f"{stem}_pairing.npz"
    lookup_path = pairing_dir / "molecule_lookup.npy"
    graph_path = graph_dir / f"{stem}_molecule_contact_edges.npz"
    intra_path = intra_dir / f"{stem}_intra_molecule_contacts.npz"

    if "pairing" in stages and (overwrite or not pairing_path.is_file() or not lookup_path.is_file()):
        params = load_params_json(run_dir / "params.json")
        r_opt = (float(params["r_att_base"]) + float(params["r_rep_base"])) / 2
        with gsd.hoomd.open(str(traj), "r") as gsd:
            data = compute_filtered_nlist_base_pairs_traj(
                traj=gsd, r_min=float(config['r_min']), r_max=float(config['r_max']),
                exclude_chain=int(config['exclude_chain']),
                stride=int(config['stride']), r_opt=r_opt)
            mol_id = make_molecule_lookup_table(gsd[0])
        pairing_dir.mkdir(parents=True, exist_ok=True)
            if overwrite or not pairing_path.is_file():
            np.savez(pairing_path, **data)
        if overwrite or not lookup_path.is_file():
            np.save(lookup_path, mol_id)
    if "graphs" in stages and (overwrite or not graph_path.is_file()):
        if not pairing_path.exists() or not lookup_path.exists():
            raise FileNotFoundError(f"Pairing inputs missing for {traj}; run --stage pairing first")
        with np.load(pairing_path) as pairing:
            graph_data = build_molecule_contact_edge_dict_from_pairing_npz(
                pairing, np.load(lookup_path))
        graph_dir.mkdir(parents=True, exist_ok=True)
        np.savez(graph_path, **graph_data)
    if "graphs" in stages and config["experiment"] == "mixture" and (overwrite or not intra_path.is_file()):
        if not pairing_path.exists() or not lookup_path.exists():
            raise FileNotFoundError(f"Intramolecular inputs missing for {traj}")
        from collections import Counter
        lookup = np.load(lookup_path)
        f_out, m_out, n_out = [], [], []
        with np.load(pairing_path) as pairing:
            frames = pairing["frame_indices"].astype(int)
            for frame in frames:
                seen = set()
                counts = Counter()
                for i,j in zip(pairing[f"qi_{frame}"], pairing[f"pi_{frame}"]):
                    i,j=int(i),int(j)
                    if i==j or lookup[i]!=lookup[j]:
                        continue
                    p=(min(i,j), max(i,j))
                    if p in seen:
                        continue
                    seen.add(p)
                    counts[int(lookup[i])] += 1
                for mol, count in counts.items():
                    f_out.append(frame); m_out.append(mol); n_out.append(count)
        intra_dir.mkdir(parents=True, exist_ok=True)
        np.savez(intra_path, frame=np.array(f_out,dtype=np.int64),
                 mol_id=np.array(m_out,dtype=np.int64),
                 n_intra_contacts=np.array(n_out,dtype=np.int64),
                 frame_indices=np.array(frames,dtype=np.int64),
                 n_molecules=np.array([len(molecule_lengths(lookup))],dtype=np.int64))


def analyze_trajectory(run_dir: Path, traj: Path, config: dict, sweep: Path):
    experiment = config["experiment"]
    stem = traj.stem
    pairing_path = run_dir / "computed_metrics" / "pairing" / f"{stem}_pairing.npz"
    lookup_path = run_dir / "computed_metrics" / "pairing" / "molecule_lookup.npy"
    graph_path = run_dir / "computed_metrics" / "mol_graph" / f"{stem}_molecule_contact_edges.npz"
    intra_path = run_dir / "computed_metrics" / "intramol_graph" / f"{stem}_intra_molecule_contacts.npz"
    if not pairing_path.is_file() or not lookup_path.is_file():
        raise FileNotFoundError(f"Missing precomputed pairing inputs: {pairing_path}")
    lengths = molecule_lengths(np.load(lookup_path))
    lookup = np.load(lookup_path)
    fraction = float(config["late_fraction"])
    min_contacts = int(config["min_contacts"])
    window = int(config.get("recurrence_window", 20))

    with np.load(pairing_path) as pairing:
        contacts = contact_frame_rows(
            pairing, lookup,
            particle_recurrence=experiment=="length_patterning",
            recurrence_window=window)
        if experiment=="single_chain":
            mfe_file = sweep / config["mfe_file"]
            if not mfe_file.exists():
                raise FileNotFoundError(f"Missing MFE reference {mfe_file}")
            mfe = single_chain_framewise(pairing,mfe_pairs_from_csv(mfe_file),
                                         int(config.get("pair_index_offset",1)))
            frame = contacts.merge(mfe,on="frame",how="left")
            frame = add_gsd_geometry(frame,traj,float(config.get("length_scale_nm",.6)))
            bins = pd.DataFrame()
            ranked = pd.DataFrame()
            extras = {}
        else:
            if not graph_path.is_file():
                raise FileNotFoundError(f"Missing precomputed graph NPZ: {graph_path}")
            with np.load(graph_path) as edges:
                species = identify_species(lengths,
                    {int(k):v for k,v in config["species_by_length"].items()}) if experiment=="mixture" else None
                graph,bins,ranked = component_frame_rows(
                    edge_frames(edges),lengths,min_contacts,window,
                    with_recurrence=(experiment=="reentrance"),
                    with_bins=True,with_ranks=experiment=="length_patterning")
                frame = contacts.merge(graph,on="frame",how="inner")
                extras={}
                if experiment=="mixture":
                    if not intra_path.exists():
                        raise FileNotFoundError(f"Mixture requires existing intra NPZ {intra_path}")
                    with np.load(intra_path) as intra:
                        ranks,types,largest = mixture_cluster_rows(edges,lengths,species,min_contacts,intra)
                    extras={"ranked_clusters":ranks,"cluster_contacts":types,"largest_cluster":largest}
    params = load_params_json(run_dir/"params.json")
    run_id = run_dir.name.split("_",1)[0] + "__" + stem.replace("trajectory_","traj_")
    meta={"run_id":run_id,"folder_name":run_dir.name,"traj_replicate":stem,
          "E_att":params.get("E_att"),"angle_k_ppp":params.get("angle_k_ppp"),
          "density":params.get("density"),"molecule_file":params.get("molecule_file")}
    for df in [frame,bins,ranked,*extras.values()]:
        if not df.empty:
            for key,value in meta.items():
                df[key]=value
    if "step" not in frame:
        import gsd.hoomd
        with gsd.hoomd.open(str(traj),"r") as gsd:
            frame["step"]=[int(gsd[int(f)].configuration.step) for f in frame.frame]
    metric_names = [c for c in frame if c in (
        "n_base_pairs","n_inter_contacts","n_intra_contacts","n_edges",
        "mean_edge_weight","mean_degree","molecule_recurrence",
        "particle_recurrence","mfe_matches","extra_pairs","precision","recall","Rg_nm")]
    summary = {**meta,"late_fraction":fraction,**summarize_numeric(frame,fraction,metric_names)}
    bin_summary = summarize_bin_frames(bins,fraction) if not bins.empty else pd.DataFrame()
    if not bin_summary.empty:
        for key,value in meta.items():
            bin_summary[key]=value
    for name,df in extras.items():
        if not df.empty:
            frames=sorted(df.frame.unique())
            keep=max(1,int(np.ceil(len(frames)*fraction)))
            extras[name]=df[df.frame.isin(frames[-keep:])]
    return frame,bin_summary,summary,extras,ranked


def write_results(sweep: Path, outputs: list[tuple], save_framewise: bool):
    target=sweep/"analysis"
    target.mkdir(exist_ok=True)
    frames, binned, summary, extras, ranks = zip(*outputs)
    pd.DataFrame(summary).to_csv(target/"replicate_summary.csv",index=False)
    if any(not x.empty for x in binned):
        pd.concat([x for x in binned if not x.empty],ignore_index=True).to_csv(
            target/"cluster_distribution.csv",index=False)
    for name in ("ranked_clusters","cluster_contacts","largest_cluster"):
        tables=[x[name] for x in extras if name in x and not x[name].empty]
        if tables:
            pd.concat(tables,ignore_index=True).to_csv(target/f"{name}.csv",index=False)
    if any(not x.empty for x in ranks):
            pd.concat([x for x in ranks if not x.empty],ignore_index=True).to_csv(
            target/"ranked_clusters_framewise.csv",index=False)
    if save_framewise:
        folder=target/"framewise"; folder.mkdir(exist_ok=True)
        for df in frames:
            if not df.empty:
                df.to_csv(folder/f"{df.iloc[0]['run_id']}.csv",index=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep",type=Path,required=True)
    p.add_argument("--experiment",choices=["single_chain","reentrance","length_patterning","mixture"],required=True)
    p.add_argument("--stage",choices=["pairing","graphs","analyze","all"],default="analyze")
    p.add_argument("--run-ids",nargs="+",default=None)
    p.add_argument("--overwrite-npz",action="store_true",help="Explicitly regenerate existing NPZ inputs")
    p.add_argument("--save-framewise",action="store_true")
    args=p.parse_args()
    if args.run_ids and args.stage in ("analyze", "all"):
        p.error("--run-ids is supported for pairing and graphs only; full analysis writes sweep-level CSVs")
    sweep=args.sweep.resolve()
    if not sweep.is_dir():
        p.error(f"Not a sweep directory: {sweep}")
    config=yaml.safe_load((ANALYSIS_DIR/"configs"/f"{args.experiment}.yaml").read_text())
    stages=set()
    if args.stage in ("pairing","all"): stages.add("pairing")
    if args.stage in ("graphs","all"): stages.add("graphs")
    results=[]
    for run_dir in run_directories(sweep,args.run_ids):
        for traj in sorted(run_dir.glob("trajectory_*.gsd")):
            if stages:
                make_missing_npzs(run_dir,traj,stages,args.overwrite_npz,config)
            if args.stage in ("analyze","all"):
                print(f"Analyzing {run_dir.name}/{traj.name}",flush=True)
                results.append(analyze_trajectory(run_dir,traj,config,sweep))
    if results:
        write_results(sweep,results,args.save_framewise)
        print(f"Saved analysis summaries: {sweep/'analysis'}")
    elif args.stage=="analyze":
        raise RuntimeError("No trajectories were found to analyze")


if __name__ == "__main__":
    main()
