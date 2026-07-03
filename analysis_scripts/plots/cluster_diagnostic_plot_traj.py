"""
Initial diagnostic plots from raw trajectories, used to confirm clustering from analysis pipeline.

Saves for each run in a sweep:
    - 2d projections (xy, xz, yz)
    - 1D density profiles
    - g(r) + cluster-size distribution + run-info text panel
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import gsd.hoomd
import freud
from json import JSONDecoder
from collections import Counter, defaultdict
from scipy.ndimage import gaussian_filter
from matplotlib.colors import PowerNorm

import sys
sys.path.insert(0, "/groups/goloborodko/projects/leona/validation/")


from analysis_scripts.phase_separation import make_molecule_lookup_table
BASE_DIR = Path("/groups/goloborodko/projects/leona/validation/bulk/output_dir")

def load_params(params_path):
    """Robust params.json loader — ignores trailing garbage after first JSON object."""
    with open(params_path) as f:
        text = f.read()
    params, _ = JSONDecoder().raw_decode(text)
    return params


def get_unique_output_dir(base_dir, suffix=""):
    """Return base_dir[_suffix][_N] where N is the lowest integer that doesn't exist."""
    tag = f"_{suffix}" if suffix else ""
    candidate = base_dir.parent / f"{base_dir.name}{tag}"
    if not candidate.exists():
        return candidate
    n = 1
    while True:
        candidate = base_dir.parent / f"{base_dir.name}{tag}_{n}"
        if not candidate.exists():
            return candidate
        n += 1


def FullDiagnostic(
    trajectory, params=None,
    frame_idx=-1,
    r_cut_cluster=0.85, min_contacts=1,
    proj_bins=120, smooth_sigma=0.5,
    r_max_gr=None, rdf_bins=100, gamma=0.5,
    figsize=(16, 12),
):
    # ---------- setup ----------
    frame = trajectory[frame_idx]
    pos = np.array(frame.particles.position)
    box = freud.box.Box.from_box(frame.configuration.box)
    Lx, Ly, Lz = frame.configuration.box[:3]
    N = len(pos)

    molecule_id = make_molecule_lookup_table(frame)
    n_mols = int(molecule_id.max()) + 1

    _, chain_lens = np.unique(molecule_id, return_counts=True)
    len_counter = Counter(chain_lens.tolist())
    chain_len_summary = ", ".join(f"{L}×{c}" for L, c in sorted(len_counter.items()))

    types_arr = np.array(frame.particles.types)
    typeid = np.array(frame.particles.typeid)
    first_bead_idx = np.concatenate([[0], np.where(np.diff(molecule_id) != 0)[0] + 1])
    species = types_arr[typeid[first_bead_idx]]
    species_counter = Counter(species.tolist())
    species_summary = ", ".join(f"{s}×{c}" for s, c in species_counter.items())

    Lmin = min(Lx, Ly, Lz)
    if r_max_gr is None:
        r_max_gr = Lmin / 2 * 0.95

    # ---------- g(r) ----------
    rdf = freud.density.RDF(bins=rdf_bins, r_max=r_max_gr)
    rdf.compute(system=(box, pos))
    gr_peak   = float(np.max(rdf.rdf))
    gr_peak_r = float(rdf.bin_centers[np.argmax(rdf.rdf)])
    gr_tail   = float(np.mean(rdf.rdf[-10:]))

    # ---------- molecule-level clustering ----------
    aq = freud.locality.AABBQuery(box, pos)
    nlist = aq.query(pos, {"r_max": r_cut_cluster, "exclude_ii": True}).toNeighborList()
    i = np.array(nlist.query_point_indices)
    j = np.array(nlist.point_indices)
    mi, mj = molecule_id[i], molecule_id[j]
    inter = mi != mj
    mi, mj = mi[inter], mj[inter]

    if min_contacts > 1:
        pair_counts = defaultdict(int)
        for a, b in zip(mi, mj):
            key = (a, b) if a < b else (b, a)
            pair_counts[key] += 1
        edges = [(a, b) for (a, b), c in pair_counts.items() if c >= min_contacts]
    else:
        edges = list(zip(mi.tolist(), mj.tolist()))

    parent = np.arange(n_mols)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    comp = np.array([find(k) for k in range(n_mols)])
    sizes = np.array(list(Counter(comp).values()))
    unique_sizes, counts = np.unique(sizes, return_counts=True)
    frac_by_size = unique_sizes * counts / n_mols

    # ---------- figure ----------
    fig, axes = plt.subplots(3, 3, figsize=figsize)

    projections = [("xy", 0, 1, Lx, Ly), ("xz", 0, 2, Lx, Lz), ("yz", 1, 2, Ly, Lz)]
    proj_data = []
    for name, ii, jj, Li, Lj in projections:
        H, xe, ye = np.histogram2d(
            pos[:, ii], pos[:, jj], bins=proj_bins,
            range=[[-Li/2, Li/2], [-Lj/2, Lj/2]],
        )
        if smooth_sigma > 0:
            H = gaussian_filter(H, sigma=smooth_sigma)
        proj_data.append((name, H, xe, ye))
    vmax = max(d[1].max() for d in proj_data)
    for ax, (name, H, xe, ye) in zip(axes[0], proj_data):
        im = ax.imshow(H.T, origin="lower",
                       extent=(xe[0], xe[-1], ye[0], ye[-1]),
                       cmap="magma",
                       norm=PowerNorm(gamma=gamma, vmin=0, vmax=vmax),
                       aspect="equal", interpolation="nearest")
        ax.set_xlabel(name[0]); ax.set_ylabel(name[1])
        ax.set_title(f"density projection: {name}")
        plt.colorbar(im, ax=ax, shrink=0.75)

    for ax, k, name, L in zip(axes[1], range(3), ["x", "y", "z"], [Lx, Ly, Lz]):
        ax.hist(pos[:, k], bins=proj_bins, range=(-L/2, L/2),
                color="steelblue", edgecolor="none")
        ax.axhline(N / proj_bins, color="red", ls="--", lw=1)
        ax.set_xlabel(name); ax.set_ylabel("particle count")
        ax.set_title(f"1D profile along {name}")

    ax = axes[2, 0]
    ax.plot(rdf.bin_centers, rdf.rdf, color="k")
    ax.axhline(1.0, color="red", ls="--", lw=1)
    ax.set_xlabel("r"); ax.set_ylabel("g(r)")
    ax.set_title("radial distribution function")

    ax = axes[2, 1]
    ax.bar(unique_sizes.astype(str), frac_by_size,
           color="steelblue", edgecolor="k")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("cluster size (molecules)")
    ax.set_ylabel("fraction of molecules")
    ax.set_title(f"mol-clusters (r_cut={r_cut_cluster}, min_contacts={min_contacts})")
    for x, (f, c) in enumerate(zip(frac_by_size, counts)):
        ax.text(x, f + 0.01, f"×{c}", ha="center", va="bottom", fontsize=8)

    ax = axes[2, 2]
    ax.axis("off")
    lines = [
        f"Frame: {frame_idx}",
        f"N particles: {N}",
        f"N molecules: {n_mols}",
        f"chain lens: {chain_len_summary}",
        f"species: {species_summary}",
        f"box: {Lx:.1f} × {Ly:.1f} × {Lz:.1f}",
        "",
        f"g(r) peak: {gr_peak:.2f} @ r={gr_peak_r:.2f}",
        f"g(r) tail: {gr_tail:.2f}",
        f"largest cluster: {int(sizes.max())} mol",
        f"# clusters: {len(sizes)}",
        "",
    ]
    if params is not None:
        for key in ["E_att", "angle_k_ppp", "density", "molecule_file"]:
            lines.append(f"{key}: {params.get(key, '—')}")
    else:
        lines.append("(no params.json provided)")
    ax.text(0.02, 0.98, "\n".join(lines),
            transform=ax.transAxes, va="top", ha="left",
            family="monospace", fontsize=10)
    ax.set_title("run info")

    fig.tight_layout()
    return fig


def main():
    p = argparse.ArgumentParser(description="Run FullDiagnostic on every run in a sweep.")
    p.add_argument("sweep_dir", type=Path, help="Directory containing run subfolders.")
    p.add_argument("--traj_name", default="trajectory_0.gsd", help="Trajectory filename inside each run dir.")
    p.add_argument("--params_name", default="params.json", help="Params filename inside each run dir.")
    p.add_argument("--output_base", type=Path,
                default=Path("/groups/goloborodko/projects/leona/validation/analysis/analysis_output/plots/diagnostic_plots_from_traj"),
                help="Base output dir.")
    p.add_argument("--folder_name_suffix", default="",
                   help="Suffix for output folder; auto-numbered if it exists.")
    p.add_argument("--frame_idx", type=int, default=-1)
    p.add_argument("--r_cut", type=float, default=0.7)
    p.add_argument("--min_contacts", type=int, default=1)
    p.add_argument("--bins", type=int, default=300)
    p.add_argument("--sigma", type=float, default=0)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing per-run pngs in the chosen output dir.")
    args = p.parse_args()

    sweep_dir = (BASE_DIR / args.sweep_dir).resolve()
    base_out = args.output_base / sweep_dir.name
    out_dir = get_unique_output_dir(base_out, args.folder_name_suffix)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] writing diagnostics to: {out_dir}")

    run_dirs = sorted([d for d in sweep_dir.iterdir()
                       if d.is_dir() and (d / args.traj_name).exists()])
    if not run_dirs:
        print(f"[warn] no runs with {args.traj_name} found under {sweep_dir}")
        return

    for run_dir in run_dirs:
        traj_path = run_dir / args.traj_name
        params_path = run_dir / args.params_name
        out_png = out_dir / f"{run_dir.name}.png"

        if out_png.exists() and not args.overwrite:
            print(f"[skip] {out_png.name} (exists)")
            continue

        try:
            params = load_params(params_path) if params_path.exists() else None
            with gsd.hoomd.open(traj_path, mode="r") as traj:
                fig = FullDiagnostic(
                    traj, params=params,
                    frame_idx=args.frame_idx,
                    r_cut_cluster=args.r_cut,
                    min_contacts=args.min_contacts,
                    proj_bins=args.bins,
                    smooth_sigma=args.sigma,
                    gamma=args.gamma,
                )
            fig.savefig(out_png, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"[ok]   {out_png.name}")
        except Exception as e:
            print(f"[err]  {run_dir.name}: {e}")


if __name__ == "__main__":
    main()