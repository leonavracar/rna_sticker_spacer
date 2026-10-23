import math
import matplotlib.pyplot as plt
import numpy as np
def plot_panels(
    items: list,                          # e.g. list of run_dir paths
    plot_fn,                              # plot_fn(item, ax) -> None
    title_fn = lambda x: str(x),
    n_cols: int = 3,
    figsize_per_panel: tuple[float, float] = (4.0, 4.0),
    suptitle: str | None = None,
    sharex: bool = False,
    sharey: bool = False,
):
    if not items:
        raise ValueError("plot_panels requires at least one item")
    if n_cols < 1:
        raise ValueError("n_cols must be positive")
    n = len(items)
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(figsize_per_panel[0]*n_cols,
                                      figsize_per_panel[1]*n_rows),
                             sharex=sharex, sharey=sharey)
    axes = np.atleast_1d(axes).ravel()
    for ax, item in zip(axes, items):
        plot_fn(item, ax)
        ax.set_title(title_fn(item), fontsize=9)
    for ax in axes[n:]:
        ax.set_visible(False)
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig, axes