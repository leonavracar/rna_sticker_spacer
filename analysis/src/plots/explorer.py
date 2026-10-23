"""Optional miniature Jupyter dashboard for exploring condition summary CSVs.

The dashboard is intentionally notebook-local: no Streamlit server, front end,
new data backend, or production-pipeline dependency.
"""
from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt

from .data import filter_rows
from .figures import plot_metric_series
from .style import PlotStyle


def summary_explorer(df: pd.DataFrame, *, style: PlotStyle, x: str = "E_att"):
    """Show metric and condition selectors in an existing Jupyter notebook."""
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError as exc:
        raise ImportError("Install ipywidgets for the optional notebook explorer") from exc

    metrics = sorted(
        name.removesuffix("_mean")
        for name in df.columns
        if name.endswith("_mean") and f"{name.removesuffix('_mean')}_std" in df
    )
    if not metrics:
        raise ValueError("No _mean/_std metric pairs found in summary data")

    metric_selector = widgets.Dropdown(options=metrics, description="Metric")
    filter_columns = [
        col for col in ("density", "angle_k_ppp", "repeat_count", "sequence_class")
        if col in df and df[col].nunique(dropna=True) > 1
    ]
    selectors = {}
    for column in filter_columns:
        values = sorted(df[column].dropna().unique().tolist())
        selectors[column] = widgets.Dropdown(
            options=[("All", None)] + [(str(v), v) for v in values],
            value=values[0], description=style.label(column),
        )
    out = widgets.Output()

    def render(_=None):
        with out:
            out.clear_output(wait=True)
            filtered = df
            for name, widget in selectors.items():
                if widget.value is not None:
                    filtered = filter_rows(filtered, **{name: widget.value})
            fig, ax = plt.subplots(figsize=(6.7, 3.7), layout="constrained")
            try:
                plot_metric_series(filtered, ax, metric=metric_selector.value,
                                   x=x, style=style, band="std")
                display(fig)
            except (KeyError, ValueError) as exc:
                print(f"Select a single parameter series: {exc}")
            finally:
                plt.close(fig)

    metric_selector.observe(render, names="value")
    for selector in selectors.values():
        selector.observe(render, names="value")
    display(widgets.VBox([metric_selector, *selectors.values(), out]))
    render()
