from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from data_config import (
    DATASETS,
    available_l_values,
    curves_from_basis,
    folder_name,
    normalize_dataset,
    normalize_scale,
    read_curve,
    read_vector,
)


CURVE_STYLES = {
    "IPR X": {"color": "#1b9e77", "label": r"$\mathrm{IPR}_X$", "linestyle": "-"},
    "IPR Z": {"color": "#d95f02", "label": r"$\mathrm{IPR}_Z$", "linestyle": "--"},
    "IPR XY": {"color": "#1b9e77", "label": r"$\mathrm{IPR}_{XY}$", "linestyle": "-"},
}


HEISENBERG_CURVE_STYLES = {
    "IPR XY": {"color": "#1b9e77", "label": r"$\mathrm{IPR}_{XY}$", "linestyle": "-"},
    "IPR Z": {"color": "#d95f02", "label": r"$\mathrm{IPR}_{Z}$", "linestyle": "--"},
    "IPR X": {"color": "#1b9e77", "label": r"$\mathrm{IPR}_{X}$ (dotted)", "linestyle": ":"},
}


def _curve_style(dataset: str, curve_name: str) -> dict:
    """Return the plot style dictionary for one curve in one dataset."""
    if dataset == "heisenberg":
        return HEISENBERG_CURVE_STYLES[curve_name]
    return CURVE_STYLES[curve_name]


def set_ipr_plot_style() -> None:
    """Apply the shared publication-style matplotlib settings."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "mathtext.fontset": "dejavusans",
        "font.size": 15,
        "axes.labelsize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 14,
    })


def _nearest_crossing(x: np.ndarray, y_a: np.ndarray, y_b: np.ndarray, reference_x: float) -> tuple[float, float] | None:
    """Return the curve crossing closest to a reference x-value, if one exists."""
    x = np.asarray(x, dtype=float)
    y_a = np.asarray(y_a, dtype=float)
    y_b = np.asarray(y_b, dtype=float)

    order = np.argsort(x)
    x = x[order]
    y_a = y_a[order]
    y_b = y_b[order]
    diff = y_a - y_b

    exact = np.where(np.isclose(diff, 0.0, atol=1e-12, rtol=1e-10))[0]
    candidates = []
    for i in exact:
        candidates.append((float(x[i]), float(y_a[i])))

    brackets = np.where(diff[:-1] * diff[1:] < 0.0)[0]
    for i in brackets:
        x0, x1 = x[i], x[i + 1]
        d0, d1 = diff[i], diff[i + 1]
        t = -d0 / (d1 - d0)
        x_cross = x0 + t * (x1 - x0)
        y_cross = y_a[i] + t * (y_a[i + 1] - y_a[i])
        candidates.append((float(x_cross), float(y_cross)))

    if not candidates:
        return None
    return min(candidates, key=lambda p: abs(p[0] - reference_x))


def plot_ipr_curves(
    dataset: str,
    L: int | str,
    curves: str | list[str] = "all",
    scale: str = "raw",
    ax=None,
    show_title: bool = False,
    mark_critical: bool = True,
    show_legend: bool = True,
    figsize=(6.2, 4.0),
    alpha: float = 1.0,
    linewidth: float = 1.8,
):
    """Plot processed IPR curves for TFIM 1D, TFIM 2D, or Heisenberg data.

    Parameters
    ----------
    dataset:
        One of "tfim1d", "tfim2d", or "heisenberg".
    L:
        System size, for example 6, "L=6x1", "L=6x6", or "L=6".
    curves:
        "all" or a list such as ["IPR X", "IPR Z"].
    scale:
        "raw" plots IPR values directly. "log10" plots log10(IPR).
    ax:
        Optional matplotlib axis.
    show_title:
        If True, add a compact title above the axis.
    mark_critical:
        Draw the accepted phase-transition marker for the selected model.
    show_legend:
        If True, draw the curve legend.
    figsize:
        Figure size in inches when a new figure is created.
    alpha:
        Curve opacity, where 1.0 is fully opaque.
    linewidth:
        Width of the plotted curve lines.
    """
    dataset = normalize_dataset(dataset)
    scale = normalize_scale(scale)

    cfg = DATASETS[dataset]
    folder = cfg["processed"] / folder_name(dataset, L)
    if not folder.exists():
        raise FileNotFoundError(f"Data folder not found: {folder}")

    available_curves = cfg["curves"]
    if curves == "all":
        selected_curves = list(available_curves)
    elif isinstance(curves, str):
        selected_curves = [curves]
    else:
        selected_curves = list(curves)

    for curve_name in selected_curves:
        if curve_name not in available_curves:
            raise ValueError(f"Unknown curve {curve_name!r}. Available: {list(available_curves)}")

    x = read_vector(folder / cfg["x_file"])

    set_ipr_plot_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.figure

    for curve_name in selected_curves:
        y = read_curve(folder, cfg, curve_name, scale)
        style = _curve_style(dataset, curve_name)
        ax.plot(
            x,
            y,
            label=style["label"],
            lw=linewidth,
            color=style["color"],
            ls=style["linestyle"],
            alpha=alpha,
        )

    if mark_critical:
        ax.axvline(cfg["critical_point"], color="0.35", lw=1.2, ls="--", zorder=0)

    ax.set_xlabel(cfg["x_label"])
    ax.set_ylabel(r"$\log_{10}(\mathrm{IPR})$" if scale == "log10" else "IPR")
    if show_title:
        ax.set_title(f"{cfg['title_name']} IPR ({folder_name(dataset, L)}, scale={scale})", pad=10)

    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.minorticks_on()
    ax.grid(which="major", alpha=0.18, lw=0.7)
    ax.grid(which="minor", alpha=0.07, lw=0.5)
    if show_legend:
        ax.legend(frameon=False, loc="best")
    return fig, ax


def plot_all_n_log_overlay(
    dataset: str,
    basis: list[str] | tuple[str, ...] = ("IPR X", "IPR Z"),
    n_min: int | None = None,
    n_max: int | None = None,
    mark_crossings: bool = False,
    mark_critical: bool = True,
    figsize=(7.2, 4.8),
    show_title: bool = False,
):
    """Overlay processed log10(IPR) curves for all available system sizes.

    Parameters
    ----------
    dataset:
        "tfim1d", "tfim2d", or "heisenberg".
    basis:
        Explicit IPR curve list, such as ["IPR X", "IPR Z"] or ["IPR XY", "IPR Z"].
    n_min, n_max:
        Optional system-size bounds.
    mark_crossings:
        If True and exactly two curves are selected, mark each L crossing with a white circle.
    mark_critical:
        Draw the model critical point.
    """
    dataset = normalize_dataset(dataset)
    scale = "log10"
    curves = curves_from_basis(dataset, basis)
    cfg = DATASETS[dataset]

    n_values = available_l_values(dataset, n_min=n_min, n_max=n_max)
    if not n_values:
        raise ValueError(f"No processed L folders found for {dataset} in the requested range.")

    set_ipr_plot_style()
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    n0, n1 = n_values[0], n_values[-1]
    crossings = []

    for n in n_values:
        t = 0.0 if n1 == n0 else (n - n0) / (n1 - n0)
        alpha = 0.85 - 0.65 * t
        linewidth = 1.7 - 0.9 * t
        line_start = len(ax.lines)

        plot_ipr_curves(
            dataset=dataset,
            L=n,
            curves=curves,
            scale=scale,
            ax=ax,
            show_title=False,
            mark_critical=False,
            show_legend=False,
            alpha=alpha,
            linewidth=linewidth,
        )

        # Z curve is dashed in all-L overlays.
        for line, curve_name in zip(ax.lines[line_start:], curves):
            if curve_name == "IPR Z":
                line.set_linestyle("--")
                line.set_dashes((5, 3))
            line.set_solid_capstyle("round")

        if mark_crossings and len(curves) == 2:
            folder = cfg["processed"] / folder_name(dataset, n)
            x = read_vector(folder / cfg["x_file"])
            y_a = read_curve(folder, cfg, curves[0], scale)
            y_b = read_curve(folder, cfg, curves[1], scale)
            crossing = _nearest_crossing(x, y_a, y_b, cfg["critical_point"])
            if crossing is not None:
                crossings.append(crossing)

    if crossings:
        ax.scatter(
            [p[0] for p in crossings],
            [p[1] for p in crossings],
            s=30,
            facecolors="white",
            edgecolors="black",
            linewidths=1.0,
            zorder=6,
        )

    if mark_critical:
        ax.axvline(cfg["critical_point"], color="0.35", lw=1.2, ls=":", zorder=0)

    ax.set_xlabel(cfg["x_label"])
    ax.set_ylabel(r"$\log_{10}(\mathrm{IPR})$")
    if show_title:
        ax.set_title(
            f"{cfg['title_name']}: all L overlays ({n_values[0]} <= L <= {n_values[-1]}, scale={scale})",
            pad=8,
        )

    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.minorticks_on()
    ax.grid(which="major", alpha=0.18, lw=0.7)
    ax.grid(which="minor", alpha=0.07, lw=0.5)

    handles = []
    for curve_name in curves:
        style = _curve_style(dataset, curve_name)
        linestyle = "--" if curve_name == "IPR Z" else style["linestyle"]
        handle = Line2D(
            [0],
            [0],
            color=style["color"],
            lw=2.0,
            ls=linestyle,
            label=style["label"] + (" (dashed)" if curve_name == "IPR Z" else ""),
        )
        if curve_name == "IPR Z":
            handle.set_dashes((5, 3))
        handles.append(handle)

    handles.extend([
        Line2D([0], [0], color="0.2", lw=2.0, alpha=0.8, label="smaller L"),
        Line2D([0], [0], color="0.2", lw=2.0, alpha=0.2, label="larger L (dimmer)"),
    ])
    if crossings:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor="white",
                markeredgecolor="black",
                markersize=6,
                label="crossing (per L)",
            )
        )

    ax.legend(handles=handles, frameon=False, loc="best", ncol=2)
    return fig, ax
