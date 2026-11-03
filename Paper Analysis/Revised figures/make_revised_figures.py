"""Generate the consolidated figures for the revised manuscript.

All figures share one style and are drawn at their final printed width
(3.4 in for one column, 7.0 in for the full page), so they can be included
in LaTeX without rescaling and all text appears at the same size.

Usage:
    python make_revised_figures.py OUT_DIR [--thermal-svm PICKLE]

The thermal SVM sweep (decision functions, balanced accuracies) is the most
expensive input. If PICKLE exists it is loaded, otherwise the sweep is run
with the settings of "TFIM Thermal/Thermal analysis.ipynb" and cached there.
TFIM balanced accuracies are read from tfim_svm_tables.json, which holds the
values of Tables I and II of the manuscript. thermal_svm_tables.json records the
resulting Table IV values for reference; the figures use the full sweep.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle, Ellipse, FancyBboxPatch, Polygon, Rectangle
from matplotlib.ticker import MultipleLocator
from sklearn.preprocessing import KernelCenterer

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from data_config import (  # noqa: E402
    DATASETS,
    available_l_values,
    available_thermal_betas,
    curves_from_basis,
    folder_name,
    read_curve,
    read_thermal_ipr_data,
    read_vector,
)
from plot_gram_matrices import (  # noqa: E402
    IPR_GRAM_CMAP,
    compute_ipr_gram_matrix,
    compute_thermal_gram_matrix,
)
from plot_ipr_curves import _nearest_crossing  # noqa: E402

COL_W = 3.4
PAGE_W = 7.0
GREEN = "#1b9e77"
ORANGE = "#d95f02"
LIGHT_BLUE = "#56b4e9"
KERNEL_COLORS = {"ipr": GREEN, "hs": ORANGE}
KERNEL_NAMES = {"ipr": "IPR", "hs": "HS"}
BASIS_COLORS = {"IPR X": GREEN, "IPR XY": GREEN, "IPR Z": ORANGE}
BASIS_LABELS = {"IPR X": r"$I^{(X)}$", "IPR XY": r"$I^{(XY)}$", "IPR Z": r"$I^{(Z)}$"}
L_MARKERS = {20: "o", 50: "s", 100: "^"}
DELTA_S_CMAP = LinearSegmentedColormap.from_list("green_white_orange", [GREEN, "#ffffff", ORANGE], N=512)


# Arial is not installed system-wide here; these copies ship with ROOT. Set
# IPR_FONT_DIR to another folder holding arial*.ttf if this path does not exist.
ARIAL_DIRS = [os.environ.get("IPR_FONT_DIR", ""), "/opt/software/root/6.36.00/fonts"]


def register_arial() -> bool:
    """Make Arial available to matplotlib without installing it; True if found."""
    for folder in filter(None, ARIAL_DIRS):
        files = sorted(Path(folder).glob("arial*.ttf"))
        if files:
            for f in files:
                font_manager.fontManager.addfont(str(f))
            return True
    return False


def apply_style() -> None:
    """Shared rcParams for every revised figure, matching the original paper plots (Arial text, DejaVu math)."""
    if not register_arial():
        print("warning: Arial not found, falling back to Liberation Sans")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        # This Arial build lacks U+2212; use the ASCII hyphen for negative ticks.
        "axes.unicode_minus": False,
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.6,
        "ytick.minor.size": 1.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.pad": 2.5,
        "ytick.major.pad": 2.5,
        "axes.labelpad": 2.0,
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "legend.handletextpad": 0.5,
        "legend.labelspacing": 0.25,
        "legend.columnspacing": 1.0,
        "legend.borderaxespad": 0.3,
        "pdf.fonttype": 42,
        "savefig.dpi": 400,
    })


# ---------------------------------------------------------------- layout helpers
def new_figure(width: float, height: float) -> plt.Figure:
    return plt.figure(figsize=(width, height))


def add_axes(fig: plt.Figure, left: float, bottom: float, width: float, height: float, **kw) -> plt.Axes:
    """Add axes with position and size given in inches."""
    W, H = fig.get_size_inches()
    return fig.add_axes([left / W, bottom / H, width / W, height / H], **kw)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.annotate(label, xy=(0, 1), xycoords="axes fraction", xytext=(-3, 3), textcoords="offset points",
                ha="right", va="bottom", fontsize=9, fontweight="bold")


def inset_text(ax: plt.Axes, text: str, loc: str = "upper left") -> None:
    x, ha = (0.05, "left") if "left" in loc else (0.95, "right")
    y, va = (0.94, "top") if "upper" in loc else (0.06, "bottom")
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.0})


def line_axes(ax: plt.Axes) -> None:
    ax.minorticks_on()
    ax.grid(which="major", alpha=0.15, lw=0.5)


def gram_panel(ax: plt.Axes, K: np.ndarray, x: np.ndarray, crit: float = 1.0):
    lo, hi = float(np.min(x)), float(np.max(x))
    im = ax.imshow(K, origin="lower", extent=[lo, hi, lo, hi], cmap=IPR_GRAM_CMAP,
                   norm=Normalize(0.0, 1.0), interpolation="nearest", aspect="auto", rasterized=True)
    ax.axvline(crit, color="0.15", lw=0.5, ls=":")
    ax.axhline(crit, color="0.15", lw=0.5, ls=":")
    # Integer major ticks on both axes, matching every other control-parameter axis.
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.minorticks_on()
    return im


def colorbar(fig: plt.Figure, im, left: float, bottom: float, height: float, label: str) -> None:
    cax = add_axes(fig, left, bottom, 0.07, height)
    cb = fig.colorbar(im, cax=cax)
    cb.set_ticks([0.0, 0.5, 1.0])
    cb.ax.tick_params(length=2.0, width=0.5, right=True, left=False)
    cb.outline.set_linewidth(0.5)
    cb.set_label(label, labelpad=3)


def save(fig: plt.Figure, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("saved", name)


# ---------------------------------------------------------------- data helpers
def ipr_overlay(ax: plt.Axes, dataset: str, basis: tuple[str, str], legend_loc: str = "best",
                n_min: int | None = None) -> None:
    """All-L log10 IPR curves for two bases with per-L crossings."""
    cfg = DATASETS[dataset]
    curves = curves_from_basis(dataset, basis)
    Ls = available_l_values(dataset, n_min=n_min)
    crossings = []
    for L in Ls:
        t = 0.0 if Ls[-1] == Ls[0] else (L - Ls[0]) / (Ls[-1] - Ls[0])
        alpha, lw = 0.85 - 0.65 * t, 1.1 - 0.6 * t
        folder = cfg["processed"] / folder_name(dataset, L)
        x = read_vector(folder / cfg["x_file"])
        ys = [read_curve(folder, cfg, c, "log10") for c in curves]
        for c, y in zip(curves, ys):
            line, = ax.plot(x, y, color=BASIS_COLORS[c], lw=lw, alpha=alpha, solid_capstyle="round")
            if c == "IPR Z":
                line.set_dashes((4, 2))
        cross = _nearest_crossing(x, ys[0], ys[1], cfg["critical_point"])
        if cross is not None:
            crossings.append(cross)
    if crossings:
        ax.scatter(*zip(*crossings), s=9, facecolors="white", edgecolors="black", linewidths=0.6, zorder=6)
    ax.axvline(cfg["critical_point"], color="0.35", lw=0.6, ls=":", zorder=0)
    ax.set_xlabel(cfg["x_label"])
    ax.set_ylabel(r"$\log_{10}(\mathrm{IPR})$")
    line_axes(ax)
    handles = []
    for c in curves:
        h = Line2D([0], [0], color=BASIS_COLORS[c], lw=1.2,
                   label=BASIS_LABELS[c] + (" (dashed)" if c == "IPR Z" else ""))
        if c == "IPR Z":
            h.set_dashes((4, 2))
        handles.append(h)
    handles += [
        Line2D([0], [0], color="0.2", lw=1.2, alpha=0.85, label=r"smaller $L$"),
        Line2D([0], [0], color="0.2", lw=1.2, alpha=0.25, label=r"larger $L$ (fainter)"),
        Line2D([0], [0], marker="o", ls="None", mfc="white", mec="black", mew=0.6, ms=3.5, label=r"crossing per $L$"),
    ]
    ax.legend(handles=handles, loc=legend_loc, ncol=2, frameon=True, facecolor="white", edgecolor="none",
              framealpha=0.85)


def load_thermal_svm(path: Path) -> dict:
    if path.exists():
        return pickle.load(open(path, "rb"))
    from run_thermal_kernel_svm import run_thermal_kernel_svm_beta_sweep
    res = {}
    for L in (20, 50, 100):
        fig, _, r = run_thermal_kernel_svm_beta_sweep(L=L, kinds=("ipr", "hs"), source="processed", test_frac=0.10,
                                                     local_delta=0.25, n_splits=100, rng_seed=12345, C=1,
                                                     show_plot_title=False)
        plt.close(fig)
        res[L] = r
    pickle.dump(res, open(path, "wb"))
    return res


def centered_kta(K: np.ndarray, x: np.ndarray, hc: float = 1.0) -> float:
    y = np.where(np.asarray(x) >= hc, 1.0, -1.0)
    Kc = KernelCenterer().fit_transform(np.asarray(K, dtype=float))
    Tc = KernelCenterer().fit_transform(np.outer(y, y))
    return float(np.sum(Kc * Tc) / (np.linalg.norm(Kc) * np.linalg.norm(Tc)))


def decision_sharpness(h: np.ndarray, f: np.ndarray, hc: float = 1.0) -> float:
    """|df/dh| at the learned zero crossing closest to hc (as in Thermal analysis.ipynb)."""
    ok = np.isfinite(h) & np.isfinite(f)
    order = np.argsort(h[ok])
    h, f = h[ok][order], f[ok][order]
    idx = np.where((f[:-1] < 0.0) & (f[1:] > 0.0))[0]
    if idx.size == 0:
        return np.nan
    stars = h[idx] - f[idx] * (h[idx + 1] - h[idx]) / (f[idx + 1] - f[idx])
    h_star = float(stars[np.argmin(np.abs(stars - hc))])
    return abs(float(np.interp(h_star, h, np.gradient(f, h))))


def ba_by_beta(result: dict, kind: str, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = result["by_kind"][kind]
    betas = np.asarray(result["betas"], dtype=float)
    return betas, np.asarray(k[f"{split}_bal_acc_mean"], dtype=float), np.asarray(k[f"{split}_bal_acc_std"], dtype=float)


# ---------------------------------------------------------------- main-text figures
def fig_tfim(out: Path, svm_tables: dict) -> None:
    """TFIM: IPR crossings, Gram matrix at L=10, global and local balanced accuracy."""
    fig = new_figure(PAGE_W, 4.25)
    ax_a = add_axes(fig, 0.45, 2.45, 3.05, 1.7)
    ax_b = add_axes(fig, 4.35, 2.45, 1.7, 1.7)
    ax_c = add_axes(fig, 0.45, 0.38, 3.05, 1.5)
    ax_d = add_axes(fig, 3.9, 0.38, 3.05, 1.5)

    ipr_overlay(ax_a, "tfim1d", ("IPR X", "IPR Z"), legend_loc="lower left")
    ax_a.set_ylim(-33, 1.5)
    ax_a.xaxis.set_major_locator(MultipleLocator(1))

    g = compute_ipr_gram_matrix("tfim1d", 10, basis=("IPR X", "IPR Z"), normalize=True)
    im = gram_panel(ax_b, g["K"], g["x"])
    ax_b.set_xlabel(r"$h_i$")
    ax_b.set_ylabel(r"$h_j$")
    inset_text(ax_b, r"$L=10$", "upper left")
    colorbar(fig, im, 6.12, 2.45, 1.7, r"$\tilde K_{\mathrm{IPR}}(h_i,h_j)$")

    shots = ["200", "500", "1000", "2000", "5000", "10000"]
    # Same green as the IPR curves in (a); opacity grows with the shot budget.
    alphas = np.linspace(0.25, 1.0, len(shots))
    for ax, col, title in ((ax_c, 1, "global"), (ax_d, 3, "local")):
        for ns, alpha in zip(shots, alphas):
            arr = np.asarray(svm_tables[ns])
            ax.errorbar(arr[:, 0], arr[:, col], yerr=arr[:, col + 1], color=GREEN, alpha=alpha, marker="o",
                        ms=2.2, lw=0.9, elinewidth=0.5, capsize=1.2, label=rf"$N_s={int(ns)}$")
        arr = np.asarray(svm_tables["exact"])
        ax.errorbar(arr[:, 0], arr[:, col], yerr=arr[:, col + 1], color=ORANGE, marker="s", ms=2.2, lw=0.9,
                    elinewidth=0.5, capsize=1.2, label="exact IPRs")
        ax.axhline(0.5, color="0.35", lw=0.6, ls=":")
        ax.set_xlabel(r"$L$")
        ax.set_xticks(np.arange(10, 101, 10))
        ax.set_xlim(6, 104)
        ax.set_ylim(0.45, 1.02)
        inset_text(ax, title, "upper right" if col == 1 else "upper right")
        line_axes(ax)
    ax_c.set_ylabel("Balanced accuracy")
    ax_d.set_yticklabels([])
    ax_d.legend(loc="center right", ncol=1)

    for ax, lab in zip((ax_a, ax_b, ax_c, ax_d), "abcd"):
        panel_label(ax, f"({lab})")
    save(fig, out, "tfim_results")


def fig_heisenberg(out: Path) -> None:
    """Heisenberg: XY/Z IPR crossings and Gram matrices for three sizes."""
    fig = new_figure(PAGE_W, 2.1)
    ax_a = add_axes(fig, 0.45, 0.38, 2.35, 1.62)
    size, gap, x0 = 1.0, 0.14, 3.2
    b = 0.38 + (1.62 - size) / 2
    axes = [add_axes(fig, x0 + k * (size + gap), b, size, size) for k in range(3)]

    ipr_overlay(ax_a, "heisenberg", ("IPR XY", "IPR Z"), legend_loc="lower right")
    ax_a.set_ylim(-5.4, 0.3)

    im = None
    for ax, L in zip(axes, (8, 16, 24)):
        g = compute_ipr_gram_matrix("heisenberg", L, basis=("IPR XY", "IPR Z"), normalize=True)
        im = gram_panel(ax, g["K"], g["x"])
        inset_text(ax, rf"$L={L}$")
        ax.set_xlabel(r"$\Delta_i$")
        ax.set_xticks([0, 1, 2, 3, 4, 5])
    axes[0].set_ylabel(r"$\Delta_j$")
    for ax in axes[1:]:
        ax.set_yticklabels([])
    colorbar(fig, im, x0 + 3 * size + 2 * gap + 0.07, b, size, r"$\tilde K_{\mathrm{IPR}}(\Delta_i,\Delta_j)$")
    panel_label(ax_a, "(a)")
    panel_label(axes[0], "(b)")
    save(fig, out, "heisenberg_results")


def fig_thermal(out: Path, svm: dict) -> None:
    """Thermal TFIM: Gram matrices at L=100, centered KTA, and local balanced accuracy."""
    fig = new_figure(PAGE_W, 3.0)
    size, gap, x0, y0 = 1.1, 0.14, 0.62, 0.38
    betas = (0.1, 0.2, 2.0)
    im = None
    gram_axes = {}
    for r, kind in enumerate(("ipr", "hs")):
        for c, beta in enumerate(betas):
            ax = add_axes(fig, x0 + c * (size + gap), y0 + r * (size + 0.06), size, size)
            g = compute_thermal_gram_matrix(kind=kind, L=100, beta=beta, source="processed", normalize=True)
            im = gram_panel(ax, g["K"], g["x"])
            ax.set_xticks([0, 1, 2, 3, 4])
            ax.set_yticks([0, 1, 2, 3, 4])
            if r == 0:
                ax.set_xlabel(r"$h_i$")
            else:
                ax.set_xticklabels([])
                ax.set_title(rf"$\beta={beta:g}$", pad=3)
            if c == 0:
                ax.set_ylabel(f"{KERNEL_NAMES[kind]}\n" + r"$h_j$")
            else:
                ax.set_yticklabels([])
            gram_axes[(r, c)] = ax
    colorbar(fig, im, x0 + 3 * size + 2 * gap + 0.07, y0, 2 * size + 0.06, r"$\tilde K(h_i,h_j)$")

    xr, wr = 5.12, 1.82
    ax_k = add_axes(fig, xr, 1.72, wr, 1.12)
    ax_l = add_axes(fig, xr, y0, wr, 1.12, sharex=ax_k)
    for kind in ("ipr", "hs"):
        for L in (20, 50, 100):
            bs = sorted(set(available_thermal_betas(kind="ipr", L=L, source="processed"))
                        & set(available_thermal_betas(kind="hs", L=L, source="processed")))
            kta = [centered_kta(**{k: v for k, v in compute_thermal_gram_matrix(
                kind=kind, L=L, beta=b, source="processed", normalize=True).items() if k in ("K", "x")})
                for b in bs]
            ax_k.plot(bs, kta, color=KERNEL_COLORS[kind], marker=L_MARKERS[L], ms=2.2, lw=0.9)
            b_ba, mean, std = ba_by_beta(svm[L], kind, "local")
            ax_l.errorbar(b_ba, mean, yerr=std, color=KERNEL_COLORS[kind], marker=L_MARKERS[L], ms=2.2, lw=0.9,
                          elinewidth=0.5, capsize=1.0)
    ax_k.set_ylabel("Centered KTA")
    ax_l.set_ylabel("Local BA")
    ax_l.set_xlabel(r"$\beta$")
    ax_k.set_xlim(2.05, 0.0)
    ax_k.tick_params(labelbottom=False)
    for ax in (ax_k, ax_l):
        line_axes(ax)
    kernel_handles = [Line2D([0], [0], color=KERNEL_COLORS[k], lw=1.2, label=KERNEL_NAMES[k]) for k in ("ipr", "hs")]
    size_handles = [Line2D([0], [0], color="0.25", marker=L_MARKERS[L], ls="None", ms=3, label=rf"$L={L}$")
                    for L in (20, 50, 100)]
    ax_l.legend(handles=kernel_handles + size_handles, loc="lower left", ncol=3, columnspacing=0.8)
    panel_label(gram_axes[(1, 0)], "(a)")
    panel_label(ax_k, "(b)")
    panel_label(ax_l, "(c)")
    save(fig, out, "thermal_results")


# ---------------------------------------------------------------- appendix figures
def fig_tfim_grams(out: Path) -> None:
    """TFIM raw and normalized IPR Gram matrices for three sizes."""
    fig = new_figure(PAGE_W, 3.9)
    size, gap, x0, y0 = 1.55, 0.14, 0.95, 0.38
    for r, (normalize, name, cbl) in enumerate(((True, "normalized", r"$\tilde K_{\mathrm{IPR}}$"),
                                               (False, "raw", r"$K_{\mathrm{IPR}}$"))):
        im = None
        for c, L in enumerate((10, 50, 90)):
            ax = add_axes(fig, x0 + c * (size + gap), y0 + r * (size + 0.24), size, size)
            g = compute_ipr_gram_matrix("tfim1d", L, basis=("IPR X", "IPR Z"), normalize=normalize)
            im = gram_panel(ax, g["K"], g["x"])
            inset_text(ax, rf"$L={L}$")
            if r == 0:
                ax.set_xlabel(r"$h_i$")
            else:
                ax.set_xticklabels([])
            if c == 0:
                ax.set_ylabel(f"{name}\n" + r"$h_j$")
                if r == 1:
                    panel_label(ax, "(a)")
                else:
                    panel_label(ax, "(b)")
            else:
                ax.set_yticklabels([])
        colorbar(fig, im, x0 + 3 * size + 2 * gap + 0.08, y0 + r * (size + 0.24), size, cbl)
    save(fig, out, "tfim_grams_raw_normalized")


def fig_heisenberg_xz(out: Path) -> None:
    """Heisenberg IPRs in all bases at L=10 and the symmetry-pinned X/Z crossing."""
    fig = new_figure(PAGE_W, 2.0)
    ax_a = add_axes(fig, 0.45, 0.38, 2.95, 1.52)
    ax_b = add_axes(fig, 3.95, 0.38, 2.95, 1.52)
    cfg = DATASETS["heisenberg"]
    folder = cfg["processed"] / folder_name("heisenberg", 10)
    x = read_vector(folder / cfg["x_file"])
    styles = {"IPR XY": "-", "IPR X": ":", "IPR Z": "--"}
    for c in ("IPR XY", "IPR X", "IPR Z"):
        name = curves_from_basis("heisenberg", (c,))[0]
        ax_a.plot(x, read_curve(folder, cfg, name, "raw"), color=BASIS_COLORS[c], ls=styles[c], lw=1.1,
                  label=BASIS_LABELS[c])
    ax_a.axvline(1.0, color="0.35", lw=0.6, ls=":")
    ax_a.set_xlabel(r"$\Delta$")
    ax_a.set_ylabel("IPR")
    inset_text(ax_a, r"$L=10$", "upper right")
    ax_a.legend(loc="center right")
    line_axes(ax_a)
    ipr_overlay(ax_b, "heisenberg", ("IPR X", "IPR Z"), legend_loc="lower right")
    panel_label(ax_a, "(a)")
    panel_label(ax_b, "(b)")
    save(fig, out, "heisenberg_ipr_xz")


def fig_heisenberg_xz_kernel(out: Path, svm_tables: dict) -> None:
    """X/Z-kernel Gram matrices and balanced accuracy of the X/Z and XY/Z kernels."""
    fig = new_figure(PAGE_W, 2.0)
    size, gap, x0, b0, h = 1.0, 0.14, 0.45, 0.38, 1.52
    b = b0 + (h - size) / 2
    axes = [add_axes(fig, x0 + k * (size + gap), b, size, size) for k in range(3)]
    im = None
    for ax, L in zip(axes, (8, 16, 24)):
        g = compute_ipr_gram_matrix("heisenberg", L, basis=("IPR X", "IPR Z"), normalize=True)
        im = gram_panel(ax, g["K"], g["x"])
        inset_text(ax, rf"$L={L}$")
        ax.set_xlabel(r"$\Delta_i$")
        ax.set_xticks([0, 1, 2, 3, 4, 5])
    axes[0].set_ylabel(r"$\Delta_j$")
    for ax in axes[1:]:
        ax.set_yticklabels([])
    cb_left = x0 + 3 * size + 2 * gap + 0.07
    colorbar(fig, im, cb_left, b, size, r"$\tilde K_{\mathrm{IPR}}(\Delta_i,\Delta_j)$")

    ax_b = add_axes(fig, 4.75, b0, 2.2, h)
    kernels = (("X_Z", r"$X$, $Z$", ORANGE), ("XY_Z", r"$XY$, $Z$", GREEN))
    for key, name, color in kernels:
        arr = np.asarray(svm_tables[key])
        ax_b.errorbar(arr[:, 0], arr[:, 1], yerr=arr[:, 2], color=color, marker="o", ms=2.4, lw=0.9,
                      elinewidth=0.5, capsize=1.2)
        line = ax_b.errorbar(arr[:, 0], arr[:, 3], yerr=arr[:, 4], color=color, marker="s", ms=2.4, lw=0.9,
                             mfc="white", elinewidth=0.5, capsize=1.2)
        line[0].set_dashes((4, 2))
    ax_b.axhline(0.5, color="0.35", lw=0.6, ls=":")
    ax_b.set_xlabel(r"$L$")
    ax_b.set_ylabel("Balanced accuracy")
    ax_b.set_xticks(np.arange(6, 25, 2))
    ax_b.set_xlim(5, 25)
    ax_b.set_ylim(0.45, 1.03)
    line_axes(ax_b)
    dashed = Line2D([0], [0], color="0.2", lw=0.9, marker="s", ms=2.4, mfc="white", label="local")
    dashed.set_dashes((4, 2))
    handles = [Line2D([0], [0], color=c, lw=1.2, label=n) for _, n, c in kernels]
    handles += [Line2D([0], [0], color="0.2", lw=0.9, marker="o", ms=2.4, label="global"), dashed]
    ax_b.legend(handles=handles, loc="center right", ncol=2, frameon=True, facecolor="white", edgecolor="none",
                framealpha=0.85)
    panel_label(axes[0], "(a)")
    panel_label(ax_b, "(b)")
    save(fig, out, "heisenberg_xz_kernel")


def bethe_hc1(delta: np.ndarray, kmax: int = 4000) -> np.ndarray:
    """AFM-XY boundary for Delta > 1 in the Pauli-operator convention: 2 sinh(eta) sum_k (-1)^k / cosh(k eta)."""
    out = np.full_like(delta, np.nan, dtype=float)
    for i, d in enumerate(delta):
        if d > 1.0:
            eta = np.arccosh(d)
            # Terms decay as exp(-|k| eta); stop once they are below double precision.
            k = np.arange(-min(kmax, int(40 / eta) + 1), min(kmax, int(40 / eta) + 1) + 1)
            out[i] = 2.0 * np.sinh(eta) * np.sum((-1.0) ** k / np.cosh(k * eta))
    return out


def fig_xxz_field(out: Path) -> None:
    """Field-XXZ phase diagram at L=14 from log10(I_Z / I_XY), with the IPR and Bethe-ansatz boundaries."""
    df = pd.read_csv(REPO_ROOT / "Data" / "Raw" / "Data Heisenberg" / "L=14_with_field" / "ipr_field_sweep_L=14.csv")
    deltas = np.round(np.sort(df["delta"].unique()), 6)
    fields = np.round(np.sort(df["h"].unique()), 6)
    grid = df.assign(r=np.log10(df["ipr_z"] / df["ipr_xy"])).pivot(index="h", columns="delta", values="r")
    ratio = grid.to_numpy()
    step = deltas[1] - deltas[0]
    edges_d = np.append(deltas - step / 2, deltas[-1] + step / 2)
    edges_h = np.append(fields - step / 2, fields[-1] + step / 2)

    # Same colour scale as the original figure: cream -> orange -> green, switching just below zero.
    mid = 0.5 * np.array(matplotlib.colors.to_rgb(ORANGE)) + 0.5 * np.array(matplotlib.colors.to_rgb(GREEN))
    t = 0.67
    cmap = LinearSegmentedColormap.from_list("xxz_field", [
        (0.0, "#FFF2E8"), (0.6 * t, "#F4A261"), (0.8 * t, ORANGE), (t, matplotlib.colors.to_hex(mid)),
        (t + 0.3 * (1 - t), GREEN), (1.0, GREEN)], N=512)

    fig = new_figure(COL_W, 2.75)
    ax = add_axes(fig, 0.42, 0.38, 2.3, 2.3)
    im = ax.pcolormesh(edges_d, edges_h, ratio, cmap=cmap, vmin=np.nanmin(ratio), vmax=np.nanmax(ratio),
                       shading="flat", rasterized=True)

    # IPR boundary: edges of the cells where the XY-basis IPR exceeds the Z-basis IPR.
    xy_local = (ratio < -1e-4).astype(float)
    fine = 20
    mask = np.kron(xy_local, np.ones((fine, fine)))
    xs = np.linspace(edges_d[0], edges_d[-1], mask.shape[1] + 1)
    ys = np.linspace(edges_h[0], edges_h[-1], mask.shape[0] + 1)
    ax.contour(0.5 * (xs[1:] + xs[:-1]), 0.5 * (ys[1:] + ys[:-1]), mask, levels=[0.5], colors="#d62728",
               linewidths=0.8)

    d_sat = np.linspace(-1.0, 1.5, 200)
    d_afm = np.linspace(1.0 + 1e-6, 5.0, 400)
    for sign in (1, -1):
        ax.plot(d_sat, sign * 2 * (1 + d_sat), color="black", lw=0.8, ls="--")
        ax.plot(d_afm, sign * bethe_hc1(d_afm), color="black", lw=0.8, ls="--")
    ax.set_xlim(edges_d[0], edges_d[-1])
    ax.set_ylim(edges_h[0], edges_h[-1])
    ax.set_xlabel(r"$\Delta$")
    ax.set_ylabel(r"$h$")
    ax.set_xticks(np.arange(-4, 5, 2))
    ax.set_yticks(np.arange(-4, 5, 2))
    ax.minorticks_on()
    for x, y, text in ((-3.0, 0.0, "Ferromagnetic"), (0.0, -1.3, "Paramagnetic"), (3.6, 0.0, "Anti-\nferro-\nmagnetic")):
        ax.text(x, y, text, ha="center", va="center", fontsize=7)
    handles = [Line2D([0], [0], color="#d62728", lw=0.8, label="IPR crossing"),
               Line2D([0], [0], color="black", lw=0.8, ls="--", label="Bethe ansatz")]
    ax.legend(handles=handles, loc="upper left", frameon=True, facecolor="white", edgecolor="none", framealpha=0.85)
    cax = add_axes(fig, 2.8, 0.38, 0.07, 2.3)
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(length=2.0, width=0.5)
    cb.outline.set_linewidth(0.5)
    cb.set_label(r"$\log_{10}\,(I^{(Z)}/I^{(XY)})$", labelpad=3)
    save(fig, out, "xxz_field_phase_diagram")


def fig_thermal_l_sweep(out: Path) -> None:
    """HS and IPR Gram matrices at beta=0.2 for increasing system size."""
    fig = new_figure(PAGE_W, 3.35)
    size, gap, row_gap, x0, y0 = 1.3, 0.14, 0.24, 0.62, 0.38
    im = None
    for r, kind in enumerate(("ipr", "hs")):
        for c, L in enumerate((10, 30, 50, 100)):
            ax = add_axes(fig, x0 + c * (size + gap), y0 + r * (size + row_gap), size, size)
            g = compute_thermal_gram_matrix(kind=kind, L=L, beta=0.2, source="raw", normalize=True)
            im = gram_panel(ax, g["K"], g["x"])
            inset_text(ax, rf"$L={L}$")
            if c == 0:
                panel_label(ax, "(a)" if kind == "hs" else "(b)")
            ax.set_xticks([0, 1, 2, 3, 4])
            ax.set_yticks([0, 1, 2, 3, 4])
            if r == 0:
                ax.set_xlabel(r"$h_i$")
            else:
                ax.set_xticklabels([])
            if c == 0:
                ax.set_ylabel(f"{KERNEL_NAMES[kind]}\n" + r"$h_j$")
            else:
                ax.set_yticklabels([])
    colorbar(fig, im, x0 + 4 * size + 3 * gap + 0.08, y0, 2 * size + row_gap, r"$\tilde K(h_i,h_j)$")
    save(fig, out, "thermal_grams_L_sweep_beta0p2")


def fig_thermal_decision(out: Path, svm: dict) -> None:
    """Rescaled global decision functions at L=100 and their steepness at the zero crossing."""
    fig = new_figure(PAGE_W, 1.95)
    w, h, y0 = 1.85, 1.45, 0.4
    ax_a = add_axes(fig, 0.5, y0, w, h)
    ax_b = add_axes(fig, 0.5 + w + 0.18, y0, w, h, sharey=ax_a)
    ax_c = add_axes(fig, 0.5 + 2 * w + 0.75, y0, w, h)
    res = svm[100]
    hc = float(res["phase_boundary"])
    for ax, kind in ((ax_a, "ipr"), (ax_b, "hs")):
        metas = sorted(res["by_kind"][kind]["gram_metadata"], key=lambda m: float(m["beta"]))
        bmin, bmax = float(metas[0]["beta"]), float(metas[-1]["beta"])
        for m in metas:
            beta = float(m["beta"])
            d = m["decision_function"]
            y = np.asarray(d["global_mean"], dtype=float)
            y = y / np.nanmax(np.abs(y))
            ax.plot(beta * (np.asarray(d["x"], dtype=float) - hc), y, color=KERNEL_COLORS[kind], lw=0.8,
                    alpha=0.12 + 0.83 * (beta - bmin) / (bmax - bmin))
        ax.axhline(0.0, color="0.35", lw=0.5, ls=":")
        ax.axvline(0.0, color="0.35", lw=0.5, ls=":")
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-1.08, 1.08)
        ax.set_xlabel(r"$\beta(h-h_c)$")
        inset_text(ax, f"{KERNEL_NAMES[kind]}, " + r"$L=100$")
        line_axes(ax)
    ax_a.set_ylabel(r"$\bar f_\beta$ (normalized)")
    ax_b.tick_params(labelleft=False)
    for kind in ("ipr", "hs"):
        for L in (20, 50, 100):
            metas = sorted(svm[L]["by_kind"][kind]["gram_metadata"], key=lambda m: float(m["beta"]))
            bs = [float(m["beta"]) for m in metas]
            s = [decision_sharpness(np.asarray(m["decision_function"]["x"], dtype=float),
                                    np.asarray(m["decision_function"]["global_mean"], dtype=float), hc)
                 for m in metas]
            ax_c.semilogy(bs, s, color=KERNEL_COLORS[kind], marker=L_MARKERS[L], ms=2.2, lw=0.9)
    ax_c.set_xlim(2.05, 0.0)
    ax_c.set_xlabel(r"$\beta$")
    ax_c.set_ylabel(r"$S(\beta)$")
    line_axes(ax_c)
    kernel_handles = [Line2D([0], [0], color=KERNEL_COLORS[k], lw=1.2, label=KERNEL_NAMES[k]) for k in ("ipr", "hs")]
    size_handles = [Line2D([0], [0], color="0.25", marker=L_MARKERS[L], ls="None", ms=3, label=rf"$L={L}$")
                    for L in (20, 50, 100)]
    ax_c.legend(handles=kernel_handles + size_handles, loc="lower left", ncol=2)
    for ax, lab in zip((ax_a, ax_b, ax_c), "abc"):
        panel_label(ax, f"({lab})")
    save(fig, out, "thermal_decision_functions")


def delta_s_grid(L: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = read_thermal_ipr_data(L=L, source="raw")
    df = df[["h", "beta", "ipr_z", "ipr_x"]].apply(pd.to_numeric, errors="coerce").dropna()
    df = df[(df["beta"] > 0) & (df["ipr_z"] > 0) & (df["ipr_x"] > 0)].copy()
    lz, lx = np.log10(df["ipr_z"]), np.log10(df["ipr_x"])
    den = np.abs(lz) + np.abs(lx)
    df = df[den != 0].copy()
    df["ds"] = (lz - lx)[den != 0] / den[den != 0]
    grid = df.pivot(index="beta", columns="h", values="ds").sort_index().sort_index(axis=1)
    return grid.columns.to_numpy(float), grid.index.to_numpy(float), grid.to_numpy(float)


def fig_thermal_phase_diagram(out: Path) -> None:
    """Delta S_P in the (h, beta) plane with the expected crossover T = |h - h_c|."""
    fig = new_figure(PAGE_W, 2.05)
    w, h, y0, x0, gap = 1.85, 1.5, 0.4, 0.5, 0.14
    data = [(L, *delta_s_grid(L)) for L in (10, 50, 100)]
    vmax = max(np.nanmax(np.abs(d)) for *_, d in data)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    mesh = None
    for k, (L, hh, bb, ds) in enumerate(data):
        ax = add_axes(fig, x0 + k * (w + gap), y0, w, h)
        mesh = ax.pcolormesh(hh, bb, ds, shading="nearest", cmap=DELTA_S_CMAP, norm=norm, rasterized=True)
        b_line = np.linspace(max(bb.min(), 1.0 / max(hh.max() - 1.0, 1.0 - hh.min())), min(6.0, bb.max()), 300)
        for sign in (1, -1):
            hl = 1.0 + sign / b_line
            m = (hl >= hh.min()) & (hl <= hh.max())
            ax.plot(hl[m], b_line[m], "k--", lw=0.8)
        ax.set_xlim(hh.min(), hh.max())
        ax.set_ylim(bb.min(), min(6.0, bb.max()))
        ax.invert_yaxis()
        ax.minorticks_on()
        ax.set_xlabel(r"$h$")
        inset_text(ax, rf"$L={L}$", "upper right")
        panel_label(ax, f"({'abc'[k]})")
        if k == 0:
            ax.set_ylabel(r"$\beta$")
        else:
            ax.set_yticklabels([])
    cax = add_axes(fig, x0 + 3 * w + 2 * gap + 0.08, y0, 0.07, h)
    cb = fig.colorbar(mesh, cax=cax)
    cb.ax.tick_params(length=2.0, width=0.5)
    cb.outline.set_linewidth(0.5)
    cb.set_label(r"$\Delta S_P(\rho_\beta(h))$", labelpad=3)
    save(fig, out, "thermal_phase_diagram")


def fig_analytical(out: Path) -> None:
    """Analytical normalized kernels between h=0 and h->infinity Gibbs states."""
    def z_x(b, L):
        return (1.0 + np.exp(-2.0 * b)) ** L

    def z_z(b, L):
        e = np.exp(-2.0 * b)
        return (1.0 + e) ** L + (1.0 - e) ** L

    def k_ipr(b, L):
        iz, ix = z_z(2 * b, L) / z_z(b, L) ** 2, z_x(2 * b, L) / z_x(b, L) ** 2
        return 2.0 ** (-L) * (iz + ix) / (np.sqrt(4.0 ** (-L) + iz ** 2) * np.sqrt(4.0 ** (-L) + ix ** 2))

    def k_hs(b, L):
        return 2.0 ** (-L) * z_z(b, L) * z_x(b, L) / (np.sqrt(z_z(2 * b, L)) * np.sqrt(z_x(2 * b, L)))

    fig = new_figure(PAGE_W, 1.9)
    w, h, y0, x0, gap = 1.95, 1.4, 0.4, 0.55, 0.18
    beta = np.linspace(0.0, 1.0, 500)
    for k, L in enumerate((10, 50, 300)):
        ax = add_axes(fig, x0 + k * (w + gap), y0, w, h)
        ax.plot(beta, k_ipr(beta, L), color=GREEN, lw=1.1, label=r"$\tilde K_{\mathrm{IPR}}$")
        ax.plot(beta, k_hs(beta, L), color=ORANGE, lw=1.1, label=r"$\tilde K_{\mathrm{HS}}$")
        ax.set_xlim(1.0, 0.0)
        ax.set_ylim(-0.03, 1.05)
        ax.set_xlabel(r"$\beta$")
        inset_text(ax, rf"$L={L}$", "lower left")
        panel_label(ax, f"({'abc'[k]})")
        line_axes(ax)
        if k == 0:
            ax.set_ylabel("Kernel value")
        else:
            ax.set_yticklabels([])
        if k == 2:
            ax.legend(loc="center left")
    save(fig, out, "analytical_kernels")


# ---------------------------------------------------------------- schematic figures
SPIN_COLORS = {"0": ORANGE, "1": LIGHT_BLUE, "+": GREEN}
# Schematics set their math in Computer Modern, the font of the equations in the manuscript.
SCHEMATIC_RC = {"mathtext.fontset": "cm"}


def lighten(color: str, amount: float) -> tuple[float, float, float]:
    """Blend a color with white; amount=0 keeps it, amount=1 gives white."""
    r, g, b = to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def canvas(fig: plt.Figure) -> plt.Axes:
    """Full-figure axes whose data coordinates are inches, for drawing schematics."""
    W, H = fig.get_size_inches()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], lw: float = 0.8) -> None:
    ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "-|>", "lw": lw, "color": "black",
                                                      "mutation_scale": 7, "shrinkA": 0, "shrinkB": 0})


def spin(ax: plt.Axes, x: float, y: float, state: str, r: float = 0.05) -> None:
    c = SPIN_COLORS[state]
    ax.add_patch(Circle((x, y), r, fc=lighten(c, 0.35), ec=c, lw=0.6, zorder=3))


def ring(ax: plt.Axes, center: tuple[float, float], radius: float, states: str, label: str = "") -> None:
    ax.add_patch(Circle(center, radius, fill=False, ec="0.35", lw=0.6, zorder=2))
    for k, s in enumerate(states):
        a = np.deg2rad(90 - 360 * k / len(states))
        spin(ax, center[0] + radius * np.cos(a), center[1] + radius * np.sin(a), s)
    if label:
        ax.text(*center, label, ha="center", va="center")


def block_arrow(ax: plt.Axes, xy: tuple[float, float], angle: float, text: str, lw: float = 0.6,
                ls: str = "-") -> None:
    """Double-headed block arrow carrying a kernel value, drawn along `angle` (degrees)."""
    ax.text(*xy, text, rotation=angle, rotation_mode="anchor", ha="center", va="center", fontsize=7, zorder=4,
            bbox={"boxstyle": "darrow,pad=0.22", "fc": "white", "ec": "black", "lw": lw, "ls": ls})


def fig_ising_example(out: Path) -> None:
    """Spin configurations of the TFIM and the values the IPR and HS kernels assign to pairs of them."""
    W, H = COL_W, 3.05
    with plt.rc_context(SCHEMATIC_RC):
        _draw_ising_example(out, W, H)


def _draw_ising_example(out: Path, W: float, H: float) -> None:
    fig = new_figure(W, H)
    ax = canvas(fig)
    ax.text(W / 2, H - 0.2, r"$H=-J\,\sum_i Z_iZ_{i+1}-h\,\sum_i X_i$", ha="center", va="center",
            bbox={"boxstyle": "round,pad=0.35", "fc": "0.93", "ec": "0.45", "lw": 0.6})

    radius, n = 0.44, 12
    top, left, right = (W / 2, 2.05), (W / 2 - 0.9, 0.55), (W / 2 + 0.9, 0.55)
    ring(ax, top, radius, "0000" + "1" + "0" * (n - 5))
    ring(ax, left, radius, "0" * n, r"$h<J$")
    ring(ax, right, radius, "+" * n, r"$h>J$")

    def pair(a, b, offset, ipr_text, ipr_lw):
        (xa, ya), (xb, yb) = a, b
        d = np.hypot(xb - xa, yb - ya)
        u = np.array([(xb - xa) / d, (yb - ya) / d])
        nrm = np.array([-u[1], u[0]])
        mid = np.array([(xa + xb) / 2, (ya + yb) / 2])
        angle = np.degrees(np.arctan2(u[1], u[0]))
        block_arrow(ax, mid + offset * nrm, angle, ipr_text, lw=ipr_lw)
        block_arrow(ax, mid - offset * nrm, angle, r"$K_{\mathrm{HS}}\approx 0$", ls=":")

    pair(left, top, 0.13, r"$K_{\mathrm{IPR}}\approx 1$", 1.4)
    pair(top, right, 0.13, r"$K_{\mathrm{IPR}}\approx 0$", 0.6)
    pair(left, right, 0.145, r"$K_{\mathrm{IPR}}\approx 0$", 0.6)

    # Legend for the single-spin states.
    x0, y0 = 0.12, 1.75
    ax.add_patch(FancyBboxPatch((x0, y0), 0.5, 0.56, boxstyle="round,pad=0.02", fc="0.93", ec="0.45", lw=0.6))
    for k, (s, lab) in enumerate((("0", r"$|0\rangle$"), ("1", r"$|1\rangle$"), ("+", r"$|{+}\rangle$"))):
        y = y0 + 0.46 - 0.18 * k
        spin(ax, x0 + 0.12, y, s)
        ax.text(x0 + 0.22, y, lab, va="center", fontsize=7)
    save(fig, out, "ising_example")


def data_stack(ax: plt.Axes, x: float, y: float, w: float, n: int, slab: float, gap: float) -> float:
    """Stack of slabs standing for a dataset of states, with an atom icon on its front; returns its top."""
    depth = np.array([0.05, 0.035])
    for k in range(n):
        yk = y + k * (slab + gap)
        front = Rectangle((x, yk), w, slab, fc="0.18", ec=GREEN, lw=0.5)
        side = Polygon([(x + w, yk), (x + w, yk + slab), (x + w + depth[0], yk + slab + depth[1]),
                        (x + w + depth[0], yk + depth[1])], fc="0.35", ec=GREEN, lw=0.5)
        topf = Polygon([(x, yk + slab), (x + w, yk + slab), (x + w + depth[0], yk + slab + depth[1]),
                        (x + depth[0], yk + slab + depth[1])], fc="0.35", ec=GREEN, lw=0.5)
        for p in (side, topf, front):
            ax.add_patch(p)
    top = y + n * (slab + gap)
    cx, cy, half = x + w / 2, (y + top) / 2, 0.15
    ax.add_patch(FancyBboxPatch((cx - half, cy - half), 2 * half, 2 * half, boxstyle="round,pad=0.01",
                                fc="0.94", ec=GREEN, lw=0.6, zorder=5))
    for a in (0, 60, 120):
        ax.add_patch(Ellipse((cx, cy), 0.26, 0.09, angle=a, fill=False, ec=ORANGE, lw=0.8, zorder=6))
    ax.add_patch(Circle((cx, cy), 0.02, fc=ORANGE, ec="none", zorder=6))
    return top


def circuit(ax: plt.Axes, x: float, yc: float, s: int) -> float:
    """Measurement circuit in basis U_s; returns the x position where the output label ends."""
    wires = (yc + 0.12, yc, yc - 0.12)
    ax.text(x, yc, r"$\psi$", ha="right", va="center")
    for y in wires:
        ax.plot([x + 0.05, x + 0.72], [y, y], color="black", lw=0.6)
    ax.add_patch(Rectangle((x + 0.17, yc - 0.17), 0.2, 0.34, fc="white", ec="black", lw=0.6, zorder=3))
    ax.text(x + 0.27, yc, rf"$U_{s}$", ha="center", va="center", fontsize=7, zorder=4)
    for y in wires:
        mx = x + 0.5
        ax.add_patch(Rectangle((mx - 0.07, y - 0.045), 0.14, 0.09, fc="white", ec="black", lw=0.5, zorder=3))
        ax.add_patch(Arc((mx, y - 0.025), 0.09, 0.07, theta1=0, theta2=180, ec="black", lw=0.5, zorder=4))
        ax.plot([mx, mx + 0.04], [y - 0.025, y + 0.035], color="black", lw=0.5, zorder=4)
    ax.text(x + 0.77, yc, rf"$I^{{({s})}}(\psi)$", ha="left", va="center")
    return x + 1.2


def fig_scheme(out: Path) -> None:
    """IPR-kernel pipeline: states, IPRs in several bases, feature vector, kernel, SVM."""
    W, H = PAGE_W, 1.75
    with plt.rc_context(SCHEMATIC_RC):
        _draw_scheme(out, W, H)


def _draw_scheme(out: Path, W: float, H: float) -> None:
    fig = new_figure(W, H)
    ax = canvas(fig)

    top = data_stack(ax, 0.1, 0.18, 0.42, 15, 0.065, 0.017)
    ax.text(0.33, top + 0.08, r"States $\psi$", ha="center", va="bottom")

    rows = (1.4, 0.87, 0.34)
    y_mid = rows[1]
    ax.plot([0.6, 0.78], [y_mid, y_mid], color="black", lw=0.8)
    ax.plot([0.78, 0.78], [rows[2], rows[0]], color="black", lw=0.8)
    x_out = 0.0
    for s, yc in enumerate(rows, start=1):
        arrow(ax, (0.78, yc), (0.98, yc))
        x_out = circuit(ax, 1.12, yc, s)
    for yc in rows:
        ax.plot([x_out - 0.05, x_out + 0.1], [yc, yc], color="black", lw=0.8)
    ax.plot([x_out + 0.1, x_out + 0.1], [rows[2], rows[0]], color="black", lw=0.8)
    arrow(ax, (x_out + 0.1, y_mid), (x_out + 0.3, y_mid))

    x = x_out + 0.36
    t = ax.text(x, y_mid, r"$\mathbf{I}(\psi)=\left(I^{(1)}(\psi),\,I^{(2)}(\psi),\,I^{(3)}(\psi)\right)$",
                ha="left", va="center")
    fig.canvas.draw()
    x = ax.transData.inverted().transform(t.get_window_extent().get_points())[1, 0] + 0.06
    arrow(ax, (x, y_mid), (x + 0.22, y_mid))
    t = ax.text(x + 0.28, y_mid, r"$K_{\mathrm{IPR}}(\psi,\phi)$", ha="left", va="center")
    fig.canvas.draw()
    x = ax.transData.inverted().transform(t.get_window_extent().get_points())[1, 0] + 0.06
    arrow(ax, (x, y_mid), (x + 0.22, y_mid))

    # SVM panel: three classes separated by linear boundaries.
    svm = add_axes(fig, x + 0.32, 0.2, W - x - 0.42, 1.25)
    rng = np.random.default_rng(7)
    r = np.sqrt(rng.uniform(0, 1, 700))
    phi = rng.uniform(0, 2 * np.pi, 700)
    px, py = r * np.cos(phi), r * np.sin(phi)
    edges = np.deg2rad([100.0, -20.0, -140.0])
    sector = np.digitize(np.mod(phi - edges[2], 2 * np.pi), np.sort(np.mod(edges - edges[2], 2 * np.pi)))
    # Keep a white margin around each boundary ray (distance to the segment from the center to the rim).
    margin = 0.06
    keep = np.ones_like(px, dtype=bool)
    for e in edges:
        t = px * np.cos(e) + py * np.sin(e)
        dist = np.where(t > 0, np.abs(px * np.sin(e) - py * np.cos(e)), np.hypot(px, py))
        keep &= dist > margin
    for k, c in enumerate((GREEN, LIGHT_BLUE, ORANGE)):
        m = keep & (sector == k + 1)
        svm.scatter(px[m], py[m], s=0.8, color=c, linewidths=0)
    for e in edges:
        svm.plot([0, np.cos(e)], [0, np.sin(e)], color="black", lw=0.6, solid_capstyle="butt")
    svm.set_xlim(-1.25, 1.35)
    svm.set_ylim(-1.25, 1.25)
    svm.set_xticks([])
    svm.set_yticks([])
    for side in ("top", "right"):
        svm.spines[side].set_visible(False)
    svm.set_title("SVM", pad=3)
    save(fig, out, "scheme")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--thermal-svm", type=Path, default=HERE / "thermal_svm_results.pkl")
    parser.add_argument("--only", nargs="*", default=None, help="subset of figure function names")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    apply_style()

    jobs = {
        "fig_tfim": lambda: fig_tfim(args.out_dir, json.load(open(HERE / "tfim_svm_tables.json"))),
        "fig_heisenberg": lambda: fig_heisenberg(args.out_dir),
        "fig_thermal": lambda: fig_thermal(args.out_dir, load_thermal_svm(args.thermal_svm)),
        "fig_tfim_grams": lambda: fig_tfim_grams(args.out_dir),
        "fig_heisenberg_xz": lambda: fig_heisenberg_xz(args.out_dir),
        "fig_heisenberg_xz_kernel": lambda: fig_heisenberg_xz_kernel(
            args.out_dir, json.load(open(HERE / "xxz_svm_tables.json"))),
        "fig_xxz_field": lambda: fig_xxz_field(args.out_dir),
        "fig_thermal_l_sweep": lambda: fig_thermal_l_sweep(args.out_dir),
        "fig_thermal_decision": lambda: fig_thermal_decision(args.out_dir, load_thermal_svm(args.thermal_svm)),
        "fig_thermal_phase_diagram": lambda: fig_thermal_phase_diagram(args.out_dir),
        "fig_analytical": lambda: fig_analytical(args.out_dir),
        "fig_scheme": lambda: fig_scheme(args.out_dir),
        "fig_ising_example": lambda: fig_ising_example(args.out_dir),
    }
    for name, job in jobs.items():
        if args.only is None or name in args.only:
            job()


if __name__ == "__main__":
    main()
