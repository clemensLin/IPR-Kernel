from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

from data_config import (
    DATASETS,
    available_l_values,
    available_thermal_betas,
    available_thermal_l_values,
    curves_from_basis,
    folder_name,
    normalize_dataset,
    normalize_thermal_kind,
    normalize_thermal_source,
    read_curve,
    read_thermal_hs_raw_data,
    read_thermal_ipr_data,
    read_vector,
    thermal_grid_for_size,
    thermal_ipr_path,
    thermal_hs_path,
)
from plot_ipr_curves import (
    set_ipr_plot_style,
)


IPR_GRAM_CMAP = LinearSegmentedColormap.from_list(
    "ipr_orange_green",
    ["#fff2e8", "#f4a261", "#d95f02", "#1b9e77"],
    N=512,
)

NORMALIZED_GRAM_TICKS = np.linspace(0.0, 1.0, 6)


def _gram_norm(K, normalized: bool, vmin: float | None = None, vmax: float | None = None):
    """Return the fixed [0, 1] color normalization used for all Gram plots."""
    return Normalize(vmin=0.0, vmax=1.0)


def _set_normalized_cbar_ticks(cbar, normalized: bool):
    """Use fixed tick labels for Gram matrix colorbars."""
    cbar.set_ticks(NORMALIZED_GRAM_TICKS)
    cbar.set_ticklabels([f"{tick:.1f}" for tick in NORMALIZED_GRAM_TICKS])


def _axis_labels(dataset: str, normalized: bool = False) -> tuple[str, str, str]:
    """Return axis labels and the raw/normalized IPR-kernel colorbar label."""
    kernel = r"\tilde{k}" if normalized else "k"
    if dataset == "heisenberg":
        return r"$\Delta_a$", r"$\Delta_b$", rf"${kernel}_{{\mathrm{{IPR}}}}(\Delta_a,\Delta_b)$"
    return r"$h_a$", r"$h_b$", rf"${kernel}_{{\mathrm{{IPR}}}}(h_a,h_b)$"


def _annotate_gram_panel(ax, label: str, fontsize: int = 11) -> None:
    """Place the panel identifier inside the Gram matrix."""
    ax.text(
        0.04,
        0.92,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
    )


def compute_ipr_gram_matrix(
    dataset: str,
    L: int | str,
    basis: list[str] | tuple[str, ...] = ("IPR X", "IPR Z"),
    normalize: bool = False,
):
    """Compute an IPR Gram matrix from processed synthetic data.

    For each sampled parameter value, the feature vector is built from the
    selected IPR curves, for example [IPR_X, IPR_Z]. The Gram matrix is then
    K_ab = features_a dot features_b.
    """
    dataset = normalize_dataset(dataset)
    cfg = DATASETS[dataset]
    curves = curves_from_basis(dataset, basis)

    folder = cfg["processed"] / folder_name(dataset, L)
    if not folder.exists():
        raise FileNotFoundError(f"Processed data folder not found: {folder}")

    x = read_vector(folder / cfg["x_file"])
    feature_columns = [read_curve(folder, cfg, curve, "raw") for curve in curves]
    features = np.column_stack(feature_columns)
    K = features @ features.T

    if normalize:
        norms = np.linalg.norm(features, axis=1)
        denom = np.outer(norms, norms)
        K = np.divide(K, denom, out=np.zeros_like(K), where=denom > 0)

    return {
        "dataset": dataset,
        "L": int(folder_name(dataset, L).split("=", 1)[1].split("x", 1)[0]),
        "folder": folder_name(dataset, L),
        "x": x,
        "features": features,
        "I_vec": features,
        "K": K,
        "curves": curves,
        "basis": basis,
        "feature_scale": "raw",
        "normalize": normalize,
    }


def plot_ipr_gram_matrix(
    dataset: str,
    L: int | str,
    basis: list[str] | tuple[str, ...] = ("IPR X", "IPR Z"),
    normalize: bool = False,
    figsize=(6.4, 5.4),
    show_title: bool = True,
    ax=None,
    vmin: float | None = None,
    vmax: float | None = None,
):
    """Plot one processed IPR Gram matrix."""
    gram = compute_ipr_gram_matrix(
        dataset=dataset,
        L=L,
        basis=basis,
        normalize=normalize,
    )
    dataset = gram["dataset"]
    cfg = DATASETS[dataset]
    x = gram["x"]
    K = gram["K"]

    set_ipr_plot_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.figure

    norm = _gram_norm(K, normalized=bool(normalize), vmin=vmin, vmax=vmax)

    im = ax.imshow(
        K,
        origin="lower",
        extent=[x.min(), x.max(), x.min(), x.max()],
        aspect="auto",
        cmap=IPR_GRAM_CMAP,
        norm=norm,
        interpolation="nearest",
    )

    xlabel, ylabel, cbar_label = _axis_labels(dataset, normalized=bool(normalize))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.minorticks_on()
    ax.axvline(cfg["critical_point"], color="0.2", lw=1.0, ls=":")
    ax.axhline(cfg["critical_point"], color="0.2", lw=1.0, ls=":")

    if show_title:
        norm_txt = "normalized" if normalize else "unnormalized"
        basis_txt = "/".join(curve.replace("IPR ", "") for curve in gram["curves"])
        ax.set_title(
            f"IPR Gram matrix ({cfg['title_name']} L={gram['L']}, {basis_txt}, raw, {norm_txt})",
            pad=8,
        )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    _set_normalized_cbar_ticks(cbar, normalized=bool(normalize))
    cbar.set_label(cbar_label)
    return fig, ax, gram


def plot_ipr_gram_matrices_up_to(
    dataset: str,
    basis: list[str] | tuple[str, ...] = ("IPR X", "IPR Z"),
    L_max: int | None = None,
    L_min: int | None = None,
    L_list: list[int] | None = None,
    normalize: bool = False,
    ncols: int = 4,
    panel_size=(3.1, 2.9),
    show_title: bool = True,
):
    """Plot processed IPR Gram matrices for several system sizes in one figure."""
    dataset = normalize_dataset(dataset)

    if L_list is None:
        values = available_l_values(dataset, n_min=L_min, n_max=L_max)
    else:
        values = sorted({int(v) for v in L_list})
        available = set(available_l_values(dataset))
        missing = [v for v in values if v not in available]
        if missing:
            raise ValueError(f"L values not found in processed data: {missing}")

    if not values:
        raise ValueError("No processed datasets found for the requested L values.")

    grams = [
        compute_ipr_gram_matrix(
            dataset=dataset,
            L=L,
            basis=basis,
            normalize=normalize,
        )
        for L in values
    ]

    norm = Normalize(vmin=0.0, vmax=1.0)

    set_ipr_plot_style()
    n = len(values)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(panel_size[0] * ncols, panel_size[1] * nrows),
        constrained_layout=False,
    )
    axes = np.atleast_1d(axes).ravel()

    cfg = DATASETS[dataset]
    xlabel, ylabel, cbar_label = _axis_labels(dataset, normalized=bool(normalize))
    last_im = None

    for ax, gram in zip(axes, grams):
        x = gram["x"]
        im = ax.imshow(
            gram["K"],
            origin="lower",
            extent=[x.min(), x.max(), x.min(), x.max()],
            aspect="auto",
            cmap=IPR_GRAM_CMAP,
            norm=norm,
            interpolation="nearest",
        )
        last_im = im
        _annotate_gram_panel(ax, rf"$L={gram['L']}$")
        ax.tick_params(direction="in", which="both", top=True, right=True, labelsize=8)
        ax.minorticks_on()
        ax.axvline(cfg["critical_point"], color="0.2", lw=0.8, ls=":")
        ax.axhline(cfg["critical_point"], color="0.2", lw=0.8, ls=":")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)

    for ax in axes[len(grams):]:
        ax.axis("off")

    fig.tight_layout(rect=[0.0, 0.0, 0.90, 1.0])
    if last_im is not None:
        cax = fig.add_axes([0.915, 0.22, 0.015, 0.56])
        cbar = fig.colorbar(last_im, cax=cax)
        _set_normalized_cbar_ticks(cbar, normalized=bool(normalize))
        cbar.set_label(cbar_label)

    if show_title:
        norm_txt = "normalized" if normalize else "unnormalized"
        basis_txt = "/".join(curve.replace("IPR ", "") for curve in grams[0]["curves"])
        title_span = f"L in {values}" if L_list is not None else f"{values[0]} <= L <= {values[-1]}"
        fig.suptitle(
            f"IPR Gram matrices for {cfg['title_name']} ({title_span}, {basis_txt}, raw, {norm_txt})",
            fontsize=12,
        )

    return fig, axes, grams






#---------------------- THERMAL IPR GRAMS ----------------------


def _normalize_feature_gram(K: np.ndarray) -> np.ndarray:
    diag = np.diag(K)
    denom = np.sqrt(np.outer(diag, diag))
    return np.divide(K, denom, out=np.zeros_like(K), where=denom > 0)


def compute_thermal_ipr_gram_matrix(
    L: int,
    beta: float,
    source: str = "processed",
    normalize: bool = True,
):
    """Compute a thermal TFIM IPR Gram matrix from raw or processed data."""
    source = normalize_thermal_source(source)
    path = thermal_ipr_path(L=L, source=source)
    df = read_thermal_ipr_data(L=L, source=source)
    sub = df[np.isclose(df["beta"], float(beta))].sort_values("h")
    if sub.empty:
        raise ValueError(f"thermal IPR L={int(L)}, source={source}: beta={beta} not found.")

    x = sub["h"].to_numpy(dtype=float)
    features = sub[["ipr_x", "ipr_z"]].to_numpy(dtype=float)
    K = features @ features.T
    if normalize:
        K = _normalize_feature_gram(K)

    return {
        "kind": "ipr",
        "source": source,
        "L": int(L),
        "beta": float(beta),
        "x": x,
        "y": x,
        "features": features,
        "I_vec": features,
        "K": K,
        "curves": ["IPR X", "IPR Z"],
        "basis": ["IPR X", "IPR Z"],
        "feature_scale": "raw",
        "normalize": bool(normalize),
        "path": path,
    }


def compute_thermal_hs_gram_matrix(
    L: int,
    beta: float,
    source: str = "processed",
):
    """Load a thermal TFIM HS Gram matrix from raw or processed data."""
    source = normalize_thermal_source(source)

    if source == "processed":
        path = thermal_hs_path(L=L, beta=beta, source=source)
        if not path.exists():
            raise FileNotFoundError(path)
        K = np.load(path, mmap_mode="r")
        x = thermal_grid_for_size(K.shape[0])
        y = thermal_grid_for_size(K.shape[1])
    else:
        path = thermal_hs_path(L=L, source=source)
        df = read_thermal_hs_raw_data(L=L)
        sub = df[np.isclose(df["beta"], float(beta))]
        if sub.empty:
            raise ValueError(f"{path.name}: beta={beta} not found.")
        x = np.sort(sub["h_a"].unique().astype(float))
        y = np.sort(sub["h_b"].unique().astype(float))
        K = (
            sub.pivot(index="h_a", columns="h_b", values="kernel_real")
            .reindex(index=x, columns=y)
            .to_numpy(dtype=float)
        )

    return {
        "kind": "hs",
        "source": source,
        "L": int(L),
        "beta": float(beta),
        "x": x,
        "y": y,
        "K": K,
        "path": path,
    }


def compute_thermal_gram_matrix(
    kind: str,
    L: int,
    beta: float,
    source: str = "processed",
    normalize: bool | None = None,
):
    """Compute/load one thermal TFIM Gram matrix for IPR or HS."""
    kind = normalize_thermal_kind(kind)
    if kind == "hs" and normalize is False:
        raise ValueError("Thermal HS Gram matrices are only available in normalized form; use normalize=None or True.")
    if kind == "ipr":
        return compute_thermal_ipr_gram_matrix(
            L=L,
            beta=beta,
            source=source,
            normalize=True if normalize is None else bool(normalize),
        )
    return compute_thermal_hs_gram_matrix(L=L, beta=beta, source=source)


def _thermal_cbar_label(kind: str, normalized: bool = False) -> str:
    """Return the raw/normalized thermal-kernel colorbar label."""
    kernel = r"\tilde{k}" if normalized else "k"
    if kind == "ipr":
        return rf"${kernel}_{{\mathrm{{IPR}}}}(h_a,h_b)$"
    return rf"${kernel}_{{\mathrm{{HS}}}}(h_a,h_b)$"


def _plot_thermal_ipr_gram_on_axis(ax, gram, norm=None):
    x = gram["x"]
    y = gram["y"]
    im = ax.imshow(
        gram["K"],
        origin="lower",
        extent=[float(y.min()), float(y.max()), float(x.min()), float(x.max())],
        aspect="auto",
        cmap=IPR_GRAM_CMAP,
        norm=norm,
        interpolation="nearest",
    )
    ax.axvline(1.0, color="0.2", lw=0.9, ls=":")
    ax.axhline(1.0, color="0.2", lw=0.9, ls=":")
    ax.set_xlabel(r"$h_a$")
    ax.set_ylabel(r"$h_b$")
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.minorticks_on()
    return im


def _plot_thermal_gram_on_axis(ax, gram, norm=None):
    im = _plot_thermal_ipr_gram_on_axis(ax, gram, norm=norm)
    return im


def plot_thermal_ipr_gram_matrix(
    L: int,
    beta: float,
    source: str = "processed",
    normalize: bool = True,
    figsize=(6.4, 5.4),
    show_title: bool = True,
    ax=None,
    vmin: float | None = None,
    vmax: float | None = None,
):
    """Plot one thermal TFIM IPR Gram matrix from raw or processed data."""
    gram = compute_thermal_ipr_gram_matrix(
        L=L,
        beta=beta,
        source=source,
        normalize=normalize,
    )
    K = gram["K"]

    set_ipr_plot_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.figure

    norm = _gram_norm(K, normalized=bool(normalize), vmin=vmin, vmax=vmax)

    im = _plot_thermal_ipr_gram_on_axis(ax, gram, norm=norm)

    if show_title:
        norm_txt = "normalized" if gram["normalize"] else "unnormalized"
        ax.set_title(
            f"Thermal TFIM IPR Gram (L={gram['L']}, beta={gram['beta']:g}, {gram['source']}, {norm_txt})",
            pad=8,
        )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    _set_normalized_cbar_ticks(cbar, normalized=bool(normalize))
    cbar.set_label(_thermal_cbar_label("ipr", normalized=bool(normalize)))
    return fig, ax, gram


def plot_thermal_gram_matrix(
    kind: str,
    L: int,
    beta: float,
    source: str = "processed",
    normalize: bool | None = None,
    figsize=(6.4, 5.4),
    show_title: bool = True,
    ax=None,
    vmin: float | None = None,
    vmax: float | None = None,
):
    """Plot one thermal TFIM IPR or HS Gram matrix."""
    gram = compute_thermal_gram_matrix(
        kind=kind,
        L=L,
        beta=beta,
        source=source,
        normalize=normalize,
    )
    kind = gram["kind"]
    K = gram["K"]

    set_ipr_plot_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.figure

    normalized_for_colorbar = kind == "hs" or bool(gram["normalize"])
    norm = _gram_norm(K, normalized=normalized_for_colorbar, vmin=vmin, vmax=vmax)

    im = _plot_thermal_gram_on_axis(ax, gram, norm=norm)

    if show_title:
        norm_txt = "normalized" if kind == "hs" or gram["normalize"] else "as stored"
        ax.set_title(
            f"Thermal TFIM {kind.upper()} Gram (L={gram['L']}, beta={gram['beta']:g}, {gram['source']}, {norm_txt})",
            pad=8,
        )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    _set_normalized_cbar_ticks(cbar, normalized=normalized_for_colorbar)
    cbar.set_label(_thermal_cbar_label(kind, normalized=normalized_for_colorbar))
    return fig, ax, gram


def plot_thermal_ipr_gram_matrices(
    L: int,
    betas: list[float] | tuple[float, ...],
    source: str = "processed",
    normalize: bool = True,
    ncols: int = 3,
    panel_size=(3.1, 2.9),
    show_title: bool = True,
):
    """Plot thermal TFIM IPR Gram matrices for selected beta values at fixed L."""
    source = normalize_thermal_source(source)
    available_l = set(available_thermal_l_values(kind="ipr", source=source))
    if int(L) not in available_l:
        raise ValueError(f"L={int(L)} not found in {source} thermal IPR data. Available: {sorted(available_l)}")

    beta_values = [float(beta) for beta in betas]
    if not beta_values:
        raise ValueError("betas must contain at least one value.")

    grams = [
        compute_thermal_ipr_gram_matrix(
            L=L,
            beta=beta,
            source=source,
            normalize=normalize,
        )
        for beta in beta_values
    ]

    norm = Normalize(vmin=0.0, vmax=1.0)

    set_ipr_plot_style()
    nrows = int(np.ceil(len(grams) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(panel_size[0] * ncols, panel_size[1] * nrows),
        constrained_layout=False,
    )
    axes = np.atleast_1d(axes).ravel()

    last_im = None
    for ax, gram in zip(axes, grams):
        last_im = _plot_thermal_ipr_gram_on_axis(ax, gram, norm=norm)
        _annotate_gram_panel(ax, rf"$\beta={gram['beta']:g}$")

    for ax in axes[len(grams):]:
        ax.axis("off")

    fig.tight_layout(rect=[0.0, 0.0, 0.90, 1.0])
    if last_im is not None:
        cax = fig.add_axes([0.915, 0.22, 0.015, 0.56])
        cbar = fig.colorbar(last_im, cax=cax)
        _set_normalized_cbar_ticks(cbar, normalized=bool(normalize))
        cbar.set_label(_thermal_cbar_label("ipr", normalized=bool(normalize)))

    if show_title:
        norm_txt = "normalized" if grams[0]["normalize"] else "unnormalized"
        fig.suptitle(
            f"Thermal TFIM IPR Gram matrices (L={int(L)}, {source}, {norm_txt})",
            fontsize=12,
        )

    return fig, axes, grams


def plot_thermal_gram_matrices(
    kind: str,
    L: int,
    betas: list[float] | tuple[float, ...],
    source: str = "processed",
    normalize: bool | None = None,
    ncols: int = 3,
    panel_size=(3.1, 2.9),
    show_title: bool = True,
):
    """Plot thermal TFIM Gram matrices for selected beta values at fixed L."""
    kind = normalize_thermal_kind(kind)
    source = normalize_thermal_source(source)
    available_l = set(available_thermal_l_values(kind=kind, source=source))
    if int(L) not in available_l:
        raise ValueError(f"L={int(L)} not found in {source} thermal {kind.upper()} data. Available: {sorted(available_l)}")

    beta_values = [float(beta) for beta in betas]
    if not beta_values:
        raise ValueError("betas must contain at least one value.")
    grams = [
        compute_thermal_gram_matrix(
            kind=kind,
            L=L,
            beta=beta,
            source=source,
            normalize=normalize,
        )
        for beta in beta_values
    ]

    norm = Normalize(vmin=0.0, vmax=1.0)

    set_ipr_plot_style()
    nrows = int(np.ceil(len(grams) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(panel_size[0] * ncols, panel_size[1] * nrows),
        constrained_layout=False,
    )
    axes = np.atleast_1d(axes).ravel()

    last_im = None
    for ax, gram in zip(axes, grams):
        last_im = _plot_thermal_gram_on_axis(ax, gram, norm=norm)
        _annotate_gram_panel(ax, rf"$\beta={gram['beta']:g}$")

    for ax in axes[len(grams):]:
        ax.axis("off")

    fig.tight_layout(rect=[0.0, 0.0, 0.90, 1.0])
    if last_im is not None:
        cax = fig.add_axes([0.915, 0.22, 0.015, 0.56])
        cbar = fig.colorbar(last_im, cax=cax)
        normalized_for_colorbar = kind == "hs" or bool(grams[0]["normalize"])
        _set_normalized_cbar_ticks(cbar, normalized=normalized_for_colorbar)
        cbar.set_label(_thermal_cbar_label(kind, normalized=normalized_for_colorbar))

    if show_title:
        norm_txt = "normalized" if kind == "hs" or grams[0]["normalize"] else "as stored"
        fig.suptitle(
            f"Thermal TFIM {kind.upper()} Gram matrices (L={int(L)}, {source}, {norm_txt})",
            fontsize=12,
        )

    return fig, axes, grams
