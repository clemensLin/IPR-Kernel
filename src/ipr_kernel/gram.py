from __future__ import annotations

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .config import (
    DATASETS,
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


IPR_GRAM_CMAP = LinearSegmentedColormap.from_list(
    "ipr_orange_green",
    ["#fff2e8", "#f4a261", "#d95f02", "#1b9e77"],
    N=512,
)

NORMALIZED_GRAM_TICKS = np.linspace(0.0, 1.0, 6)


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


