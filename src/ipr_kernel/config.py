"""Dataset locations and readers shared by the notebooks and the figure script."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


DATASETS = {
    "tfim1d": {
        "processed": PROCESSED_DIR / "tfim_1d",
        "x_file": "hs.csv",
        "x_label": r"$h$",
        "title_name": "TFIM 1D",
        "critical_point": 1.0,
        "curves": {"IPR X": "ipr_x.csv", "IPR Z": "ipr_z.csv"},
        "log_curves": {"IPR X": "log10_ipr_x.csv", "IPR Z": "log10_ipr_z.csv"},
    },
    "heisenberg": {
        "processed": PROCESSED_DIR / "xxz",
        "x_file": "Delta.csv",
        "x_label": r"$\Delta$",
        "title_name": "Heisenberg",
        "critical_point": 1.0,
        "curves": {"IPR X": "IPR X.csv", "IPR XY": "IPR XY.csv", "IPR Z": "IPR Z.csv"},
        "log_curves": {"IPR X": "log10_IPR_X.csv", "IPR XY": "log10_IPR_XY.csv", "IPR Z": "log10_IPR_Z.csv"},
    },
}


RAW_THERMAL_DIR = RAW_DIR / "tfim_thermal"
RAW_THERMAL_IPR_DIR = RAW_THERMAL_DIR / "ipr"
RAW_THERMAL_HS_DIR = RAW_THERMAL_DIR / "hs"
RAW_XXZ_FIELD_DIR = RAW_DIR / "xxz_field"

PROCESSED_THERMAL_DIR = PROCESSED_DIR / "tfim_thermal"
PROCESSED_THERMAL_IPR_DIR = PROCESSED_THERMAL_DIR / "ipr"
PROCESSED_THERMAL_HS_DIR = PROCESSED_THERMAL_DIR / "hs"
PROCESSED_THERMAL_GRID = PROCESSED_THERMAL_DIR / "h_dense_4000.csv"

# Validate and return one canonical dataset key.
def normalize_dataset(dataset: str) -> str:
    key = str(dataset).lower().strip()
    if key not in DATASETS:
        raise ValueError(f"dataset must be one of {list(DATASETS)}")
    return key


# Validate whether curves should be loaded as raw values or log10 values.
def normalize_scale(scale: str) -> str:
    scale = str(scale).lower().strip()
    if scale not in {"raw", "log10"}:
        raise ValueError("scale must be 'raw' or 'log10'")
    return scale


# Validate an explicit list of IPR curve names used as features.
def curves_from_basis(dataset: str, basis: list[str] | tuple[str, ...]) -> list[str]:
    if not isinstance(basis, (list, tuple)):
        raise ValueError('basis must be an explicit list such as ["IPR X", "IPR Z"].')

    available = DATASETS[dataset]["curves"]
    curves = list(basis)
    for curve_name in curves:
        if curve_name not in available:
            raise ValueError(f"Unknown curve {curve_name!r}. Available: {list(available)}")
    return curves


# Convert a system size into the processed-data folder name for one dataset.
def folder_name(dataset: str, L: int | str) -> str:
    if isinstance(L, str) and L.startswith("L="):
        return L
    if dataset == "tfim1d":
        return f"L={int(L)}x1"
    if dataset == "heisenberg":
        return f"L={int(L)}"
    raise ValueError(f"Unknown dataset: {dataset}")


# Read a one-dimensional numeric CSV into a NumPy float array.
def read_vector(path: Path) -> np.ndarray:
    data = pd.read_csv(path, header=None)
    if data.shape[1] == 1:
        values = pd.to_numeric(data.iloc[:, 0], errors="coerce").dropna()
    else:
        values = pd.to_numeric(data.iloc[:, -1], errors="coerce").dropna()
    return values.to_numpy(dtype=float)


# Read one raw or log10 IPR curve from a processed dataset folder.
def read_curve(folder: Path, cfg: dict, curve_name: str, scale: str) -> np.ndarray:
    if scale == "log10":
        log_file = folder / cfg["log_curves"][curve_name]
        if log_file.exists():
            return read_vector(log_file)
        return np.log10(read_vector(folder / cfg["curves"][curve_name]))
    return read_vector(folder / cfg["curves"][curve_name])


# List available processed system sizes for one dataset, optionally bounded.
def available_l_values(dataset: str, n_min: int | None = None, n_max: int | None = None) -> list[int]:
    cfg = DATASETS[dataset]
    values = []
    for folder in cfg["processed"].iterdir():
        if not folder.is_dir() or not folder.name.startswith("L="):
            continue
        number = folder.name.split("=", 1)[1].split("x", 1)[0]
        if number.isdigit():
            n = int(number)
            if (n_min is None or n >= n_min) and (n_max is None or n <= n_max):
                values.append(n)
    return sorted(values)



# --------------------------- THERMAL DATA ---------------------------

# Validate whether thermal data should be loaded from raw or processed files.
def normalize_thermal_source(source: str) -> str:
    key = str(source).lower().strip()
    if key not in {"raw", "processed"}:
        raise ValueError("source must be 'raw' or 'processed'.")
    return key


# Validate whether thermal helpers should use IPR or HS kernel data.
def normalize_thermal_kind(kind: str) -> str:
    key = str(kind).lower().strip()
    if key not in {"ipr", "hs"}:
        raise ValueError("kind must be 'ipr' or 'hs'.")
    return key


# Build the thermal IPR CSV path for one system size and source.
def thermal_ipr_path(L: int, source: str = "processed") -> Path:
    source = normalize_thermal_source(source)
    root = PROCESSED_THERMAL_IPR_DIR if source == "processed" else RAW_THERMAL_IPR_DIR
    return root / f"ipr_thermal_L={int(L)}.csv"


# Build the thermal HS path for raw CSVs, processed folders, or processed beta matrices.
def thermal_hs_path(L: int, beta: float | None = None, source: str = "processed") -> Path:
    source = normalize_thermal_source(source)
    if source == "raw":
        return RAW_THERMAL_HS_DIR / f"HS Kernel L={int(L)}.csv"
    if beta is None:
        return PROCESSED_THERMAL_HS_DIR / f"L={int(L)}"
    return PROCESSED_THERMAL_HS_DIR / f"L={int(L)}" / f"HS_kernel_beta={float(beta):.1f}.npy"


# List available thermal IPR system sizes for one source.
def available_thermal_ipr_l_values(source: str = "processed") -> list[int]:
    source = normalize_thermal_source(source)
    root = PROCESSED_THERMAL_IPR_DIR if source == "processed" else RAW_THERMAL_IPR_DIR
    prefix = "ipr_thermal_L="
    return sorted(int(path.stem.replace(prefix, "")) for path in root.glob(f"{prefix}*.csv"))


# List available thermal HS system sizes for one source.
def available_thermal_hs_l_values(source: str = "processed") -> list[int]:
    source = normalize_thermal_source(source)
    if source == "processed":
        return sorted(
            int(path.name.replace("L=", ""))
            for path in PROCESSED_THERMAL_HS_DIR.glob("L=*")
            if path.is_dir()
        )
    prefix = "HS Kernel L="
    return sorted(int(path.stem.replace(prefix, "")) for path in RAW_THERMAL_HS_DIR.glob(f"{prefix}*.csv"))


# List available thermal system sizes for one kernel kind and source.
def available_thermal_l_values(kind: str = "ipr", source: str = "processed") -> list[int]:
    kind = normalize_thermal_kind(kind)
    if kind == "ipr":
        return available_thermal_ipr_l_values(source=source)
    return available_thermal_hs_l_values(source=source)


# Read one thermal IPR CSV from raw or processed data.
def read_thermal_ipr_data(L: int, source: str = "processed") -> pd.DataFrame:
    path = thermal_ipr_path(L=L, source=source)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


# Read one raw thermal HS CSV.
def read_thermal_hs_raw_data(L: int) -> pd.DataFrame:
    path = thermal_hs_path(L=L, source="raw")
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


# Return the processed dense thermal h-grid when it matches a matrix size.
def thermal_grid_for_size(n: int) -> np.ndarray:
    if PROCESSED_THERMAL_GRID.exists():
        h = read_vector(PROCESSED_THERMAL_GRID)
        if h.size == int(n):
            return h
    return np.linspace(0.0, 4.0, int(n))


# List beta values available in one thermal IPR file.
def available_thermal_ipr_betas(L: int = 20, source: str = "processed") -> list[float]:
    df = read_thermal_ipr_data(L=L, source=source)
    if "beta" not in df.columns:
        raise ValueError(f"{thermal_ipr_path(L, source).name}: missing 'beta' column.")
    return sorted(float(beta) for beta in df["beta"].unique())


# List beta values available for thermal HS data.
def available_thermal_hs_betas(L: int = 20, source: str = "processed") -> list[float]:
    source = normalize_thermal_source(source)
    if source == "raw":
        df = read_thermal_hs_raw_data(L=L)
        if "beta" not in df.columns:
            raise ValueError(f"{thermal_hs_path(L, source='raw').name}: missing 'beta' column.")
        return sorted(float(beta) for beta in df["beta"].unique())

    root = thermal_hs_path(L=L, source="processed")
    if not root.exists():
        raise FileNotFoundError(root)
    prefix = "HS_kernel_beta="
    return sorted(float(path.stem.replace(prefix, "")) for path in root.glob(f"{prefix}*.npy"))


# List beta values available for one thermal kernel kind and source.
def available_thermal_betas(kind: str = "ipr", L: int = 20, source: str = "processed") -> list[float]:
    kind = normalize_thermal_kind(kind)
    if kind == "ipr":
        return available_thermal_ipr_betas(L=L, source=source)
    return available_thermal_hs_betas(L=L, source=source)
