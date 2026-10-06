# -*- coding: utf-8 -*-
"""Utilidades compartidas del benchmark BETO."""
from __future__ import annotations

import json
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parents[1]
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
DATA_VALIDATION = REPO_ROOT / "data" / "validation"
ARTIFACTS = MODULE_DIR / "artifacts"
RESULTS = MODULE_DIR / "results"
LOGS = MODULE_DIR / "logs"
CONFIG_PATH = MODULE_DIR / "config.json"


def ensure_dirs() -> None:
    for path in (ARTIFACTS, RESULTS, LOGS):
        path.mkdir(parents=True, exist_ok=True)


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def load_corpus() -> pd.DataFrame:
    """Carga id_doc y texto_analisis; prefiere Parquet y conserva CSV como respaldo."""
    parquet = DATA_PROCESSED / "corpus_limpio.parquet"
    csv = DATA_PROCESSED / "corpus_limpio.csv"
    if parquet.exists():
        return pd.read_parquet(parquet, columns=["id_doc", "texto_analisis"])
    if csv.exists():
        return pd.read_csv(csv, usecols=["id_doc", "texto_analisis"])
    raise FileNotFoundError("No se encontró corpus_limpio.parquet ni corpus_limpio.csv")


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "no_disponible"


def environment_report() -> dict[str, Any]:
    import sklearn
    import transformers

    try:
        import pyarrow
        pyarrow_version = pyarrow.__version__
    except Exception:
        pyarrow_version = "no_disponible"

    report: dict[str, Any] = {
        "python": sys.version,
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "transformers": transformers.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "pyarrow": pyarrow_version,
        "git_commit": git_commit(),
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        report.update({
            "gpu": torch.cuda.get_device_name(0),
            "gpu_total_memory_gib": round(p.total_memory / 1024**3, 3),
            "gpu_compute_capability": f"{p.major}.{p.minor}",
        })
    return report


def validate_unique(df: pd.DataFrame, name: str, key: str = "id_doc") -> None:
    if df[key].isna().any():
        raise ValueError(f"{name}: hay {key} faltantes")
    if df[key].duplicated().any():
        dup = df.loc[df[key].duplicated(), key].head().tolist()
        raise ValueError(f"{name}: {key} duplicados, ejemplos={dup}")


def weighted_metrics(
    y: np.ndarray, w: np.ndarray, pred: np.ndarray, domain: np.ndarray
) -> tuple[float, float, float]:
    m = domain.astype(bool)
    y_ = y[m].astype(float)
    w_ = w[m].astype(float)
    p_ = pred[m].astype(float)
    tp = float(np.sum(w_ * y_ * p_))
    den_precision = float(np.sum(w_ * p_))
    den_recall = float(np.sum(w_ * y_))
    precision = tp / den_precision if den_precision > 0 else 0.0
    recall = tp / den_recall if den_recall > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def threshold_grid(cfg: dict[str, Any]) -> np.ndarray:
    lo = float(cfg["threshold_min"])
    hi = float(cfg["threshold_max"])
    step = float(cfg["threshold_step"])
    n = int(round((hi - lo) / step))
    return np.round(np.linspace(lo, hi, n + 1), 10)


def hardware_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    path = ARTIFACTS / "hardware.json"
    if not path.exists():
        raise FileNotFoundError(
            "Falta artifacts/hardware.json. Ejecute primero 01_probar_memoria.py."
        )
    settings = read_json(path)
    for key in ("max_length", "fp16"):
        if key not in settings:
            raise ValueError(f"hardware.json no contiene {key}")
    return settings


def original_design_weights(
    df: pd.DataFrame, stratum_col: str, population_col: str
) -> pd.Series:
    """Pesos N_h/n_h, con n_h calculado antes de excluir DUDOSO."""
    n_h = df.groupby(stratum_col)["id_doc"].count()
    if set(n_h.index) != set(range(5)):
        raise ValueError(f"Estratos inesperados: {sorted(n_h.index.tolist())}")
    return df[population_col] / df[stratum_col].map(n_h)


def batch_iterable(values: Iterable[Any], batch_size: int) -> Iterable[list[Any]]:
    batch: list[Any] = []
    for value in values:
        batch.append(value)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
