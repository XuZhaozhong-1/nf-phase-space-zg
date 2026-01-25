import os
import ctypes
import numpy as np
from pathlib import Path

_lib = None

def _resolve_lib_path() -> Path:
    env = os.environ.get("ZG_MG5_LIB", "")
    if not env:
        raise FileNotFoundError(
            "ZG_MG5_LIB is not set.\n"
            "Set it to the full path of your MadGraph dylib, e.g.\n"
            "  export ZG_MG5_LIB=/.../libzgme_uux.dylib"
        )
    p = Path(env).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"ZG_MG5_LIB points to a non-existent file: {p}")
    return p

def _ensure_cards_dir():
    # MadGraph lha_read searches for ident_card.dat relative to *cwd*.
    cards = os.environ.get("MG5_CARDS_DIR", "")
    if not cards:
        return
    cpath = Path(cards).expanduser()
    if not cpath.exists():
        raise FileNotFoundError(f"MG5_CARDS_DIR does not exist: {cpath}")
    os.chdir(str(cpath))

def _load():
    global _lib
    if _lib is None:
        _ensure_cards_dir()
        lib_path = _resolve_lib_path()
        _lib = ctypes.CDLL(str(lib_path))
        _lib.zg_msq.argtypes = [ctypes.POINTER(ctypes.c_double)]
        _lib.zg_msq.restype = ctypes.c_double
    return _lib

def me2(p_all: np.ndarray) -> float:
    """
    p_all shape (4,4): rows [p1, p2, pZ, pg], cols [E,px,py,pz]
    """
    p = np.asarray(p_all, dtype=np.float64).reshape(-1)  # length 16
    val = float(_load().zg_msq(p.ctypes.data_as(ctypes.POINTER(ctypes.c_double))))
    if not np.isfinite(val) or val < 0.0:
        raise RuntimeError(
            f"MadGraph returned invalid |M|^2={val}. "
            "Check MG5_CARDS_DIR contains param_card.dat and ident_card.dat."
        )
    return val
