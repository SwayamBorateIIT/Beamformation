"""Loader for the PICMUS single-plane-wave (0 deg) RF files.

The files used here are the preprocessed copies in github.com/Tio-Panda/inf577-project
(`dataset_test/*.hdf5`): `rfdata` is the central 0-degree plane wave (128 x 2800, RF,
normalised), `img` is their 75-angle compounded reference (mean of per-angle MV images,
RF domain) on a 2048 x 256 grid over z = 5-50 mm, x = +-19.2 mm, and `das` their 1-PW DAS.
Only the 0-degree plane wave is available, so PICMUS is used for held-out evaluation only.
"""
import glob
import os

import h5py
import numpy as np
from scipy.signal import hilbert

from .common import ChannelData, rf_to_iq

PULSE_LAG = 2 * 0.19e-3 / 1540.0

NAMES = [
    "resolution_distorsion_simu", "resolution_distorsion_expe",
    "contrast_speckle_simu", "contrast_speckle_expe",
    "carotid_cross_expe", "carotid_long_expe",
]


def load(name, root="data/picmus"):
    f = h5py.File(os.path.join(root, f"{name}_dataset_rf.hdf5"), "r")
    rf = f["rfdata"][:].astype(np.float64)
    fs = float(f.attrs["fs"])
    iq, fs_iq = rf_to_iq(rf, fs=fs)
    # +0.19 mm systematic axial offset of all PICMUS simulated point targets (pulse lag)
    # corrected by a fixed time offset (applied to all six datasets).
    cd = ChannelData(iq=iq, fs=fs_iq, t0=float(f.attrs["t0"]) - PULSE_LAG, angle=float(f.attrs["angle"]),
                     c=float(f.attrs["c0"]))
    grid = f["grid"][:]
    ref_rf = f["img"][:]
    ref_env = np.abs(hilbert(ref_rf, axis=0))       # 75-angle compounded reference envelope
    das_env = np.abs(hilbert(f["das"][:], axis=0))
    probe = f["probe_geometry"][:]
    return dict(cd=cd, grid=grid, ref_env=ref_env, das_env=das_env, probe=probe, name=name)
