"""Shared physics utilities: probe model, RF->IQ, plane-wave time of flight,
receive aperture support and reference (fixed) apodization.

Conventions (match PICMUS): linear array on z = 0, 128 elements, pitch 0.3 mm,
f0 = 5.208 MHz, fs(RF) = 20.832 MHz, t = 0 when the plane wave crosses x = 0, z = 0
for a 0-degree transmit. Pixel grid coordinates are in metres, (z, x).
"""
from dataclasses import dataclass

import numpy as np
import torch
from scipy.signal import hilbert, resample_poly

C0 = 1540.0
F0 = 5.208e6
FS_RF = 20.832e6
PITCH = 0.3e-3
N_EL = 128
ELEMENT_X = (np.arange(N_EL) - (N_EL - 1) / 2) * PITCH
LAMBDA = C0 / F0


@dataclass
class ChannelData:
    """Baseband IQ channel data of one plane-wave transmit."""
    iq: np.ndarray          # (N_el, N_t) complex64, baseband, fs_iq
    fs: float               # IQ sampling rate
    t0: float               # time of first sample
    angle: float            # steering angle [rad]
    c: float = C0


def rf_to_iq(rf, fs=FS_RF, f0=F0, up=2):
    """Real RF (N_el, N_t) -> baseband IQ upsampled by `up` (analytic signal, demodulated)."""
    if up > 1:
        rf = resample_poly(rf, up, 1, axis=-1)
    fs_iq = fs * up
    t = np.arange(rf.shape[-1]) / fs_iq
    iq = hilbert(rf, axis=-1) * np.exp(-2j * np.pi * f0 * t)[None, :]
    return iq.astype(np.complex64), fs_iq


def make_grid(zlim=(5e-3, 50e-3), xlim=(-19.2e-3, 19.2e-3), nz=512, nx=256):
    z = np.linspace(zlim[0], zlim[1], nz, dtype=np.float32)
    x = np.linspace(xlim[0], xlim[1], nx, dtype=np.float32)
    return z, x


def delay_gather(iq_t, fs, t0, angle, zz, xx, c, f0=F0, el_x=None):
    """Differentiable plane-wave ToF + linear interpolation + phase rotation.

    iq_t: (N_el, N_t) complex torch tensor. zz, xx: (P,) pixel coordinates.
    c may be a torch scalar (learnable sound speed). Returns d: (P, N_el) complex.
    """
    if el_x is None:
        el_x = torch.as_tensor(ELEMENT_X, dtype=zz.dtype, device=zz.device)
    ang = torch.as_tensor(angle, dtype=zz.dtype)
    tx = (xx * torch.sin(ang) + zz * torch.cos(ang))[:, None]
    rx = torch.sqrt((xx[:, None] - el_x[None, :]) ** 2 + zz[:, None] ** 2)
    tau = (tx + rx) / c                                  # (P, N_el)
    s = (tau - t0) * fs
    n_t = iq_t.shape[-1]
    s = s.clamp(0, n_t - 2.001)
    i0 = s.floor().long()
    a = (s - i0.to(s.dtype)).to(torch.float32)
    ch = torch.arange(iq_t.shape[0], device=zz.device)[None, :].expand_as(i0)
    v0 = iq_t[ch, i0]
    v1 = iq_t[ch, i0 + 1]
    d = v0 * (1 - a) + v1 * a
    ph = (2 * np.pi * f0) * (tau - t0)          # re-modulation (baseband referenced to sample 0)
    return d * torch.polar(torch.ones_like(ph), ph)


def aperture_mask(zz, xx, fnum=1.5, el_x=None, soft=0.15):
    """Soft F-number receive support M_i(x) in [0,1] (Tukey-like edge)."""
    if el_x is None:
        el_x = torch.as_tensor(ELEMENT_X, dtype=zz.dtype, device=zz.device)
    half = (zz / (2 * fnum))[:, None].clamp_min(2 * PITCH)
    r = (xx[:, None] - el_x[None, :]).abs() / half        # 0 at centre, 1 at edge
    edge = soft
    m = torch.clamp((1 - r) / edge, 0, 1)
    return 0.5 - 0.5 * torch.cos(np.pi * m)


def hann_apod(zz, xx, fnum=1.5, el_x=None):
    """Reference apodization: Hann over the F-number aperture, normalised to sum 1."""
    if el_x is None:
        el_x = torch.as_tensor(ELEMENT_X, dtype=zz.dtype, device=zz.device)
    half = (zz / (2 * fnum))[:, None].clamp_min(2 * PITCH)
    r = ((xx[:, None] - el_x[None, :]) / half).clamp(-1, 1)
    w = 0.5 + 0.5 * torch.cos(np.pi * r)
    w = w * ((xx[:, None] - el_x[None, :]).abs() < half)
    return w / w.sum(-1, keepdim=True).clamp_min(1e-6)


def beamform_image(cd: ChannelData, z, x, apod_fn=hann_apod, chunk=16, fnum=1.5,
                   return_looks=None):
    """Full-image DAS with a fixed apodization. Returns complex (nz, nx) image and,
    optionally, K sub-aperture looks (K, nz, nx) using windows `return_looks` (K, N_el)."""
    iq_t = torch.from_numpy(cd.iq)
    zt, xt = torch.from_numpy(z), torch.from_numpy(x)
    out = np.zeros((len(z), len(x)), np.complex64)
    looks = None if return_looks is None else np.zeros((len(return_looks), len(z), len(x)), np.complex64)
    with torch.no_grad():
        for i in range(0, len(z), chunk):
            zz, xx = torch.meshgrid(zt[i:i + chunk], xt, indexing="ij")
            zz, xx = zz.reshape(-1), xx.reshape(-1)
            d = delay_gather(iq_t, cd.fs, cd.t0, cd.angle, zz, xx, cd.c)
            w = apod_fn(zz, xx, fnum=fnum)
            out[i:i + chunk] = (d * w).sum(-1).reshape(-1, len(x)).numpy()
            if looks is not None:
                a = torch.from_numpy(return_looks)
                for k in range(a.shape[0]):
                    wk = w * a[k][None, :]
                    wk = wk / wk.sum(-1, keepdim=True).clamp_min(1e-6)
                    looks[k, i:i + chunk] = (d * wk).sum(-1).reshape(-1, len(x)).numpy()
    return out, looks


def look_windows(K=4, overlap=0.5, n=N_EL):
    """K Hann receive sub-aperture windows over the array with given fractional overlap,
    scaled so that they sum to ~1 (complementary)."""
    L = n / (K - (K - 1) * overlap)
    step = L * (1 - overlap)
    idx = np.arange(n)
    wins = []
    for k in range(K):
        s = k * step
        r = (idx - s) / L
        w = np.where((r >= 0) & (r <= 1), 0.5 - 0.5 * np.cos(2 * np.pi * r), 0.0)
        wins.append(w)
    wins = np.array(wins)
    # make complementary where they overlap; edge windows keep flat tops at array ends
    wins[0, : int(L / 2)] = 1.0
    wins[-1, n - int(L / 2):] = 1.0
    wins = wins / wins.sum(0, keepdims=True).clip(1e-6)
    return wins.astype(np.float32)
