"""Field-II-like 2-D pulse-echo simulator for plane-wave channel data, matched to the
PICMUS probe (128 el., pitch 0.3 mm, f0 5.208 MHz, fs 20.832 MHz).

Model (first-order Born, point scatterers, 2-D):
  rf_e(t) = sum_s a_s * D(phi_es) / sqrt(r_es) * p(t - tau_es),
  tau_es  = (x_s sin(th) + z_s cos(th)) / c + r_es / c,
with a Gaussian-modulated two-way pulse p, element directivity D (sinc of element width),
cylindrical spreading. Components are generated separately and are linear, so exact
"truth" maps exist for test phantoms:
  * tissue    : random scatterers with amplitude variance = echogenicity map E(z, x)
  * clutter   : reverberation of a shallow body-wall layer (echoes of z < z_wall re-emitted
                with extra round trips m * 2 d / c and per-channel jitter -> mis-focused,
                low spatial coherence; changes with transmit angle like real reverberation)
  * noise     : band-limited white Gaussian channel noise
Training uses only the summed channel data (no labels are stored or used).
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt

from .common import C0, F0, FS_RF, PITCH, ELEMENT_X, ChannelData, rf_to_iq

EL_W = 0.27e-3
BW = 0.67                       # -6 dB fractional bandwidth (two-way)
SIG_T = np.sqrt(2 * np.log(2)) / (np.pi * BW * F0)
HALF_TAPS = int(np.ceil(3.5 * SIG_T * FS_RF)) + 1
T0 = -3e-6                      # start of recording
N_T = 1800                      # RF samples (covers ~ 50 mm depth incl. steering)
ZLIM = (3e-3, 52e-3)
XLIM = (-22e-3, 22e-3)
ATT_DB_M_MHZ = 30.0            # 0.3 dB/cm/MHz (two-way path used)


def pulse(t):
    return np.exp(-t ** 2 / (2 * SIG_T ** 2)) * np.cos(2 * np.pi * F0 * t)


def simulate_rf(zs, xs, amp, angle, n_t=N_T, extra_delay=None, jitter=None):
    """Sum of point-scatterer echoes. extra_delay: (N_s,) additional delay per scatterer.
    jitter: (N_el,) per-channel delay offsets. Returns (N_el, n_t) float64 RF."""
    rf = np.zeros((len(ELEMENT_X), n_t))
    tx = (xs * np.sin(angle) + zs * np.cos(angle)) / C0
    if extra_delay is not None:
        tx = tx + extra_delay
    taps = np.arange(-HALF_TAPS, HALF_TAPS + 1)
    lam = C0 / F0
    for e, xe in enumerate(ELEMENT_X):
        dx = xs - xe
        r = np.sqrt(dx ** 2 + zs ** 2)
        tau = tx + r / C0 - T0
        if jitter is not None:
            tau = tau + jitter[e]
        sin_phi = dx / r
        att = 10 ** (-ATT_DB_M_MHZ * (F0 / 1e6) * ((xs * np.sin(angle) + zs * np.cos(angle)) + r) / 20)
        g = amp * att * np.sinc(EL_W / lam * sin_phi) / np.sqrt(r / 1e-2)
        n0 = np.round(tau * FS_RF).astype(np.int64)
        idx = n0[:, None] + taps[None, :]
        t = idx / FS_RF - tau[:, None]
        val = g[:, None] * pulse(t)
        ok = (idx >= 0) & (idx < n_t)
        rf[e] = np.bincount(idx[ok], weights=val[ok], minlength=n_t)[:n_t]
    return rf


def random_phantom(rng, density=60e6, test_design=None):
    """Echogenicity map as a function + scatterers. Returns dict with scatterers and a
    callable E(z, x) (linear power), and metadata of inclusions / points / wall."""
    area = (ZLIM[1] - ZLIM[0]) * (XLIM[1] - XLIM[0])
    n = rng.poisson(density * area)
    zs = rng.uniform(*ZLIM, n)
    xs = rng.uniform(*XLIM, n)
    incl = []
    if test_design is not None:
        incl = list(test_design["inclusions"])
        pts = list(test_design["points"])
        layer = test_design.get("layer")
        wall = test_design["wall"]
    else:
        for _ in range(rng.integers(2, 6)):
            r = rng.uniform(1.5e-3, 5e-3)
            c_db = rng.choice([-np.inf, -20, -12, -6, 6, 10])
            incl.append((rng.uniform(10e-3, 47e-3), rng.uniform(-15e-3, 15e-3), r, c_db))
        pts = [(rng.uniform(8e-3, 48e-3), rng.uniform(-16e-3, 16e-3)) for _ in range(rng.integers(0, 6))]
        layer = None
        if rng.random() < 0.5:
            z0 = rng.uniform(12e-3, 40e-3)
            layer = (z0, z0 + rng.uniform(2e-3, 6e-3), rng.choice([-6, 4, 8]))
        wall = (rng.uniform(4e-3, 7e-3), rng.uniform(0.8, 2.5))   # wall depth, reverb strength

    def E(z, x):
        e = np.ones_like(z)
        if layer is not None:
            e = np.where((z > layer[0]) & (z < layer[1]), 10 ** (layer[2] / 10), e)
        e = np.where(z < wall[0], 10 ** (8 / 10), e)                 # bright body wall
        for (zc, xc, r, cdb) in incl:
            inside = (z - zc) ** 2 + (x - xc) ** 2 < r ** 2
            e = np.where(inside, 0.0 if np.isinf(cdb) else 10 ** (cdb / 10), e)
        return e

    amp = rng.standard_normal(n) * np.sqrt(E(zs, xs))
    if pts:
        pz, px = np.array(pts).T
        zs = np.concatenate([zs, pz]); xs = np.concatenate([xs, px])
        amp = np.concatenate([amp, 25.0 * np.ones(len(pz))])
    return dict(zs=zs, xs=xs, amp=amp, E=E, inclusions=incl, points=pts, layer=layer, wall=wall)


def band_noise(rng, shape):
    sos = butter(4, [2e6, 8.5e6], btype="band", fs=FS_RF, output="sos")
    return sosfiltfilt(sos, rng.standard_normal(shape), axis=-1)


def simulate_components(ph, angle, rng, snr_db=20.0, n_reverb=6):
    """Returns tissue, clutter, noise RF components (each (N_el, N_T))."""
    tissue = simulate_rf(ph["zs"], ph["xs"], ph["amp"], angle)
    # reverberation of the shallow body wall: echoes from z < wall depth re-emitted
    wd, strength = ph["wall"]
    sel = ph["zs"] < wd
    clutter = np.zeros_like(tissue)
    for m in range(1, n_reverb + 1):
        extra = np.full(sel.sum(), m * 2 * wd / C0)
        jit = rng.normal(0, 0.12 / F0, len(ELEMENT_X))           # ~0.12 period channel jitter
        clutter += (strength * 0.7 ** m) * simulate_rf(ph["zs"][sel], ph["xs"][sel], ph["amp"][sel],
                                                         angle, extra_delay=extra, jitter=jit)
    # noise level relative to tissue RMS around 25-35 mm depth (two-way time window)
    t = T0 + np.arange(N_T) / FS_RF
    win = (t > 2 * 25e-3 / C0) & (t < 2 * 35e-3 / C0)
    p_sig = np.mean(tissue[:, win] ** 2)
    noise = band_noise(rng, tissue.shape)
    noise *= np.sqrt(p_sig / np.mean(noise ** 2) / 10 ** (snr_db / 10))
    return tissue, clutter, noise


def to_cd(rf, angle):
    iq, fs = rf_to_iq(rf, fs=FS_RF)
    return ChannelData(iq=iq, fs=fs, t0=T0, angle=angle)
