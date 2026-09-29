"""Generate the simulated datasets.

train/: unlabeled multi-angle channel data only (summed tissue + clutter + noise), 7 angles.
test/ : designed phantoms, 0-degree input + all angles, with separate component channel
        data for truth maps, and R extra tissue realisations (0 deg) to estimate the
        speckle-ensemble echogenicity. Also a motion sequence for phase-fidelity tests.
"""
import argparse
import os
from multiprocessing import Pool

import zlib

import numpy as np

from . import sim
from .common import rf_to_iq

ANGLES = np.deg2rad(np.linspace(-16, 16, 7)).astype(np.float32)


def _iq(rf):
    return rf_to_iq(rf)[0]


def make_train(i, out):
    rng = np.random.default_rng(1000 + i)
    ph = sim.random_phantom(rng)
    snr = rng.uniform(10, 30)
    iq, iq2 = [], []
    for a in ANGLES:
        t, c, n = sim.simulate_components(ph, float(a), rng, snr_db=snr)
        n2 = sim.band_noise(rng, n.shape)
        n2 *= n.std() / n2.std()
        iq.append(_iq(t + c + n))
        iq2.append(_iq(t + c + n2))       # repeated acquisition of the static scene (new noise)
    np.savez(os.path.join(out, f"train_{i:03d}.npz"), iq=np.array(iq), iq2=np.array(iq2),
             angles=ANGLES, t0=sim.T0)
    return i


TEST_DESIGNS = {
    # six lesions of known contrast + a row of point targets
    "T1_lesions_snr20": dict(snr=20, wall=(5e-3, 1.2), layer=None),
    "T2_lesions_snr10": dict(snr=10, wall=(5e-3, 1.2), layer=None),
    "T3_lesions_strong_reverb": dict(snr=20, wall=(6e-3, 2.5), layer=None),
    "T4_lesions_layer": dict(snr=25, wall=(4.5e-3, 0.8), layer=(36e-3, 40e-3, 6)),
}


def design(spec):
    incl = []
    contrasts = [-np.inf, -20, -12, -6, 6, 12]
    pos = [(15e-3, -10e-3), (15e-3, 0.0), (15e-3, 10e-3), (29e-3, -10e-3), (29e-3, 0.0), (29e-3, 10e-3)]
    for (zc, xc), cdb in zip(pos, contrasts):
        incl.append((zc, xc, 3.5e-3, cdb))
    pts = [(44e-3, x) for x in (-12e-3, -6e-3, 0.0, 6e-3, 12e-3)] + [(22e-3, -15e-3), (22e-3, 15e-3)]
    return dict(inclusions=incl, points=pts, layer=spec["layer"], wall=spec["wall"])


def make_test(name, out, n_real=8):
    spec = TEST_DESIGNS[name]
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    ph = sim.random_phantom(rng, test_design=design(spec))
    comps = {}
    tot = []
    for j, a in enumerate(ANGLES):
        t, c, n = sim.simulate_components(ph, float(a), rng, snr_db=spec["snr"])
        tot.append(_iq(t + c + n))
        if abs(a) < 1e-6:
            comps = dict(tissue=_iq(t), clutter=_iq(c), noise=_iq(n))
    # extra speckle realisations (same echogenicity map, new scatterers), 0 deg tissue only
    reals = []
    for r in range(n_real):
        rr = np.random.default_rng(10_000 + r)
        ph_r = sim.random_phantom(rr, test_design=design(spec))
        reals.append(_iq(sim.simulate_rf(ph_r["zs"], ph_r["xs"], ph_r["amp"], 0.0)))
    np.savez(os.path.join(out, f"{name}.npz"), iq=np.array(tot), angles=ANGLES, t0=sim.T0,
             tissue=comps["tissue"], clutter=comps["clutter"], noise=comps["noise"],
             tissue_reals=np.array(reals),
             inclusions=np.array([list(v) for v in ph["inclusions"]]), points=np.array(ph["points"]))
    return name


def make_motion(out, n_frames=4, step=None):
    """Axial translation by lambda/16 per frame (0 deg); noise independent per frame."""
    lam = sim.C0 / sim.F0
    step = lam / 16 if step is None else step
    rng = np.random.default_rng(77)
    ph = sim.random_phantom(rng, test_design=design(TEST_DESIGNS["T1_lesions_snr20"]))
    frames, tissue = [], []
    for f in range(n_frames):
        t = sim.simulate_rf(ph["zs"] + f * step, ph["xs"], ph["amp"], 0.0)
        noise = sim.band_noise(rng, t.shape)
        tt = sim.T0 + np.arange(sim.N_T) / sim.FS_RF
        win = (tt > 2 * 25e-3 / sim.C0) & (tt < 2 * 35e-3 / sim.C0)
        noise *= np.sqrt(np.mean(t[:, win] ** 2) / np.mean(noise ** 2) / 10 ** 2.0)
        frames.append(_iq(t + noise)); tissue.append(_iq(t))
    np.savez(os.path.join(out, "motion.npz"), iq=np.array(frames), tissue=np.array(tissue),
             step=step, t0=sim.T0)
    return "motion"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/sim")
    ap.add_argument("--n_train", type=int, default=24)
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(f"{a.root}/train", exist_ok=True)
    os.makedirs(f"{a.root}/test", exist_ok=True)
    with Pool(a.procs) as pool:
        jobs = [pool.apply_async(make_train, (i, f"{a.root}/train")) for i in range(a.n_train)]
        jobs += [pool.apply_async(make_test, (n, f"{a.root}/test")) for n in TEST_DESIGNS]
        jobs += [pool.apply_async(make_motion, (f"{a.root}/test",))]
        for j in jobs:
            print("done", j.get(), flush=True)
