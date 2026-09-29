"""Figures for the results report."""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import picmus
from .common import make_grid

Z, X = make_grid(nz=512)
EXT = [X[0] * 1e3, X[-1] * 1e3, Z[-1] * 1e3, Z[0] * 1e3]


def bmode(env, dr=60):
    b = 20 * np.log10(env / np.percentile(env, 99.9) + 1e-12)
    return np.clip(b, -dr, 0)


def picmus_grid(run, out):
    I = np.load(os.path.join(run, "picmus_images.npz"))
    methods = ["DAS (1 PW)", "CF-DAS", "Rx-compound (4 looks)", "LC-PIB y_pp (IQ)", "LC-PIB echogenicity", "PICMUS 75-PW reference"]
    fig, ax = plt.subplots(len(picmus.NAMES), len(methods), figsize=(3.0 * len(methods), 3.6 * len(picmus.NAMES)))
    for i, n in enumerate(picmus.NAMES):
        for j, m in enumerate(methods):
            ax[i, j].imshow(bmode(I[f"{n}|{m}"]), cmap="gray", vmin=-60, vmax=0, extent=EXT, aspect="equal")
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            if i == 0:
                ax[i, j].set_title(m, fontsize=10)
            if j == 0:
                ax[i, j].set_ylabel(n.replace("_", "\n"), fontsize=9)
    plt.tight_layout()
    plt.savefig(out, dpi=70)
    plt.close()


def diag_maps(run, out, name="carotid_cross_expe"):
    I = np.load(os.path.join(run, "picmus_images.npz"))
    s2, c, n = I[f"{name}|_sigma2"], I[f"{name}|_clutter"], I[f"{name}|_noise"]
    tot = s2 + c + n
    panels = [("DAS (1 PW)", bmode(I[f"{name}|DAS (1 PW)"]), "gray", (-60, 0)),
              ("echogenicity  σ̂² [dB]", bmode(np.sqrt(s2)), "gray", (-60, 0)),
              ("clutter fraction ĉ/total", c / tot, "magma", (0, 1)),
              ("noise fraction n̂/total", n / tot, "magma", (0, 1)),
              ("Wiener gain g", I[f"{name}|_gain"], "viridis", (0, 1)),
              ("local Gamma shape L̂", I[f"{name}|_Lhat"], "coolwarm", (0, 2)),
              ("L_eff (4 looks)", I[f"{name}|_Leff"], "viridis", (1, 4))]
    fig, ax = plt.subplots(1, len(panels), figsize=(3.1 * len(panels), 4.2))
    for a, (t, im, cm, (lo, hi)) in zip(ax, panels):
        h = a.imshow(im, cmap=cm, vmin=lo, vmax=hi, extent=EXT)
        a.set_title(t, fontsize=10); a.set_xticks([]); a.set_yticks([])
        plt.colorbar(h, ax=a, fraction=0.046)
    plt.suptitle(f"LC-PIB outputs and diagnostic maps: PICMUS {name}")
    plt.tight_layout(); plt.savefig(out, dpi=70); plt.close()


def sim_truth(run, out):
    S = np.load(os.path.join(run, "sim_T1_images.npz"))
    db = lambda a: 10 * np.log10(a / np.percentile(a, 99.9) + 1e-12)
    panels = [("DAS (1 PW)", bmode(S["DAS (1 PW)"]), (-60, 0)),
              ("LC-PIB echogenicity", bmode(S["LC-PIB echogenicity"]), (-60, 0)),
              ("TRUTH tissue expectation", bmode(S["TRUTH (tissue expectation)"]), (-60, 0)),
              ("LC-PIB y_pp", bmode(S["LC-PIB y_pp (IQ)"]), (-60, 0)),
              ("clutter est. [dB]", db(S["clutter"]), (-60, 0)),
              ("clutter TRUTH [dB]", db(S["clutter_truth"]) + 10 * np.log10(np.percentile(S["clutter_truth"], 99.9) / np.percentile(S["clutter"], 99.9)), (-60, 0)),
              ("noise est. [dB]", 10 * np.log10(S["noise"]), None),
              ("noise TRUTH [dB]", 10 * np.log10(S["noise_truth"]), None)]
    fig, ax = plt.subplots(1, len(panels), figsize=(2.9 * len(panels), 4.2))
    lo_n = np.percentile(10 * np.log10(S["noise_truth"]), 1); hi_n = np.percentile(10 * np.log10(S["noise_truth"]), 99.5)
    for a, (t, im, rng) in zip(ax, panels):
        lo, hi = rng if rng else (lo_n - 10, hi_n + 5)
        h = a.imshow(im, cmap="gray" if "noise" not in t and "clutter" not in t else "magma", vmin=lo, vmax=hi, extent=EXT)
        a.set_title(t, fontsize=9); a.set_xticks([]); a.set_yticks([])
    plt.suptitle("Simulation test phantom T1 (lesions -inf/-20/-12 dB top row, -6/+6/+12 dB middle row, points at 44 mm)")
    plt.tight_layout(); plt.savefig(out, dpi=70); plt.close()


def curves(runs, out):
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    for r in runs:
        p = os.path.join(r, "train_log.json")
        if not os.path.exists(p):
            continue
        log = json.load(open(p))["log"]
        tot = [sum(v for k, v in e.items() if k in ("E", "xv", "t")) for e in log]
        k = 50
        sm = np.convolve(tot, np.ones(k) / k, mode="valid")
        ax.plot(sm, label=os.path.basename(r))
    ax.set_xlabel("step"); ax.set_ylabel("L_E + L_xv + L_t (moving avg)"); ax.legend(); ax.set_yscale("symlog")
    plt.tight_layout(); plt.savefig(out, dpi=80); plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/full")
    ap.add_argument("--runs", nargs="*", default=[])
    a = ap.parse_args()
    fd = os.path.join(a.run, "figures"); os.makedirs(fd, exist_ok=True)
    picmus_grid(a.run, os.path.join(fd, "picmus_all.png"))
    for n in ("carotid_cross_expe", "contrast_speckle_expe"):
        diag_maps(a.run, os.path.join(fd, f"diag_{n}.png"), n)
    sim_truth(a.run, os.path.join(fd, "sim_T1.png"))
    if a.runs:
        curves(a.runs, os.path.join(fd, "training_curves.png"))
