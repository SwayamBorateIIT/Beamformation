"""Evaluation of LC-PIB and baselines.

PICMUS (held out, single 0-degree PW input): contrast (CR, CNR, gCNR, speckle SNR) on the
contrast/speckle phantoms, axial/lateral FWHM on the resolution phantoms, images of all six.
Simulation test set (truth available): contrast linearity, echogenicity/clutter/noise map
accuracy, Gamma-shape calibration (PIT/KS), L_eff validation, phase fidelity (Kasai
displacement on a moving phantom), hallucination tests (null input, point insertion), runtime.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch
from scipy import ndimage, stats
from scipy.signal import hilbert

from . import picmus, sim
from .common import ChannelData, delay_gather, make_grid, look_windows, hann_apod, aperture_mask, F0, C0
from .model import LCPIB, smooth, abs2

torch.set_num_threads(4)
NZ, NX = 512, 256
Z, X = make_grid(nz=NZ)
DZ, DX = Z[1] - Z[0], X[1] - X[0]


# ------------------------------------------------------------------ inference
@torch.no_grad()
def infer(model, cd, keep_w=False):
    zz, xx = torch.meshgrid(torch.from_numpy(Z), torch.from_numpy(X), indexing="ij")
    zz, xx = zz.reshape(-1), xx.reshape(-1)
    t = time.time()
    o = model(torch.from_numpy(cd.iq), cd.fs, cd.t0, cd.angle, zz, xx, NZ, NX, c=cd.c)
    rt = time.time() - t
    d, Mk, w0 = o["d"], o["Mk"], o["w0"]
    act = (Mk > 0.5).float()
    # baselines from the same delayed data
    cf = (abs2((d * act).sum(-1)) / (act.sum(-1).clamp_min(1) * (abs2(d) * act).sum(-1).clamp_min(1e-20)))
    looks = torch.from_numpy(look_windows())
    zk = []
    for k in range(looks.shape[0]):
        wk = w0 * looks[k][None]
        wk = wk / wk.sum(-1, keepdim=True).clamp_min(1e-6)
        zk.append((wk * d).sum(-1))
    zk = torch.stack(zk)
    rx_comp = abs2(zk).mean(0)
    out = dict(
        y_ref=o["y_ref"].numpy(), y=o["y"].numpy(), y_pp=o["y_pp"].numpy(),
        sigma2=o["sigma2"].numpy(), clutter=o["clutter"].numpy(), noise=o["noise"].numpy(),
        gain=o["gain"].numpy(), L_hat=o["L_hat"].numpy(), L_eff=o["L_eff"].numpy(),
        cf=cf.reshape(NZ, NX).numpy(), rx_comp=rx_comp.reshape(NZ, NX).numpy(),
        zk=zk.reshape(-1, NZ, NX).numpy(), runtime=rt, n_w=o["n_w"])
    if keep_w:
        out["w"], out["Mk"] = o["w"], o["Mk"]
    return out


def envelopes(o):
    """Linear envelope (amplitude) images of each method."""
    return {
        "DAS (1 PW)": np.abs(o["y_ref"]),
        "CF-DAS": o["cf"] * np.abs(o["y_ref"]),
        "Rx-compound (4 looks)": np.sqrt(o["rx_comp"]),
        "LC-PIB y_pp (IQ)": np.abs(o["y_pp"]),
        "LC-PIB echogenicity": np.sqrt(o["sigma2"]),
    }


# ------------------------------------------------------------------ metrics
def gcnr(a, b, bins=256):
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    ha, _ = np.histogram(a, bins=bins, range=(lo, hi), density=True)
    hb, _ = np.histogram(b, bins=bins, range=(lo, hi), density=True)
    w = (hi - lo) / bins
    return 1 - np.sum(np.minimum(ha, hb)) * w


def contrast_metrics(env, inside, outside):
    ei, eo = env[inside], env[outside]
    ii, io = ei ** 2, eo ** 2
    return dict(CR_dB=10 * np.log10(ii.mean() / io.mean()),
                CNR=abs(ei.mean() - eo.mean()) / np.sqrt(ei.var() + eo.var()),
                gCNR=gcnr(ei, eo), SNR_out=eo.mean() / eo.std())


def disk(zc, xc, r):
    zz, xx = np.meshgrid(Z, X, indexing="ij")
    return (zz - zc) ** 2 + (xx - xc) ** 2 < r ** 2


def ring(zc, xc, r1, r2):
    zz, xx = np.meshgrid(Z, X, indexing="ij")
    rr = (zz - zc) ** 2 + (xx - xc) ** 2
    return (rr > r1 ** 2) & (rr < r2 ** 2)


def detect_cysts(ref_env):
    """Circular hypo/anechoic regions from the (heavily smoothed) 75-angle reference."""
    zoomed = ndimage.zoom(ref_env, (NZ / ref_env.shape[0], 1), order=1)
    lg = 20 * np.log10(ndimage.gaussian_filter(zoomed, (6, 3)) + 1e-12)
    lg -= np.median(lg)
    lab, n = ndimage.label(lg < -6)
    cysts = []
    for i in range(1, n + 1):
        m = lab == i
        area = m.sum() * DZ * DX
        if area < np.pi * (1.0e-3) ** 2:
            continue
        zc = (np.meshgrid(Z, X, indexing="ij")[0][m]).mean()
        xc = (np.meshgrid(Z, X, indexing="ij")[1][m]).mean()
        r = np.sqrt(area / np.pi)
        if abs(xc) > 17e-3 or zc < 8e-3:
            continue
        cysts.append((zc, xc, r))
    return cysts


def fwhm_1d(profile, spacing):
    p = profile / profile.max()
    i = int(np.argmax(p))
    l = i
    while l > 0 and p[l] > 0.5:
        l -= 1
    r = i
    while r < len(p) - 1 and p[r] > 0.5:
        r += 1
    if l == 0 or r == len(p) - 1:
        return np.nan
    fl = l + (0.5 - p[l]) / (p[l + 1] - p[l])
    fr = r - 1 + (p[r - 1] - 0.5) / (p[r - 1] - p[r])
    return (fr - fl) * spacing


def point_fwhm(env, zc, xc, up=8):
    iz, ix = int(round((zc - Z[0]) / DZ)), int(round((xc - X[0]) / DX))
    hz, hx = 12, 10
    if iz - hz < 0 or iz + hz >= NZ or ix - hx < 0 or ix + hx >= NX:
        return np.nan, np.nan
    patch = env[iz - hz:iz + hz + 1, ix - hx:ix + hx + 1]
    pu = ndimage.zoom(patch, up, order=3)
    pu = np.clip(pu, 0, None)
    jz, jx = np.unravel_index(np.argmax(pu), pu.shape)
    return fwhm_1d(pu[:, jx], DZ / up), fwhm_1d(pu[jz, :], DX / up)


def detect_points(env, thresh_db=-25):
    lg = 20 * np.log10(env / env.max() + 1e-12)
    mx = ndimage.maximum_filter(lg, size=(25, 13))
    pk = np.argwhere((lg == mx) & (lg > thresh_db))
    pts = [(Z[i], X[j]) for i, j in pk if 7e-3 < Z[i] < 48e-3 and abs(X[j]) < 17e-3]
    return pts


# ------------------------------------------------------------------ PICMUS
def eval_picmus(model, outdir):
    res, imgs = {}, {}
    for name in picmus.NAMES:
        dset = picmus.load(name)
        o = infer(model, dset["cd"])
        env = envelopes(o)
        ref = ndimage.zoom(dset["ref_env"], (NZ / dset["ref_env"].shape[0], 1), order=1)
        env_all = dict(env)
        env_all["PICMUS 75-PW reference"] = ref
        imgs[name] = {k: v.astype(np.float32) for k, v in env_all.items()}
        imgs[name]["_gain"] = o["gain"]; imgs[name]["_clutter"] = o["clutter"]; imgs[name]["_noise"] = o["noise"]
        imgs[name]["_Lhat"] = o["L_hat"]; imgs[name]["_Leff"] = o["L_eff"]; imgs[name]["_sigma2"] = o["sigma2"]
        r = dict(runtime_s=o["runtime"])
        if name.startswith("contrast"):
            cysts = detect_cysts(dset["ref_env"])
            per = {k: [] for k in env_all}
            for (zc, xc, rad) in cysts:
                ins, out = disk(zc, xc, 0.8 * rad), ring(zc, xc, 1.3 * rad, 1.3 * rad + 3e-3)
                for k, e in env_all.items():
                    per[k].append(contrast_metrics(e, ins, out))
            r["cysts_mm"] = [[round(1e3 * v, 2) for v in c] for c in cysts]
            r["contrast"] = {k: {m: float(np.mean([p[m] for p in v])) for m in v[0]} for k, v in per.items()} if cysts else {}
            r["contrast_per_cyst"] = {k: [{m: float(p[m]) for m in p} for p in v] for k, v in per.items()}
        if name.startswith("resolution"):
            pts = detect_points(np.abs(o["y_ref"]))
            fw = {}
            for k, e in env_all.items():
                vals = np.array([point_fwhm(e, zc, xc) for zc, xc in pts])
                fw[k] = dict(axial_mm=float(np.nanmedian(vals[:, 0]) * 1e3), lateral_mm=float(np.nanmedian(vals[:, 1]) * 1e3),
                             n_points=int(np.sum(np.isfinite(vals[:, 1]))))
            r["points_mm"] = [[round(1e3 * p[0], 2), round(1e3 * p[1], 2)] for p in pts]
            r["fwhm"] = fw
        # calibration: single-look intensity vs predicted Gamma(L_hat, sigma+c+n) (PIT/KS)
        r["calibration"] = pit_stats(o, mask=None)
        res[name] = r
    np.savez_compressed(os.path.join(outdir, "picmus_images.npz"),
                        **{f"{n}|{k}": v for n, d in imgs.items() for k, v in d.items()})
    return res


def pit_stats(o, mask=None):
    I1 = np.abs(o["y_ref"]) ** 2
    m = o["sigma2"] + o["clutter"] + o["noise"]
    L = o["L_hat"]
    if mask is None:
        mask = np.ones_like(I1, bool)
    mask = mask & np.isfinite(L)
    sub = (slice(None, None, 4), slice(None, None, 3))       # thin to reduce spatial correlation
    msk = mask[sub]
    u_model = stats.gamma.cdf(I1[sub][msk], a=L[sub][msk], scale=m[sub][msk] / L[sub][msk])
    u_exp = stats.expon.cdf(I1[sub][msk], scale=m[sub][msk])
    ks = lambda u: float(stats.kstest(u, "uniform").statistic)
    return dict(KS_model_gamma=ks(u_model), KS_exponential=ks(u_exp),
                mean_ratio_I_over_m=float(np.mean(I1[mask] / m[mask])),
                median_L_hat=float(np.median(L[mask])), n=int(msk.sum()))


# ------------------------------------------------------------------ simulation test set
def fixed_bf(iq, t0, angle=0.0):
    cd = ChannelData(iq=iq, fs=2 * sim.FS_RF, t0=t0, angle=angle)
    zz, xx = torch.meshgrid(torch.from_numpy(Z), torch.from_numpy(X), indexing="ij")
    zz, xx = zz.reshape(-1), xx.reshape(-1)
    with torch.no_grad():
        d = delay_gather(torch.from_numpy(iq), cd.fs, t0, angle, zz, xx, C0)
        w0 = hann_apod(zz, xx, 1.5)
        return (w0 * d).sum(-1).reshape(NZ, NX).numpy()


def eval_sim(model, root, outdir):
    res = {}
    kw = model.kw
    sm = lambda a: smooth(torch.from_numpy(np.ascontiguousarray(a)).float(), kw).numpy()
    for f in sorted(glob.glob(os.path.join(root, "test", "T*.npz"))):
        name = os.path.basename(f)[:-4]
        D = np.load(f)
        t0 = float(D["t0"])
        cd = ChannelData(iq=D["iq"][3], fs=2 * sim.FS_RF, t0=t0, angle=0.0)
        o = infer(model, cd)
        env = envelopes(o)
        # truth maps (reference-DAS units)
        ti = [fixed_bf(D["tissue"], t0)] + [fixed_bf(r, t0) for r in D["tissue_reals"]]
        E_t = sm(np.mean([np.abs(t) ** 2 for t in ti], 0))
        C_t = sm(np.abs(fixed_bf(D["clutter"], t0)) ** 2)
        N_t = sm(np.abs(fixed_bf(D["noise"], t0)) ** 2)
        incl = D["inclusions"]
        r = {"runtime_s": o["runtime"]}
        # lesion contrast linearity (finite designed contrasts), gCNR / CNR for all lesions
        bgmask = np.ones((NZ, NX), bool)
        for (zc, xc, rad, _) in incl:
            bgmask &= ~disk(zc, xc, rad + 1.5e-3)
        per = {k: [] for k in list(env) + ["TRUTH (tissue expectation)"]}
        env_t = dict(env); env_t["TRUTH (tissue expectation)"] = np.sqrt(E_t)
        for (zc, xc, rad, cdb) in incl:
            ins = disk(zc, xc, rad - 1.0e-3)
            out = ring(zc, xc, rad + 1.5e-3, rad + 4e-3) & bgmask
            for k, e in env_t.items():
                cm = contrast_metrics(e, ins, out); cm["design_dB"] = float(cdb)
                per[k].append(cm)
        lin = {}
        for k, v in per.items():
            fin = [p for p in v if np.isfinite(p["design_dB"])]
            xdb = np.array([p["design_dB"] for p in fin]); ydb = np.array([p["CR_dB"] for p in fin])
            sl, ic, rv, _, _ = stats.linregress(xdb, ydb)
            an = [p for p in v if not np.isfinite(p["design_dB"])]
            lin[k] = dict(slope=float(sl), r2=float(rv ** 2), CR_by_design={str(p["design_dB"]): round(float(p["CR_dB"]), 2) for p in v},
                          gCNR_anechoic=float(an[0]["gCNR"]) if an else np.nan,
                          gCNR_mean=float(np.mean([p["gCNR"] for p in v])),
                          CNR_mean=float(np.mean([p["CNR"] for p in v])),
                          speckle_SNR_bg=float(np.mean([p["SNR_out"] for p in v])))
        r["lesions"] = lin
        # component-map accuracy (log-domain correlation and median ratio)
        valid = (Z[:, None] > 6e-3) & np.ones((1, NX), bool)
        def cmp(est, tru, msk):
            a, b = np.log10(est[msk]), np.log10(tru[msk])
            return dict(pearson_log=float(np.corrcoef(a, b)[0, 1]), median_ratio_dB=float(10 * np.median(a - b)))
        strong_clutter = valid & (C_t > 0.1 * E_t)
        r["noise_map"] = cmp(o["noise"], N_t, valid)
        r["clutter_map"] = cmp(o["clutter"], C_t, strong_clutter) if strong_clutter.sum() > 100 else None
        r["clutter_fraction_pixels"] = float(strong_clutter.mean())
        # echogenicity accuracy vs truth after one global scale (units: coherent vs total power)
        la, lb = np.log10(o["sigma2"][valid]), np.log10(E_t[valid])
        off = np.median(la - lb)
        r["echo_map"] = dict(pearson_log=float(np.corrcoef(la, lb)[0, 1]), rmse_dB_after_global_scale=float(10 * np.sqrt(np.mean((la - lb - off) ** 2))))
        das_s = np.log10(sm(np.abs(o["y_ref"]) ** 2)[valid])
        off2 = np.median(das_s - lb)
        r["echo_map_smoothedDAS_baseline"] = dict(pearson_log=float(np.corrcoef(das_s, lb)[0, 1]), rmse_dB_after_global_scale=float(10 * np.sqrt(np.mean((das_s - lb - off2) ** 2))))
        r["calibration_bg"] = pit_stats(o, bgmask & valid)
        r["L_eff_validation"] = leff_validation(o, bgmask & valid & (Z[:, None] > 15e-3) & (Z[:, None] < 40e-3))
        res[name] = r
        if name.startswith("T1"):
            np.savez_compressed(os.path.join(outdir, "sim_T1_images.npz"), **{k: v.astype(np.float32) for k, v in env_t.items()},
                                clutter=o["clutter"], clutter_truth=C_t, noise=o["noise"], noise_truth=N_t, sigma2=o["sigma2"], E_truth=E_t)
            res["hallucination_insertion"] = insertion_test(model, D, t0, o)
    res["phase_fidelity"] = phase_test(model, root)
    res["hallucination_null"] = null_test(model)
    return res


def leff_validation(o, mask):
    """Empirical L_eff of the 4 adaptive looks vs speckle-SNR of their incoherent average."""
    zk = o["zk"]
    I = np.abs(zk) ** 2
    comp = I.mean(0)
    # measured intensity SNR of the look-compound in homogeneous speckle -> L_meas = SNR^2
    # (use fixed-apodization looks for a clean check; region-wide correlation matrix)
    v = zk[:, mask]
    Cm = v @ v.conj().T
    p = np.real(np.diag(Cm))
    rho2 = np.abs(Cm) ** 2 / (p[:, None] * p[None, :])
    L_pred = (p.sum() ** 2) / (p[:, None] * p[None, :] * rho2).sum()
    single = I[1][mask]
    L_single = single.mean() ** 2 / single.var()
    L_meas = comp[mask].mean() ** 2 / comp[mask].var()
    return dict(K=int(zk.shape[0]), L_naive=float(zk.shape[0]), L_pred_from_rho=float(L_pred),
                L_measured_ENL=float(L_meas), single_look_ENL=float(L_single),
                L_measured_over_single=float(L_meas / L_single),
                median_pixelwise_L_eff_estimator=float(np.median(o["L_eff"][mask])),
                rho2_offdiag=np.round(rho2, 3).tolist())


@torch.no_grad()
def phase_test(model, root):
    D = np.load(os.path.join(root, "test", "motion.npz"))
    t0 = float(D["t0"]); step = float(D["step"])
    frames = D["iq"]
    outs = [infer(model, ChannelData(iq=f, fs=2 * sim.FS_RF, t0=t0, angle=0.0), keep_w=True) for f in frames]
    zz, xx = torch.meshgrid(torch.from_numpy(Z), torch.from_numpy(X), indexing="ij")
    zz, xx = zz.reshape(-1), xx.reshape(-1)
    # frozen weights/gain from frame 0 (recommended Doppler mode)
    w0f, g0 = outs[0]["w"], outs[0]["gain"]
    frozen = []
    for f in frames:
        d = delay_gather(torch.from_numpy(f), 2 * sim.FS_RF, t0, 0.0, zz, xx, C0) * outs[0]["Mk"]
        frozen.append(g0 * (w0f * d).sum(-1).reshape(NZ, NX).numpy())
    series = {"DAS (1 PW)": [o["y_ref"] for o in outs], "LC-PIB y_pp (per-frame weights)": [o["y_pp"] for o in outs],
              "LC-PIB y_pp (frozen weights+gain)": frozen, "LC-PIB y (adaptive apod only, frozen)": [ (fr / np.maximum(g0, 1e-6)) for fr in frozen]}
    kw = model.kw
    region = (Z[:, None] > 8e-3) & (Z[:, None] < 45e-3) & (np.abs(X)[None, :] < 15e-3)
    tis = D["tissue"]
    res = {"true_axial_step_um": step * 1e6}
    for k, ys in series.items():
        est = []
        for a, b in zip(ys[:-1], ys[1:]):
            r = smooth(torch.from_numpy(b * np.conj(a)), kw).numpy()
            ph = np.angle(r)
            est.append(-ph * C0 / (4 * np.pi * F0))          # axial displacement [m]
        est = np.array(est)[:, region]
        wts = np.abs(np.array([ys[0]]))[:, region] ** 0     # uniform
        res[k] = dict(bias_um=float((np.mean(est) - step) * 1e6), std_um=float(np.std(est) * 1e6),
                      rmse_um=float(np.sqrt(np.mean((est - step) ** 2)) * 1e6))
    # phase difference of y_pp vs DAS on the same frame (intensity-weighted, speckle region)
    dphi = np.angle(outs[0]["y_pp"] * np.conj(outs[0]["y_ref"]))[region]
    wgt = (np.abs(outs[0]["y_ref"]) ** 2)[region]
    res["phase_diff_ypp_vs_DAS_deg"] = dict(weighted_median_abs=float(np.degrees(np.median(np.abs(dphi)))),
                                            weighted_mean_abs=float(np.degrees(np.sum(np.abs(dphi) * wgt) / wgt.sum())))
    return res


@torch.no_grad()
def null_test(model):
    """Pure band-limited channel noise: no tissue anywhere. Echogenicity should not exceed noise."""
    rng = np.random.default_rng(5)
    rf = sim.band_noise(rng, (128, sim.N_T))
    cd = sim.to_cd(rf, 0.0)
    o = infer(model, cd)
    region = Z[:, None] > 6e-3
    s, n, c = o["sigma2"][np.broadcast_to(region, o["sigma2"].shape)], o["noise"][np.broadcast_to(region, o["sigma2"].shape)], o["clutter"][np.broadcast_to(region, o["sigma2"].shape)]
    tot = s + n + c
    return dict(median_echo_fraction=float(np.median(s / tot)), p95_echo_fraction=float(np.percentile(s / tot, 95)),
                median_noise_fraction=float(np.median(n / tot)),
                echo_image_speckle_SNR=float(np.mean(np.sqrt(s)) / np.std(np.sqrt(s))))


@torch.no_grad()
def insertion_test(model, D, t0, o_base):
    """Add the channel data of one point scatterer (linear superposition) in background and check
    that the response appears only at the inserted location (no ghosts)."""
    zc, xc = 38e-3, -14e-3
    rf_pt = sim.simulate_rf(np.array([zc]), np.array([xc]), np.array([6.0]), 0.0)
    iq_pt = sim.to_cd(rf_pt, 0.0).iq
    cd = ChannelData(iq=D["iq"][3] + iq_pt, fs=2 * sim.FS_RF, t0=t0, angle=0.0)
    o = infer(model, cd)
    near = disk(zc, xc, 1.5e-3)
    res = {}
    for k, (a, b) in {"DAS intensity": (np.abs(o["y_ref"]) ** 2, np.abs(o_base["y_ref"]) ** 2),
                      "LC-PIB echogenicity": (o["sigma2"], o_base["sigma2"]),
                      "LC-PIB y_pp intensity": (np.abs(o["y_pp"]) ** 2, np.abs(o_base["y_pp"]) ** 2)}.items():
        diff = a - b
        pos = np.clip(diff, 0, None)
        res[k] = dict(energy_fraction_near_insert=float(pos[near].sum() / pos.sum()),
                      peak_near_insert=bool(np.unravel_index(np.argmax(diff), diff.shape) in set(map(tuple, np.argwhere(near)))),
                      rel_change_far=float(np.median(np.abs(diff[~disk(zc, xc, 4e-3)]) / b[~disk(zc, xc, 4e-3)])))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/full")
    ap.add_argument("--data", default="data/sim")
    ap.add_argument("--skip_picmus", action="store_true")
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.run, "train_log.json")))["args"]
    model = LCPIB(use_apod=not cfg.get("no_apod", False))
    model.load_state_dict(torch.load(os.path.join(a.run, "model.pt")))
    model.eval()
    res = {}
    if not a.skip_picmus:
        res["picmus"] = eval_picmus(model, a.run)
    res["sim"] = eval_sim(model, a.data, a.run)
    json.dump(res, open(os.path.join(a.run, "metrics.json"), "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float)[:3000])
