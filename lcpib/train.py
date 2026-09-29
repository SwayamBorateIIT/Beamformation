"""Self-supervised training of LC-PIB on unlabeled simulated multi-angle channel data.

Objective (no clean targets anywhere):
  L = l_E  * L_E    energy consistency: window-averaged single-PW DAS power ~ Gamma(N_w, sig+c+n)
    + l_xv * L_xv   cross-view likelihood: intensities of *other* transmit angles whose measured
                    speckle correlation with the input is |rho|^2 <= eps, averaged, Gamma(L_B)
                    with L_B from the measured view-correlation matrix (look calibration)
    + l_t  * L_t    noise identification from a repeated acquisition with independent noise
    + l_ch * L_ch   masked-channel likelihood, rank-1 focal signal + incoherent residual
                    (Wiener / coherence-factor model) evaluated on held-out receive channels
    + l_coh* L_coh  complex coherence of y_pp with adjacent-angle DAS (phase-preserving output)
    + l_L  * L_disp local Gamma shape (dispersion) of the single-look intensity (uncertainty map)
    + R_w           apodization white-noise-gain cap + total variation
"""
import argparse
import glob
import json
import math
import os
import time

import numpy as np
import torch

from .common import ChannelData, beamform_image, delay_gather, make_grid, aperture_mask, N_EL
from .model import LCPIB, smooth, abs2

torch.set_num_threads(4)
NZ, NX = 512, 256
NB = 8   # depth bands for correlation statistics


def precompute(path, z, x):
    """Reference-DAS image for every angle and band-wise complex correlations between angles."""
    cache = path.replace(".npz", "_pre.npz")
    if os.path.exists(cache):
        c = np.load(cache)
        return c["Y"], c["rho2"]
    f = np.load(path)
    Y = []
    for j, a in enumerate(f["angles"]):
        cd = ChannelData(iq=f["iq"][j], fs=2 * 20.832e6, t0=float(f["t0"]), angle=float(a))
        Y.append(beamform_image(cd, z, x)[0])
    Y = np.array(Y)                                               # (A, NZ, NX)
    A = len(Y)
    rho2 = np.zeros((NB, A, A), np.float32)
    bs = NZ // NB
    for b in range(NB):
        yb = Y[:, b * bs:(b + 1) * bs].reshape(A, -1)
        num = yb @ yb.conj().T
        p = np.real(np.diag(num))
        rho2[b] = np.abs(num) ** 2 / (p[:, None] * p[None, :])
    np.savez(cache, Y=Y, rho2=rho2)
    return Y, rho2


def gamma_nll(J, m, L=1.0, tau=20.0):
    """Gamma negative log-likelihood (mean m, shape L) up to terms constant in m.
    Robustified for J/m > tau (log growth instead of linear) so that a collapsed estimate on
    a single patch cannot dominate the gradient; identical to the exact NLL for J/m <= tau."""
    rho = J / m
    rob = torch.where(rho <= tau, rho, tau * (1 + torch.log(rho.clamp_min(tau) / tau)))
    return L * (rob + torch.log(m))


def train(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    z, x = make_grid(nz=NZ)
    files = sorted(glob.glob(os.path.join(args.data, "train", "train_*.npz")))
    files = [f for f in files if not f.endswith("_pre.npz")]
    data = []
    for f in files:
        Y, rho2 = precompute(f, z, x)
        d = np.load(f)
        data.append(dict(iq=d["iq"], iq2=d["iq2"], angles=d["angles"], t0=float(d["t0"]), Y=Y, rho2=rho2))
    print(f"{len(data)} training phantoms; median rho2 to farthest angle per band:",
          np.round(np.median([d["rho2"][:, 3, 0] for d in data], 0), 3), flush=True)

    dev = torch.device(args.device)
    model = LCPIB(use_apod=not args.no_apod).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1)
    H = W = args.patch
    zt, xt = torch.from_numpy(z).to(dev), torch.from_numpy(x).to(dev)
    log = []
    t_start = time.time()
    for step in range(args.steps):
        d = data[rng.integers(len(data))]
        A = len(d["angles"])
        a = 3 if rng.random() < 0.5 else int(rng.integers(A))
        frame = int(rng.integers(2))
        iq1 = torch.from_numpy((d["iq"], d["iq2"])[frame][a]).to(dev)
        iq2 = torch.from_numpy((d["iq2"], d["iq"])[frame][a]).to(dev)
        i0 = int(rng.integers(0, NZ - H)); j0 = int(rng.integers(0, NX - W))
        zz, xx = torch.meshgrid(zt[i0:i0 + H], xt[j0:j0 + W], indexing="ij")
        zz, xx = zz.reshape(-1), xx.reshape(-1)
        ang = float(d["angles"][a])
        with torch.no_grad():
            d_all = delay_gather(iq1, 2 * 20.832e6, d["t0"], ang, zz, xx, 1540.0)
            d_two = delay_gather(iq2, 2 * 20.832e6, d["t0"], ang, zz, xx, 1540.0)
        masked = (not args.no_ch) and rng.random() < 0.5
        keep = torch.ones(N_EL, device=dev)
        if masked:
            xc = float(xt[j0 + W // 2]); ec = int(round(xc / 0.3e-3 + (N_EL - 1) / 2))
            c0 = int(np.clip(ec + rng.integers(-8, 9) - 8, 0, N_EL - 16))
            keep[c0:c0 + 16] = 0
        o = model(iq1, 2 * 20.832e6, d["t0"], ang, zz, xx, H, W, chan_keep=keep, d_pre=d_all)
        sig, clu, noi = o["sigma2"], o["clutter"], o["noise"]
        m_tot = sig + clu + noi
        n_w = o["n_w"]
        losses = {}
        # ---- L_E: energy consistency with the same single-PW view
        losses["E"] = gamma_nll(o["I_ref"], m_tot, n_w).mean() - torch.log(o["I_ref"]).mean() * n_w
        # ---- L_t: noise from repeated acquisition (static scene): E|y1 - y2|^2 = 2 n
        with torch.no_grad():
            y2 = (o["w0"] * d_two * o["Mk"]).sum(-1).reshape(H, W)
            Dn = smooth((o["y_ref"] - y2).abs() ** 2, model.kw) / 2
        losses["t"] = gamma_nll(Dn, noi, n_w).mean() - torch.log(Dn.clamp_min(1e-12)).mean() * n_w
        # ---- L_xv: correlation-selected cross-view likelihood (other transmit angles)
        if not args.no_xv:
            band = min(NB - 1, (i0 + H // 2) // (NZ // NB))
            r2 = d["rho2"][band][a]
            others = [b for b in range(A) if b != a]
            if args.xv_nocorr:
                V = others
                L_B = float(len(V))                            # assumes independent looks
            else:
                V = [b for b in others if r2[b] <= args.eps]
                if len(V) == 0:
                    V = [others[int(np.argmin(r2[others]))]]
                R = d["rho2"][band][np.ix_(V, V)]
                L_B = float(len(V) ** 2 / R.sum())             # effective number of looks
            J = torch.from_numpy(np.mean(np.abs(d["Y"][V, i0:i0 + H, j0:j0 + W]) ** 2, 0)).to(dev)
            if args.xv_mse:
                losses["xv"] = ((torch.log(m_tot) - torch.log(J.clamp_min(1e-12))) ** 2).mean()
            else:
                losses["xv"] = gamma_nll(J, m_tot, L_B).mean() - L_B * torch.log(J.clamp_min(1e-12)).mean()
            losses["xv"] = losses["xv"] / max(L_B, 1.0)
        # ---- L_ch: held-out receive channels, rank-1 focal signal + incoherent residual
        if masked:
            M = aperture_mask(zz, xx, fnum=1.5) > 0.5
            obs = M & (keep[None, :] > 0)
            hold = M & (keep[None, :] == 0)
            m_cnt = hold.sum(-1)
            ok = (m_cnt >= 4) & (obs.sum(-1) >= 8)
            if ok.any():
                s2 = sig.reshape(-1)[ok]
                q = ((clu + noi).reshape(-1)[ok] / (o["w0"][ok] ** 2).sum(-1)).clamp_min(1e-12)
                dd = d_all[ok]
                ob, ho, mc = obs[ok].float(), hold[ok].float(), m_cnt[ok].float()
                n_obs = ob.sum(-1)
                V_ = 1.0 / (1.0 / s2 + n_obs / q)
                s_hat = V_ * (dd * ob).sum(-1) / q
                r = (dd - s_hat[:, None]) * ho
                rr = abs2(r).sum(-1)
                rs = abs2(r.sum(-1))
                nll = (mc - 1) * torch.log(q) + torch.log(q + mc * V_) + (rr - V_ * rs / (q + mc * V_)) / q
                base = (mc * torch.log((dd.abs() ** 2 * ho).sum(-1) / mc + 1e-12))
                losses["ch"] = ((nll - base) / mc).mean()
        # ---- L_coh: phase-preserving output coherent with adjacent-angle DAS
        if not args.no_coh:
            band = min(NB - 1, (i0 + H // 2) // (NZ // NB))
            nb = [b for b in (a - 1, a + 1) if 0 <= b < A]
            lc = 0
            for b in nb:
                yb = torch.from_numpy(d["Y"][b, i0:i0 + H, j0:j0 + W]).to(dev)
                num = smooth(o["y_pp"] * yb.conj(), model.kw)
                den = torch.sqrt((smooth(abs2(o["y_pp"]), model.kw) * smooth(abs2(yb), model.kw)).clamp_min(1e-20))
                lc = lc + math.sqrt(d["rho2"][band][a, b]) * (1 - num.real / den).mean()
            losses["coh"] = lc / len(nb)
        # ---- L_disp: local Gamma shape of single-look intensity (mean detached)
        I1 = abs2(o["y_ref"])
        Lh = o["L_hat"]
        md = m_tot.detach()
        losses["disp"] = (torch.lgamma(Lh) - Lh * torch.log(Lh) + Lh * torch.log(md)
                          - (Lh - 1) * torch.log(I1.clamp_min(1e-12)) + Lh * I1 / md + torch.log(I1.clamp_min(1e-12))).mean()
        # ---- regularisers on the adaptive apodization
        if o["dw"] is not None:
            wn = (o["w"].abs() ** 2).sum(-1) / (o["w0"] ** 2).sum(-1).clamp_min(1e-12)
            losses["wng"] = torch.relu(wn - 2.0).mean()
            dw = o["dw"]
            losses["tv"] = (dw[1:] - dw[:-1]).abs().mean() + (dw[:, 1:] - dw[:, :-1]).abs().mean()
        lam = dict(E=1.0, t=1.0, xv=1.0, ch=0.2, coh=args.l_coh, disp=0.1, wng=0.1, tv=10.0)
        if args.coh_only:
            losses = {k: v for k, v in losses.items() if k in ("E", "t", "coh", "wng", "tv", "disp")}
        loss = sum(lam[k] * v for k, v in losses.items())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); sched.step()
        rec = {k: float(v.detach() if torch.is_tensor(v) else v) for k, v in losses.items()}
        rec["step"] = step
        log.append(rec)
        if step % 50 == 0 or step == args.steps - 1:
            mean = {k: np.mean([r[k] for r in log[-50:] if k in r]) for k in rec if k != "step"}
            print(step, f"{time.time() - t_start:.0f}s", {k: round(v, 4) for k, v in mean.items()}, flush=True)
    os.makedirs(args.out, exist_ok=True)
    torch.save(model.cpu().state_dict(), os.path.join(args.out, "model.pt"))
    json.dump(dict(args=vars(args), log=log), open(os.path.join(args.out, "train_log.json"), "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sim")
    ap.add_argument("--out", default="results/full")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--l_coh", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no_ch", action="store_true")
    ap.add_argument("--no_xv", action="store_true")
    ap.add_argument("--no_coh", action="store_true")
    ap.add_argument("--no_apod", action="store_true")
    ap.add_argument("--xv_mse", action="store_true")
    ap.add_argument("--xv_nocorr", action="store_true")
    ap.add_argument("--coh_only", action="store_true")
    train(ap.parse_args())
