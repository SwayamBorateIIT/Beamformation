"""LC-PIB: Look-Calibrated Physics-Informed Beamformer (v1 implementation).

raw channel IQ -> differentiable ToF + F-number support -> constrained adaptive complex
apodization -> K complementary sub-aperture looks -> statistical estimator (coherence,
LOC noise, look correlation, L_eff, N_eff) -> unrolled Gamma-MAP estimator ->
{echogenicity sigma^2, clutter c, noise n, dispersion L} + phase-preserving IQ y_pp = g*y.

All intensities are in units of the reference beamformer (Hann apodization, F# = 1.5).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.signal import firwin

from .common import C0, F0, ELEMENT_X, N_EL, aperture_mask, delay_gather, hann_apod, look_windows

FNUM = 1.5
EPS = 1e-12


def abs2(t):
    """|t|^2 with a well-defined gradient at 0."""
    return t.real ** 2 + t.imag ** 2


def sabs(t, eps=1e-20):
    """|t| with a finite gradient at 0."""
    return torch.sqrt(abs2(t) + eps)


def gauss_kernel(sz, sx):
    """Separable Gaussian window stored as (kz, kx) 1-D kernels packed in a 2-D tensor
    of shape (1, 1, len(kz), len(kx)) = outer product (used for N_eff), plus 1-D factors."""
    rz, rx = int(3 * sz), int(3 * sx)
    z = torch.arange(-rz, rz + 1, dtype=torch.float32)
    x = torch.arange(-rx, rx + 1, dtype=torch.float32)
    kz = torch.exp(-z ** 2 / (2 * sz ** 2)); kz = kz / kz.sum()
    kx = torch.exp(-x ** 2 / (2 * sx ** 2)); kx = kx / kx.sum()
    return (kz[:, None] * kx[None, :])[None, None]


def smooth(img, k):
    """Gaussian window average of (..., H, W) real or complex images (reflect padding),
    implemented separably using the row/column marginals of the 2-D kernel k."""
    if img.is_complex():
        return torch.complex(smooth(img.real, k), smooth(img.imag, k))
    kz = k[0, 0].sum(1); kx = k[0, 0].sum(0)
    sh = img.shape
    x = img.reshape(-1, 1, sh[-2], sh[-1])
    pz, px = len(kz) // 2, len(kx) // 2
    x = F.pad(x, (0, 0, pz, pz), mode="reflect")
    x = F.conv2d(x, kz.view(1, 1, -1, 1))
    x = F.pad(x, (px, px, 0, 0), mode="reflect")
    x = F.conv2d(x, kx.view(1, 1, 1, -1))
    return x.reshape(sh)


class ApodNet(nn.Module):
    """Per coarse-pixel 1-D conv over the channel axis -> complex apodization update."""

    def __init__(self, cin=5, ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(cin, ch, 7, padding=3), nn.GELU(),
            nn.Conv1d(ch, ch, 7, padding=6, dilation=2), nn.GELU(),
            nn.Conv1d(ch, ch, 7, padding=12, dilation=4), nn.GELU(),
            nn.Conv1d(ch, 2, 1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, f):
        o = self.net(f)
        return torch.complex(o[:, 0], o[:, 1])


class Stage(nn.Module):
    def __init__(self, cin, ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, ch, 3, padding=1), nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=2, dilation=2), nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=4, dilation=4), nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.GELU(),
            nn.Conv2d(ch, 4, 1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


class LCPIB(nn.Module):
    def __init__(self, K=4, overlap=0.5, T=4, beta=3.0, coarse=4, win=(4.0, 3.0), use_apod=True, dz=45e-3 / 511):
        super().__init__()
        self.K, self.T, self.beta, self.coarse, self.use_apod = K, T, beta, coarse, use_apod
        self.register_buffer("looks", torch.from_numpy(look_windows(K, overlap)))
        self.register_buffer("el_x", torch.as_tensor(ELEMENT_X, dtype=torch.float32))
        self.register_buffer("kw", gauss_kernel(*win))
        self.register_buffer("kbig", gauss_kernel(16.0, 12.0))
        # axial high-pass on beamformed baseband IQ: keeps |f - f0| > 2.8 MHz (outside the
        # two-way echo band, inside the receiver noise band) -> single-frame noise cue.
        fs_z = C0 / (2 * dz)
        self.register_buffer("hp", torch.tensor(firwin(31, 2.8e6 / (fs_z / 2), pass_zero=False), dtype=torch.float32))
        self.apod = ApodNet()
        self.n_feat = 13 + K
        self.stages = nn.ModuleList([Stage(self.n_feat + 6) for _ in range(T)])
        self.eta = nn.Parameter(torch.full((T,), 0.3))

    def forward(self, f):
        o = self.net(f)
        return torch.complex(o[:, 0], o[:, 1])


class Stage(nn.Module):
    def __init__(self, cin, ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, ch, 3, padding=1), nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=2, dilation=2), nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=4, dilation=4), nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.GELU(),
            nn.Conv2d(ch, 4, 1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


class LCPIB(nn.Module):
    def __init__(self, K=4, overlap=0.5, T=4, beta=3.0, coarse=4, win=(4.0, 3.0), use_apod=True, dz=45e-3 / 511):
        super().__init__()
        self.K, self.T, self.beta, self.coarse, self.use_apod = K, T, beta, coarse, use_apod
        self.register_buffer("looks", torch.from_numpy(look_windows(K, overlap)))
        self.register_buffer("el_x", torch.as_tensor(ELEMENT_X, dtype=torch.float32))
        self.register_buffer("kw", gauss_kernel(*win))
        self.register_buffer("kbig", gauss_kernel(16.0, 12.0))
        # axial high-pass on beamformed baseband IQ: keeps |f - f0| > 2.8 MHz (outside the
        # two-way echo band, inside the receiver noise band) -> single-frame noise cue.
        fs_z = C0 / (2 * dz)
        self.register_buffer("hp", torch.tensor(firwin(31, 2.8e6 / (fs_z / 2), pass_zero=False), dtype=torch.float32))
        self.apod = ApodNet()
        self.n_feat = 13 + K
        self.stages = nn.ModuleList([Stage(self.n_feat + 6) for _ in range(T)])
        self.eta = nn.Parameter(torch.full((T,), 0.3))

    # ------------------------------------------------------------------ apodization
    def weights(self, d, M, zz, H, W, chan_keep):
        """Constrained complex apodization. d: (P, N) delayed data, M: (P, N) support."""
        Mk = M * chan_keep[None, :]
        w0 = hann_apod(zz, torch.zeros_like(zz), fnum=FNUM, el_x=self.el_x)  # placeholder shape
        return Mk, w0

    def forward(self, iq_t, fs, t0, angle, zz, xx, H, W, chan_keep=None, c=1540.0, d_pre=None):
        P = zz.numel()
        d = d_pre if d_pre is not None else delay_gather(iq_t, fs, t0, angle, zz, xx, c, el_x=self.el_x)
        M = aperture_mask(zz, xx, fnum=FNUM, el_x=self.el_x)
        if chan_keep is None:
            chan_keep = torch.ones(N_EL)
        Mk = M * chan_keep[None, :]
        d = d * Mk
        # reference apodization (Hann over F# aperture, masked channels removed, sum 1)
        w0 = hann_apod(zz, xx, fnum=FNUM, el_x=self.el_x) * chan_keep[None, :]
        w0 = w0 / w0.sum(-1, keepdim=True).clamp_min(1e-6)
        y_ref = (w0 * d).sum(-1)

        n_act = (Mk > 0.5).sum(-1, keepdim=True).clamp_min(1).float()
        if self.use_apod:
            # ---- features on coarse grid
            c_ = self.coarse
            e = (abs2(d)).reshape(H, W, N_EL)
            q = (d[:, 1:] * d[:, :-1].conj()).reshape(H, W, N_EL - 1)
            pool = lambda t: F.avg_pool2d(t.permute(2, 0, 1)[None], c_)[0].permute(1, 2, 0)
            e_c = pool(e)
            q_c = torch.complex(pool(q.real), pool(q.imag))
            en = e_c / e_c.mean(-1, keepdim=True).clamp_min(EPS)
            qn = q_c / torch.sqrt((e_c[..., 1:] * e_c[..., :-1]).clamp_min(EPS))
            qn = F.pad(qn, (0, 1))
            m_c = pool(Mk.reshape(H, W, N_EL))
            z_c = pool(zz.reshape(H, W, 1).expand(H, W, 1)) / 0.05
            feat = torch.stack([torch.log(en + 1e-3), qn.real, qn.imag, m_c, z_c.expand_as(m_c)], 2)
            hc, wc = feat.shape[:2]
            dw = self.apod(feat.reshape(-1, 5, N_EL))                               # (hc*wc, N)
            dw = dw.reshape(hc, wc, N_EL).permute(2, 0, 1)[None]
            dw = torch.complex(F.interpolate(dw.real, size=(H, W), mode="bilinear", align_corners=False),
                               F.interpolate(dw.imag, size=(H, W), mode="bilinear", align_corners=False))
            dw = dw[0].permute(1, 2, 0).reshape(P, N_EL)
            bound = self.beta / n_act
            mag = sabs(dw)
            dw = dw * (bound * torch.tanh(mag / bound) / mag.clamp_min(EPS))            # |dw| < bound
            w = (w0 + dw) * (Mk > 0.5)
            for _ in range(2):   # distortionless projection + magnitude clamp
                sup = (Mk > 0.5).float()
                w = w + (1 - w.sum(-1, keepdim=True)) * sup / sup.sum(-1, keepdim=True).clamp_min(1)
                mag = sabs(w)
                w = w * torch.clamp(bound / mag.clamp_min(EPS), max=1.0)
            w = w + (1 - w.sum(-1, keepdim=True)) * sup / sup.sum(-1, keepdim=True).clamp_min(1)
            dw_img = dw.reshape(H, W, N_EL)
        else:
            w = w0.to(torch.complex64)
            dw_img = None
        y = (w * d).sum(-1)

        # ---- complementary looks
        g = w[:, None, :] * self.looks[None]                                       # (P, K, N)
        gs = g.sum(-1, keepdim=True)
        valid = (sabs(gs) > 0.05).float()
        g = g / torch.where(sabs(gs) > 0.05, gs, torch.ones_like(gs))
        z = (g * d[:, None, :]).sum(-1) * valid[..., 0]                              # (P, K)

        img = lambda t: t.reshape(H, W)
        Y, Yref = img(y), img(y_ref)
        Z = z.T.reshape(self.K, H, W)
        k = self.kw
        I_ref = smooth(abs2(Yref), k)
        I_y = smooth(abs2(Y), k)
        I_k = smooth(abs2(Z), k)
        # empirical look correlation and L_eff (equal-weight incoherent combination)
        C = smooth(Z[:, None] * Z[None].conj(), k)                                  # (K, K, H, W)
        dg = torch.sqrt((I_k[:, None] * I_k[None]).clamp_min(EPS))
        rho2 = (abs2(C) / dg ** 2).clamp(0, 1)
        n_w = self.n_window(Yref)
        rho2c = ((rho2 - 1 / n_w) / (1 - 1 / n_w)).clamp(0, 1)
        eye = torch.eye(self.K)[:, :, None, None]
        rho2c = rho2c * (1 - eye) + eye
        L_eff = I_k.sum(0) ** 2 / (I_k[:, None] * I_k[None] * rho2c).sum((0, 1)).clamp_min(EPS)
        # channel coherence (LOC-type) at lags 1, 2, 4, 8
        E_ch = smooth(img((abs2(d)).sum(-1) / Mk.sum(-1).clamp_min(1)), k)
        gam = []
        for lag in (1, 2, 4, 8):
            num = img((d[:, lag:] * d[:, :-lag].conj()).sum(-1))
            den = img(torch.sqrt((abs2(d[:, lag:]).sum(-1) * abs2(d[:, :-lag]).sum(-1)).clamp_min(EPS)))
            gam.append((smooth(num, k).real / smooth(den, k).clamp_min(EPS)).clamp(-1, 1))
        gam = torch.stack(gam)
        s_coh = (2 * gam[0] - gam[1]).clamp(1e-3, 1.0)   # lag-0 extrapolated coherence (feature)
        w0n = img((w0 ** 2).sum(-1))
        # expected reference-DAS power if all channels were mutually incoherent (noise-like):
        # for single-PW data the tissue channel coherence is ~0 beyond lag 1, so LOC cannot
        # separate noise; this is only an initialisation / feature.
        n0 = (E_ch * w0n).clamp_min(EPS)

        # out-of-band (noise) power of the reference beamformer output
        hp = self.hp.view(1, 1, -1, 1)
        pad = hp.shape[2] // 2
        yr = (Yref * torch.polar(torch.ones_like(Zimg := img(zz)), -4 * np.pi * F0 * Zimg / C0))[None, None]
        yhp = torch.complex(F.conv2d(F.pad(yr.real, (0, 0, pad, pad), mode="reflect"), hp),
                            F.conv2d(F.pad(yr.imag, (0, 0, pad, pad), mode="reflect"), hp))[0, 0]
        P_oob = smooth(abs2(yhp), k).clamp_min(EPS)

        # ---- unrolled Gamma-MAP estimator (scale-equivariant via local log-normalisation)
        l0 = torch.log(smooth(I_ref, self.kbig).clamp_min(EPS))
        lg = lambda t: torch.log(t.clamp_min(EPS)) - l0
        zn = img(zz) / 0.05
        feat = torch.stack([lg(I_ref), lg(I_y), lg(I_k.mean(0)), *[lg(I_k[i]) for i in range(self.K)],
                            *gam, s_coh, lg(n0), L_eff / self.K, zn, lg(E_ch * w0n), lg(P_oob)])[None]
        u = torch.log((I_ref - n0).clamp_min(0.05 * I_ref).clamp_min(EPS))
        v = torch.log((0.05 * I_ref).clamp_min(EPS))
        nu = torch.log(3.0 * P_oob)
        logL = torch.zeros_like(u)
        for t, st in enumerate(self.stages):
            m = torch.exp(u) + torch.exp(v) + torch.exp(nu)
            r = 1 - I_ref / m
            gu, gv, gn = r * torch.exp(u) / m, r * torch.exp(v) / m, r * torch.exp(nu) / m
            x = torch.cat([feat, torch.stack([u - l0, v - l0, nu - l0, gu, gv, gn])[None]], 1)
            o = st(x)[0]
            u = u - self.eta[t] * gu + o[0].clamp(-3, 3)
            v = v - self.eta[t] * gv + o[1].clamp(-3, 3)
            nu = nu - self.eta[t] * gn + o[2].clamp(-3, 3)
            logL = logL + o[3]
        sig, clu, noi = torch.exp(u), torch.exp(v), torch.exp(nu)
        gain = sig / (sig + clu + noi)
        return dict(y=Y, y_ref=Yref, y_pp=gain * Y, sigma2=sig, clutter=clu, noise=noi, gain=gain,
                    L_hat=torch.exp(logL.clamp(-4, 6)), L_eff=L_eff, rho2=rho2c, I_ref=I_ref, n0=n0, P_oob=P_oob,
                    gamma=gam, s_coh=s_coh, E_ch=E_ch, w=w, w0=w0, d=d, Mk=Mk, n_w=n_w, dw=dw_img,
                    z_looks=Z)

    @torch.no_grad()
    def n_window(self, Y):
        """Effective number of independent samples in the Gaussian window:
        N_eff = (sum v)^2 / sum_ij v_i v_j |rho_sp(i-j)|^2, rho_sp from the IQ autocorrelation."""
        H, W = Y.shape
        Yc = Y - Y.mean()
        Fy = torch.fft.fft2(Yc, s=(2 * H, 2 * W))
        ac = torch.fft.ifft2(abs2(Fy))
        ac = ac / ac[0, 0].real
        ac = torch.fft.fftshift(abs2(ac))
        kz, kx = self.kw.shape[-2] // 2, self.kw.shape[-1] // 2
        cz, cx = H, W
        rho2 = ac[cz - 2 * kz:cz + 2 * kz + 1, cx - 2 * kx:cx + 2 * kx + 1]
        v = self.kw[0, 0]
        vv = F.conv2d(v[None, None], v[None, None], padding=(2 * kz, 2 * kx))[0, 0]  # autocorr of window
        vv = vv[: rho2.shape[0], : rho2.shape[1]]
        n = (v.sum() ** 2) / (vv * rho2).sum().clamp_min(EPS)
        return float(n.clamp(1.0, 1e4))
