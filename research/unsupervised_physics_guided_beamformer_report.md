# Unsupervised, Physics-Guided Single-Plane-Wave Beamforming with Look-Calibrated Speckle Statistics

**Research report: literature review, novelty assessment, and a proposed architecture**
Date of search: 29 September 2026. Search window: 2024 to September 2026, plus the older work that the design depends on.

---

## 0. Scope, repository status, and how sources were checked

**Repository files.** The four paths you listed (`my_method/PE-PIB_implementation_spec.md`, `my_method/pe_pib.py`, `datasets_picmus_cubdl/dcl/`, `datasets_picmus_cubdl/unet_supervised/`) are **not in the repository**. The branch `claude/physics-guided-ultrasound-beamformer-2r1nhe` had no commits and the remote had no branches when I looked. So this report is based on your written description of PE-PIB, not on the spec or code. Once those files are committed, Section D should be checked against them. It is written so it can be mapped onto an existing PE-PIB module list.

**How sources were checked.** Direct page fetches (arXiv, DOI resolvers, PubMed Central, MDPI) were blocked by this environment's network proxy. Every citation was therefore checked through search-engine metadata: title, authors, venue, DOI and abstract. **I did not read full texts.** Method details below come from abstracts and publisher summaries. Where a detail matters for a novelty claim, it is marked *"verify in full text"*. A few classical references could not be link-checked here. They are marked **[DOI not verified in this session]**.

**Status labels used throughout:**
**[PR]** peer-reviewed journal. **[PR-conf]** peer-reviewed conference. **[Preprint]** arXiv/TechRxiv, not yet peer-reviewed. **[Accepted]** accepted according to the preprint page.

---

## A. Executive conclusion

**Recommended direction.** Build one **constrained, complex-apodization beamformer plus a small unrolled statistical estimator**. Train it with a **look-calibrated, cross-view speckle likelihood** as the main objective, supported by a **held-out-channel raw-data likelihood** and a **DCL-style complex coherence term** for the phase-preserving output. Give the method **two outputs by design**:

1. **Echogenicity output** (despeckled). This is the estimated *expected intensity over speckle realizations*: the local mean backscatter power, separated from clutter and thermal noise.
2. **Speckle-preserving coherent IQ output.** This is `y_pp = g(x) · wᴴ(x) d(x)`: a bounded linear combination of the delayed channel data times a *real, non-negative* Wiener-type gain `g ≤ 1`. Speckle is kept, so Doppler, elastography and speckle tracking still work. Clutter and noise are attenuated. Phase is unchanged up to the known phase of the apodized sum.

**Why this direction and not others:**

- **Coherence-only unsupervised beamforming is already published.** Deep Coherence Learning (DCL) (Cho et al., 2024) and its 2025 follow-ups (DCL-A, UBF-DCL_opt) train single-plane-wave beamformers with no labels, using coherence across plane-wave angles ([Cho 2024](https://doi.org/10.1016/j.ultras.2024.107408); [Kim 2025](https://doi.org/10.3390/diagnostics15243193); [UBF-DCL_opt](https://doi.org/10.3390/diagnostics16010058)). A new coherence loss alone is an incremental contribution.
- **No reviewed work had the following gaps filled.** In what I found, none of the unsupervised beamformers:
  - (a) defines its target as the *speckle-ensemble mean* instead of a particular compounded speckle realization;
  - (b) accounts for correlation between training "views" (overlapping sub-apertures, nearby angles) through a calibrated **effective number of looks, L_eff**;
  - (c) scores the result with a **proper likelihood on linear intensity** plus a **goodness-of-fit / calibration check**;
  - (d) keeps the phase-preserving IQ output separate from the despeckled echogenicity output.
  These gaps are real. They are also statistically well-posed, which makes them publishable without overclaiming.
- **The statistics are mature and can be borrowed directly.** Multi-look Gamma/Wishart models and equivalent-number-of-looks (ENL) estimation are well established in SAR ([Anfinsen 2009](https://www.researchgate.net/publication/220823003_Estimation_of_the_Equivalent_Number_of_Looks_in_Polarimetric_SAR_Imagery)). So are self-supervised likelihood despeckling ([MERLIN](https://doi.org/10.1109/TGRS.2021.3128621)) and speckle correlation versus aperture translation ([Trahey 1986](https://doi.org/10.1109/T-UFFC.1986.26827); [Mallart & Fink 1991](https://doi.org/10.1121/1.401867)). Carrying them into a *raw-channel* ultrasound beamformer has strong theoretical support and is low-risk.
- **Neural operators should not be the beamformer in version 1** (Section G). At clinical pulse-echo frequencies the operator you would need is the delay/Born operator, which is already known exactly enough. The unknown parts are statistical: clutter, noise and speckle. An FNO does not address those.

**One-sentence thesis you can defend:** *"A single-plane-wave beamformer can be trained without labels to estimate echogenicity in an unbiased and calibrated way, and to output a speckle-preserving coherent IQ signal, if training views are chosen and weighted by their measured speckle correlation (L_eff) and the loss is a proper likelihood on linear intensity."*

---

## B. Literature table (2024 to Sep 2026, plus required foundations)

### B.1 Unsupervised / self-supervised beamforming and plane-wave enhancement

| Year | Paper | Method | Input domain | Supervision | Contribution | Limitations (vs. your goals) | Link | Status |
|---|---|---|---|---|---|---|---|---|
| 2024 | Cho, Park, Kang, Yoo, *Deep coherence learning: an unsupervised deep beamformer for high-quality single plane-wave imaging* (Ultrasonics 143:107408) | Network trained to predict highly correlated signals across a set of PW frames; complex-baseband formulation | Complex baseband channel data (single PW at inference) | Unsupervised (multi-PW coherence at training) | First explicitly unsupervised single-PW deep beamformer; resolution comparable to 1-PW DMAS / 75-PW DAS per abstract | Target is coherence with *other realizations*, so speckle is treated implicitly and no statistical calibration is claimed; no echogenicity/uncertainty outputs; phase fidelity for Doppler not reported in abstract (*verify in full text*) | [DOI](https://doi.org/10.1016/j.ultras.2024.107408) · [arXiv](https://arxiv.org/abs/2311.11169) · [PubMed](https://pubmed.ncbi.nlm.nih.gov/39094387/) | [PR] |
| 2025 | Kim, Hwang, Song, Kang, *DCL-A: adaptive deep coherence loss for SPWI* (Diagnostics 15(24):3193) | DCL with linear/nonlinear weights by angular distance between input and target frames | Channel data, single PW | Unsupervised | 7 dB (sim) / 14 dB (phantom) better peak range sidelobe level vs. comparisons | Weights set heuristically by angle, not by measured correlation; no speckle statistics | [DOI](https://doi.org/10.3390/diagnostics15243193) · [PubMed](https://pubmed.ncbi.nlm.nih.gov/41464197/) | [PR] |
| 2025/26 | *Unsupervised beamforming with optimized coherence loss for clutter suppression in SPWI* (Diagnostics 16(1):58) | Chooses target frames by a physics criterion (axial displacement ≥ 2λ from steering) so clutter decorrelates across angles while tissue stays coherent | Channel data, single PW | Unsupervised | ~16% gain over plain DCL; explicitly targets reverberation clutter | Decorrelation criterion is geometric and fixed, not measured per region; closest prior for "clutter decorrelates across angles" | [DOI](https://doi.org/10.3390/diagnostics16010058) · [MDPI](https://www.mdpi.com/2075-4418/16/1/58) | [PR] |
| 2025 | Liu, Jiang, Dai, Zhang, *Single-PW imaging with implicit multi-angle acoustic synthesis* (IEEE TUFFC 72(4):479–497) | Network generates virtual steered PWs from one PW and combines them | Channel/beamformed PW data | Trained against multi-angle data (*verify exact supervision*) | Implicit angular synthesis; public code | Uses compounded / multi-angle targets; not a statistical-echogenicity method | [DOI](https://doi.org/10.1109/TUFFC.2025.3541113) · [PubMed](https://pubmed.ncbi.nlm.nih.gov/40031850/) · [code](https://github.com/yijiaLiu12/Implicit-Plane-Wave-Synthesis) | [PR] |
| 2024 | Asgariandehkordi, Goudarzi, Sharifzadeh, Basarab, Rivaz, *Denoising PW images using diffusion probabilistic models* (TUFFC 71(11):1526–1539) | DDPM on beamformed RF; treats low-angle vs. high-angle difference as "noise" | Beamformed RF | Paired low/high-angle (compounding-derived targets) | Keeps speckle pattern | Needs compounded targets; iterative sampling is slow; post-beamforming | [DOI](https://doi.org/10.1109/TUFFC.2024.3448209) · [PubMed](https://pubmed.ncbi.nlm.nih.gov/39186422/) | [PR] |
| 2025 | Asgariandehkordi, Sharifzadeh, Rivaz, *Lightweight physics-aware zero-shot PW denoising* | Splits angles into two disjoint subsets, forms two compounds, trains a 2-layer CNN per test sample (Noise2Noise-like) | Beamformed compounds | Self-supervised, zero-shot | Physics-motivated angle-split pairing | Needs several angles at *inference*; assumes the two subsets have independent artifacts (correlation not calibrated) | [arXiv](https://arxiv.org/abs/2506.21499) | [Preprint] |
| 2025/26 | *Self-supervised deep learning for denoising in ultrasound microvascular imaging* (HA2HA) (Biomed. Signal Process. Control) | Half-angle-to-half-angle pairs from complementary angle subsets | Beamformed RF blood-flow data | Self-supervised (N2N) | >15 dB CNR/SNR gain; in vivo pig/human | Blood-flow domain; assumes the two noise realizations are independent | [arXiv](https://arxiv.org/abs/2507.05451) · [journal](https://www.sciencedirect.com/science/article/abs/pii/S1746809426009225) | [PR] |
| 2024 | Stevens et al., *Dehazing ultrasound using diffusion models* (IEEE TMI 43(10):3546–3558) | Joint posterior sampling with two diffusion priors (tissue and haze) trained on RF | RF (beamformed) | Unsupervised (separate priors) | Clutter/haze separation; shows RF-domain training is better than image-domain | Needs samples to train the "haze" prior; slow; no calibrated likelihood | [DOI](https://doi.org/10.1109/TMI.2024.3363460) · [arXiv](https://arxiv.org/abs/2307.11204) | [PR] |
| 2025 | Stevens, Nolan, Robert, van Sloun, *Sequential posterior sampling with diffusion models* (ICASSP 2025) | Models transition dynamics to speed up sequential DPS | Ultrasound image sequences | Generative prior | Real-time direction for DPS | Image-level; not a beamformer | [arXiv](https://arxiv.org/abs/2409.05399) | [PR-conf] |
| 2025 | van de Schaft et al., *Off-grid ultrasound imaging by stochastic optimization* (INFER) (TUFFC 72(9)) | Jointly optimizes scatterer positions, reflectivities and model parameters incl. speed of sound | Raw channel data | None (model-based inverse problem) | 2–3× lateral resolution, 6–68% gCNR gain in vivo; robust to ±100 m/s SoS | Per-image optimization (slow); point-reflectivity model, not a speckle-statistical echogenicity model | [DOI](https://doi.org/10.1109/TUFFC.2025.3586377) · [arXiv](https://arxiv.org/abs/2407.02285) · [code](https://github.com/vincentvdschaft/off-grid-ultrasound) | [PR] |
| 2025 | Zhang et al., *PW imaging as an inverse problem with joint sparse regularization* (J Ultrasound Med) | ℓ1 + ℓ2,1 regularized inverse problem, ADMM | Channel data | None | Better texture fidelity than ℓ1 | Hand-crafted prior; 5.3 s/frame | [DOI](https://doi.org/10.1002/jum.70095) | [PR] |
| 2025 | Amar et al., *Deep task-based beamforming and channel-data augmentations* | Beamformer trained jointly with a classifier; channel-data augmentations | Channel data | Supervised (task labels) | Channel-domain augmentation recipes | Needs labels | [arXiv](https://arxiv.org/abs/2502.00524) | [Preprint] |
| 2025 | Stevens et al., *zea: a toolbox for cognitive ultrasound imaging* | Differentiable, modular pipeline (Keras 3; JAX/PyTorch/TF), includes a DBUA example | Channel to image | Tooling | Practical differentiable beamforming stack | Tooling, not a method | [arXiv](https://arxiv.org/abs/2512.01433) · [docs](https://zea.readthedocs.io/) | [Preprint] |

### B.2 Self-supervised speckle reduction (post-beamforming) and statistical losses

| Year | Paper | Method | Input | Supervision | Contribution | Limitations | Link | Status |
|---|---|---|---|---|---|---|---|---|
| 2025 | Li, Navab, Jiang, *Speckle2Self* (Med. Image Anal. 2025, 103755) | Multi-scale perturbation; clean image modeled as low-rank, speckle as sparse | Single B-mode image | Self-supervised | Argues explicitly that Noise2Noise fails because speckle is tissue-dependent | Image domain (post-log/post-beamforming); no raw-data physics; no likelihood calibration | [DOI](https://doi.org/10.1016/j.media.2025.103755) · [arXiv](https://arxiv.org/abs/2507.06828) | [PR] |
| 2026 | Li et al., *Mask2Restore: self-supervised despeckling via inpainting* | Block-wise masking and region inpainting, because speckle is spatially correlated (pixel blind-spot methods fail) | Single image | Self-supervised | Correct observation that blind-spot masks must exceed the speckle correlation length | Image domain; preprint from 26 Sep 2026 | [arXiv](https://arxiv.org/abs/2609.32844) | [Preprint] |
| 2022 | Göbl, Hennersperger, Navab, *Speckle2Speckle* | N2N on repeated images plus an interface term | Images | Unsupervised | Early ultrasound N2N | Needs decorrelated speckle pairs (not checked statistically) | [arXiv](https://arxiv.org/abs/2208.00402) | [Preprint] |
| 2023 | Viñals, Thiran, *KL-divergence-based loss for in vivo ultrafast image enhancement* (J. Imaging 9(12):256) | Loss that preserves the distribution of echogenicity values (KL) over high-dynamic-range RF | RF/IQ single PW | Supervised (compounded targets) | Distribution-aware loss; notes Rayleigh envelope SNR = 1.91 | Supervised; matches histograms, not a per-pixel likelihood | [DOI](https://doi.org/10.3390/jimaging9120256) | [PR] |
| 2026 | Lee, Choi, *Null-space diffusion restoration with uncertainty-guided fusion* (UGNS) | Range-null-space DDNM on an envelope proxy obtained by inverse log compression | B-mode (inverse-log) | Label-free (generative prior) | Uncertainty-guided fusion | Works from log-compressed data (your constraint rules this out); image-domain | [arXiv](https://arxiv.org/abs/2608.29820) | [Accepted] (IEEE Access, per preprint) |
| 2022 | Dalsasso, Denis, Tupin, *MERLIN* (IEEE TGRS 60) | Predicts reflectivity from the real part; scores with the likelihood of the imaginary part (they are independent in SAR SLC) | Complex SAR (SLC) | Self-supervised, single image | Template for **likelihood-based self-supervision on complex data** | SAR-specific independence assumption. **Does not transfer directly to ultrasound IQ** (see F.2) | [DOI](https://doi.org/10.1109/TGRS.2021.3128621) · [arXiv](https://arxiv.org/abs/2110.13148) | [PR] |
| 2026 | Pillai et al., *Physics-guided self-supervised statistical residual learning for sonar despeckling* | Log-ratio residual constrained to follow multiplicative speckle statistics (variance-targeted loss) | Sonar images | Self-supervised | Closest *statistical-residual* self-supervision in acoustics | Sonar images, log domain, image-level | [arXiv](https://arxiv.org/abs/2605.24716) | [Preprint] |
| 2026 | Smith, Raza, *Compression hurts, pooling helps: information loss in Rayleigh-scale estimation from B-mode* | Fisher-information analysis of log compression | B-mode vs. envelope | Theory | Unknown log compression inflates the variance bound ≥ 34.98× per window; ~340 windows needed to get inflation below 1.1 | Supports your rule "do not fit PDFs to log B-mode" | [arXiv](https://arxiv.org/abs/2609.19525) | [Preprint] |

### B.3 Adaptive / coherence-based beamforming, complex networks, aperture statistics

| Year | Paper | Contribution relevant here | Link | Status |
|---|---|---|---|---|
| 2020 | Luijten et al., *Adaptive ultrasound beamforming using deep learning (ABLE)* (IEEE TMI 39(12)) | Network predicts per-pixel apodization weights (MV-inspired structure). **Closest to "constrained adaptive complex apodization"**, but trained with supervision | [DOI](https://doi.org/10.1109/TMI.2020.3008537) | [PR] |
| 2022 | Lu et al., *Complex CNNs for ultrafast imaging from IQ* (CID-Net) (TUFFC 69(2)) | Complex-valued processing beats split real/imag networks; motivates complex weights | [DOI](https://doi.org/10.1109/TUFFC.2021.3127916) | [PR] |
| 2019 | Hyun, Brickson, Looby, Dahl, *Beamforming and speckle reduction using neural networks* (TUFFC 66(5)) | Channel data to speckle-reduced echogenicity; trained in silico, works in vivo. **Closest prior for "echogenicity from channel data"**, but supervised by simulation | [DOI](https://doi.org/10.1109/TUFFC.2019.2903795) | [PR] |
| 2020 | Wiacek et al., *CohereNet* (TUFFC) | DNN estimates spatial-coherence functions for SLSC | [DOI](https://doi.org/10.1109/TUFFC.2020.2982848) | [PR] |
| 2011 | Lediju et al., *Short-lag spatial coherence (SLSC)* (TUFFC 58(7)) | Coherence as an image; basis for the coherence features in D | [DOI](https://doi.org/10.1109/TUFFC.2011.1957) | [PR] |
| 2018 | Long, Bottenus, Trahey, *Lag-one coherence (LOC)* (TUFFC 65(10)) | Nearest-element coherence gives a **local estimate of thermal and acoustic noise**. Used for the noise map n̂ | [DOI](https://doi.org/10.1109/TUFFC.2018.2855653) | [PR] |
| 2010 | Nilsen, Holm, *Wiener beamforming and the coherence factor* (TUFFC 57(6)) | Wiener postfilter derivation; basis for the real gain `g(x)` in the phase-preserving output | [DOI](https://doi.org/10.1109/TUFFC.2010.1553) | [PR] |
| 2009 | Montaldo et al., *Coherent plane-wave compounding* (TUFFC 56(3)) | CPWC; physics of coherence across angles | [DOI](https://doi.org/10.1109/TUFFC.2009.1067) | [PR] |
| 1991 | Mallart, Fink, *Van Cittert–Zernike theorem in pulse-echo* (JASA 90(5)) | Spatial covariance of backscatter across the aperture; theory for **predicting look correlation from apodizations** | [DOI](https://doi.org/10.1121/1.401867) | [PR] |
| 1986 | Trahey, Smith, von Ramm, *Speckle correlation with lateral aperture translation* (TUFFC 33(3)) | Speckle correlation versus sub-aperture translation: **overlapping looks are correlated** | [DOI](https://doi.org/10.1109/T-UFFC.1986.26827) | [PR] |
| 1988 | O'Donnell, Silverstein, *Optimum displacement for compound image generation* (TUFFC 35(4)) | Compounding trade-off of decorrelation vs. resolution | [DOI](https://doi.org/10.1109/58.4184) | [PR] |
| 2017 | Rindal, Rodriguez-Molares, Austeng, *Dark region artifact in adaptive beamforming* (IUS 2017) | Adaptive beamformers can create false dark regions, which inflates contrast | [IEEE](https://ieeexplore.ieee.org/document/8092255/) | [PR-conf] |
| 2019 | Rindal et al., *Effect of dynamic-range alterations on contrast* (TUFFC 66(7)) | CR/CNR gains can come from simple gray-level transforms (R²≈0.88–0.98). **Required hallucination/metric control** | [DOI](https://doi.org/10.1109/TUFFC.2019.2911267) | [PR] |

### B.4 Sound-speed / aberration autofocus

| Year | Paper | Method | Supervision | Link | Status |
|---|---|---|---|---|---|
| 2023 | Simson et al., *Differentiable beamforming for ultrasound autofocusing (DBUA)* (MICCAI 2023) | Backpropagates a focus criterion through differentiable DAS to a SoS map | None (per-acquisition optimization) | [DOI](https://doi.org/10.1007/978-3-031-43999-5_41) · [code](https://github.com/waltsims/dbua) | [PR-conf] |
| 2025/26 | Simson et al., *Ultrasound autofocusing: common-midpoint phase error (CMPE) optimization via differentiable beamforming* (IEEE TMI 45(2):681–692) | CMPE quantifies phase aberration in diffuse media; optimizes velocity field | None | [DOI](https://doi.org/10.1109/TMI.2025.3607875) · [arXiv](https://arxiv.org/abs/2410.03008) | [PR] |
| 2026 | *Lens-aware differentiable beamforming for in vivo distributed aberration correction with curvilinear transducers* | CMPE extended with differentiable bent-ray lens refraction; 321 acquisitions from 81 high-BMI liver subjects | None | [arXiv](https://arxiv.org/abs/2608.06853) | [Preprint] |
| 2025/26 | *Minimal angular compounding required for coherence-based sound speed estimation* (Ultrasonics) | SLSC-maximizing SoS; 3–7 angles match 31–75 angles | None | [journal](https://www.sciencedirect.com/science/article/abs/pii/S0041624X2500352X) · [PubMed](https://pubmed.ncbi.nlm.nih.gov/41401623/) | [PR] |
| 2025/26 | *Robust deep learning for pulse-echo SoS imaging via time-shift maps* (IEEE TMI) | Physics-guided DL on multi-angle time-shift maps; ray-tracing pretrain then full-wave fine-tune | Supervised (simulation) | [DOI](https://doi.org/10.1109/TMI.2025.3602000) · [PubMed](https://pubmed.ncbi.nlm.nih.gov/40844937/) | [PR] |
| 2026 | Sode, Pinton, *IQ-JEPA: Hermitian ViT for SoS and attenuation from IQ* | Self-supervised JEPA pretraining on complex IQ, then fine-tuning (8.71 m/s error) | SSL + supervised fine-tune | [arXiv](https://arxiv.org/abs/2607.22351) | [Preprint] |
| 2022 | Khan, Huh, Ye, *Phase-aberration-robust PW beamformer via self-supervised learning* | Models SoS variation as stochastic; self-supervised 3D CNN | Self-supervised | [arXiv](https://arxiv.org/abs/2202.08262) | [Preprint] (pre-2024 background) |

### B.5 Masked / raw-channel self-supervision and clutter

| Year | Paper | Relevance | Link | Status |
|---|---|---|---|---|
| 2025 | Roßteutscher et al., *Masked autoencoders for ultrasound signals* | MAE on 1-D raw ultrasound (NDT/SHM). Shows masked pretraining works on raw A-lines, but not medical beamforming | [arXiv](https://arxiv.org/abs/2508.20622) | [Preprint] |
| 2025 | *RF-MAE: adaptive frequency masked autoencoder* | Frequency-domain masking on RF (radio signals, not ultrasound) | [ResearchGate](https://www.researchgate.net/publication/397697529_RF-MAE_A_Self-Supervised_Adaptive_Frequency_Masked_Autoencoder_With_Radio-Frequency_Signal_Processing_Applications) | (venue unclear) |
| 2025 | *U2-rPCA: unsupervised unfolded rPCA for clutter filtering* | Unrolled IRLS-rPCA trained without labels. Template for **unsupervised unrolling** | [arXiv](https://arxiv.org/abs/2510.00660) | [Preprint] |
| — | **Masked-channel self-supervision for a medical ultrasound beamformer** | **No peer-reviewed or preprint work found** that holds out receive channels and scores a learned beamformer by the likelihood of the held-out *raw* channels. Channel augmentation exists ([Amar 2025](https://arxiv.org/abs/2502.00524)); channel MAE exists outside medical ultrasound. **Treat as a gap, but confirm with IEEE Xplore / IUS proceedings full-text search.** | — | — |

### B.6 Speckle statistics (foundations; linear envelope/intensity only)

| Year | Paper | Model | Link | Status |
|---|---|---|---|---|
| 1983 | Wagner et al., *Statistics of speckle in ultrasound B-scans* | Rayleigh envelope; speckle SNR 1.91; speckle cell size from PSF autocorrelation | [DOI](https://doi.org/10.1109/T-SU.1983.31404) | [PR] |
| 1988 | Tuthill, Sperry, Parker, *Deviations from Rayleigh statistics* | Rician (coherent component); shows how detection, compression and bandwidth change the PDF | [DOI](https://doi.org/10.1016/0161-7346(88)90051-X) | [PR] |
| 1993 | Shankar et al., *Non-Rayleigh statistics for breast tumors* (IEEE TMI 12(4)) | K-distribution (low effective scatterer number) | [DOI](https://doi.org/10.1109/42.251119) | [PR] |
| 1994 | Dutt, Greenleaf, *Homodyned-K signal model* (Ultrason. Imaging 16(4)) | Homodyned-K (clustering α + coherent-to-diffuse ratio k) | [DOI](https://doi.org/10.1177/016173469401600404) | [PR] |
| 2000 | Shankar, *General statistical model: Nakagami* (TUFFC 47(3)) | Nakagami-m | [DOI](https://doi.org/10.1109/58.842062) | [PR] |
| 2002 | Raju, Srinivasan, *Envelope statistics of HF backscatter from human skin* (TUFFC) | Generalized Gamma fitted best of 6 families (KS test) | **[DOI not verified in this session]**; [search](https://pubmed.ncbi.nlm.nih.gov/?term=Raju+Srinivasan+envelope+high-frequency+ultrasonic+backscatter+skin) | [PR] |
| 2010 | Destrempes, Cloutier, *Critical review of envelope distributions* (UMB 36(7)) | Unified HK/K/Rice/Rayleigh/Nakagami view; **requires no log compression and no nonlinear filtering** | [journal](https://www.sciencedirect.com/science/article/abs/pii/S0301562910001675) | [PR] |
| 2024 | Tehrani et al., *HK parameter estimation: autoencoder and Bayesian NN* (TUFFC) | Amortized HK estimation with uncertainty | [DOI](https://doi.org/10.1109/TUFFC.2024.3357438) · [arXiv](https://arxiv.org/abs/2401.11006) | [PR] |
| 2022 | Tehrani, Rosado-Mendez, Rivaz, *Deep estimation of speckle-statistics parametric images* (EMBC) | Patchless CNN estimation of QUS parameters | [DOI](https://doi.org/10.1109/EMBC48229.2022.9871883) | [PR-conf] |
| 2024 | Lee et al., *UNICORN: Nakagami imaging via score matching* | Per-pixel Nakagami m from a learned score of the envelope | [arXiv](https://arxiv.org/abs/2403.06275) | [Preprint] |
| 2009 | Anfinsen, Doulgeris, Eltoft, *ENL estimation in PolSAR* (IEEE TGRS 47(11)) | Wishart-based ENL; ML via log-determinant; scene-wide robust estimation | [ResearchGate](https://www.researchgate.net/publication/220823003_Estimation_of_the_Equivalent_Number_of_Looks_in_Polarimetric_SAR_Imagery) | [PR] |

### B.7 Neural operators and learned wave solvers

| Year | Paper | Relevance | Link | Status |
|---|---|---|---|---|
| 2025 | Wang et al., *Luna: lung aeration map via physics-aware neural operators* | FNO maps RF directly to an aeration map; trained on simulation, fine-tuned on a little real data (9% error ex vivo) | [arXiv](https://arxiv.org/abs/2501.01157) | [Preprint] |
| 2025 | Zeng et al., *OpenBreastUS* | 8,000 phantoms, 16M frequency-domain simulations; benchmarks neural operators for USCT forward and inverse problems | [arXiv](https://arxiv.org/abs/2507.15035) | [Preprint] |
| 2023 | Stanziola, Arridge, Cox, Treeby, *A learned Born series for highly-scattering media* (JASA-EL 3(5)) | Unrolled Born-series solver with learned components; more accurate than the convergent Born series at equal iterations | [JASA-EL](https://pubs.aip.org/asa/jel/article/3/5/052401/2887637/A-learned-Born-series-for-highly-scattering-media) · [code](https://github.com/ucl-bug/lbs) | [PR] |
| 2023 | *Neural Born series operator for USCT* | Operator learning for transmission USCT | [arXiv](https://arxiv.org/abs/2312.15575) | [Preprint] |
| 2021 | Li et al., *Fourier neural operator* (ICLR) / Li et al., *PINO* / Lu et al., *DeepONet* (Nat. Mach. Intell.) | Base methods | [FNO](https://arxiv.org/abs/2010.08895) · [PINO](https://arxiv.org/abs/2111.03794) · [DeepONet](https://doi.org/10.1038/s42256-021-00302-5) | [PR-conf]/[Preprint]/[PR] |

### B.8 Benchmarks and metrics

| Paper | Link | Status |
|---|---|---|
| Liebgott et al., *PICMUS* (IEEE IUS 2016) | [DOI](https://doi.org/10.1109/ULTSYM.2016.7728908) · [site](https://www.creatis.insa-lyon.fr/Challenge/IEEE_IUS_2016/home) | [PR-conf] |
| Hyun et al., *CUBDL evaluation framework and open datasets* (TUFFC 68(12)) | [DOI](https://doi.org/10.1109/TUFFC.2021.3094849) · [data](https://ieee-dataport.org/competitions/challenge-ultrasound-beamforming-deep-learning-cubdl-datasets) | [PR] |
| Rodriguez-Molares et al., *gCNR* (TUFFC 67) | [DOI](https://doi.org/10.1109/TUFFC.2019.2956855) | [PR] |

---

## C. Novelty assessment

### C.1 Components that are already established (do not claim these)

| Component | Established by |
|---|---|
| Unsupervised single-PW deep beamforming with multi-angle coherence loss | [DCL 2024](https://doi.org/10.1016/j.ultras.2024.107408), [DCL-A](https://doi.org/10.3390/diagnostics15243193), [UBF-DCL_opt](https://doi.org/10.3390/diagnostics16010058) |
| Physics-based choice of decorrelated target angles for clutter | [UBF-DCL_opt](https://doi.org/10.3390/diagnostics16010058) |
| Angle-split Noise2Noise pairs | [HA2HA](https://arxiv.org/abs/2507.05451), [zero-shot PW denoising](https://arxiv.org/abs/2506.21499) |
| Learned per-pixel adaptive apodization | [ABLE](https://doi.org/10.1109/TMI.2020.3008537) (supervised) |
| Complex-valued IQ networks | [CID-Net](https://doi.org/10.1109/TUFFC.2021.3127916) |
| Differentiable beamforming for SoS autofocus (DBUA, CMPE) | [DBUA](https://doi.org/10.1007/978-3-031-43999-5_41), [CMPE](https://doi.org/10.1109/TMI.2025.3607875) |
| Coherence-maximizing SoS estimation | [Ultrasonics 2025/26](https://pubmed.ncbi.nlm.nih.gov/41401623/) |
| Channel data to echogenicity (speckle-reduced) | [Hyun 2019](https://doi.org/10.1109/TUFFC.2019.2903795) (supervised, simulation) |
| Likelihood-based self-supervised despeckling of complex data | [MERLIN](https://doi.org/10.1109/TGRS.2021.3128621) (SAR) |
| ENL / multi-look Gamma–Wishart statistics | [Anfinsen 2009](https://www.researchgate.net/publication/220823003_Estimation_of_the_Equivalent_Number_of_Looks_in_Polarimetric_SAR_Imagery) and SAR literature |
| Rayleigh/Rice/K/HK/Nakagami/GG envelope models; deep QUS estimators | B.6 |
| LOC-based noise estimation; Wiener postfilter | [LOC](https://doi.org/10.1109/TUFFC.2018.2855653), [Nilsen & Holm](https://doi.org/10.1109/TUFFC.2010.1553) |
| Unrolled unsupervised clutter filtering | [U2-rPCA](https://arxiv.org/abs/2510.00660) |
| Diffusion posterior sampling for ultrasound | [Stevens 2024](https://doi.org/10.1109/TMI.2024.3363460) |

### C.2 Potentially novel contributions (no match found in this search; must be confirmed)

1. **Correlation-aware cross-view likelihood (the core contribution).** A Gamma / Itakura–Saito likelihood on *linear intensity*. The target is a set of held-out views (other angles, other sub-apertures). The target's shape parameter is **L_eff, computed from the measured or predicted complex correlation matrix of the looks**. The resulting **speckle-leakage bias (∝|ρ|², derived in E.2) is bounded and reported**. DCL, DCL-A and UBF-DCL_opt choose or weight targets by angle geometry. Based on their abstracts, none of them models look correlation statistically or uses an intensity likelihood.
2. **Held-out raw-channel likelihood.** A structured complex-Gaussian covariance model for the delayed channel vector: van Cittert–Zernike tissue coherence + short-coherence clutter + white noise. Its NLL on *masked receive channels* is used as a training loss for a learned beamformer. I found no ultrasound beamforming paper doing this.
3. **Dual-output design with passivity guarantees.** A despeckled echogenicity output plus a speckle-preserving IQ output formed with a real gain `0 ≤ g ≤ 1` on a bounded, distortionless apodized sum. Weights and gain are frozen across a Doppler ensemble. Each part exists separately (Wiener postfilter, MV-style constraints). The combination as a *learned, unsupervised* design with phase-fidelity tests appears new.
4. **A "view-independence design".** Temporal frames, transmit angles and receive sub-apertures are used as views with *different, known* independence structure. This makes echogenicity, clutter and thermal noise separately identifiable (table in E.1).
5. **Calibrated diagnostic maps.** L_eff, N_eff, log-cumulant model selection, and PIT/KS goodness-of-fit computed on the beamformer's own looks, output together with the image.

**Confidence:** moderate. The search covered IEEE/Elsevier/MDPI/arXiv metadata up to 29 Sep 2026. IUS proceedings abstracts, SPIE Medical Imaging 2026 and non-English venues were **not** fully searched. **Before submitting:** run full-text searches for "effective number of looks" + "beamform*", "equivalent number of looks" + "ultrasound" + "deep", and "held-out channel" / "masked channel" + "beamformer", on IEEE Xplore, Scopus and Google Scholar.

### C.3 Claims that must not be made

- ✗ "First unsupervised / self-supervised deep beamformer" (DCL 2024 predates this).
- ✗ "First physics-informed deep beamformer" or "first differentiable beamformer" (ABLE, DBUA, zea).
- ✗ "Recovers true / ground-truth echogenicity from a single plane wave." Echogenicity is an *ensemble mean*. The method estimates it at the resolution of the target-view intensity PSF, with variance set by L_eff·N_eff. Claim only **"unbiased relative to the speckle-mean of the training-time target views, within the clutter level of those views."**
- ✗ "Removes clutter" as an absolute statement. Stationary reverberation that stays coherent across the training angle span cannot be identified by angle diversity (Section K).
- ✗ "Phase-preserving" without reporting Doppler velocity bias/variance and displacement-tracking error against DAS on the same data.
- ✗ "Sub-aperture looks are independent" or "L looks give √L SNR" without the measured L_eff.
- ✗ "Outperforms 75-angle compounding" unless shown with dynamic-range-invariant metrics (gCNR, contrast linearity) and the Rindal DRA controls.
- ✗ "Homodyned-K parameters from single-PW data" as a validated quantitative biomarker. HK estimation is unstable at small N_eff. Report it only as an exploratory diagnostic.
- ✗ "Neural-operator-based beamformer" (not recommended for v1; Section G).
- ✗ "Generalizes across sites" if only PICMUS plus one CUBDL institution were tested.
- ✗ "MERLIN applies to ultrasound IQ" without the spectral-symmetry check in F.2.

---

## D. Proposed architecture: **LC-PIB** (Look-Calibrated Physics-Informed Beamformer)

(The name is a placeholder. It maps onto your PE-PIB pipeline: ToF → constrained apodization → complementary looks → statistical estimator → unrolled reconstruction → outputs.)

### D.1 Diagram

```
             single PW raw channel IQ  S ∈ C^{N_t × N_ch}   (angle θ_a; demod f_d; fs)
                                   │
       ┌───────────────────────────▼────────────────────────────┐
  (1)  │ Differentiable ToF + F-number support                  │  params: c (global or B-spline
       │  τ_i(x;c) = (z cosθ + x sinθ)/c + √((x−x_i)²+z²)/c     │  slowness, straight-ray), F∈[1,2]
       │  d_i(x) = S_i(τ_i(x)) · e^{j2πf_d τ_i(x)} · M_i(x;F)   │  (phase rotation for baseband)
       └───────────────────────────┬────────────────────────────┘
                 delayed tensor D ∈ C^{N_z × N_x × N_ch}
                                   │
       ┌───────────────────────────▼────────────────────────────┐
  (2)  │ Constrained adaptive complex apodization  A_φ          │  coarse grid (N_z/4 × N_x/4),
       │  w(x) = Π_C[ M(x)⊙(w₀ + Δw_φ(features)) ]              │  bilinear upsample; hard constraints:
       │  y(x) = w(x)ᴴ d(x)                                     │  1ᴴw = 1, |w_i| ≤ β/|M|, supp ⊆ M
       └───────┬───────────────────────────────┬────────────────┘
               │ y (full aperture)              │ looks
       ┌───────▼────────┐        ┌─────────────▼─────────────────┐
  (3)  │                │        │ K complementary sub-apertures │  a_k: overlapping Hann windows,
       │                │        │ z_k(x) = (a_k⊙w)ᴴ d / ‖·‖      │  Σ_k a_k ≈ 1 (so Σ z_k ≈ y)
       │                │        └─────────────┬─────────────────┘
       │                │   ┌──────────────────▼──────────────────────────────┐
  (4)  │                │   │ Statistical estimator (non-learned, differentiable)│
       │                │   │  Γ̂(Δ): lag coherence of d (LOC, SLSC), n̂ from LOC │
       │                │   │  ρ_kl(x) = a_kᴴR a_l/√(…)  (VCZ-structured R)     │
       │                │   │  L_eff(x), N_eff(x), log-cumulants κ̂_1..3,        │
       │                │   │  GoF statistic, model label                       │
       │                │   └──────────────────┬───────────────────────────────┘
       │                │        features F(x) ∈ R^{N_z×N_x×C_f}
       │         ┌──────▼────────────────────────▼─────────────┐
  (5)  │         │ Unrolled MAP estimator (T = 4–6 stages)      │  variables: u = log σ², c ≥ 0
       │         │ u ← u − η_t ∇_u NLL_Γ(I_k | e^u + γ_k c + n̂, │  (echogenicity, clutter)
       │         │        L_eff) + CNN_t(u, c, F)               │  n̂ fixed from LOC
       │         └──────┬──────────────────────────┬────────────┘
       │                │                          │
  (6)  ▼                ▼                          ▼
  y_pp = g·y,     σ̂² (echogenicity),        ĉ (clutter), n̂ (noise),
  g = σ̂²/(σ̂²+ĉ+n̂_bf) linear intensity      Var[u] ≈ ψ₁(L_eff N_eff)+learned corr.,
  (speckle kept,                          L_eff, N_eff, ρ_kl, GoF/PIT, model label,
   phase kept)                            ρ̄ (coherence), CMPE/phase-error map
```

### D.2 Tensor and data flow (typical linear-array sizes, e.g. PICMUS/CUBDL-like 128-element probes)

| Stage | Tensor | Shape (example) | Notes |
|---|---|---|---|
| Input | `S` | `C^{N_t × 128}`, N_t ≈ 1–3k | IQ (or RF → analytic → baseband). Store `f_d`, `fs`, `θ_a`, element positions |
| ToF | `τ` | `R^{N_z × N_x × 128}` | Recomputed when `c` changes; straight-ray; differentiable interpolation (linear or cubic) |
| Delayed | `D` | `C^{512 × 256 × 128}` ≈ 128 MB (complex64) | Pixel grid ≈ λ/2 lateral, λ/4 axial; use pixel batching in training |
| Aperture mask | `M` | `[0,1]^{N_z × N_x × 128}` | Soft Tukey edge; F-number learnable in a box |
| Apod features | per coarse pixel | `R^{C_in}` | e.g. normalized `|d_i|`, phase differences `∠(d_i d*_{i+1})`, local Γ̂(1..8), depth |
| Weights | `w` | `C^{128 × 64 × 128}` (coarse), then upsampled | Complex; projected onto constraints (D.3) |
| Full output | `y` | `C^{512 × 256}` | Coherent; linear in `D` for fixed `w` |
| Looks | `z_k` | `C^{K × 512 × 256}`, K = 4–8 | 50% overlap by default; K and overlap are ablations |
| Stats | `Γ̂, ρ, L_eff, N_eff, κ̂` | `R^{…×512×256}` | Gaussian-window convolutions (σ_w ≈ 3–5 speckle cells) |
| Unrolled | `u, c` | `R^{512×256}` each | T stages; each CNN_t ~ 50–100k params (small U-Net or residual block) |
| Outputs | `σ̂², ĉ, n̂, y_pp`, maps | — | Display: `10 log10(σ̂²)` *after* all statistics are computed |

### D.3 Constraints that carry the physics (hard, by construction)

- **Distortionless (unit gain on the delay-aligned signal):** project onto `1ᴴw = 1`: `w ← w + (1 − 1ᴴw)·M/(1ᴴM)`. A perfectly focused on-axis scatterer passes with unit gain, so the network cannot amplify it. This is the same constraint as Capon/MV.
- **Bounded weights:** `|w_i| ≤ β/|M|` (e.g. β = 3) via a magnitude clamp before projection (iterate twice). This limits white-noise gain `‖w‖²`, which causes the MV-type dark-region artifact ([Rindal 2017](https://ieeexplore.ieee.org/document/8092255/)).
- **Support:** `supp(w) ⊆ M(x;F)`. No contributions from elements outside the acceptance angle.
- **Spatial smoothness:** predict `w` on a ¼-resolution grid and upsample. This prevents pixel-wise "painting" of structure, and keeps `w` quasi-constant over a speckle cell, so `y` is locally linear in the channel data (important for phase fidelity).
- **Passivity of the IQ output:** `0 ≤ g ≤ 1` and real. So `|y_pp| ≤ |y|` and `∠y_pp = ∠y`. No signal can be created in the coherent output.
- **Doppler/elastography mode:** compute `w` and `g` from the ensemble-averaged statistics of a slow-time packet and **freeze them across the packet**. Frame-varying weights would modulate phase and bias Kasai/autocorrelation estimates.

---

## E. Objective function

### E.1 Views and what each one identifies

Training data: unlabeled multi-angle PW acquisitions (angles `Θ`), several slow-time frames per angle where available, and all receive channels. At inference, only one PW.

| View pair | Speckle | Clutter (sidelobe/reverb) | Thermal noise | What it identifies |
|---|---|---|---|---|
| Same angle, consecutive frames (static or motion-compensated) | same | ~same | **independent** | noise `n` |
| Different angles, `|ρ_ab|² ≤ ε` | ~independent | changes (sidelobes, grating lobes, off-axis reverberation) | independent | echogenicity vs. (speckle + angle-varying clutter) |
| Different angles, `|ρ_ab|` high | shared | changes | independent | coherent IQ (speckle kept), clutter suppression (DCL-type) |
| Disjoint / masked receive channels | partially correlated (VCZ) | low spatial coherence | independent per channel | raw-data consistency, noise vs. coherent signal, focus/SoS |

This table is what makes the separation well-posed. Each loss term below uses one row.

### E.2 Loss terms

Notation: `x` = pixel. `a` = input angle. `𝓑(a)` = target angle set. `G_σ *` = Gaussian-window average. `m̂(x) = σ̂²(x)` in target-beamformer units, using the known deterministic normalization between apodizations. `n̂_B` = noise power of the target view (LOC-estimated).

**(1) Look-calibrated cross-view intensity likelihood (primary; echogenicity).**

Build the target from angles `b ∈ 𝓑(a)` with measured `|ρ_ab|² ≤ ε` (e.g. ε = 0.05). Measure `ρ_ab` in homogeneous regions or predict it (F.3). Split `𝓑(a)` into `G` groups. **Coherently** compound within each group (suppresses sidelobe/grating clutter as in CPWC). Then **incoherently** average across groups:

$$J_B(x)=\sum_{g=1}^{G}\omega_g\,\Big|\sum_{b\in\mathcal B_g}y^{\rm DAS}_b(x)\Big|^2,\qquad L_B(x)=\frac{\big(\sum_g\omega_g\mu_g\big)^2}{\sum_{g,h}\omega_g\omega_h\mu_g\mu_h|\rho_{gh}|^2}.$$

The loss is the Gamma negative log-likelihood (up to constants) with the noise floor modeled rather than subtracted:

$$\mathcal L_{\rm xv}=\frac{1}{|\Omega|}\sum_{x}\frac{L_B(x)}{N_{\rm eff}(x)}\left[\frac{J_B(x)}{\hat m(x)+\hat n_B(x)}+\log\big(\hat m(x)+\hat n_B(x)\big)\right].$$

*Why no clean target is needed.* For any shape `L > 0`, the Gamma NLL `L(J/μ + log μ)` is minimized in expectation at `μ* = E[J | input]`. It is the Itakura–Saito Bregman divergence, which is a proper scoring rule for the mean. `J_B` is a *measurement*, not a label. Its conditional expectation given the input is `E[σ² ⊛ |h_B|²] + ⟨clutter_B⟩ + n_B`. That is the speckle-ensemble mean, as long as `J_B`'s speckle is independent of the input view. The `1/N_eff` factor accounts for correlated pixels. **L_eff does not move the minimizer. It sets the heteroscedastic weighting and makes the uncertainty head calibrated**, so a moderate L_eff error does not bias echogenicity.

*Speckle leakage bound (why correlation must be calibrated).* For jointly circular-Gaussian `z_a, z_b` with complex correlation `ρ`:

$$\mathbb E\big[|z_b|^2\,\big|\,z_a\big]=\sigma_b^2+|\rho|^2\frac{\sigma_b^2}{\sigma_a^2}\big(|z_a|^2-\sigma_a^2\big).$$

So a network trained on a correlated target learns to reproduce a fraction `|ρ|²` of the input speckle. The residual speckle contrast of the echogenicity output is bounded to leading order by `|ρ|²` times the input speckle contrast. This is the quantitative reason to select targets by *measured* correlation. **Report the achieved `|ρ|²` distribution.**

**(2) Held-out raw-channel likelihood (masked channels; forward-model consistency).**

Randomly mask a channel subset `𝒪` (contiguous blocks longer than 1 element, 15–30% of the aperture). The network sees `d_{𝒪̄}` only (apodization renormalized). From its outputs, form the structured covariance of the delayed channel vector:

$$C(x)=\hat\sigma^2(x)\,\Gamma_s(x)+\hat c(x)\,\Gamma_c+\hat n(x)\,I,$$

where `Γ_s` is Toeplitz with `Γ_s[i,j] = γ_s(|i−j|; depth)`, parameterized as the Fourier transform of a **non-negative** source-intensity profile (Bochner, so it is positive semi-definite by construction, consistent with van Cittert–Zernike ([Mallart & Fink](https://doi.org/10.1121/1.401867))). `Γ_c` is a short-coherence kernel (e.g. exponential, length ≤ 2 elements). Loss = complex-Gaussian NLL of the held-out channels conditioned on the observed ones:

$$\mathcal L_{\rm ch}=\sum_{x\in\mathcal P}\frac{1}{N_{\rm eff}}\Big[\log\det S(x)+r(x)^{\!H}S(x)^{-1}r(x)\Big],\quad r=d_{\mathcal O}-C_{\mathcal O\bar{\mathcal O}}C_{\bar{\mathcal O}\bar{\mathcal O}}^{-1}d_{\bar{\mathcal O}},\ S=C_{\mathcal O\mathcal O}-C_{\mathcal O\bar{\mathcal O}}C_{\bar{\mathcal O}\bar{\mathcal O}}^{-1}C_{\bar{\mathcal O}\mathcal O}.$$

Evaluate on a random subset `𝒫` of about 4k pixels per batch; the Toeplitz structure keeps it cheap. *No clean target:* the held-out raw channels are the measurement. The term penalizes over-smoothing and hallucination: a fabricated `σ̂²` mispredicts the held-out channel power and cross-covariance. It also penalizes wrong delays, because misfocus lowers `Γ_s` and raises the NLL. This is the "raw-channel forward-model consistency" you asked for, written in a form that respects speckle statistics.

**(3) Complex coherence for the phase-preserving IQ output (DCL-type, near angles).**

For angle pairs with **high** expected angular coherence (the opposite choice from (1)), and weights `ν_ab` from the predicted `|ρ_ab|` (DCL-A-style weighting, but from measured correlation):

$$\mathcal L_{\rm coh}=\sum_{(a,b)}\nu_{ab}\sum_{W}\Big(1-\frac{\operatorname{Re}\langle y^{a}_{pp},\,y^{\rm DAS}_b\rangle_W}{\|y^a_{pp}\|_W\,\|y^{\rm DAS}_b\|_W}\Big).$$

Using `Re⟨·,·⟩` rather than `|⟨·,·⟩|` penalizes phase bias. Speckle is **kept** because the partner view shares it. Clutter and noise are suppressed because they are not coherent across angles. Without labels this trains the Doppler/elastography output. *Guard against a trivial solution:* the loss is scale-invariant, so pair it with (6) below. Otherwise `g → 0` everywhere except bright coherent structures would score well.

**(4) Temporal noise identification (optional; static or motion-compensated data only).**

$$\mathcal L_{\rm t}=\sum_x m_{\rm static}(x)\,\Big|\,G*\big(|y_t-y_{t+1}|^2\big)(x)-2\,\hat n_{\rm bf}(x)\Big|$$

where `n̂_bf = ‖w‖² n̂` is the beamformed noise power and `m_static` masks pixels with inter-frame correlation above 0.99. This anchors the noise map to a physically independent measurement. Where no static frames exist, LOC alone ([Long 2018](https://doi.org/10.1109/TUFFC.2018.2855653)) provides `n̂` and `𝓛_t` is dropped.

**(5) Self-supervised autofocus (sound speed).**

For receive sub-aperture pairs `(k,l)` with a common midpoint (and, at training, angle pairs):

$$\mathcal L_{\rm af}=\sum_x\sum_{(k,l)}|\hat\rho_{kl}(x)|\big(1-\cos\angle\hat\rho_{kl}(x)\big)$$

This is a CMPE-like phase-error criterion ([CMPE](https://doi.org/10.1109/TMI.2025.3607875)). Update only the low-dimensional `c` (global or coarse B-spline slowness), not the apodization network, to avoid confounding. `𝓛_ch` also carries focus information.

**(6) Energy consistency and anti-hallucination.**

(a) The *decomposition* must explain the measured local energy of the plain DAS image from the same single PW:

$$\mathcal L_{\rm E}=\sum_x L_{\rm DAS}N_{\rm eff}\Big[\frac{G*|y^{\rm DAS}_a|^2}{G*(\hat\sigma^2+\hat c+\hat n_{\rm bf})}+\log G*(\hat\sigma^2+\hat c+\hat n_{\rm bf})\Big].$$

Echogenicity can therefore only fall below DAS energy by the amount assigned to clutter and noise. It cannot exceed DAS energy beyond sampling error.
(b) Hard constraints: `σ̂², ĉ, n̂ ≥ 0` (softplus); `g ∈ [0,1]`; distortionless, bounded `w` (D.3).

**(7) Apodization regularizer.**

$$\mathcal R_w=\lambda_{\rm TV}\,\mathrm{TV}(w)+\lambda_{\rm WNG}\sum_x\max\big(0,\ \|w(x)\|^2-\beta_{\rm WNG}/|M(x)|\big).$$

The second term is a white-noise-gain cap, which limits noise amplification and dark-region artifacts.

**(8) Dispersion / calibration head (optional).** The unrolled network also predicts a local shape `L̂(x)`. Train it by the same Gamma NLL with respect to `L` (the mean minimizer is unaffected). This produces the uncertainty map. Calibration is checked, not trained, with the PIT tests in F.5.

**Total:**

$$\boxed{\mathcal L=\lambda_{\rm xv}\mathcal L_{\rm xv}+\lambda_{\rm ch}\mathcal L_{\rm ch}+\lambda_{\rm coh}\mathcal L_{\rm coh}+\lambda_{\rm E}\mathcal L_{\rm E}+\lambda_{\rm af}\mathcal L_{\rm af}+\lambda_{\rm t}\mathcal L_{\rm t}+\mathcal R_w}$$

Suggested curriculum: (i) freeze `c` at 1540 m/s, train `𝓛_E + 𝓛_ch` (a stable statistical beamformer); (ii) add `𝓛_xv`; (iii) add `𝓛_coh`; (iv) unfreeze `c` with `𝓛_af`. Set λ by uncertainty weighting or a fixed grid. Always report the λ values.

**Why no term needs a clean target:** every target in (1)–(6) is either another *physical measurement* whose noise and/or speckle realization is independent of the input in a known way (other angles, held-out channels, other frames, the same PW's DAS energy), or a *likelihood of the observed data* under a physical covariance model. No compounded image is used as a pixel label. The compound in (1) enters only through its *expected intensity*, not its speckle.

---

## F. PDF estimator: domains, models, L_eff calibration, and model selection

### F.1 Correct models by domain (linear data only)

| Domain | Fully developed speckle (many random scatterers, no coherent part) | Coherent component (specular / periodic) | Low effective scatterer number / clustering | General |
|---|---|---|---|---|
| **IQ** `z = I + jQ` | circular complex Gaussian `CN(0, σ²)` | non-zero mean `CN(s, σ²)` | compound Gaussian (Gaussian with Gamma texture) | — |
| **Envelope** `A = |z|` | **Rayleigh** (SNR = 1.91) ([Wagner 1983](https://doi.org/10.1109/T-SU.1983.31404)) | **Rician** ([Tuthill 1988](https://doi.org/10.1016/0161-7346(88)90051-X)) | **K** ([Shankar 1993](https://doi.org/10.1109/42.251119)) | **Homodyned-K** (covers all three; [Dutt & Greenleaf 1994](https://doi.org/10.1177/016173469401600404)); **Nakagami-m** as a 2-parameter approximation ([Shankar 2000](https://doi.org/10.1109/58.842062)); **generalized Gamma** as an empirical 3-parameter fit ([Raju 2002], DOI not verified) |
| **Intensity** `I = |z|²` | **Exponential** (SNR = 1) | noncentral χ² (2 dof, scaled) | K-intensity | L-look average: **Gamma(L, σ²/L)** if looks independent; exactly a *sum of exponentials with means λ_m = eig(Σ^{1/2} W Σ^{1/2})* if correlated; Gamma(L_eff) is its 2-moment match |

**Key facts that constrain the design:**

- Nakagami on the envelope is exactly Gamma on intensity (`A² ~ Gamma(m, Ω/m)`). Use the intensity form: it is simpler, and look averaging happens in intensity.
- **Look averaging and tissue statistics both change the shape parameter.** An estimated `m` or `L` on a compounded image is roughly `L_eff × m_tissue` (exact for Gamma looks, approximate otherwise). Estimate tissue shape on **single looks** and report `L_eff` separately. Otherwise "post-Rayleigh" tissue cannot be told apart from "more looks".
- **Thermal noise is circular Gaussian, just like speckle.** First-order statistics cannot separate noise from diffuse echogenicity (`I ~ Exp(σ² + n)`). Separation needs *channel coherence* (LOC; `𝓛_ch`) or *temporal independence* (`𝓛_t`). This is why the architecture does not ask the PDF head to do it.
- **Log-compressed B-mode is not used for any fit.** Unknown compression parameters destroy scale information ([Smith & Raza 2026](https://arxiv.org/abs/2609.19525)), and the classical models assume linear, unfiltered envelopes ([Destrempes & Cloutier 2010](https://www.sciencedirect.com/science/article/abs/pii/S0301562910001675)). Log compression is applied only for display, after all statistics are computed.
- **Edges and mixtures produce spurious non-Rayleigh statistics.** Any window straddling two echogenicities looks pre-Rayleigh (K-like). Model-selection maps must be masked or down-weighted near strong gradients of `σ̂²`.

### F.2 Why MERLIN does not transfer directly to ultrasound IQ

MERLIN relies on the real and imaginary parts of SAR SLC data being independent ([Dalsasso 2022](https://doi.org/10.1109/TGRS.2021.3128621)). For a baseband ultrasound signal, `I` and `Q` are independent processes **only if the RF power spectrum is symmetric about the demodulation frequency** (standard narrowband random-process theory). Frequency-dependent attenuation shifts the spectral centroid downward with depth. With a fixed `f_d`, `I` and `Q` become correlated fields, and a MERLIN-style loss would leak speckle. Use it only as an ablation, with depth-adaptive demodulation at the local spectral centroid and a measured check that `corr(I, Q)` at spatial lags is ≈ 0.

### F.3 L_eff calibration under overlapping apertures

**Model-based (primary).** The looks are `z_k = a_kᴴ d`. With the delayed channel covariance `R(x) = σ² Γ_s + c Γ_c + n I` from Section E(2):

$$\rho_{kl}(x)=\frac{a_k^HR\,a_l}{\sqrt{a_k^HR\,a_k\;a_l^HR\,a_l}},\qquad \mathrm{Cov}(I_k,I_l)=\mu_k\mu_l|\rho_{kl}|^2\ \ (\text{Siegert / Isserlis, circular Gaussian}),$$

$$L_{\rm eff}(x)=\frac{\big(\sum_k\omega_k\mu_k\big)^2}{\sum_{k,l}\omega_k\omega_l\mu_k\mu_l|\rho_{kl}|^2}\quad(\text{equal } \mu,\omega:\ L_{\rm eff}=K^2/\textstyle\sum_{kl}|\rho_{kl}|^2).$$

`Γ_s(Δ)` is estimated with many samples (averaged over a depth band and a wide lateral region), so it is well conditioned. `ρ_kl` then follows *exactly* for any overlap pattern, including the *learned* apodizations `a_k ⊙ w`. This is the van Cittert–Zernike argument ([Mallart & Fink](https://doi.org/10.1121/1.401867); [Trahey 1986](https://doi.org/10.1109/T-UFFC.1986.26827)) applied to arbitrary sub-apertures.

**Empirical (validation).** In homogeneous regions (selected automatically by a low-gradient mask), compute `ρ̂_kl = Σ_W z_k z_l* / √(Σ_W|z_k|² Σ_W|z_l|²)`. Correct the magnitude bias (≈ `1/N_eff` when the true ρ is 0) using the standard coherence-estimator bias correction, or by averaging over many windows as in scene-wide ENL estimation ([Anfinsen 2009](https://www.researchgate.net/publication/220823003_Estimation_of_the_Equivalent_Number_of_Looks_in_Polarimetric_SAR_Imagery)). **Report the agreement between model-based and empirical L_eff** (Bland–Altman) as a validation figure.

**Spatial correlation (N_eff).** Pixels within a speckle cell are correlated. For a weighted window `v_i`:

$$N_{\rm eff}=\frac{(\sum_i v_i)^2}{\sum_{i,j}v_iv_j|\rho_{\rm sp}(x_i-x_j)|^2},$$

with `ρ_sp` the normalized IQ autocorrelation (≈ PSF autocorrelation; [Wagner 1983](https://doi.org/10.1109/T-SU.1983.31404)), estimated locally from `y`. It has the same form as L_eff. For a joint space×look estimate, use the full covariance if it is small, or the product `L_eff·N_eff` when it is separable. **Every variance, test statistic and loss weight uses these effective counts, never raw pixel counts.**

### F.4 Differentiable local estimators

- **Local moments:** `μ̂ = G_σ * I`, `v̂ = G_σ * I² − μ̂²` (convolutions; differentiable).
- **Shape via log-cumulants** (robust, closed-form variance). For Gamma(L): `κ₁ = ψ(L) − log L + log μ`, `κ₂ = ψ₁(L)`, `κ₃ = ψ₂(L)`. Invert `ψ₁` by 3–4 Newton steps. Gradients pass through the fixed point via the implicit-function theorem (`∂L/∂κ₂ = 1/ψ₂(L)`). *Mellin-kind log-cumulant methods come from the SAR literature (Nicolas 2002; Anfinsen & Eltoft 2011) [DOI not verified in this session].*
- **Moment estimator (alternative):** `m̂ = μ̂²/v̂` with finite-sample correction by `N_eff`.
- **Coherent-component (Rician) parameter:** from `κ̂₂ < ψ₁(L_eff)` (post-Rayleigh), fit the noncentral-χ² intensity by 2-moment matching.
- **Homodyned-K:** use as a **diagnostic only**, through an amortized estimator ([Tehrani 2024](https://doi.org/10.1109/TUFFC.2024.3357438)), never inside a loss. It is unstable at the N_eff available per window.
- **Soft windows:** use Gaussian windows (σ_w of 3–5 speckle cells). Report the effective resolution of every statistical map.

### F.5 Local model selection and goodness of fit

1. **Null hypothesis H₀ (diffuse speckle with calibrated looks):** combined intensity ~ Gamma(L_eff(x), μ(x)). Test statistic `Z₂ = (κ̂₂ − ψ₁(L_eff)) / sd`, where `sd² ≈ [ψ₃(L_eff) + 2ψ₁(L_eff)²] / N_eff` (the variance of the sample second log-cumulant). `Z₂ ≫ 0` means pre-Rayleigh (K-like: clustering, low density, *or edges/mixtures*). `Z₂ ≪ 0` means post-Rayleigh (coherent component, *or L_eff under-estimated*).
2. **Alternatives:** Gamma with free L; K-intensity (α); Rician-Gamma (k); generalized Gamma. Select by AIC/BIC computed with **N_eff** in place of N. Output a model label **and** its posterior weight. Show the (κ₂, κ₃) diagram for representative ROIs.
3. **Goodness of fit:** PIT values `u = F̂(I; θ̂)` per pixel. Test uniformity with **block-bootstrap** Anderson–Darling / KS, with block size ≥ the speckle cell (N_eff-adjusted critical values).
4. **Residual PDF calibration of the echogenicity output:** normalized residuals `r = J_B / (m̂ + n̂_B)` on held-out views should follow Gamma(L_B, 1/L_B) in speckle regions. Report reliability diagrams and PIT histograms per depth band.
5. **Diagnostic map semantics:** L_eff, N_eff, Z₂, model label, `ρ̄` (mean look coherence), LOC-SNR, clutter fraction `ĉ/(σ̂²+ĉ+n̂)`, and the held-out-channel NLL residual (a per-pixel "consistency" map). These are the uncertainty and diagnostic outputs.

---

## G. Neural operators (FNO / PINO / DeepONet / learned wave solvers)

**Recommendation: do not use a neural operator as the beamformer or as the data-consistency forward model in version 1.** Reasons:

- **The operator you would learn is already known well enough.** For single-scattering pulse-echo at 3–10 MHz, the delay (Born / straight-ray) operator in D.1 is exact to within the aberration error. That error is low-dimensional (a slowness map) and handled by `c`. What is unknown is *statistical*: speckle, clutter, noise. An FNO surrogate adds approximation error without addressing those unknowns.
- **Phase accuracy requirement.** `𝓛_ch` and `𝓛_coh` need phase errors well below λ/8 at the carrier. Domains of about 100–200 λ with speckle-scale structure are where current neural operators are weakest. Benchmarks such as [OpenBreastUS](https://arxiv.org/abs/2507.15035) exist precisely to measure forward/inverse accuracy and generalization at lower USCT frequencies. A phase-biased surrogate would corrupt a likelihood that assumes an exact forward model.
- **Direct-inversion successes are task-specific and supervised by simulation.** [Luna](https://arxiv.org/abs/2501.01157) (FNO, RF to lung aeration) is a supervised, simulation-trained estimator of a low-dimensional target, not a speckle-preserving beamformer.

**Where they could help (version 2 or later, each with a clear test):**

| Use case | Role | Required simulation data | Expected benefit | Main risk | v1? |
|---|---|---|---|---|---|
| **Evaluation data generator** | Fast surrogate of full-wave (k-Wave/Fullwave) channel-data simulation with heterogeneous SoS and layered reverberating walls | ~10³–10⁴ full-wave simulations of abdominal-wall/tissue phantoms with *known echogenicity maps*; held-out geometry families | Scale up the only data that provide **echogenicity ground truth and reverberation truth** for *evaluation* | Surrogate bias leaks into the evaluation; always spot-check against the full-wave solver | **No** (use the full-wave solver directly for the v1 test set) |
| **Aberration operator (travel-time/eikonal or bent-ray surrogate)** | Replaces straight-ray `τ(x;c)` in heterogeneous media (cf. the lens-aware bent-ray model in [arXiv 2608.06853](https://arxiv.org/abs/2608.06853)) | Ray-tracing / eikonal solutions over random slowness fields (cheap, millions) | Better delays under strong heterogeneity; differentiable; low-dimensional input/output | Small gain if SoS contrast is modest; must beat plain differentiable ray tracing | **No** (v2 ablation) |
| **Learned iterative wave solver for reverberation modeling** | Learned Born-series-style unrolled solver ([LBS](https://pubs.aip.org/asa/jel/article/3/5/052401/2887637/A-learned-Born-series-for-highly-scattering-media)) to model multiple scattering in the clutter term `Γ_c` | Helmholtz solutions in layered media | Physically motivated clutter prior | Cost; frequency-domain vs. broadband pulse mismatch | **No** (research extension) |
| **Direct FNO beamformer** | Channel data → image | — | Not justified | Loses the phase-preservation guarantees; needs labels or simulation | **Never recommended** here |

---

## H. Experimental protocol

### H.1 Data roles (strict separation)

- **Training (unlabeled):** your own multi-angle PW acquisitions (e.g. ±16°, 31–75 angles; slow-time packets where possible) on tissue-mimicking phantoms (including anechoic/hyperechoic cysts and a Doppler flow phantom) and in vivo (carotid, thyroid, muscle, liver; ethics approval required). Record probe, `f_c`, `fs`, `f_d`, TGC, and all per-angle channel data. **No PICMUS or CUBDL data in training.**
- **Development (hyperparameters, early stopping):** a held-out subset of your own data, plus *simulation* (Field II / k-Wave) with known echogenicity maps. Do not tune on PICMUS/CUBDL.
- **Held-out evaluation (report once, pre-registered):**
  - **PICMUS** ([Liebgott 2016](https://doi.org/10.1109/ULTSYM.2016.7728908)): simulated and experimental resolution/contrast phantoms and in vivo carotid. Input = the single 0° PW. The 75-angle compound is a **reference**, not ground truth. Simulated contrast phantoms give known lesion geometry for echogenicity-bias checks.
  - **CUBDL** ([Hyun 2021](https://doi.org/10.1109/TUFFC.2021.3094849)): plane-wave data from multiple institutions/systems. Use the **official evaluation code**. Report **per institution** (leave-institution-out by construction, since none were in training).
  - **Own simulation test set:** Field II (speckle statistics, known echogenicity including −20…+6 dB lesions and graded-scatterer-density regions for K/HK) and k-Wave/Fullwave (layered abdominal wall for reverberation; SoS inclusions ±30–60 m/s). **The only source of absolute echogenicity and clutter truth.**
  - **Motion/Doppler set:** Field II with known scatterer displacement (axial/lateral, 0–λ/4 per frame) and a flow phantom or in vivo carotid packets. **Required for phase-fidelity claims**; PICMUS/CUBDL are essentially single-frame and cannot support them.

### H.2 Baselines

- DAS 1-PW; DAS 75-PW (reference); MV/Capon; coherence factor and generalized CF; Wiener postfilter ([Nilsen & Holm](https://doi.org/10.1109/TUFFC.2010.1553)); SLSC; DMAS.
- **DCL** (your `dcl/` implementation, or the authors' code if available); DCL-A-style angle weighting; UBF-DCL_opt-style frame selection.
- Supervised U-Net (your `unet_supervised/`), trained on compounded targets.
- ABLE-style supervised apodization.
- Post-hoc despeckling on DAS (Speckle2Self / Mask2Restore if code is available; otherwise a Gamma-MAP filter). This shows whether integrating statistics *into* the beamformer matters.
- DBUA/CMPE autofocus + DAS (for the SoS component).

### H.3 Protocol details

- Fix the input normalization, pixel grid and display dynamic range across methods. **Compute all metrics on linear intensity/IQ** before any log compression.
- Repeat training with ≥ 3 seeds. Report mean ± SD across seeds and bootstrap 95% CIs across images. Use paired Wilcoxon tests with Holm correction for method comparisons.
- Pre-register ROIs (PICMUS/CUBDL ROI definitions where provided).

---

## I. Ablations (each tied to one claim)

| # | Ablation | Claim tested |
|---|---|---|
| A1 | Replace `𝓛_xv`'s Gamma NLL with L1/L2 on log-intensity | Proper intensity likelihood → lower echogenicity bias and calibrated uncertainty |
| A2 | Set L_eff = K (assume independent looks) vs. calibrated L_eff | Correlation calibration → calibrated PIT/uncertainty (the mean should be ≈ unchanged; see E.2) |
| A3 | Target angles selected by geometry (UBF-DCL_opt rule) vs. by measured `|ρ|² ≤ ε`, sweeping ε ∈ {0.01, 0.05, 0.2, 0.5} | Leakage law: residual speckle contrast ∝ `|ρ|²` |
| A4 | Remove `𝓛_ch` (masked raw-channel likelihood) | Raw-data consistency reduces hallucination and improves focus |
| A5 | Masking pattern: random single elements vs. contiguous blocks; mask ratio sweep | Block masking needed because channel noise/speckle are correlated over short lags |
| A6 | Remove `𝓛_coh` | IQ phase fidelity / Doppler bias depend on the complex coherence term |
| A7 | `Re⟨·⟩` vs. `|⟨·⟩|` in `𝓛_coh` | Phase-bias penalty |
| A8 | Remove hard constraints (distortionless, magnitude bound, coarse grid) | Constraints prevent dark-region artifacts and DRA-driven "contrast gains" |
| A9 | Real gain `g` vs. unconstrained complex post-network for the IQ output | Passivity/real gain preserves phase |
| A10 | Frozen vs. per-frame `w, g` in Doppler packets | Weight freezing is needed for unbiased Doppler |
| A11 | K ∈ {2,4,8} looks; overlap ∈ {0, 25, 50, 75}% | L_eff trade-off; overlap is fine *when calibrated* |
| A12 | Remove unrolling (single CNN on the features) vs. T ∈ {2,4,6} | Value of the physics-structured estimator |
| A13 | Remove `𝓛_af` / fix c = 1540 m/s; test on SoS-mismatched simulation | Autofocus contribution |
| A14 | Remove `𝓛_E` | Energy consistency prevents hallucinated echogenicity |
| A15 | MERLIN-style I/Q split with fixed vs. depth-adaptive demodulation | Tests the claim in F.2 |
| A16 | Training on phantoms only vs. phantoms + in vivo | Domain robustness |

---

## J. Metrics

**Image quality (linear domain; PICMUS/CUBDL ROIs):**
- **gCNR** ([Rodriguez-Molares 2020](https://doi.org/10.1109/TUFFC.2019.2956855)) on linear intensity. Also report it on `σ̂²`, since it is invariant to monotone transforms.
- **Contrast (CR) and CNR**, *always* with the dynamic-range controls from [Rindal 2019](https://doi.org/10.1109/TUFFC.2019.2911267): (i) **contrast linearity**, i.e. the slope and R² of measured vs. true lesion contrast (dB) over −20…+6 dB simulated lesions (ideal slope = 1); (ii) the same metrics after histogram-matching each output to DAS.
- **FWHM** axial/lateral on PICMUS/CUBDL point targets, from the *coherent* output `y_pp` (and `σ̂²` separately). Echogenicity maps are expected to be slightly smoother; report both.
- **Speckle SNR** (envelope 1.91 / intensity 1.0 for fully developed speckle) on `y_pp` (should stay near DAS values, i.e. speckle kept) and on `σ̂²` (should rise ≈ √(L_eff·N_eff)).

**Echogenicity accuracy (simulation truth):**
- Bias and RMSE of `10 log10 σ̂²` vs. the true PSF-smoothed echogenicity, per lesion and per depth. Clutter-fraction accuracy in reverberation phantoms. Noise-map error vs. known added noise.

**Phase fidelity (motion/Doppler sets):**
- Phase error `∠(y_pp · y_DAS*)` distribution in high-SNR speckle (median should be ≈ 0).
- **Kasai/lag-1 velocity bias and SD** vs. ground-truth motion, and vs. DAS on the same packets.
- **Speckle-tracking displacement error** (normalized cross-correlation tracker) and peak correlation coefficient.
- Doppler clutter-to-blood ratio after the standard SVD clutter filter.

**Statistical calibration:**
- PIT histograms / reliability diagrams; block-bootstrap KS/AD p-values; coverage of predicted intervals (target 90% → observed ≈ 90%); model vs. empirical L_eff agreement.

**Hallucination tests:**
- **Null input:** noise-only channel data (probe in air, or synthetic white noise). `σ̂²` must ≈ 0 relative to `n̂`, with no structure.
- **Insertion/deletion:** add or remove simulated channel data of a point target or cyst *by linear superposition* into real channel data. The structure must appear or disappear correctly, with no ghost.
- **Anechoic fill-in:** mean `σ̂²` inside anechoic cysts vs. noise floor.
- **Channel permutation / aperture shift:** outputs must degrade as physics predicts, not stay "pretty".
- **Out-of-distribution SoS/angles/probes:** degradation must be flagged by the consistency and GoF maps.

**Runtime:** ms/frame (GPU and CPU), peak memory and FLOPs at the PICMUS grid, broken down by stage (ToF, apodization, looks, statistics, unrolled). Real-time target: ≥ 30 fps for B-mode. Doppler mode is budgeted separately.

**Cross-site generalization:** per-institution CUBDL metrics, the spread across institutions, and the degradation relative to in-distribution data.

---

## K. Practical risks, assumptions, and failure modes

1. **Identifiability of stationary reverberation.** Reverberation from near-field layers can remain partly coherent across a small angle span. Angle diversity cannot then separate it from tissue, and `σ̂²` will be biased high. *Mitigation:* use the widest usable training angle span; rely on low spatial coherence of reverberation via `Γ_c` in `𝓛_ch`; quantify in k-Wave/Fullwave layered-wall phantoms; report the clutter fraction honestly.
2. **Motion between training angles (in vivo).** Tissue motion across the angle sequence decorrelates views. Decorrelation is then attributed to clutter, and `𝓛_coh` penalizes true tissue. *Mitigation:* short angle sequences, motion estimation and masking, and frame-to-frame correlation gating.
3. **SoS error during training.** Wrong delays misregister angles, which inflates apparent clutter and biases `𝓛_coh`. *Mitigation:* the curriculum with autofocus, or pre-computing a DBUA/CMPE SoS per training acquisition.
4. **Depth-dependent spectrum.** Attenuation downshift changes the PSF, `Γ_s` and I/Q symmetry. *Mitigation:* depth-banded `Γ_s`; depth-adaptive demodulation.
5. **Correlation estimates at small N_eff.** Biased `|ρ̂|`. *Mitigation:* the model-based `ρ` from a well-estimated `Γ_s`; empirical `ρ̂` used only for validation.
6. **Gamma approximation at low L_eff.** The exact sum-of-exponentials law differs in the tails. *Mitigation:* use the exact eigenvalue form for small K in GoF tests; the mean estimate is unaffected (proper-scoring argument).
7. **Non-Gaussian tissue (K/HK).** The circular-Gaussian covariance model in `𝓛_ch` is misspecified for strongly clustered or specular tissue. *Mitigation:* treat it as a quasi-likelihood; the model-selection map flags these regions; use robust (Student-t-like) weighting as an ablation.
8. **Adaptive-weight artifacts.** Even with constraints, learned weights can darken regions near bright targets ([Rindal 2017](https://ieeexplore.ieee.org/document/8092255/)). *Mitigation:* the white-noise-gain cap, `𝓛_E`, and DRA-controlled metrics.
9. **Training/inference gap.** Multi-angle views exist only at training. The single-PW network must infer from features alone what compounding reveals. Some clutter types may simply not be predictable from one PW. *Report failure regions*; do not average them away.
10. **Data heterogeneity across sites.** CUBDL institutions differ in probes, sampling and IQ vs. RF formats. *Mitigation:* a physics-normalized input (delays from metadata; per-channel gain normalization); per-site reporting.
11. **Compute and memory.** `N_z·N_x·N_ch` complex tensors plus per-pixel K×K statistics. *Mitigation:* pixel batching in training, coarse-grid weights, Toeplitz/low-rank covariances, mixed precision (keep phase-critical ops in fp32).
12. **Evaluation leakage.** Tuning on PICMUS/CUBDL would invalidate the held-out claims. *Mitigation:* pre-registration; a single final evaluation run.

**Assumptions stated explicitly in the paper:** first-order single-scattering delay model (plus a clutter term); circular-Gaussian speckle as the null model; noise independent across channels and frames; echogenicity defined at the resolution of the target-view intensity PSF; the look and spatial correlation structure estimable from the data.

---

## Appendix: open verification items before submission

- Read DCL (2024), DCL-A and UBF-DCL_opt in full. Confirm how targets are formed (particular compounded realization vs. expectation), whether IQ phase fidelity was evaluated, and whether any statistical weighting of looks is used.
- Confirm the exact supervision used by Liu et al. (TUFFC 2025).
- Full-text novelty searches listed in C.2 (IEEE Xplore, IUS 2024–2026, SPIE MI 2025–2026).
- Verify the DOIs marked "not verified in this session" (Raju & Srinivasan 2002; the Mellin log-cumulant references).
- Commit `my_method/PE-PIB_implementation_spec.md` and `pe_pib.py` so the architecture in D can be reconciled with the existing spec.
