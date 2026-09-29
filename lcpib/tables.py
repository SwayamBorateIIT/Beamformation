"""Print Markdown tables from metrics.json files of one or more runs."""
import json
import os
import sys

import numpy as np


def f(v, d=2):
    try:
        return "n/a" if v is None or not np.isfinite(v) else f"{v:.{d}f}"
    except TypeError:
        return str(v)


def picmus_tables(m):
    P = m["picmus"]
    out = []
    for n in ("contrast_speckle_simu", "contrast_speckle_expe"):
        c = P[n]["contrast"]
        out.append(f"\n**PICMUS {n}** ({len(P[n]['cysts_mm'])} cysts detected; mean over cysts)\n")
        out.append("| Method | CR [dB] | CNR | gCNR | speckle SNR (bg) |\n|---|---|---|---|---|")
        for k, v in c.items():
            out.append(f"| {k} | {f(v['CR_dB'])} | {f(v['CNR'])} | {f(v['gCNR'], 3)} | {f(v['SNR_out'])} |")
    for n in ("resolution_distorsion_simu", "resolution_distorsion_expe"):
        fw = P[n]["fwhm"]
        out.append(f"\n**PICMUS {n}** (median FWHM over {list(fw.values())[0]['n_points']} points)\n")
        out.append("| Method | axial FWHM [mm] | lateral FWHM [mm] |\n|---|---|---|")
        for k, v in fw.items():
            out.append(f"| {k} | {f(v['axial_mm'], 3)} | {f(v['lateral_mm'], 3)} |")
    out.append("\n**Calibration on PICMUS (single-look DAS intensity vs predicted Gamma(L̂, σ̂²+ĉ+n̂))**\n")
    out.append("| Dataset | KS (LC-PIB Gamma) | KS (Exponential, same mean) | mean I/m | median L̂ |\n|---|---|---|---|---|")
    for n, r in P.items():
        c = r["calibration"]
        out.append(f"| {n} | {f(c['KS_model_gamma'], 3)} | {f(c['KS_exponential'], 3)} | {f(c['mean_ratio_I_over_m'], 3)} | {f(c['median_L_hat'], 2)} |")
    rt = np.mean([r["runtime_s"] for r in P.values()])
    out.append(f"\nMean LC-PIB full-image inference time (512×256 px, 128 ch, 4 CPU threads): **{rt:.2f} s**.")
    return "\n".join(out)


def sim_tables(m):
    S = m["sim"]
    out = []
    names = [k for k in S if k.startswith("T")]
    methods = list(S[names[0]]["lesions"].keys())
    out.append("\n**Contrast linearity on simulated lesions (designed −20, −12, −6, +6, +12 dB); mean over 4 test phantoms**\n")
    out.append("| Method | slope (ideal 1) | R² | gCNR anechoic | gCNR mean | CNR mean | speckle SNR bg |\n|---|---|---|---|---|---|---|")
    for k in methods:
        g = lambda key: np.mean([S[n]["lesions"][k][key] for n in names])
        out.append(f"| {k} | {f(g('slope'))} | {f(g('r2'), 3)} | {f(g('gCNR_anechoic'), 3)} | {f(g('gCNR_mean'), 3)} | {f(g('CNR_mean'))} | {f(g('speckle_SNR_bg'))} |")
    out.append("\n**Measured lesion contrast [dB] by designed contrast (phantom T1)**\n")
    des = list(S[names[0]]["lesions"][methods[0]]["CR_by_design"].keys())
    out.append("| Method | " + " | ".join(des) + " |\n|" + "---|" * (len(des) + 1))
    for k in methods:
        out.append(f"| {k} | " + " | ".join(f(S[names[0]]['lesions'][k]['CR_by_design'][d]) for d in des) + " |")
    out.append("\n**Component maps vs truth (log-domain Pearson r; median ratio in dB)**\n")
    out.append("| Phantom | echogenicity r | echo RMSE dB | smoothed-DAS r | smoothed-DAS RMSE dB | noise r | noise bias dB | clutter r | clutter bias dB | clutter px fraction |\n|---|---|---|---|---|---|---|---|---|---|")
    for n in names:
        r = S[n]; cm = r["clutter_map"] or {"pearson_log": np.nan, "median_ratio_dB": np.nan}
        out.append(f"| {n} | {f(r['echo_map']['pearson_log'], 3)} | {f(r['echo_map']['rmse_dB_after_global_scale'])} | "
                   f"{f(r['echo_map_smoothedDAS_baseline']['pearson_log'], 3)} | {f(r['echo_map_smoothedDAS_baseline']['rmse_dB_after_global_scale'])} | "
                   f"{f(r['noise_map']['pearson_log'], 3)} | {f(r['noise_map']['median_ratio_dB'])} | {f(cm['pearson_log'], 3)} | {f(cm['median_ratio_dB'])} | {f(r['clutter_fraction_pixels'], 3)} |")
    out.append("\n**Look calibration: effective number of looks of the 4 overlapping sub-aperture looks (background speckle, 15–40 mm)**\n")
    out.append("| Phantom | naive K | L_eff predicted from measured ρ | measured ENL gain (compound/single) | median pixel-wise L_eff estimator |\n|---|---|---|---|---|")
    for n in names:
        v = S[n]["L_eff_validation"]
        out.append(f"| {n} | {v['L_naive']:.0f} | {f(v['L_pred_from_rho'])} | {f(v['L_measured_over_single'])} | {f(v['median_pixelwise_L_eff_estimator'])} |")
    out.append("\n**Gamma-shape calibration in background (KS distance of PIT to uniform; lower is better)**\n")
    out.append("| Phantom | KS LC-PIB Gamma(L̂) | KS Exponential | mean I/m | median L̂ |\n|---|---|---|---|---|")
    for n in names:
        c = S[n]["calibration_bg"]
        out.append(f"| {n} | {f(c['KS_model_gamma'], 3)} | {f(c['KS_exponential'], 3)} | {f(c['mean_ratio_I_over_m'], 3)} | {f(c['median_L_hat'])} |")
    p = S["phase_fidelity"]
    out.append(f"\n**Phase fidelity: Kasai axial displacement, true step {p['true_axial_step_um']:.2f} µm/frame (λ/16)**\n")
    out.append("| Signal | bias [µm] | std [µm] | RMSE [µm] |\n|---|---|---|---|")
    for k, v in p.items():
        if isinstance(v, dict) and "bias_um" in v:
            out.append(f"| {k} | {f(v['bias_um'], 3)} | {f(v['std_um'], 3)} | {f(v['rmse_um'], 3)} |")
    pd = p["phase_diff_ypp_vs_DAS_deg"]
    out.append(f"\nPhase of y_pp vs DAS on the same frame: median |Δφ| = {pd['weighted_median_abs']:.2f}°, intensity-weighted mean |Δφ| = {pd['weighted_mean_abs']:.2f}°.")
    h = S["hallucination_null"]
    out.append(f"\n**Null test (pure channel noise):** median echogenicity fraction σ̂²/(σ̂²+ĉ+n̂) = {h['median_echo_fraction']:.3f} (95th pct {h['p95_echo_fraction']:.3f}); median noise fraction = {h['median_noise_fraction']:.3f}.")
    ins = S["hallucination_insertion"]
    out.append("\n**Insertion test (one point scatterer added by channel-data superposition)**\n")
    out.append("| Output | fraction of added energy within 1.5 mm | peak at insert | median rel. change elsewhere |\n|---|---|---|---|")
    for k, v in ins.items():
        out.append(f"| {k} | {f(v['energy_fraction_near_insert'], 3)} | {v['peak_near_insert']} | {f(v['rel_change_far'], 4)} |")
    return "\n".join(out)


def ablation_table(runs):
    out = ["| Run | PICMUS sim gCNR (echo) | PICMUS expe gCNR (echo) | sim slope (echo) | sim gCNR mean (echo) | sim echo r | noise r | noise bias dB | Kasai RMSE y_pp frozen [µm] | KS Gamma bg |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        p = os.path.join(r, "metrics.json")
        if not os.path.exists(p):
            continue
        m = json.load(open(p)); S = m["sim"]
        names = [k for k in S if k.startswith("T")]
        e = "LC-PIB echogenicity"
        g = lambda key: np.mean([S[n]["lesions"][e][key] for n in names])
        pg = lambda n: m["picmus"][n]["contrast"][e]["gCNR"] if "picmus" in m else np.nan
        out.append(f"| {os.path.basename(r)} | {f(pg('contrast_speckle_simu'), 3)} | {f(pg('contrast_speckle_expe'), 3)} | {f(g('slope'))} | {f(g('gCNR_mean'), 3)} | "
                   f"{f(np.mean([S[n]['echo_map']['pearson_log'] for n in names]), 3)} | {f(np.mean([S[n]['noise_map']['pearson_log'] for n in names]), 3)} | "
                   f"{f(np.mean([S[n]['noise_map']['median_ratio_dB'] for n in names]))} | {f(S['phase_fidelity']['LC-PIB y_pp (frozen weights+gain)']['rmse_um'], 3)} | "
                   f"{f(np.mean([S[n]['calibration_bg']['KS_model_gamma'] for n in names]), 3)} |")
    return "\n".join(out)


if __name__ == "__main__":
    m = json.load(open(os.path.join(sys.argv[1], "metrics.json")))
    print(picmus_tables(m))
    print(sim_tables(m))
    if len(sys.argv) > 2:
        print(ablation_table(sys.argv[2:]))
