import os

import numpy as np
from scipy.signal import welch
from astropy import units as u
import simuran
import matplotlib.pyplot as plt
import seaborn as sns

from lfp_atn_simuran.analysis.lfp_clean import LFPClean


def plot_psd(x, ax, fs=250, group="ATNx", region="SUB", fmin=1, fmax=100):
    f, Pxx = welch(
        x.samples.to(u.uV).value,
        fs=fs,
        nperseg=2 * fs,
        return_onesided=True,
        scaling="density",
        average="mean",
    )

    f = f[np.nonzero((f >= fmin) & (f <= fmax))]
    Pxx = Pxx[np.nonzero((f >= fmin) & (f <= fmax))]

    sns.lineplot(x=f, y=Pxx, ax=ax)
    simuran.despine()
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (\u00b5V\u00b2 / Hz)")

    return np.array([f, Pxx, [group] * len(f), [region] * len(f)])


def define_recording_group(base_dir):
    dirs = base_dir.split(os.sep)
    if dirs[-1].startswith("CS") or dirs[-2].startswith("CS"):
        group = "Control"
    elif dirs[-1].startswith("LS") or dirs[-2].startswith("LS"):
        group = "Lesion"
    else:
        group = "Undefined"
    return group


def name_plot(recording, base_dir, end):
    return recording.get_name_for_save(base_dir) + end


def powers(
    recording, base_dir, figures, clean_method="avg", fmin=1, fmax=100, **kwargs
):
    # TODO refactor the cleaning
    clean_kwargs = kwargs.get("clean_kwargs", {})
    lc = LFPClean(method=clean_method, visualise=False)
    signals_grouped_by_region = lc.clean(
        recording.signals, fmin, fmax, method_kwargs=clean_kwargs
    )["signals"]
    fmt = kwargs.get("image_format", "png")

    results = {}
    window_sec = 2
    simuran.set_plot_style()

    for name, signal in signals_grouped_by_region.items():
        results["{} delta".format(name)] = np.nan
        results["{} theta".format(name)] = np.nan
        results["{} low gamma".format(name)] = np.nan
        results["{} high gamma".format(name)] = np.nan
        results["{} total".format(name)] = np.nan

        results["{} delta rel".format(name)] = np.nan
        results["{} theta rel".format(name)] = np.nan
        results["{} low gamma rel".format(name)] = np.nan
        results["{} high gamma rel".format(name)] = np.nan

        # TODO find good bands from a paper
        sig_in_use = signal.to_neurochat()
        delta_power = sig_in_use.bandpower(
            [1.5, 4], window_sec=window_sec, band_total=True
        )
        theta_power = sig_in_use.bandpower(
            [6, 10], window_sec=window_sec, band_total=True
        )
        low_gamma_power = sig_in_use.bandpower(
            [30, 55], window_sec=window_sec, band_total=True
        )
        high_gamma_power = sig_in_use.bandpower(
            [65, 90], window_sec=window_sec, band_total=True
        )

        if not (
            delta_power["total_power"]
            == theta_power["total_power"]
            == low_gamma_power["total_power"]
            == high_gamma_power["total_power"]
        ):
            raise ValueError("Unequal total powers")

        results["{} delta".format(name)] = delta_power["bandpower"]
        results["{} theta".format(name)] = theta_power["bandpower"]
        results["{} low gamma".format(name)] = low_gamma_power["bandpower"]
        results["{} high gamma".format(name)] = high_gamma_power["bandpower"]
        results["{} total".format(name)] = delta_power["total_power"]

        results["{} delta rel".format(name)] = delta_power["relative_power"]
        results["{} theta rel".format(name)] = theta_power["relative_power"]
        results["{} low gamma rel".format(name)] = low_gamma_power["relative_power"]
        results["{} high gamma rel".format(name)] = high_gamma_power["relative_power"]

        # Do power spectra
        out_name = name_plot(recording, base_dir, f"power_{name}")
        sr = signal.sampling_rate
        fig, ax = plt.subplots()
        group = define_recording_group(base_dir)
        results["{} welch".format(name)] = plot_psd(
            signal, ax, sr, group, name, fmin=fmin, fmax=fmax
        )
        fig = simuran.SimuranFigure(fig, out_name, dpi=400, done=True, format=fmt)
        figures.append(fig)

    return results
