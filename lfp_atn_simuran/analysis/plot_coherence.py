import os

import numpy as np
from scipy.signal import coherence
from scipy.signal import welch
import matplotlib.pyplot as plt
import seaborn as sns
import simuran
import astropy.units as u

from lfp_atn_simuran.analysis.lfp_clean import LFPClean


def plot_coherence(x, y, ax, fs=250, group="ATNx", fmin=1, fmax=100):
    sns.set_style("ticks")
    sns.set_palette("colorblind")

    f, Cxy = coherence(x.samples, y.samples, fs, nperseg=2 * fs)

    f = f[np.nonzero((f >= fmin) & (f <= fmax))]
    Cxy = Cxy[np.nonzero((f >= fmin) & (f <= fmax))]

    sns.lineplot(x=f, y=Cxy, ax=ax)
    simuran.despine()

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Coherence")
    plt.ylim(0, 1)

    return np.array([f, Cxy, [group] * len(f)])


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


def plot_recording_coherence(
    recording, figures, base_dir, clean_method="avg", fmin=1, fmax=30, **kwargs
):
    fmt = kwargs.get("image_format", "png")
    clean_kwargs = kwargs.get("clean_kwargs", {})
    group = define_recording_group(base_dir)
    result = {}

    # Firstly, clean
    lfp_clean = LFPClean(method=clean_method, visualise=False)
    clean_res = lfp_clean.clean(
        recording, min_f=fmin, max_f=fmax, method_kwargs=clean_kwargs
    )
    cleaned_signal_dict = clean_res["signals"]

    keys = sorted(list(cleaned_signal_dict.keys()))
    if len(keys) != 2:
        raise RuntimeError("This method is designed for signals from two brain regions")

    # Do coherence
    k1, k2 = keys
    v1, v2 = cleaned_signal_dict[k1], cleaned_signal_dict[k2]
    name = name_plot(recording, base_dir, f"_coherence_{k1}-{k2}")
    simuran.set_plot_style()
    fig, ax = plt.subplots()
    sr = v1.sampling_rate
    result = plot_coherence(v1, v2, ax, sr, group, fmin=fmin, fmax=fmax)
    ax.set_ylim(0, 1)
    figures.append(simuran.SimuranFigure(fig, name, dpi=400, done=True, format=fmt))

    return result