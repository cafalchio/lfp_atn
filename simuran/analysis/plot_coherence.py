import os

import numpy as np
from scipy.signal import coherence
from scipy.signal import welch
import matplotlib.pyplot as plt
import seaborn as sns

from simuran.plot.figure import SimuranFigure

from neurochat.nc_utils import butter_filter


def plot_coherence(x, y, ax, fs=250, group="ATNx", fmax=100):
    sns.set_style("ticks")
    sns.set_palette("colorblind")

    f, Cxy = coherence(x, y, fs, nperseg=1024)

    f = f[np.nonzero(f <= fmax)]
    Cxy = Cxy[np.nonzero(f <= fmax)]

    sns.lineplot(x=f, y=Cxy, ax=ax)
    sns.despine()

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Coherence")
    plt.ylim(0, 1)

    return np.array([f, Cxy, [group] * len(f)])


def plot_psd(x, ax, fs=250, group="ATNx", fmax=100):
    f, Pxx = welch(x, fs=fs, nperseg=1024, return_onesided=True, scaling="density",)

    f = f[np.nonzero(f <= fmax)]
    Pxx = Pxx[np.nonzero(f <= fmax)]

    sns.lineplot(x=f, y=Pxx, ax=ax)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")

    return np.array([f, Pxx, [group] * len(f)])


def sig_avg(arr, at, _filter):
    sig1 = arr[4 * at].samples
    sig2 = arr[4 * at + 1].samples
    dead_first = np.all(sig1 == 0)
    dead_second = np.all(sig2 == 0)

    if dead_first and dead_second:
        return None

    if dead_first:
        return butter_filter(sig2, arr[0].sampling_rate, *_filter)
    if dead_second:
        return butter_filter(sig1, arr[0].sampling_rate, *_filter)
    return butter_filter((sig1 + sig2) / 2, arr[0].sampling_rate, *_filter)


def plot_recording_coherence(recording, figures, base_dir, sig_type="first"):
    location = os.path.splitext(recording.source_file)[0]

    dirs = base_dir.split(os.sep)
    if dirs[-1].startswith("CS") or dirs[-2].startswith("CS"):
        group = "Control"
    elif dirs[-1].startswith("LS") or dirs[-2].startswith("LS"):
        group = "Lesion"
    else:
        group = "Undefined"

    name = (
        "--".join(os.path.dirname(location)[len(base_dir + os.sep) :].split(os.sep))
        + "--"
        + os.path.basename(location)
        + "_coherence"
        + ".png"
    )

    sub_signals = recording.signals.group_by_property("region", "SUB")[0]
    rsc_signals = recording.signals.group_by_property("region", "RSC")[0]

    # filter signals to use
    _filter = [10, 1.5, 100, "bandpass"]

    if sig_type == "first":
        # Remove dead channels
        sub_signals = [s for s in sub_signals if not np.all((s.samples == 0))]
        rsc_signals = [s for s in rsc_signals if not np.all((s.samples == 0))]

        sub_signal = sig_avg(sub_signals, 0, _filter)
        # sub_signal2 = np.mean(np.array([s.samples for s in sub_signals[:2]]), axis=0)
        # sub_signal2 = butter_filter(sub_signal2, sub_signals[0].sampling_rate, *_filter)
        rsc_signal = sig_avg(rsc_signals, 0, _filter)

    elif sig_type == "avg":
        # Remove dead channels
        sub_signals = [s for s in sub_signals if not np.all((s.samples == 0))]
        rsc_signals = [s for s in rsc_signals if not np.all((s.samples == 0))]

        sub_signal = np.mean(np.array([s.samples for s in sub_signals]), axis=0)
        sub_signal = butter_filter(sub_signal, sub_signals[0].sampling_rate, *_filter)
        rsc_signal = np.mean(np.array([s.samples for s in rsc_signals]), axis=0)
        rsc_signal = butter_filter(rsc_signal, rsc_signals[0].sampling_rate, *_filter)

    elif sig_type == "dist":
        result = {}
        fs = sub_signals[0].sampling_rate
        # Get the main result
        rsc_signal = sig_avg(rsc_signals, 0, _filter)
        sub_signal = sig_avg(sub_signals, 0, _filter)
        f, main_result = coherence(rsc_signal, sub_signal, fs, nperseg=1024)

        for i in range(len(main_result)):
            result["Coh_{:.2f}".format(f[i])] = main_result[i]

        # Compute the other results
        if len(sub_signals) > 2:
            matrix_data = np.zeros((int(len(recording.signals) / 4), len(main_result)))
            matrix_data[0] = main_result
            for i in range(1, 8):
                sub_signal = sig_avg(sub_signals, i, _filter)
                _, matrix_data[i] = coherence(rsc_signal, sub_signal, fs, nperseg=1024)

            # Compare results by getting the variance of each col
            var_res = np.var(matrix_data, axis=1)

            for i in range(len(main_result)):
                result["Var_{:.2f}".format(f[i])] = var_res[i]

        else:
            for i in range(len(main_result)):
                result["Var_{:.2f}".format(f[i])] = np.nan

        return result

    # TODO handle x -> y and y -> x
    fig, ax = plt.subplots()
    result = plot_coherence(
        sub_signal, rsc_signal, ax, sub_signals[0].sampling_rate, group=group
    )

    figures.append(SimuranFigure(fig, name, dpi=400, done=True, format="png"))

    fig, ax = plt.subplots()
    plot_psd(sub_signal, ax, sub_signals[0].sampling_rate, group=group)

    name = (
        "--".join(os.path.dirname(location)[len(base_dir + os.sep) :].split(os.sep))
        + "--"
        + os.path.basename(location)
        + "_psd_sub"
        + ".png"
    )

    figures.append(SimuranFigure(fig, name, dpi=400, done=True, format="png"))

    fig, ax = plt.subplots()
    plot_psd(rsc_signal, ax, rsc_signals[0].sampling_rate, group=group)

    name = (
        "--".join(os.path.dirname(location)[len(base_dir + os.sep) :].split(os.sep))
        + "--"
        + os.path.basename(location)
        + "_psd_rsc"
        + ".png"
    )

    figures.append(SimuranFigure(fig, name, dpi=400, done=True, format="png"))

    return result
