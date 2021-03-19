import os
import math

import numpy as np
import matplotlib.pyplot as plt

from simuran.plot.figure import SimuranFigure
from neurochat.nc_utils import butter_filter


def get_normalised_diff(s1, s2):
    return np.sum(np.square(s1 - s2)) / (np.sum(np.square(s1) + np.square(s2)) / 2)


def main(recording, figures, base_dir, split_len=100):
    location = os.path.splitext(recording.source_file)[0]

    sub_signals = recording.signals.group_by_property("region", "SUB")[0]
    sub_signals = [s for s in sub_signals if not np.all((s.samples == 0))]
    sub_mean = np.mean(np.array([s.samples for s in sub_signals]), axis=0)
    _filter = [10, 1.5, 90, "bandpass"]
    sub_mean = butter_filter(sub_mean, sub_signals[0].sampling_rate, *_filter)
    rsc_signals = recording.signals.group_by_property("region", "RSC")[0]
    rsc_signals = [s for s in rsc_signals if not np.all((s.samples == 0))]
    rsc_mean = np.mean(np.array([s.samples for s in rsc_signals]), axis=0)
    rsc_mean = butter_filter(rsc_mean, rsc_signals[0].sampling_rate, *_filter)

    recording_signals = [
        butter_filter(
            lfp.get_samples(), recording.signals[0].get_sampling_rate(), *_filter
        )
        for lfp in recording.signals
    ]
    in_range = (0, max([lfp.underlying.get_duration() for lfp in recording.signals]))
    y_axis_max = max([max(lfp) for lfp in recording_signals])
    y_axis_min = min([min(lfp) for lfp in recording_signals])
    seg_splits = np.arange(in_range[0], in_range[1], split_len)
    if np.abs(in_range[1] - seg_splits[-1]) > 0.0001:
        seg_splits = np.concatenate([seg_splits, [in_range[1]]])
        max_split_len = max(np.diff(seg_splits))

    for j, split in enumerate(seg_splits[:-1]):
        fig, axes = plt.subplots(
            nrows=len(recording.signals) + 2,
            figsize=(40, (len(recording.signals) + 2) * 2),
        )
        a = np.round(split, 2)
        b = np.round(min(seg_splits[j + 1], in_range[1]), 2)
        out_name = "--".join(
            os.path.dirname(location)[len(base_dir + os.sep) :].split(os.sep)
        ) + "raw--0_{}_{:.2f}s_to_{:.2f}s.png".format(j, a, b)
        for i, lfp in enumerate(recording_signals):
            convert = recording.signals[0].get_sampling_rate()
            c_start, c_end = math.floor(a * convert), math.floor(b * convert)
            lfp_sample = lfp[c_start:c_end]
            x_pos = recording.signals[i].get_timestamps()[c_start:c_end]
            axes[i].plot(x_pos, lfp_sample, color="k")
            axes[i].text(
                0.03,
                1.02,
                "Channel " + str(i + 1),
                transform=axes[i].transAxes,
                color="k",
                fontsize=15,
            )
            axes[i].set_ylim(y_axis_min, y_axis_max)
            axes[i].tick_params(labelsize=12)
            axes[i].set_xlim(a, a + max_split_len)

        # plot sub and rsc mean
        i = -2
        axes[i].plot(x_pos, sub_mean[c_start:c_end], color="r")
        axes[i].text(
            0.03, 1.02, "Sub mean", transform=axes[i].transAxes, color="r", fontsize=15,
        )
        axes[i].set_ylim(y_axis_min, y_axis_max)
        axes[i].tick_params(labelsize=12)
        axes[i].set_xlim(a, a + max_split_len)

        i = -1
        axes[i].plot(x_pos, rsc_mean[c_start:c_end], color="r")
        axes[i].text(
            0.03, 1.02, "RSC mean", transform=axes[i].transAxes, color="r", fontsize=15,
        )
        axes[i].set_ylim(y_axis_min, y_axis_max)
        axes[i].tick_params(labelsize=12)
        axes[i].set_xlim(a, a + max_split_len)

        figures.append(SimuranFigure(fig, out_name, dpi=100, format="png", done=True))

    results = {
        "sub": get_normalised_diff(sub_signals[0].samples, sub_signals[1].samples),
        "rsc": get_normalised_diff(rsc_signals[0].samples, rsc_signals[1].samples),
    }

    return results
