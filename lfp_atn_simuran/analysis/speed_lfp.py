import logging
from copy import deepcopy
from math import floor, ceil
from collections import OrderedDict

from neurochat.nc_utils import butter_filter
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from icecream import ic

# TODO scatter plot the speed vs other things
# Maybe can do one summary plot of speed vs other things

# 1. Compare speed and firing rate
def speed_firing(self, spike_train, speed_sr, **kwargs):
    lim = kwargs.get("range", [0, self.get_duration()])
    skip_rate = int(self.get_sampling_rate() / speed_sr)
    slicer = slice(skip_rate, -skip_rate, skip_rate)
    index_to_grab = np.logical_and(self.get_time() >= lim[0], self.get_time() <= lim[1])

    speed = self.get_speed()[index_to_grab][slicer]

    f_times = spike_train.flatten()
    f_times_speed = f_times * speed_sr
    nearest_speed_samples = f_times_speed.round().astype(int)
    f_speeds = speed[nearest_speed_samples]

    isi = np.diff(f_times)

    return isi, f_speeds[1:]


# 2. Compare speed and interburst interval
def calc_ibi(spike_train, speed, speed_sr, burst_thresh=5):
    unitStamp = spike_train
    isi = 1000 * np.diff(unitStamp)

    burst_start = []
    burst_end = []
    burst_duration = []
    spikesInBurst = []
    bursting_isi = []
    num_burst = 0
    ibi = []
    k = 0
    ibi_speeds = []
    while k < isi.size:
        if isi[k] <= burst_thresh:
            burst_start.append(k)
            spikesInBurst.append(2)
            bursting_isi.append(isi[k])
            burst_duration.append(isi[k])
            m = k + 1
            while m < isi.size and isi[m] <= burst_thresh:
                spikesInBurst[num_burst] += 1
                bursting_isi.append(isi[m])
                burst_duration[num_burst] += isi[m]
                m += 1
            # to compensate for the span of the last spike
            burst_duration[num_burst] += 1
            burst_end.append(m)
            k = m + 1
            num_burst += 1
        else:
            k += 1

    if num_burst:
        for j in range(0, num_burst - 1):
            time_end = unitStamp[burst_start[j + 1]]
            time_start = unitStamp[burst_end[j]]
            ibi.append(time_end - time_start)
            speed_time_idx1 = int(time_start * speed_sr)
            speed_time_idx2 = int(time_end * speed_sr)
            burst_speed = speed[speed_time_idx1:speed_time_idx2]
            avg_speed = np.mean(burst_speed)
            ibi_speeds.append(avg_speed)

        # ibi in sec, burst_duration in ms
    else:
        logging.warning("No burst detected")
    ibi = 1000 * np.array(ibi)

    return ibi, np.array(ibi_speeds)


def speed_ibi(self, spike_train, **kwargs):
    samples_per_sec = kwargs.get("samplesPerSec", 10)
    binsize = kwargs.get("binsize", 1)
    min_speed, max_speed = kwargs.get("range", [0, 40])

    speed = self.get_speed()
    max_speed = min(max_speed, np.ceil(speed.max() / binsize) * binsize)
    min_speed = max(min_speed, np.floor(speed.min() / binsize) * binsize)
    bins = np.arange(min_speed, max_speed, binsize)

    ibi, ibi_speeds = calc_ibi(spike_train, speed, samples_per_sec)
    return ibi, ibi_speeds
    # visit_time = np.histogram(speed, bins)[0]
    # speedInd = np.digitize(speed, bins) - 1
    # visit_time = visit_time / self.get_sampling_rate()

    # rate = (
    #     np.array([sum(vid_count[speedInd == i]) for i in range(len(bins))]) / visit_time
    # )
    # rate[np.isnan(rate)] = 0

    # _results["Speed Skaggs"] = self.skaggs_info(rate, visit_time)

    # rate = rate[visit_time > 1]
    # bins = bins[visit_time > 1]

    # fit_result = np.linfit(bins, rate)

    # _results["Speed Pears R"] = fit_result["Pearson R"]
    # _results["Speed Pears P"] = fit_result["Pearson P"]
    # graph_data["bins"] = bins
    # graph_data["rate"] = rate
    # graph_data["fitRate"] = fit_result["yfit"]

    # if update:
    #     self.update_result(_results)
    # return graph_data


# 3. Compare theta and speed
def speed_vs_amp(self, lfp_signal, low_f, high_f, filter_kwargs=None, **kwargs):
    lim = kwargs.get("range", [0, self.get_duration()])
    samples_per_sec = kwargs.get("samplesPerSec", 10)
    do_once = True

    if filter_kwargs is None:
        filter_kwargs = {}
    try:
        lfp_signal = lfp_signal.filter(low_f, high_f, **filter_kwargs)
    except BaseException:
        lfp_signal = deepcopy(lfp_signal)
        _filt = [10, low_f, high_f, "bandpass"]
        lfp_signal._set_samples(
            butter_filter(
                lfp_signal.get_samples(), lfp_signal.get_sampling_rate(), *_filt
            )
        )

    # Calculate the LFP power
    skip_rate = int(self.get_sampling_rate() / samples_per_sec)
    slicer = slice(skip_rate, -skip_rate, skip_rate)
    index_to_grab = np.logical_and(self.get_time() >= lim[0], self.get_time() <= lim[1])
    time_to_use = self.get_time()[index_to_grab][slicer]
    speed = self.get_speed()[index_to_grab][slicer]
    binsize = kwargs.get("binsize", 2)
    min_speed, max_speed = kwargs.get("range", [0, 40])

    max_speed = min(max_speed, np.ceil(speed.max() / binsize) * binsize)
    min_speed = max(min_speed, np.floor(speed.min() / binsize) * binsize)
    bins = np.arange(min_speed, max_speed, binsize)

    visit_time = np.histogram(speed, bins)[0]
    speedInd = np.digitize(speed, bins) - 1

    # visit_time = visit_time / samples_per_sec

    lfp_amplitudes = np.zeros_like(time_to_use)
    lfp_samples = lfp_signal.get_samples()
    if hasattr(lfp_samples, "unit"):
        import astropy.units as u

        lfp_samples = lfp_samples.to(u.uV).value
    else:
        lfp_samples = lfp_samples * 1000

    for i, t in enumerate(time_to_use):
        low_sample = floor((t - 0.05) * lfp_signal.get_sampling_rate())
        high_sample = ceil((t + 0.05) * lfp_signal.get_sampling_rate())
        if high_sample < len(lfp_samples):
            lfp_amplitudes[i] = np.mean(
                np.abs(lfp_samples[low_sample : high_sample + 1])
            )
        elif do_once:
            logging.warning(
                "Position data ({}s) is longer than EEG data ({}s)".format(
                    time_to_use[-1], len(lfp_samples) / lfp_signal.get_sampling_rate()
                )
            )
            do_once = False

    binned_lfp = [np.sum(lfp_amplitudes[speedInd == i]) for i in range(len(bins) - 1)]
    rate = np.array(binned_lfp) / visit_time

    return rate, bins[:-1], lfp_amplitudes, speed


def main(self, lfp_signal, spike_train, binsize=1, speed_sr=10):

    # Speed vs LFP power
    r, b, r1, b1 = speed_vs_amp(
        self, lfp_signal, 5, 12, binsize=binsize, samplesPerSec=speed_sr
    )

    fig, axes = plt.subplots(2, 1)

    sns.lineplot(x=b, y=r, ax=axes[0])
    sns.scatterplot(x=b1, y=r1, ax=axes[1])
    fig.savefig("speed_theta.png", dpi=400)
    plt.close(fig)

    # Speed vs firing rate
    data = self.speed(spike_train, binsize=binsize, samplesPerSec=speed_sr)
    r, b = data["rate"], data["bins"]
    r1, b1 = speed_firing(self, spike_train, speed_sr)

    fig, axes = plt.subplots(2, 1)
    sns.lineplot(x=b, y=r, ax=axes[0])
    sns.scatterplot(x=b1, y=r1, ax=axes[1])
    fig.savefig("speed_frate.png", dpi=400)
    plt.close(fig)

    # Speed vs IBI
    r, b = speed_ibi(self, spike_train, samplesPerSec=speed_sr)
    fig, ax = plt.subplots()

    sns.scatterplot(x=b, y=r, ax=ax)
    fig.savefig("speed_ibi.png", dpi=400)
    plt.close(fig)
