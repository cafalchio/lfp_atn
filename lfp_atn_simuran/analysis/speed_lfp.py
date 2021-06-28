import logging
from copy import deepcopy
from math import floor, ceil
import os
import seaborn as sns

from neurochat.nc_utils import butter_filter
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from skm_pyutils.py_table import list_to_df
from skm_pyutils.py_plot import UnicodeGrabber
import simuran
import pandas as pd
import scipy.stats

from lfp_atn_simuran.analysis.lfp_clean import LFPClean

# 1. Compare speed and firing rate
def speed_firing(self, spike_train, **kwargs):
    graph_results = self.speed(spike_train, **kwargs)
    ax = kwargs.get("ax", None)

    results = {}
    results["lin_fit_r"] = self.get_results()["Speed Pears R"]
    results["lin_fit_p"] = self.get_results()["Speed Pears P"]

    bins = np.array([int(b) for b in graph_results["bins"]])
    rate = np.array([float(r) for r in graph_results["rate"]])

    if ax is None:
        fig, ax = plt.subplots()

    df = list_to_df([bins, rate], transpose=True, headers=["Speed", "Firing rate"])

    sns.lineplot(data=df, x="Speed", y="Firing rate", ax=ax)
    ax.set_title("Speed vs Firing rate")
    ax.set_xlabel("Speed (cm / s)")
    ax.set_ylabel("Firing rate (spike / s)")

    return results, ax


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
            speed_time_idx1 = int(floor(time_start * speed_sr))
            speed_time_idx2 = int(ceil(time_end * speed_sr))
            burst_speed = speed[speed_time_idx1:speed_time_idx2]
            avg_speed = np.mean(burst_speed)
            ibi_speeds.append(avg_speed)

        # ibi in sec, burst_duration in ms
    else:
        logging.warning("No burst detected")
        return None, None
    ibi = np.array(ibi) / 1000

    return ibi, np.array(ibi_speeds)


def speed_ibi(self, spike_train, **kwargs):
    samples_per_sec = kwargs.get("samplesPerSec", 10)
    binsize = kwargs.get("binsize", 1)
    min_speed, max_speed = kwargs.get("range", [0, 40])
    ax = kwargs.get("ax", None)

    speed = self.get_speed()
    max_speed = min(max_speed, np.ceil(speed.max() / binsize) * binsize)
    min_speed = max(min_speed, np.floor(speed.min() / binsize) * binsize)
    bins = np.arange(min_speed, max_speed, binsize)

    ibi, ibi_speeds = calc_ibi(spike_train, speed, samples_per_sec)
    if ibi is None:
        return None, None, np.nan, np.nan, 0
    elif len(ibi) < 10:
        return None, None, np.nan, np.nan, 0
    spear_r, spear_p = scipy.stats.spearmanr(ibi_speeds, ibi)

    pd_df = list_to_df([ibi_speeds, ibi], transpose=True, headers=["Speed", "IBI"])
    pd_df = pd_df[pd_df["Speed"] <= 40]
    pd_df["Speed"] = np.around(pd_df["Speed"]).astype(int)

    if ax is None:
        _, ax = plt.subplots()
    sns.lineplot(data=pd_df, x="Speed", y="IBI", ci=None, ax=ax)
    ax.set_ylabel("IBI (s)")
    ax.set_xlabel("Speed (cm / s)")
    ax.set_title("Speed vs IBI")

    return pd_df, ax, spear_r, spear_p, len(ibi) + 1


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

    # binsize = kwargs.get("binsize", 2)
    # min_speed, max_speed = kwargs.get("range", [0, 40])

    # max_speed = min(max_speed, np.ceil(speed.max() / binsize) * binsize)
    # min_speed = max(min_speed, np.floor(speed.min() / binsize) * binsize)
    # bins = np.arange(min_speed, max_speed, binsize)

    # visit_time = np.histogram(speed, bins)[0]
    # speedInd = np.digitize(speed, bins) - 1

    # visit_time = visit_time / samples_per_sec
    # binned_lfp = [np.sum(lfp_amplitudes[speedInd == i]) for i in range(len(bins) - 1)]
    # rate = np.array(binned_lfp) / visit_time

    pd_df = list_to_df(
        [speed, lfp_amplitudes], transpose=True, headers=["Speed", "LFP amplitude"]
    )
    pd_df = pd_df[pd_df["Speed"] <= 40]
    pd_df["Speed"] = np.around(pd_df["Speed"])

    return pd_df


def speed_lfp_amp(
    recording,
    figures,
    base_dir,
    clean_method="avg",
    fmin=5,
    fmax=12,
    speed_sr=10,
    **kwargs,
):
    # TODO should get this to work with non-NC
    clean_kwargs = kwargs.get("clean_kwargs", {})
    lc = LFPClean(method=clean_method, visualise=False)
    signals_grouped_by_region = lc.clean(
        recording.signals, 0.5, 100, method_kwargs=clean_kwargs
    )["signals"]
    fmt = kwargs.get("image_format", "png")

    # Single values
    spatial = recording.spatial.underlying
    simuran.set_plot_style()
    results = {}
    skip_rate = int(spatial.get_sampling_rate() / speed_sr)
    slicer = slice(skip_rate, -skip_rate, skip_rate)
    speed = spatial.get_speed()[slicer]
    results["mean_speed"] = np.mean(speed)
    results["duration"] = spatial.get_duration()
    results["distance"] = results["mean_speed"] * results["duration"]

    basename = recording.get_name_for_save(base_dir)

    # Figures
    simuran.set_plot_style()
    for name, signal in signals_grouped_by_region.items():  #
        lfp_signal = signal

        # Speed vs LFP power
        pd_df = speed_vs_amp(spatial, lfp_signal, fmin, fmax, samplesPerSec=speed_sr)

        results[f"{name}_df"] = pd_df

        fig, ax = plt.subplots()
        sns.lineplot(data=pd_df, x="Speed", y="LFP amplitude", ax=ax)
        simuran.despine()
        fname = basename + "_speed_theta_{}".format(name)
        speed_amp_fig = simuran.SimuranFigure(
            fig, filename=fname, done=True, format=fmt, dpi=400
        )
        figures.append(speed_amp_fig)

    return results


def define_recording_group(base_dir):
    # TODO change this to allow running on other places
    main_dir = r"D:\SubRet_recordings_imaging"
    dirs = base_dir[len(main_dir + os.sep) :].split(os.sep)
    dir_to_check = dirs[0]
    if dir_to_check.startswith("CS"):
        group = "Control"
    elif dir_to_check.startswith("LS"):
        group = "Lesion"
    else:
        group = "Undefined"
    return group


def combine_results(info, extra_info):
    """This uses the pickle output from SIMURAN."""
    simuran.set_plot_style()
    data_animal_list, fname_animal_list = info
    out_dir, name = extra_info
    os.makedirs(out_dir, exist_ok=True)

    n_ctrl_animals = 0
    n_lesion_animals = 0
    df_lists = []
    for item_list, fname_list in zip(data_animal_list, fname_animal_list):
        r_ctrl = 0
        r_les = 0
        for item_dict, fname in zip(item_list, fname_list):
            item_dict = item_dict["speed_lfp_amp"]
            data_set = define_recording_group(os.path.dirname(fname))
            if data_set == "Control":
                r_ctrl += 1
            else:
                r_les += 1

            for r in ["SUB", "RSC"]:
                id_ = item_dict[r + "_df"]
                id_["Group"] = data_set
                id_["region"] = r
                df_lists.append(id_)

        n_ctrl_animals += r_ctrl / len(fname_list)
        n_lesion_animals += r_les / len(fname_list)
    print(f"{n_ctrl_animals} CTRL animals, {n_lesion_animals} Lesion animals")

    df = pd.concat(df_lists, ignore_index=True)
    df.replace("Control", "Control (ATN,   N = 6)", inplace=True)
    df.replace("Lesion", "Lesion  (ATNx, N = 6)", inplace=True)

    print("Saving plots to {}".format(os.path.join(out_dir, "summary")))
    for ci, oname in zip([95, None], ["_ci", ""]):
        sns.lineplot(
            data=df[df["region"] == "SUB"],
            x="Speed",
            y="LFP amplitude",
            style="Group",
            hue="Group",
            ci=ci,
            estimator=np.median,
        )
        simuran.despine()
        plt.xlabel("Speed (cm / s)")
        plt.ylabel("Amplitude ({}V)".format(UnicodeGrabber.get("micro")))
        plt.title("Subicular LFP power (median)")

        os.makedirs(os.path.join(out_dir, "summary"), exist_ok=True)
        plt.savefig(
            os.path.join(
                out_dir, "summary", name + "--sub--speed--theta{}.png".format(oname)
            ),
            dpi=400,
        )

        plt.close("all")

        sns.lineplot(
            data=df[df["region"] == "RSC"],
            x="Speed",
            y="LFP amplitude",
            style="Group",
            hue="Group",
            ci=ci,
            estimator=np.median,
        )
        simuran.despine()
        plt.xlabel("Speed (cm / s)")
        plt.ylabel("Amplitude ({}V)".format(UnicodeGrabber.get("micro")))
        plt.title("Retrosplenial LFP power (median)")

        plt.savefig(
            os.path.join(
                out_dir, "summary", name + "--rsc--speed--theta{}.png".format(oname)
            ),
            dpi=400,
        )

        plt.close("all")


def recording_ibi_headings():
    return [
        "IBI R",
        "IBI P",
        "Number of bursts",
        "Speed R",
        "Speed P",
        "Mean speed",
        "Mean firing rate",
    ]


def recording_speed_ibi(recording, out_dir, base_dir, **kwargs):
    """This is performed per cell in the recording."""
    # How many results expected in a row?
    NUM_RESULTS = len(recording_ibi_headings())
    img_format = kwargs.get("img_format", ".png")

    simuran.set_plot_style()

    output = {}
    # To avoid overwriting what has been set to analyse
    all_analyse = deepcopy(recording.get_set_units())

    # Unit contains probe/tetrode info, to_analyse are list of cells
    spatial_error = False
    try:
        recording.spatial.load()
    except BaseException:
        print(
            "WARNING: Unable to load spatial information for {}".format(
                recording.source_file
            )
        )
        spatial_error = True

    spatial = recording.spatial.underlying
    for unit, to_analyse in zip(recording.units, all_analyse):

        # Two cases for empty list of cells
        if to_analyse is None:
            continue
        if len(to_analyse) == 0:
            continue

        unit.load()
        # Loading can overwrite units_to_use, so reset these after load
        unit.units_to_use = to_analyse
        out_str_start = str(unit.group)
        no_data_loaded = unit.underlying is None
        if not no_data_loaded:
            available_units = unit.underlying.get_unit_list()

        for cell in to_analyse:
            name_for_save = out_str_start + "_" + str(cell)
            output[name_for_save] = [np.nan] * NUM_RESULTS

            if spatial_error:
                continue
            # Check to see if this data is ok
            if no_data_loaded:
                continue
            if cell not in available_units:
                continue

            op = [np.nan] * NUM_RESULTS
            fig, axes = plt.subplots(2, 1)
            unit.underlying.set_unit_no(cell)
            spike_train = unit.underlying.get_unit_stamp()
            ibi_df, ibi_ax, sr, sp, nb = speed_ibi(spatial, spike_train, ax=axes[0])
            op[0] = sr
            op[1] = sp
            op[2] = nb

            res, speed_ax = speed_firing(spatial, spike_train, ax=axes[1], binsize=2)

            op[3] = res["lin_fit_r"]
            op[4] = res["lin_fit_p"]
            op[5] = np.mean(np.array(spatial.get_speed()))
            op[6] = len(spike_train) / unit.underlying.get_duration()

            simuran.despine()
            plt.tight_layout()
            out_name_end = recording.get_name_for_save(base_dir)
            out_name_end += "_T{}_SS{}".format(out_str_start, str(cell))
            out_name = os.path.join(out_dir, out_name_end) + img_format
            print("Saving plot to {}".format(out_name))
            fig.savefig(out_name, dpi=400)
            plt.close(fig)

            output[name_for_save] = op
            # Do analysis on that unit
            unit.underlying.reset_results()

    return output