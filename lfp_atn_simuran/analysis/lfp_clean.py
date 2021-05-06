"""Clean LFP signals."""
from collections import OrderedDict
import logging

import numpy as np
import simuran


def detect_outlying_signals(signals, z_threshold=1.1):
    """
    Detect signals that are outliers from the average.

    Parameters
    ----------
    signals : np.ndarray
        Assumed to be an N_chans * N_samples iterable.
    z_threshold : float
        The threshold for the mean signal z-score to be an outlier.

    Returns
    -------
    good : np.ndarray
        The clean signals
    outliers : np.ndarray
        The outliers
    good_idx : list
        The indices of the good signals
    outliers_idx : list
        The indices of the bad signals

    """
    avg_sig = np.mean(signals, axis=0)
    std_sig = np.std(signals, axis=0)
    # Use this with axis = 0 for per signal
    std_sig = np.where(std_sig == 0, 1, std_sig)

    z_score_abs = np.zeros(shape=(len(signals), len(signals[0])))

    for i, s in enumerate(signals):
        z_score_abs[i] = np.abs(s - avg_sig) / std_sig

    z_score_means = np.nanmean(z_score_abs, axis=1)

    # TODO test this more
    z_threshold = z_threshold * np.median(z_score_means)

    good, bad = [], []
    for i, val in enumerate(z_score_means):
        if val > z_threshold:
            bad.append(i)
        else:
            good.append(i)

    good_signals = np.array([signals[i] for i in good])
    bad_signals = np.array([signals[i] for i in bad])

    return good_signals, bad_signals, good, bad


def average_signals(signals, z_threshold=1.1, verbose=False, clean=True):
    """
    Clean and average a set of signals.

    Parameters
    ----------
    signals : iterable
        Assumed to be an N_chans * N_samples iterable.
    sampling_rate : int
        The sampling rate of the signals in samples/s.
    filter_ : tuple
        Butter filter parameters.
    z_threshold : float, optional.
        The threshold for the mean signal z-score to be an outlier.
        Defaults to 1.1. This means z > 1.1 * z.median is outlier.
    verbose : bool, optional.
        Whether to print further information, defaults to False.

    Returns
    -------
    np.ndarray
        The cleaned and averaged signals.

    """
    if type(signals) is not np.ndarray:
        signals_ = np.array(signals)
    else:
        signals_ = signals

    # 1. Try to identify dead channels
    if clean:
        good_signals, bad_signals, good_idx, bad_idx = detect_outlying_signals(
            signals_, z_threshold=z_threshold
        )
        if verbose:
            if len(bad_idx) != 0:
                print(
                    "Excluded {} signals with indices {}".format(len(bad_idx), bad_idx)
                )
    else:
        good_signals = signals

    # 1a. Consider trying to remove noise per channel? Or after avg?

    # 2. Average the good signals
    avg_sig = np.mean(good_signals, axis=0)

    if hasattr(signals[0], "unit"):
        return avg_sig * signals[0].unit
    else:
        return avg_sig


class LFPClean(object):
    """
    Class to clean LFP signals.

    Attributes
    ----------
    method : string
        The method to use for cleaning.
        Currently supports "avg".
    visualise : bool
        Whether to visualise the cleaning.

    Parameters
    ----------
    method : string
        The method to use for cleaning.
        Currently supports "avg".
    visualise : bool
        Whether to visualise the cleaning.
    show_vis : bool
        Whether to visualise on the fly or return figs

    Methods
    -------
    clean(recording/signals)

    """

    def __init__(self, method="avg", visualise=False, show_vis=True):
        self.method = method
        self.visualise = visualise
        self.show_vis = show_vis

    def compare_methods(self, methods, data, min_f, max_f, **filter_kwargs):
        results = {}
        temp = self.visualise
        for method in methods:
            self.method = method
            self.visualise = False
            result = self.clean(data, min_f, max_f, **filter_kwargs)["signals"]
            for k, v in result.items():
                v.set_channel(method[:5])
                results[f"{method}-{k}"] = v
        self.visualise = temp

        if isinstance(data, simuran.Recording):
            signals = data.signals
        else:
            signals = data

        fig = self.vis_cleaning(results, signals)

        return fig

    def clean(self, data, min_f=None, max_f=None, method_kwargs=None, **filter_kwargs):
        """
        Clean the lfp signals.

        Parameters
        ----------
        data : simuran.recording.Recording or simuran.EegArray
            also accepts simuran.GenericContainer of simuran.BaseSignal
            The signals to clean

        Returns
        -------
        dict with keys "signals", "fig"

        """
        if method_kwargs is None:
            method_kwargs = {}
        if isinstance(data, simuran.Recording):
            signals = data.signals
        else:
            signals = data

        if min_f is not None:
            filter_kwargs["verbose"] = filter_kwargs.get("verbose", "WARNING")
            signals = self.filter_sigs(signals, min_f, max_f, **filter_kwargs)

        if self.method == "avg":
            z_threshold = method_kwargs.get("z_threshold", 1.1)
            result = self.avg_method(
                signals,
                min_f,
                max_f,
                clean=True,
                z_threshold=z_threshold,
                **filter_kwargs,
            )
        elif self.method == "avg_raw":
            result = self.avg_method(
                signals, min_f, max_f, clean=False, **filter_kwargs
            )
        else:
            logging.warning(f"{self.method} is not a valid clean method, using avg")

        results = {"signals": result, "fig": None}
        if self.visualise:
            fig = self.vis_cleaning(result, signals)
            results["fig"] = fig

        return results

    def vis_cleaning(self, result, signals):
        if isinstance(result, dict):
            eeg_array = simuran.EegArray()
            _, eeg_idxs = signals.group_by_property("channel_type", "eeg")
            eeg_sigs = signals.subsample(idx_list=eeg_idxs, inplace=False)
            eeg_array.set_container([simuran.Eeg(signal=eeg) for eeg in eeg_sigs])

            for k, v in result.items():
                eeg_array.append(v)
        else:
            eeg_array = simuran.EegArray()
            eeg_array.set_container([simuran.Eeg(signal=eeg) for eeg in result])

        fig = eeg_array.plot(proj=False, show=self.show_vis)

        return fig

    def avg_method(
        self, signals, min_f, max_f, clean=True, z_threshold=1.1, **filter_kwargs
    ):
        lfp_signals = signals

        signals_grouped_by_region = lfp_signals.split_into_groups("region")

        output_dict = OrderedDict()

        for region, (signals, _) in signals_grouped_by_region.items():
            val = average_signals(
                [s.samples for s in signals],
                z_threshold=z_threshold,
                verbose=True,
                clean=clean,
            )
            eeg = simuran.Eeg()
            eeg.from_numpy(val, sampling_rate=signals[0].sampling_rate)
            eeg.set_region(region)
            eeg.set_channel("avg")
            eeg.filter(min_f, max_f, inplace=True, **filter_kwargs)
            output_dict[region] = eeg

        return output_dict

    def filter_sigs(self, signals, min_f, max_f, **filter_kwargs):
        eeg_array = simuran.EegArray()
        for signal in signals:
            filt_s = signal.filter(min_f, max_f, inplace=False, **filter_kwargs)
            eeg_array.append(simuran.Eeg(signal=filt_s))
        return eeg_array