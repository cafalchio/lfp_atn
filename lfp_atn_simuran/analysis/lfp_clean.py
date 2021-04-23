"""Clean LFP signals."""
from collections import OrderedDict

import numpy as np
import simuran

from neurochat.nc_utils import butter_filter


def detect_outlying_signals(signals, z_threshold=3):
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

    z_score_abs = np.zeros(shape=(len(signals), len(signals[0])))

    for i, s in enumerate(signals):
        z_score_abs[i] = np.abs(s - avg_sig) / std_sig

    z_score_means = np.mean(z_score_abs, axis=1)

    good, bad = [], []
    for i, val in enumerate(z_score_means):
        if val > z_threshold:
            bad.append(i)
        else:
            good.append(i)

    good_signals = [signals[i] for i in good]
    bad_signals = [signals[i] for i in bad]

    return good_signals, bad_signals, good, bad


def clean_and_average_signals(
    signals, sampling_rate, filter_, z_threshold=3, verbose=False
):
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
        Defaults to 3.
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
    good_signals, bad_signals, good_idx, bad_idx = detect_outlying_signals(
        signals_, z_threshold=z_threshold
    )
    if verbose:
        print("Excluded {} signals with indices {}".format(len(bad_idx), bad_idx))

    # 1a. Consider trying to remove noise per channel? Or after avg?

    # 2. Average the good signals
    avg_sig = np.mean(good_signals, axis=0)

    # 3. Smooth the signal
    # TODO replace by own smoothing
    smooth_sig = butter_filter(avg_sig, sampling_rate, *filter_)

    if hasattr(signals[0], "unit"):
        return smooth_sig * signals[0].unit
    else:
        return smooth_sig


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

    """

    def __init__(self, method="avg", visualise=False):
        self.method = method
        self.visualise = visualise

    @staticmethod
    def clean_lfp_signals(
        recording, min_f=1.5, max_f=100, append_avg=False, verbose=False, vis=False
    ):
        """
        Clean the lfp signals in a recording.

        Parameters
        ----------
        recording : simuran.recording.Recording

        Returns
        -------
        EegArray or tuple of EegArray
            The cleaned signals, with average signals appended
            Or (the cleaned signals, the average signals)

        """
        sig_dict = LFPClean._clean_avg_signals(recording, min_f, max_f, verbose)
        appended_sigs = recording.get_eeg_signals(copy=False)

        if append_avg:
            appended_sigs = recording.get_eeg_signals(copy=False)
        else:
            appended_sigs = simuran.EegArray()

        for k, v in sig_dict.items():
            eeg = simuran.Eeg()
            eeg.from_numpy(v, sampling_rate=recording.signals[0].sampling_rate)
            eeg.set_region(k)
            eeg.set_channel("avg")
            appended_sigs.append(eeg)

        if vis:
            appended_sigs.plot(proj=False)

        return appended_sigs

    @staticmethod
    def avg_signals(signals, min_f, max_f):
        # TODO reconsider the filtering and method here
        filter_ = [10, min_f, max_f, "bandpass"]
        signals_grouped_by_region = signals.split_into_groups("region")

        output_dict = OrderedDict()
        for region, (signals, idxs) in signals_grouped_by_region.items():
            val = clean_and_average_signals(
                [s.samples for s in signals],
                signals[0].sampling_rate,
                filter_,
            )
            eeg = simuran.Eeg()
            eeg.from_numpy(val, sampling_rate=signals[0].sampling_rate)
            eeg.set_region(region)
            eeg.set_channel("avg")
            output_dict[region] = eeg

        return output_dict

    @staticmethod
    def _clean_avg_signals(recording, min_f=1.5, max_f=100, verbose=False):
        filter_ = [10, min_f, max_f, "bandpass"]

        lfp_signals = recording.get_signals()

        signals_grouped_by_region = lfp_signals.split_into_groups("region")

        output_dict = OrderedDict()
        for region, (signals, idxs) in signals_grouped_by_region.items():
            output_dict[region] = clean_and_average_signals(
                [s.samples for s in signals],
                signals[0].sampling_rate,
                filter_,
                verbose=verbose,
            )

        return output_dict
