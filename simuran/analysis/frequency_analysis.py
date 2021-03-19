from copy import deepcopy

import numpy as np

from scipy.signal import welch


def detect_outlying_signals(signals):
    avg_sig = np.mean(np.array([s.samples for s in signals]), axis=0)

    diff = np.zeros(shape=(len(signals), len(signals[0].samples)))

    for i, s in enumerate(signals):
        diff[i] = np.square(s.samples - avg_sig) / (
            np.sum(np.square(s.samples) + np.square(avg_sig)) / 2
        )

    return diff


def powers(recording, **kwargs):
    results = {}

    sub_signals = recording.signals.group_by_property("region", "SUB")[0]
    # Remove dead channels
    sub_signals = [s for s in sub_signals if not np.all((s.samples == 0))]
    rsc_signals = recording.signals.group_by_property("region", "RSC")[0]
    rsc_signals = [s for s in rsc_signals if not np.all((s.samples == 0))]
    all_signals = [sub_signals, rsc_signals]
    names = ["sub", "rsc"]

    # For now, lets just take the first non dead channels
    for sig_list, name in zip(all_signals, names):
        results["{} delta".format(name)] = np.nan
        results["{} theta".format(name)] = np.nan
        results["{} low gamma".format(name)] = np.nan
        results["{} high gamma".format(name)] = np.nan
        results["{} total".format(name)] = np.nan

        results["{} delta rel".format(name)] = np.nan
        results["{} theta rel".format(name)] = np.nan
        results["{} low gamma rel".format(name)] = np.nan
        results["{} high gamma rel".format(name)] = np.nan

        if len(sig_list) > 0:
            diff = detect_outlying_signals(sig_list)
            for i in range(len(sig_list)):
                results["{} {} diff".format(name, sig_list[i].channel)] = np.mean(
                    diff[i]
                )
            window_sec = 2
            avg_sig = np.mean(np.array([s.samples for s in sig_list]), axis=0)
            sig_in_use = sig_list[0]
            temp = deepcopy(sig_in_use.samples)
            sig_in_use.samples = avg_sig
            # TODO find good bands from a paper
            delta_power = sig_in_use.underlying.bandpower(
                [1.5, 4], window_sec=window_sec, band_total=True
            )
            theta_power = sig_in_use.underlying.bandpower(
                [6, 10], window_sec=window_sec, band_total=True
            )
            low_gamma_power = sig_in_use.underlying.bandpower(
                [30, 55], window_sec=window_sec, band_total=True
            )
            high_gamma_power = sig_in_use.underlying.bandpower(
                [65, 90], window_sec=window_sec, band_total=True
            )
            sig_in_use.samples = temp

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
            results["{} high gamma rel".format(name)] = high_gamma_power[
                "relative_power"
            ]

            low, high = [0.5, 120]
            window_sec = kwargs.get("window_sec", 2 / (low + 0.000001))
            unit = kwargs.get("unit", "micro")
            scale = 1000 if unit == "micro" else 1
            sf = sig_list[0].get_sampling_rate()
            lfp_samples = avg_sig * scale

            # Compute the modified periodogram (Welch)
            nperseg = int(window_sec * sf)
            freqs, psd = welch(lfp_samples, sf, nperseg=nperseg)
            idx_band = np.logical_and(freqs >= low, freqs <= high)
            results["{} welch".format(name)] = [freqs[idx_band], psd[idx_band]]

    return results
