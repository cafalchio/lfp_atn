import numpy as np
from scipy.signal import welch
from astropy import units as u

from lfp_atn_simuran.analysis.lfp_clean import LFPClean


def detect_outlying_signals(signals):
    avg_sig = np.mean(np.array([s.samples for s in signals]), axis=0)

    diff = np.zeros(shape=(len(signals), len(signals[0].samples)))

    for i, s in enumerate(signals):
        diff[i] = np.square(s.samples - avg_sig) / (
            np.sum(np.square(s.samples) + np.square(avg_sig)) / 2
        )

    return diff

def grouped_powers(recording, **kwargs):
    """Signal power in clusters."""
    s_part = kwargs.get("win_len", 2)
    print(s_part)
    cluster_features = np.zeros((len(recording.signals), 10)) * u.uV
    for i, sig in enumerate(recording.signals):
        for j, val in enumerate(np.arange(0, s_part, step=0.1)):
            sample = sig.in_range(val, val+0.1)
            res = np.sum(sample)
            cluster_features[i][j] = res
    for i, clust in enumerate(cluster_features):
        print(f"{i}: {clust}")



def powers(recording, **kwargs):
    signals = LFPClean.clean_lfp_signals(recording, 0.5, 125)
    return signal_powers(signals, **kwargs)


def signal_powers(signals, **kwargs):
    results = {}

    signals_grouped_by_region = signals.split_into_groups("region")

    for name, (signal, idxs) in signals_grouped_by_region.items():
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

        low, high = [0.5, 120]
        window_sec = kwargs.get("window_sec", 2 / (low + 0.000001))
        unit = kwargs.get("unit", "micro")
        scale = u.uV if unit == "micro" else u.mV
        sf = signal.get_sampling_rate()
        lfp_samples = np.array(signal.samples * scale)

        # Compute the modified periodogram (Welch)
        nperseg = int(window_sec * sf)
        freqs, psd = welch(lfp_samples, sf, nperseg=nperseg)
        idx_band = np.logical_and(freqs >= low, freqs <= high)
        results["{} welch".format(name)] = [freqs[idx_band], psd[idx_band]]

    return results
