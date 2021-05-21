import numpy as np
from scipy.signal import welch
from astropy import units as u
from sklearn.cluster import KMeans

from lfp_atn_simuran.analysis.lfp_clean import LFPClean


def powers(recording, clean_method="avg", fmin=1, fmax=100, **kwargs):
    # TODO refactor the cleaning
    clean_kwargs = kwargs.get("clean_kwargs", {})
    lc = LFPClean(method=clean_method, visualise=False)
    signals = lc.clean(recording.signals, fmin, fmax, method_kwargs=clean_kwargs)[
        "signals"
    ]
    return signal_powers(signals, **kwargs)


def signal_powers(signals_grouped_by_region, **kwargs):
    results = {}
    window_sec = kwargs.get("window_sec", 2)

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

        low, high = [0.1, 125]
        window_sec = kwargs.get("window_sec", 2)
        unit = kwargs.get("unit", "micro")
        scale = u.uV if unit == "micro" else u.mV
        sf = signal.get_sampling_rate()
        lfp_samples = np.array(signal.samples.to(scale))

        # Compute the modified periodogram (Welch)
        nperseg = int(window_sec * sf)
        freqs, psd = welch(lfp_samples, sf, nperseg=nperseg)
        idx_band = np.logical_and(freqs >= low, freqs <= high)
        results["{} welch".format(name)] = [freqs[idx_band], psd[idx_band]]

    return results
