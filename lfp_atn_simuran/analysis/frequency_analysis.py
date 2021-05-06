import numpy as np
from scipy.signal import welch
from astropy import units as u
from sklearn.cluster import KMeans

from lfp_atn_simuran.analysis.lfp_clean import LFPClean
from sklearn.metrics import silhouette_score


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
    n_clusters = kwargs.get("n_clusters", None)
    step = kwargs.get("step", 0.1)
    min_f = kwargs.get("min_f", 1.0)
    max_f = kwargs.get("max_f", 100)
    n_features = int(s_part / step)
    cluster_features = np.zeros((len(recording.signals), n_features)) * u.uV
    for i, sig in enumerate(recording.signals):
        for j, val in enumerate(np.arange(0, s_part, step=step)):
            sample = sig.in_range(val, val + 0.1)
            res = np.sum(np.abs(sample))
            cluster_features[i][j] = res

    results = {"clustering": {}}
    sigs = {}

    if n_clusters is None:
        sc = []
        for clusts in range(2, 6):
            cluster = KMeans(n_clusters=clusts, random_state=42)
            cluster_labels = cluster.fit_predict(cluster_features)
            sc.append(silhouette_score(cluster_features, cluster_labels))
        results["silhoeutte"] = sc
        best_sc, best_id = -1, -1
        for i in range(4):
            if sc[i] > best_sc:
                best_sc, best_id = sc[i], i + 2
        cluster = KMeans(n_clusters=best_id, random_state=42)
        cluster_labels = cluster.fit_predict(cluster_features)
        sc = silhouette_score(cluster_features, cluster_labels)
        results["silhoeutte_final"] = sc
        n_clusters = best_id

    else:
        cluster = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = cluster.fit_predict(cluster_features)
        sc = silhouette_score(cluster_features, cluster_labels)
        results["silhoeutte"] = sc

    for i in range(n_clusters):
        idxs = np.nonzero(cluster_labels == i)[0]
        sigs[i] = recording.signals.subsample(idxs)
        results["clustering"][i] = [s.channel for s in sigs[i]]

    for i in range(n_clusters):
        lc = LFPClean()
        avg_sig = lc.clean(sigs[i], min_f, max_f)["signals"]
        res = signal_powers(avg_sig, **kwargs)
        for k, v in res.items():
            results[f"Cluster {i} -- {k}"] = v

    return results


def powers(recording, **kwargs):
    # TODO refactor the cleaning
    min_f = kwargs.get("min_f", 1.0)
    max_f = kwargs.get("max_f", 100)
    lc = LFPClean(method="avg")
    signals = lc.clean(recording.signals, min_f, max_f)["signals"]
    return signal_powers(signals, **kwargs)


def signal_powers(signals_grouped_by_region, **kwargs):
    results = {}
    window_sec = kwargs.get("window_sec", 4)

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
