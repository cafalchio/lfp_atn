import numpy as np
import matplotlib.pyplot as plt
from neurochat.nc_plot import plv, plv_bs, plv_tr


def nc_sfc(lfp, spike_train):
    g_data = lfp.plv(spike_train)
    print(lfp.get_results())
    figs = plv(g_data)
    for i, f in enumerate(figs):
        plt.savefig(f, f"out_{i}.png", dpi=300)
        plt.close(f)


def sfc(lfp, spike_train):
    plt.rcParams["figure.figsize"] = (12, 3)  # Change the default figure size

    # Load the data and plot it.
    y = lfp.get_samples()
    n = spike_train
    t = np.array([(1.0 / lfp.get_sampling_rate()) * i for i in range(len(y))])
    K = 500
    N = len(lfp)
    dt = t[1] - t[0]  # Get the sampling interval.

    SYY = np.zeros(int(N / 2 + 1))  # Variable to store field spectrum.
    SNN = np.zeros(int(N / 2 + 1))  # Variable to store spike spectrum.
    SYN = np.zeros(int(N / 2 + 1), dtype=complex)  # Variable to store cross spectrum.

    for k in np.arange(K):  # For each trial,
        yf = np.fft.rfft(
            (y[k, :] - np.mean(y[k, :])) * np.hanning(N)
        )  # Hanning taper the field,
        nf = np.fft.rfft(
            (n[k, :] - np.mean(n[k, :]))
        )  # ... but do not taper the spikes.
        SYY = SYY + (np.real(yf * np.conj(yf))) / K  # Field spectrum
        SNN = SNN + (np.real(nf * np.conj(nf))) / K  # Spike spectrum
        SYN = SYN + (yf * np.conj(nf)) / K  # Cross spectrum

    cohr = np.real(SYN * np.conj(SYN)) / SYY / SNN  # Spike-field coherence
    f = np.fft.rfftfreq(N, dt)  # Frequency axis for plotting

    plt.plot(f, cohr)  # Plot the result.
    plt.xlim([0, 100])
    plt.ylim([0, 1])
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Coherence")
    plt.show()


def recording_spike_lfp(recording, **kwargs):
    pass