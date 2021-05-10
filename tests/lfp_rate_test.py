import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from lfp_atn_simuran.analysis.lfp_rate_map import lfp_rate, lfp_rate_plot
from lfp_atn_simuran.analysis.lfp_clean import LFPClean
import astropy.units as u

sys.path.insert(0, "..")
from lib.plots import plot_pos_over_time

from default_recording import load_recording

here = os.path.dirname(os.path.abspath(__file__))


def fake_sig_version():
    recording = load_recording()
    nc_spatial = recording.spatial.underlying
    sigs = LFPClean.avg_signals(recording.signals, min_f=0.1, max_f=100)
    time = nc_spatial.get_duration()  # time in seconds
    rate1 = int(time * nc_spatial.get_sampling_rate())
    for region, signal in sigs.items():
        posX = nc_spatial._pos_x[
            np.logical_and(nc_spatial.get_time() >= 0, nc_spatial.get_time() <= time)
        ]
        posY = nc_spatial._pos_y[
            np.logical_and(nc_spatial.get_time() >= 0, nc_spatial.get_time() <= time)
        ]
        max_x, max_y = max(posX), max(posY)
        _, positions, _ = nc_spatial.get_event_loc(
            np.array(signal.get_timestamps()), keep_zero_idx=True
        )
        for i in range(len(positions[0])):
            pos = [positions[0][i], positions[1][i]]
            if (pos[0] > (max_x / 2)) and (pos[1] > (max_y / 2)):
                val = 1
            elif (pos[0] < (max_x / 2)) and (pos[1] < (max_y / 2)):
                val = -1
            else:
                val = 0
            signal.samples[i] = val * u.mV

        data = lfp_rate(nc_spatial, signal, range=(0, time), pixel=3, brAdjust=False)
        fig, ax = plt.subplots()
        lfp_rate_plot(data, ax=ax)
        fig.savefig(os.path.join(here, f"{region}_lfp_rate.png"), dpi=400)
        plt.close(fig)
    fig, ax = plt.subplots()
    ax.plot(nc_spatial._pos_x[:rate1], nc_spatial._pos_y[:rate1], c="k")
    ax.invert_yaxis()
    fig.savefig(os.path.join(here, f"Path.png"), dpi=400)
    plt.close(fig)


def main(set_file, low_f, high_f):
    for type_ in (".pos", ".txt"):
        recording = load_recording(set_file_location=set_file, type_=type_)
        nc_spatial = recording.spatial.underlying
        lc = LFPClean()
        sigs = lc.clean(recording.signals, min_f=0.5, max_f=100)["signals"]
        time = nc_spatial.get_duration()  # time in seconds
        rate1 = int(time * nc_spatial.get_sampling_rate())
        for region, signal in sigs.items():
            data = lfp_rate(
                nc_spatial,
                signal,
                low_f=low_f,
                high_f=high_f,
                range=(0, time),
                pixel=5,
                filter_kwargs={"verbose": "WARNING"},
                filter=["b", 5],
            )
            fig, ax = plt.subplots()
            lfp_rate_plot(data, ax=ax)
            fig.savefig(
                os.path.join(here, f"{region}-{type_[1:]}_lfp_rate.png"), dpi=400
            )
            plt.close(fig)
        fig, ax = plt.subplots()
        ax.plot(nc_spatial._pos_x[:rate1], nc_spatial._pos_y[:rate1], c="k")
        ax.invert_yaxis()
        fig.savefig(os.path.join(here, f"Path-{type_[1:]}.png"), dpi=400)
        plt.close(fig)

    # plot_pos_over_time(nc_spatial._pos_x[:rate1], nc_spatial._pos_y[:rate1], rate=1)


if __name__ == "__main__":
    low_f_ = 7
    high_f_ = 12
    set_file = r"D:\SubRet_recordings_imaging\CSR6\small  sq_big sq\22032018\S1_small\22032018_CSR6_smallbig_small_1.set"
    main(set_file, low_f_, high_f_)
    # fake_sig_version()
