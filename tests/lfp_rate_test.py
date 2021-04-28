import os
import sys

import matplotlib.pyplot as plt
from lfp_atn_simuran.analysis.lfp_rate_map import lfp_rate, lfp_rate_plot
from lfp_atn_simuran.analysis.lfp_clean import LFPClean

sys.path.insert(0, "..")
from lib.plots import plot_pos_over_time

from default_recording import load_recording

here = os.path.dirname(os.path.abspath(__file__))


def main():
    recording = load_recording()
    nc_spatial = recording.spatial.underlying
    sigs = LFPClean.avg_signals(recording.signals, min_f=0.1, max_f=100)
    time = nc_spatial.get_duration()  # time in seconds
    rate1 = int(time * nc_spatial.get_sampling_rate())
    for region, signal in sigs.items():
        data = lfp_rate(nc_spatial, signal, range=(0, time), pixel=3)
        fig, ax = plt.subplots()
        lfp_rate_plot(data, ax=ax)
        fig.savefig(os.path.join(here, f"{region}_lfp_rate.png"), dpi=400)
        plt.close(fig)
    fig, ax = plt.subplots()
    ax.plot(nc_spatial._pos_x[:rate1], nc_spatial._pos_y[:rate1], c="k")
    ax.invert_yaxis()
    fig.savefig(os.path.join(here, f"Path.png"), dpi=400)
    plt.close(fig)

    # plot_pos_over_time(nc_spatial._pos_x[:rate1], nc_spatial._pos_y[:rate1], rate=1)


if __name__ == "__main__":
    main()
