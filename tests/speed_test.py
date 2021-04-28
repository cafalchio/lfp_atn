import sys

sys.path.insert(0, "..")
import os

import matplotlib.pyplot as plt
from neurochat.nc_spatial import NSpatial

from lib.data_pos import RecPos

here = os.path.dirname(os.path.abspath(__file__))


def main(file):
    pos = RecPos(file)
    pos.calculate_speed(num_samples=5, smooth_size=5, smooth=True)
    x, y = pos.get_position()
    speed = pos.get_speed()

    spatial = NSpatial(system="Axona")
    spatial.load(file[:-4] + "_3.txt")

    # Some comparison plots
    rate1 = 500
    rate2 = 2500

    # Compare position and speed
    fig, axes = plt.subplots(3, 2)
    fig.tight_layout()
    ax = axes[0][0]
    ax.set_title("Matheus position")
    ax.plot(x[:rate2], y[:rate2], c="k")
    ax.invert_yaxis()

    ax = axes[0][1]
    ax.set_title("NeuroChaT position")
    ax.plot(spatial._pos_x[:rate1], spatial._pos_y[:rate1], c="k")
    ax.invert_yaxis()

    ax = axes[1][0]
    ax.set_title("Matheus speed")
    ax.plot(speed[:rate2], c="k")

    ax = axes[1][1]
    ax.set_title("NeuroChaT speed")
    ax.plot(spatial._speed[:rate1], c="k")

    ax = axes[2][0]
    ax.set_title("Matheus speed hist")
    ax.hist(speed[:rate2], color="k", density=True)

    ax = axes[2][1]
    ax.set_title("NeuroChaT speed hist")
    ax.hist(spatial._speed[:rate1], color="k", density=True)

    fig.savefig(os.path.join(here, "compare.png"), dpi=400)


if __name__ == "__main__":
    # main_set_file = (
    #     r"D:/SubRet_recordings_imaging/CanCSR7/sleeps/S2/13072018_CanCSR7_sleep_1_2.set"
    # )
    main_set_file = os.path.join(
        "D:\\",
        "SubRet_recordings_imaging",
        os.path.normpath(
            "LSubRet5/recording/Small sq up_small sq down/01122017/S1_small sq up/01122017_smallsqdownup_up_1_1.set"
        ),
    )
    main(main_set_file)
