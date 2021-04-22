import sys; sys.path.insert(0, "..")
import os

import matplotlib.pyplot as plt
from neurochat.nc_spatial import NSpatial

from lib.data_pos import RecPos

here = os.path.dirname(os.path.abspath(__file__))

def main(file):
    pos = RecPos(file)
    x, y = pos.get_position()
    speed, x, y = pos.get_speed()

    spatial = NSpatial(system="Axona")
    spatial.load(file[:-4] + "_3.txt")
    
    # Some comparison plots

    # Compare position and speed
    fig, axes = plt.subplots(2, 2)
    fig.tight_layout()
    ax = axes[0][0]
    ax.set_title("Matheus position")
    ax.plot(x, y, c="k")
    ax.invert_yaxis()

    ax = axes[0][1]
    ax.set_title("NeuroChaT position")
    ax.plot(spatial._pos_x, spatial._pos_y, c="k")
    ax.invert_yaxis()
    
    ax = axes[1][0]
    ax.set_title("Matheus speed")
    ax.plot(speed, c="k")

    ax = axes[1][1]
    ax.set_title("NeuroChaT speed")
    ax.plot(spatial._speed, c="k")

    fig.savefig(os.path.join(here, "compare.png"), dpi=400)


if __name__ == "__main__":
    main_set_file = r"D:/SubRet_recordings_imaging/CanCSR7/sleeps/S2/13072018_CanCSR7_sleep_1_2.set"
    main(main_set_file)
