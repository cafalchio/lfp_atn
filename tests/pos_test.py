import os

from neurochat.nc_data import NData
from neurochat.nc_plot import loc_firing_and_place
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))


def main():
    folder = r"E:\Google_Drive\NeuroScience\Recordings\recording_example"
    spike_file = os.path.join(folder, "010416b-LS3-50Hz10V5ms.2")
    txt_file = os.path.join(folder, "010416b-LS3-50Hz10V5ms_2.txt")
    pos_file = os.path.join(folder, "010416b-LS3-50Hz10V5ms.pos")
    lfp_file = os.path.join(folder, "010416b-LS3-50Hz10V5ms.eeg")
    unit_no = 7

    for f in (txt_file, pos_file):
        ndata = NData()
        ndata.set_spike_file(spike_file)
        ndata.set_spatial_file(f)
        ndata.set_lfp_file(lfp_file)
        ndata.load()
        ndata.set_unit_no(unit_no)

        result = ndata.place()
        fig = loc_firing_and_place(result)
        fig.savefig(
            os.path.join(here, os.path.basename(f).replace(".", "--") + ".png"), dpi=400
        )
        plt.close(fig)

if __name__ == "__main__":
    main()
