import numpy as np
import matplotlib.pyplot as plt
from neurochat.nc_plot import plv, plv_bs, plv_tr
from icecream import ic


def nc_sfc(lfp, spike_train):
    g_data = lfp.plv(spike_train)
    print(lfp.get_results())
    figs = plv(g_data)
    for i, f in enumerate(figs):
        f.savefig(f"out_{i}.png", dpi=300)
        plt.close(f)


def recording_spike_lfp(recording, **kwargs):
    pass