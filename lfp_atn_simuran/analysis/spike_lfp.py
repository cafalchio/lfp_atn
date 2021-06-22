from copy import deepcopy

import simuran
import numpy as np

from lfp_atn_simuran.analysis.lfp_clean import LFPClean


def recording_spike_lfp(recording, clean_method="avg", **kwargs):
    clean_kwargs = kwargs.get("clean_kwargs", {})
    lc = LFPClean(method=clean_method, visualise=False)
    fmin = 0
    fmax = 100
    signals_grouped_by_region = lc.clean(
        recording.signals, fmin, fmax, method_kwargs=clean_kwargs
    )["signals"]

    simuran.set_plot_style()
    fmt = kwargs.get("image_format", "png")

    sub_sig = signals_grouped_by_region["SUB"]

    # SFC here
    nc_sig = sub_sig.to_neurochat()

    NUM_RESULTS = 2

    output = {}
    # To avoid overwriting what has been set to analyse
    all_analyse = deepcopy(recording.get_set_units())

    # Unit contains probe/tetrode info, to_analyse are list of cells
    for unit, to_analyse in zip(recording.units, all_analyse):

        # Two cases for empty list of cells
        if to_analyse is None:
            continue
        if len(to_analyse) == 0:
            continue

        unit.load()
        # Loading can overwrite units_to_use, so reset these after load
        unit.units_to_use = to_analyse
        out_str_start = str(unit.group)
        no_data_loaded = unit.underlying is None
        available_units = unit.underlying.get_unit_list()

        for cell in to_analyse:
            name_for_save = out_str_start + "_" + str(cell)
            output[name_for_save] = [np.nan] * NUM_RESULTS

            # Check to see if this data is ok
            if no_data_loaded:
                continue
            if cell not in available_units:
                continue

            unit.underlying.set_unit_no(cell)
            # Do analysis on that unit
            spike_train = unit.underlying.get_unit_stamp()
            g_data = nc_sig.plv(spike_train, mode="bs", fwin=[0, 20])
            sta = g_data["STAm"]
            sfc = g_data["SFC"]

            output[name_for_save] = [sta, sfc]
            unit.underlying.reset_results()

    return output


# TODO finish this function
def combine_results(info, extra):
    data = info[0]

    print(data)
    return
