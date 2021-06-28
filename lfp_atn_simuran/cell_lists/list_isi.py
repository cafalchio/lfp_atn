import os
from copy import deepcopy

import simuran
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))

def stat_per_cell(recording):
    output = {}
    # To avoid overwriting what has been set to analyse
    all_analyse = deepcopy(recording.get_set_units())

    for unit, to_analyse in zip(recording.units, all_analyse):
        loaded = False
        if to_analyse is None:
            continue
        if len(to_analyse) == 0:
            continue
        if not loaded:
            unit.load()
            unit.units_to_use = to_analyse
        out_str_start = str(unit.group)
        if unit.underlying is None:
            for cell in to_analyse:
                output[out_str_start + "_" + str(cell)] = [
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ]
        else:
            for cell in to_analyse:
                if cell in unit.underlying.get_unit_list():
                    unit.underlying.set_unit_no(cell)

                    wave_prop = unit.underlying.wave_property()
                    isi_data = unit.underlying.isi()
                    results = unit.underlying.get_results()
                    output[out_str_start + "_" + str(cell)] = [
                        results["Mean width"],
                        results["Mean Spiking Freq"],
                        results["Median ISI"],
                        results["Std ISI"],
                        results["CV ISI"],
                    ]
                    unit.underlying.reset_results()
                else:
                    output[out_str_start + "_" + str(cell)] = [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ]
    # pdf_file.close()
    return output

def main():
    cell_list = os.path.join(here, "CTRL_Lesion_cells_filled.xlsx")
    headers = [
        "Directory",
        "Filename",
        "Group",
        "Unit",
        "Mean Width",
        "Firing Rate",
        "Median ISI",
        "Std ISI",
        "CV ISI",
    ]
    simuran.analyse_cell_list(cell_list, stat_per_cell, headers)

if __name__ == "__main__":
    main()