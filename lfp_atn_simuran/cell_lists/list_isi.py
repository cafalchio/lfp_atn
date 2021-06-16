import os

import simuran
from simuran.analysis.custom.nc import stat_per_cell

here = os.path.dirname(os.path.abspath(__file__))

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