import os

import simuran

from lfp_atn_simuran.analysis.spike_lfp import recording_spike_lfp
from lfp_atn_simuran.analysis.speed_lfp import (
    recording_speed_ibi,
    recording_ibi_headings,
)
from lfp_atn_simuran.analysis.parse_cfg import parse_cfg_info

here = os.path.dirname(os.path.abspath(__file__))
herename = os.path.splitext(os.path.basename(__file__))[0]


def spike_ibi_and_locking(recording, out_dir, base_dir, **kwargs):
    res1 = recording_speed_ibi(recording, out_dir, base_dir, **kwargs)

    # TODO bring in spike lfp here
    res2 = []

    return res1


def main():
    cfg = parse_cfg_info()
    base_dir = cfg["cfg_base_dir"]
    out_dir = os.path.abspath(os.path.join(here, "..", "sim_results", herename))
    os.makedirs(out_dir, exist_ok=True)
    cell_list = os.path.join(here, "CTRL_Lesion_cells_filled.xlsx")
    headers = recording_ibi_headings()
    simuran.analyse_cell_list(
        cell_list,
        spike_ibi_and_locking,
        headers,
        None,
        out_dir,
        fn_args=(out_dir, base_dir),
    )


if __name__ == "__main__":
    main()