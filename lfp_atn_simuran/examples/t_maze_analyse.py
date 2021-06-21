import os
from site import addsitedir
from json import loads

import simuran
import pandas as pd
import matplotlib.pyplot as plt
import astropy.units as u
from neurochat.nc_lfp import NLfp
import numpy as np

try:
    from lfp_atn_simuran.analysis.lfp_clean import LFPClean
    from lfp_atn_simuran.analysis.plot_coherence import plot_recording_coherence
    from lfp_atn_simuran.analysis.frequency_analysis import powers
    from lfp_atn_simuran.analysis.parse_cfg import parse_cfg_info

    do_analysis = True
except ImportError:
    do_analysis = False

lib_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
addsitedir(lib_folder)
from lib.plots import plot_pos_over_time

here = os.path.dirname(os.path.abspath(__file__))


def main(excel_location, base_dir):
    df = pd.read_excel(excel_location)
    cfg = parse_cfg_info()
    delta_min = cfg["delta_min"]
    delta_max = cfg["delta_max"]
    theta_min = cfg["theta_min"]
    theta_max = cfg["theta_max"]
    window_sec = 2

    ituples = df.itertuples()
    num_rows = len(df)

    results = []
    for j in range(num_rows // 2):
        row1 = next(ituples)
        row2 = next(ituples)
        recording_location = os.path.join(base_dir, row1.location)
        recording_location = recording_location.replace("--", os.sep)
        param_file = os.path.join(here, "..", "recording_mappings", row1.mapping)

        recording = simuran.Recording(
            param_file=param_file, base_file=recording_location
        )
        spatial = recording.spatial.underlying
        lfp_clean = LFPClean(method="pick", visualise=False)
        clean_kwargs = cfg["clean_kwargs"]
        sig_dict = lfp_clean.clean(
            recording, min_f=0.5, max_f=100, method_kwargs=clean_kwargs
        )["signals"]

        fig, ax = plt.subplots()
        for r in (row1, row2):
            t1, t2 = r.start, r.end
            lfpt1, lfpt2 = int(t1 * 250), int(t2 * 250)

            st1, st2 = int(t1 * 50), int(t2 * 50)
            x_time = spatial.get_pos_x()[st1:st2]
            y_time = spatial.get_pos_y()[st1:st2]

            if r.test == "first":
                c = "k"
            else:
                c = "r"

            ax.plot(x_time, y_time, c=c, label=r.test)

            res_dict = {}
            for region, signal in sig_dict.items():
                lfp = NLfp()
                lfp.set_channel_id(signal.channel)
                lfp._timestamp = np.array(signal.timestamps[lfpt1:lfpt2] * u.mV)
                lfp._samples = np.array(signal.samples[lfpt1:lfpt2] * u.s)
                lfp._record_info["Sampling rate"] = signal.sampling_rate
                delta_power = lfp.bandpower(
                    [delta_min, delta_max], window_sec=window_sec, band_total=True
                )
                theta_power = lfp.bandpower(
                    [theta_min, theta_max], window_sec=window_sec, band_total=True
                )
                res_dict["{}_delta".format(region)] = delta_power["relative_power"]
                res_dict["{}_theta".format(region)] = theta_power["relative_power"]

            res_list = [r.location, r.session, r.animal, r.test]
            res_list += [
                res_dict["SUB_delta"],
                res_dict["SUB_theta"],
                res_dict["RSC_delta"],
                res_dict["RSC_theta"],
            ]
            results.append(res_list)

            name = os.path.splitext(r.location)[0]

        ax.invert_yaxis()
        ax.legend()
        base_dir_new = os.path.dirname(excel_location)
        figname = os.path.join(base_dir_new, name) + "_tmaze.png"
        fig.savefig(figname, dpi=400)
        plt.close(fig)

    headers = [
        "location",
        "session",
        "animal",
        "test",
        "SUB_delta",
        "SUB_theta",
        "RSC_delta",
        "RSC_theta",
    ]

    df = pd.DataFrame(results, columns=headers)

    split = os.path.splitext(excel_location)
    out_name = split[0] + "_results" + split[1]
    df.to_excel(out_name, index=False)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    main_output_location = os.path.join(here, "results", "tmaze")

    base_dir = r"D:\SubRet_recordings_imaging"
    xls_location = os.path.join(main_output_location, "tmaze-times.xlsx")

    main(xls_location, base_dir)
