import os
import shutil
from site import addsitedir

import simuran
import pandas as pd
from skm_pyutils.py_save import save_mixed_dict_to_csv
from skm_pyutils.py_path import get_all_files_in_dir

try:
    from lfp_atn_simuran.analysis.lfp_clean import LFPClean
    from lfp_atn_simuran.analysis.plot_coherence import plot_recording_coherence
    from lfp_atn_simuran.analysis.frequency_analysis import powers

    do_analysis = True
except ImportError:
    do_analysis = False

lib_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
addsitedir(lib_folder)
from lib.plots import plot_pos_over_time


def analyse_recording(
    recording,
    output_location,
    set_file_location,
    min_f,
    max_f,
    clean_method,
    clean_kwargs,
):
    output_location = os.path.join(output_location, clean_method)
    os.makedirs(output_location, exist_ok=True)

    # lfp_clean = LFPClean(method=clean_method, visualise=False, show_vis=False)
    # result = lfp_clean.clean(
    #     recording, min_f=min_f, max_f=max_f, method_kwargs=clean_kwargs
    # )
    # sigs = result["signals"]

    spatial = recording.spatial.underlying
    times = plot_pos_over_time(spatial.get_pos_x(), spatial.get_pos_y())
    return times

    # # Then plot coherence and spectograms
    # figures = []
    # plot_recording_coherence(
    #     recording,
    #     figures,
    #     os.path.dirname(set_file_location),
    #     fmin=min_f,
    #     fmax=max_f,
    #     clean_method=clean_method,
    #     clean_kwargs=clean_kwargs,
    # )
    # for figure in figures:
    #     figure.set_filename(os.path.join(output_location, figure.get_filename()))
    #     figure.save()
    #     figure.close()

    # # Then integrate the periodogram to get powers
    # power_res = powers(
    #     recording,
    #     clean_method,
    #     fmin=1,
    #     fmax=100,
    #     clean_kwargs=clean_kwargs,
    # )
    # save_mixed_dict_to_csv(power_res, output_location, f"power_{clean_method}.csv")


def main(
    t_maze_dir,
    output_location,
    base_dir,
    mapping_file,
    xls_location=None,
    animal="",
    do_analysis=False,
    min_f=0.5,
    max_f=30,
):
    """Create a single recording for analysis."""
    # loop over the sessions in each t-maze folder
    columns = ["location", "start", "end", "session", "animal", "test", "mapping"]
    data_list = []
    df = None
    if xls_location is not None:
        if os.path.exists(xls_location):
            df = pd.read_excel(xls_location)
    if df is None:
        df = pd.DataFrame(columns=columns)

    if not do_analysis:
        return
    else:
        os.makedirs(output_location, exist_ok=True)
    for folder in os.listdir(t_maze_dir):
        dir_loc = os.path.join(t_maze_dir, folder)
        set_file_locations = get_all_files_in_dir(dir_loc, ext=".set")
        if len(set_file_locations) == 0:
            raise ValueError(f"No set files were found in {dir_loc}")
        set_file_location = set_file_locations[0]
        main_file_name = set_file_location[len(base_dir + os.sep) :].replace(
            os.sep, "--"
        )
        recording = simuran.Recording(
            param_file=mapping_file, base_file=set_file_location
        )
        inp = input(f"Analyse {set_file_location}? (y/n) ")
        if inp == "n":
            break
        if main_file_name not in df["location"]:
            done = False
            while not done:
                clean_kwargs = {"channels": [17, 18, 19, 20]}
                clean_method = "pick"
                # for clean_method in ("avg", "pick"):
                times = analyse_recording(
                    recording,
                    output_location,
                    set_file_location,
                    min_f,
                    max_f,
                    clean_method,
                    clean_kwargs,
                )
                if len(times) != 4:
                    print("Incorrect number of times, retrying")
                else:
                    done = True
            data = [
                main_file_name,
                times[0],
                times[1],
                folder[-1],
                animal,
                "first",
                os.path.basename(mapping_file),
            ]
            data_list.append(data)
            data = [
                main_file_name,
                times[2],
                times[3],
                folder[-1],
                animal,
                "second",
                os.path.basename(mapping_file),
            ]
            data_list.append(data)

    print("Saving results to {}".format(xls_location))
    process_list_to_df(df, data_list, columns, xls_location)


def process_list_to_df(orig_df, list_, columns, out_name):
    df = pd.DataFrame(list_, columns=columns)
    df = pd.concat((orig_df, df))
    if os.path.exists(out_name):
        split = os.splitext(out_name)
        new_name = split[0] + "__copy" + split[1]
        shutil.copy(out_name, new_name)
    df.to_excel(out_name, index=False)


if __name__ == "__main__":
    main_t_maze_dir = (
        r"D:\SubRet_recordings_imaging\LSubRet5\recording\plus maze\29112017_t1"
    )
    here = os.path.dirname(os.path.abspath(__file__))
    main_output_location = os.path.join(here, "results", "tmaze")

    base_dir = r"D:\SubRet_recordings_imaging"
    xls_location = os.path.join(main_output_location, "tmaze-times.xlsx")

    mapping_location = os.path.join(
        here, "..", "recording_mappings", "CL-SR_4-6-no-cells.py"
    )

    main(
        main_t_maze_dir,
        main_output_location,
        base_dir,
        xls_location,
        animal="LSR5",
        do_analysis=True,
    )
