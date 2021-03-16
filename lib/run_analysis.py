from lib.utils import create_dataframe, clean_data
import random

# import re
# import math
import pandas as pd
from lib.data_pos import RecPos
from lib.data_lfp import load_lfp_Axona
from lib.plots import plot_tmaze
import mne


if __name__ == "__main__":

    ## Pre-processing
    path = "d:/beths"
    df = create_dataframe(path)
    df = clean_data(df)
    df.to_csv(path + "/data_df.csv", header=False)

    # Load T-maze files
    df = df.loc[df.maze != "screening"]
    df = df.loc[df.habituation == 0]
    tmaze_files = (
        df.loc[df.maze == "tmaze", ["folder", "filename"]].agg("/".join, axis=1).values
    )

    # Save table:

    # Select a random file (test)
    i = random.randint(0, len(tmaze_files))
    file = tmaze_files[i]
    print(f"Selected file:\n\t{file}")

    # Load position
    position = RecPos(file)
    x, y = position.get_position()
    start = position.get_tmaze_start()
    w_view = position.get_window_view()
    plot_tmaze(x, y, w_view)

    # # Test MNE
    # test_eeg_loc1 = r"D:\SubRet_recordings_imaging\CSubRet1\CSubRet1_recording\CSR1_small sq\04092017\04092017_CSubRet1_smallsq_d2_1.eeg"
    # test_eeg_loc2 = r"D:\SubRet_recordings_imaging\CSubRet1\CSubRet1_recording\CSR1_small sq\04092017\04092017_CSubRet1_smallsq_d2_1.eeg2"

    # mne_object = mne_example(test_eeg_loc1, test_eeg_loc2)

    # name = os.path.splitext(os.path.basename(test_eeg_loc1))[0]
    # plot_mne(mne_object, name)
