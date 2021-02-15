import os
import mne as mne
import numpy as np

from neurochat.nc_lfp import NLfp
def mne_example(eeg_filename1, eeg_filename2):
    print(f"Loading eeg data from {eeg_filename1}")
    lfp1 = NLfp()
    lfp1.load_lfp_Axona(eeg_filename1)

    print(f"Loading eeg data from {eeg_filename2}")
    lfp2 = NLfp()
    lfp2.load_lfp_Axona(eeg_filename2)

    # Idea of what MNE wants
    # n_channels * n_samples or Transpose -> as numpy
    lfp_list = []

    for lfp in [lfp1, lfp2]:
        lfp_list.append(lfp.get_samples())
    raw_data = np.array(lfp_list, float)
    # Convert to volts
    raw_data = raw_data / 1000

    ch_names = ["1", "2"]
    sfreq = lfp1.get_sampling_rate()
    ch_types = ["eeg"] * 2
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    raw = mne.io.RawArray(raw_data, info)

    return raw


def plot_mne(raw_array, base_name):
    raw_array.load_data()
    raw_array.plot(
        n_channels=2,
        block=True,
        duration=50,
        show=True,
        clipping="transparent",
        title="Raw LFP Data from {}".format(base_name),
        remove_dc=False,
        scalings=dict(eeg=350e-6),
    )


if __name__ == "__main__":
    # Test MNE
    test_eeg_loc1 = r"D:\Beths\CanCSRetCa2\small_bigsq\10082018\S1_smallsq\10082018_CanCSubRetCa2_smallsq_2_1.eeg"
    test_eeg_loc2 = r"D:\Beths\CanCSRetCa2\small_bigsq\10082018\S1_smallsq\10082018_CanCSubRetCa2_smallsq_2_1.eeg2"

    mne_object = mne_example(test_eeg_loc1, test_eeg_loc2)

    name = os.path.splitext(os.path.basename(test_eeg_loc1))[0]
    plot_mne(mne_object, name)