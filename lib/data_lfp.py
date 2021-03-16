import os
import re
import mne
import numpy as np


# Function from NeuroChat - read LFP
def load_lfp_Axona(file_name):
    file_directory, file_basename = os.path.split(file_name)
    file_tag, file_extension = os.path.splitext(file_basename)
    file_extension = file_extension[1:]
    set_file = os.path.join(file_directory, file_tag + ".set")
    if os.path.isfile(file_name):
        with open(file_name, "rb") as f:
            while True:
                line = f.readline()
                try:
                    line = line.decode("latin-1")
                except BaseException:
                    break

                if line == "":
                    break
                if line.startswith("trial_date"):
                    # Blank eeg file
                    if line.strip() == "trial_date":
                        total_samples = 0
                        return
                    date = " ".join(line.replace(",", " ").split()[1:])
                if line.startswith("trial_time"):
                    time = line.split()[1]
                if line.startswith("experimenter"):
                    experimenter = " ".join(line.split()[1:])
                if line.startswith("comments"):
                    comments = " ".join(line.split()[1:])
                if line.startswith("duration"):
                    duration = float("".join(line.split()[1:]))
                if line.startswith("sw_version"):
                    file_version = line.split()[1]
                if line.startswith("num_chans"):
                    total_channel = int("".join(line.split()[1:]))
                if line.startswith("sample_rate"):
                    sampling_rate = float("".join(re.findall(r"\d+.\d+|\d+", line)))
                if line.startswith("bytes_per_sample"):
                    bytes_per_sample = int("".join(line.split()[1:]))
                if line.startswith("num_" + file_extension[:3].upper() + "_samples"):
                    total_samples = int("".join(line.split()[1:]))
                if line.startswith("data_start"):
                    break

            num_samples = total_samples
            f.seek(0, 0)
            header_offset = []
            while True:
                try:
                    buff = f.read(10).decode("UTF-8")
                except BaseException:
                    break
                if buff == "data_start":
                    header_offset = f.tell()
                    break
                else:
                    f.seek(-9, 1)

            eeg_ID = re.findall(r"\d+", file_extension)
            file_tag = 1 if not eeg_ID else int(eeg_ID[0])
            max_ADC_count = 2 ** (8 * bytes_per_sample - 1) - 1
            max_byte_value = 2 ** (8 * bytes_per_sample)

            with open(set_file, "r", encoding="latin-1") as f_set:
                lines = f_set.readlines()
                channel_lines = dict(
                    [
                        tuple(map(int, re.findall(r"\d+.\d+|\d+", line)[0].split()))
                        for line in lines
                        if line.startswith("EEG_ch_")
                    ]
                )
                channel_id = channel_lines[file_tag]
                channel_id = channel_id

                gain_lines = dict(
                    [
                        tuple(map(int, re.findall(r"\d+.\d+|\d+", line)[0].split()))
                        for line in lines
                        if "gain_ch_" in line
                    ]
                )
                gain = gain_lines[channel_id - 1]

                for line in lines:
                    if line.startswith("ADC_fullscale_mv"):
                        fullscale_mv = int(re.findall(r"\d+.\d+|d+", line)[0])
                        break
                AD_bit_uvolt = (
                    2 * fullscale_mv / (gain * np.power(2, 8 * bytes_per_sample))
                )

            record_size = bytes_per_sample
            sample_le = 256 ** (np.arange(0, bytes_per_sample, 1))

            if not header_offset:
                print("Error: data_start marker not found!")
            else:
                f.seek(header_offset, 0)
                byte_buffer = np.fromfile(f, dtype="uint8")
                len_bytebuffer = len(byte_buffer)
                end_offset = len("\r\ndata_end\r")
                lfp_wave = np.zeros([num_samples,], dtype=np.float64,)
                for k in np.arange(0, bytes_per_sample, 1):
                    byte_offset = k
                    sample_value = (
                        sample_le[k]
                        * byte_buffer[
                            byte_offset : byte_offset
                            + len_bytebuffer
                            - end_offset
                            - record_size : record_size
                        ]
                    )
                    if sample_value.size < num_samples:
                        sample_value = np.append(
                            sample_value, np.zeros([num_samples - sample_value.size,]),
                        )
                    sample_value = sample_value.astype(
                        np.float64, casting="unsafe", copy=False
                    )
                    np.add(lfp_wave, sample_value, out=lfp_wave)
                np.putmask(
                    lfp_wave, lfp_wave > max_ADC_count, lfp_wave - max_byte_value
                )

                samples = lfp_wave * AD_bit_uvolt
                # timestamp = (
                #     np.arange(0, num_samples, 1) / sampling_rate)
                return samples

    else:
        print("No lfp file found for file {}".format(file_name))


def mne_lfp_Axona(file_name):
    """
    Create a mne object from a Axona recording.
    ------
    Load all channels from an Axona recording into a mne object


    Parameters:
    ------
    file_name (str): Axona .set file in the same folder as the EEG recordings referents to the set file

    Returns:
    ------
    MNE object with N channels named as ch_0 - ch_N

    """
    file_directory, file_basename = os.path.split(file_name)
    file_tag, file_extension = os.path.splitext(file_basename)
    set_file = os.path.join(file_directory, file_tag + ".set")

    # Open Set files configurations
    with open(file_name, "r", encoding="latin-1") as f_set:
        lines = f_set.readlines()
        for line in lines:
            if line.startswith("ADC_fullscale_mv"):
                fullscale_mv = int(re.findall(r"\d+.\d+|d+", line)[0])
        channel_map = dict(  # map internal channels from Axona set
            [
                tuple(map(int, re.findall(r"\d+.\d+|\d+", line)[0].split()))
                for line in lines
                if line.startswith("EEG_ch_")
            ]
        )
        recorded_channels = dict(  # map or recorded channels from Axona set
            [
                tuple(map(int, re.findall(r"\d+.\d+|\d+", line)[0].split()))
                for line in lines
                if line.startswith("saveEEG_ch_")
            ]
        )
        channel_ids = [
            ch for ch in recorded_channels.keys() if recorded_channels[ch]
        ]  # All recorded EEG channels
        gains = [
            int((re.findall(r"\d+.\d+|\d+", line)[0].split()[1]))
            for line in lines
            if "gain_ch_" in line
        ]  # List of gains

    data = []
    labels = []
    ch_types = []
    for ch in channel_ids:  # Loop for all channels
        if ch == 1:
            eeg_file = (
                file_directory + "/" + file_tag + ".eeg"
            )  # if it is the first eeg channel
        else:
            eeg_file = file_directory + "/" + file_tag + ".eeg" + str(ch)

        if os.path.isfile(eeg_file):  # open eeg file
            with open(eeg_file, "rb") as f:
                while True:
                    line = f.readline()
                    try:
                        line = line.decode("latin-1")
                    except:
                        try:
                            line = line.decode("UTF-8")
                        except BaseException:
                            break
                    if line == "":
                        break
                    if line.startswith("trial_date"):
                        # Blank eeg file
                        if line.strip() == "trial_date":
                            total_samples = 0
                            break
                        date = " ".join(line.replace(",", " ").split()[1:])
                    if line.startswith("trial_time"):
                        time = line.split()[1]
                    if line.startswith("experimenter"):
                        experimenter = " ".join(line.split()[1:])
                    if line.startswith("comments"):
                        comments = " ".join(line.split()[1:])
                    if line.startswith("duration"):
                        duration = float("".join(line.split()[1:]))
                    if line.startswith("sw_version"):
                        file_version = line.split()[1]
                    if line.startswith("num_chans"):
                        total_channel = int("".join(line.split()[1:]))
                    if line.startswith("sample_rate"):
                        sampling_rate = float("".join(re.findall(r"\d+.\d+|\d+", line)))
                    if line.startswith("bytes_per_sample"):
                        bytes_per_sample = int("".join(line.split()[1:]))
                    if line.startswith("num_EEG_samples"):
                        total_samples = int("".join(line.split()[1:]))
                    if line.startswith("data_start"):
                        break

                f.seek(0, 0)
                header_offset = []
                while True:
                    try:
                        buff = f.read(10).decode("UTF-8")
                    except BaseException:
                        break
                    if buff == "data_start":
                        header_offset = f.tell()
                        break
                    else:
                        f.seek(-9, 1)
                if not header_offset:
                    print("Error: data_start marker not found!")
                else:
                    AD_bit_uvolt = (
                        2
                        * fullscale_mv
                        / (
                            gains[channel_map[ch] - 1]
                            * np.power(2, 8 * bytes_per_sample)
                        )
                    )
                    num_samples = total_samples
                    max_ADC_count = 2 ** (8 * bytes_per_sample - 1) - 1
                    max_byte_value = 2 ** (8 * bytes_per_sample)
                    record_size = bytes_per_sample
                    sample_le = 256 ** (np.arange(0, bytes_per_sample, 1))
                    f.seek(header_offset, 0)
                    byte_buffer = np.fromfile(f, dtype="uint8")
                    len_bytebuffer = len(byte_buffer)
                    end_offset = len("\r\ndata_end\r")
                    lfp_wave = np.zeros([num_samples,], dtype=np.float64,)
                    for k in np.arange(0, bytes_per_sample, 1):
                        byte_offset = k
                        sample_value = (
                            sample_le[k]
                            * byte_buffer[
                                byte_offset : byte_offset
                                + len_bytebuffer
                                - end_offset
                                - record_size : record_size
                            ]
                        )
                        if sample_value.size < num_samples:
                            sample_value = np.append(
                                sample_value,
                                np.zeros([num_samples - sample_value.size,]),
                            )
                        sample_value = sample_value.astype(
                            np.float64, casting="unsafe", copy=False
                        )
                        np.add(lfp_wave, sample_value, out=lfp_wave)
                    np.putmask(
                        lfp_wave, lfp_wave > max_ADC_count, lfp_wave - max_byte_value
                    )
                    samples = lfp_wave * AD_bit_uvolt

                    timestamp = np.arange(0, num_samples, 1) / sampling_rate
                    lfp_data = lfp_wave * AD_bit_uvolt
                    if max(lfp_data) > 0:
                        data.append((lfp_wave * AD_bit_uvolt) / 1000)
                        labels.append(f"ch_{ch}")
                        ch_types.append("eeg")

    info = mne.create_info(ch_names=labels, sfreq=sampling_rate, ch_types=ch_types)
    return mne.io.RawArray(np.array(data), info)
