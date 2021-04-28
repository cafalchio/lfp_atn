import os
import simuran
from pprint import pprint

# Import from my files
from lfp_atn_simuran.analysis.frequency_analysis import grouped_powers, powers
from lfp_atn_simuran.analysis.lfp_clean import LFPClean

# Establish the recording layout
def recording_info():
    def setup_signals():
        """Set up the signals (such as eeg or lfp)."""
        # The total number of signals in the recording
        num_signals = 32

        # What brain region each signal was recorded from
        regions = ["RSC"] * 2 + ["SUB"] * 30

        # If the wires were bundled, or any other kind of grouping existed
        # If no grouping, groups = [i for i in range(num_signals)]
        groups = [0, 0, 1, 1]
        for i in range(2, 9):
            groups.append(i)
            groups.append(i)
            groups.append(i)
            groups.append(i)

        # The sampling rate in Hz of each signal
        sampling_rate = [250] * num_signals
        channel_type = ["eeg"] * num_signals

        # This just passes the information on
        output_dict = {
            "num_signals": num_signals,
            "region": regions,
            "group": groups,
            "sampling_rate": sampling_rate,
            "channel_type": channel_type,
        }

        return output_dict

    def setup_units():
        """Set up the single unit data."""
        # The number of tetrodes, probes, etc - any kind of grouping
        num_groups = 8

        # The region that each group belongs to
        regions = ["SUB"] * num_groups

        # A group number for each group, for example the tetrode number
        groups = [1, 2, 3, 4, 9, 10, 11, 12]

        output_dict = {
            "num_groups": num_groups,
            "region": regions,
            "group": groups,
        }

        return output_dict

    def setup_spatial():
        """Set up the spatial data."""
        arena_size = "default"

        output_dict = {
            "arena_size": arena_size,
        }
        return output_dict

    def setup_loader():
        """
        Set up the loader and keyword arguments for the loader.

        See also
        --------
        simuran.loaders.loader_list.py

        """
        # The type of loader to use, see simuran.loaders.loader_list.py for options
        # For now nc_loader is the most common option
        # loader = "params_only"
        loader = "nc_loader"

        # Keyword arguments to pass to the loader.
        loader_kwargs = {
            "system": "Axona",
            "pos_extension": ".txt",
        }

        output_dict = {
            "loader": loader,
            "loader_kwargs": loader_kwargs,
        }

        return output_dict

    load_params = setup_loader()

    mapping = {
        "signals": setup_signals(),
        "units": setup_units(),
        "spatial": setup_spatial(),
        "loader": load_params["loader"],
        "loader_kwargs": load_params["loader_kwargs"],
    }
    return mapping

def main(set_file_location, output_location):
    recording = simuran.Recording(
        params=recording_info(), base_file=set_file_location)
    sigs = LFPClean.clean_lfp_signals(
            recording, verbose=False, vis=False, append_avg=True
        )
    fig = sigs.plot(title=os.path.basename(set_file_location), duration=20, show=False)
    fig.savefig(output_location[:-4] + ".png", dpi=300)
    analysis_handler = simuran.AnalysisHandler()
    analysis_handler.add_fn(grouped_powers, recording, min_f=1, max_f=100, win_len=2)
    analysis_handler.add_fn(powers, recording, min_f=1, max_f=100, win_len=1)
    analysis_handler.run_all_fns()
    analysis_handler.save_results(output_location=output_location)


if __name__ == "__main__":
    # Establish data paths
    main_set_file_location = os.path.join(
        "D:\\", "SubRet_recordings_imaging", 
        os.path.normpath("LSubRet5/recording/Small sq up_small sq down/01122017/S1_small sq up/01122017_smallsqdownup_up_1_1.set"))
    main_output_location = "compare_results.csv"
    main(main_set_file_location, main_output_location)