"""
simuran_multi_params describes running main.py for multiple
different recordings, functions, etc.
In theory, this could describe a full experiment.
"""


def create_new_entry(batch_param_loc, fn_param_loc, add=""):
    def make_default_dict(add=""):
        param_names = {}

        # file_list_name is often specific to each analysis
        # as it lists all recordings found under the given regex
        param_names["file_list_name"] = "file_list{}.txt".format(add)

        param_names["cell_list_name"] = "cell_list{}.txt".format(add)

        # These will often stay the same since the parameters that describe the
        # layout of a recording don't often change
        param_names["base_param_name"] = "simuran_base_params.py"
        param_names["batch_find_name"] = "simuran_params.py"

        return param_names

    output_dict = make_default_dict(add=add)

    output_dict["batch_param_loc"] = batch_param_loc
    output_dict["fn_param_loc"] = fn_param_loc

    return output_dict


def set_file_locations():
    import os

    output = []

    for val in [1, 2, 3, 4, 5, 6]:
        output.append(
            (
                os.path.join(
                    "__dirname__", "..", "batch_params", "CSR{}-openfield.py"
                ).format(val),
                os.path.join("__dirname__", "..", "functions", "avg_coherence.py"),
                "CSR{}".format(val),
            )
        )

    for val in [1, 3, 4, 5, 6]:
        output.append(
            (
                os.path.join(
                    "__dirname__", "..", "batch_params", "LSR{}-openfield.py"
                ).format(val),
                os.path.join("__dirname__", "..", "functions", "avg_coherence.py"),
                "LSR{}".format(val),
            )
        )

    return output


def set_fixed_params(in_dict):
    in_dict["default_param_folder"] = None

    # Can set a function to run after all analysis here
    # For example, it could plot a summary of all the data
    from lfp_atn_simuran.analysis.do_wt_figure import do_fig

    in_dict["after_batch_fn"] = do_fig

    # If the after batch function needs the full dataset
    # Pass this as true
    # For example, this could be used to concatenate
    # EEG signals that were recorded in two second long trials
    in_dict["keep_all_data"] = False

    # What folders to merge
    in_dict["to_merge"] = ["avg_coherence"]
    return in_dict


# Setup the actual parameters
params = {"run_list": []}
params = set_fixed_params(params)

for val in set_file_locations():
    param_dict = create_new_entry(val[0], val[1], val[2])
    params["run_list"].append(param_dict)
