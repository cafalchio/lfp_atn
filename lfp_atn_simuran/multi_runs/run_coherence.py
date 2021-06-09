"""
simuran_multi_params describes running main.py for multiple
different recordings, functions, etc.
In theory, this could describe a full experiment.
"""
import os


def create_new_entry(batch_param_loc, fn_param_loc, add=""):
    def make_default_dict(add=""):
        param_names = {}

        # file_list_name is often specific to each analysis
        # as it lists all recordings found under the given regex
        param_names["file_list_name"] = "file_list{}.txt".format(add)

        param_names["cell_list_name"] = "cell_list{}.txt".format(add)

        return param_names

    output_dict = make_default_dict(add=add)

    output_dict["batch_param_loc"] = batch_param_loc
    output_dict["fn_param_loc"] = fn_param_loc

    return output_dict


def set_file_locations():

    output = []

    for val in [1, 2, 3, 4, 5, 6]:
        output.append(
            (
                os.path.join(
                    "__thisdirname__", "..", "batch_params", "CSR{}-openfield.py"
                ).format(val),
                os.path.join("__thisdirname__", "..", "functions", "fn_coherence.py"),
                "CSR{}".format(val),
            )
        )

    for val in [1, 3, 4, 5, 6]:
        output.append(
            (
                os.path.join(
                    "__thisdirname__", "..", "batch_params", "LSR{}-openfield.py"
                ).format(val),
                os.path.join("__thisdirname__", "..", "functions", "fn_coherence.py"),
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

    return in_dict


# Setup the actual parameters
params = {"run_list": [], "to_merge": []}
params = set_fixed_params(params)

for val in set_file_locations():
    param_dict = create_new_entry(val[0], val[1], val[2])
    fn_name = os.path.splitext(os.path.basename(val[1]))[0]
    if fn_name not in params["to_merge"]:
        params["to_merge"].append(fn_name)
    params["run_list"].append(param_dict)
