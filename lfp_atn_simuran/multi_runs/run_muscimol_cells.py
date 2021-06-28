import os


def make_default_dict(add=""):
    param_names = {
        "file_list_name": "file_list{}.txt".format(add),
        "cell_list_name": "cell_list{}.txt".format(add),
        "fn_param_name": "simuran_fn_params{}.py".format(add),
        "base_param_name": "simuran_base_params.py",
        "batch_param_name": "simuran_batch_params.py",
        "batch_find_name": "simuran_params.py",
    }
    return param_names


main_dir = r"D:\SubRet_recordings_imaging\muscimol_data"
can8_05 = os.path.join(main_dir, "CanCSR8_muscimol", "05102018")
can8_12 = os.path.join(main_dir, "CanCSR8_muscimol", "12112018")
can7_03 = os.path.join(main_dir, "CanCSR7_muscimol", "2_03082018")
can1_09 = os.path.join(main_dir, "CanCSCa1_muscimol", "09082018")
directory_list = [
    can8_05,
    can8_05,
#    can8_05,
    can8_05,
    can8_12,
    can8_12,
    can8_12,
    can7_03,
    can7_03,
    can7_03,
    can7_03,
    can1_09
]

param_list = [
    make_default_dict("_3_3"),
    make_default_dict("_11_6"),
#    make_default_dict("_9_5"),
    make_default_dict("_3_2"),
    make_default_dict("_10_2"),
    make_default_dict("_10_1"),
    make_default_dict("_10_3"),
    make_default_dict("_9_1"),
    make_default_dict("_9_2"),
    make_default_dict("_4_1"),
    make_default_dict("_3_1"),
    make_default_dict("_10_1"),
]

if len(param_list) != len(directory_list):
    raise ValueError(
        "Parameter list and directory list must be the same length")

default_param_folder = os.path.join(main_dir, "simuran_default")
check_params = False

params = {
    "directory_list": directory_list,
    "param_list": param_list,
    "default_param_folder": default_param_folder,
    "check_params": check_params,
}

if __name__ == "__main__":
    print(params)
