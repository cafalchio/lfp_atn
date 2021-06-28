# For use with Python doit.
import os

from simuran.main.doit import create_task
from skm_pyutils.py_config import read_cfg
from skm_pyutils.py_path import get_all_files_in_dir
from doit.tools import title_with_actions

here = os.path.dirname(os.path.abspath(__file__))
cfg = read_cfg(os.path.join(here, "dodo.cfg"), verbose=False)
num_workers = cfg.get("DEFAULT", "num_workers")
dirname = cfg.get("DEFAULT", "dirname")


def task_list_openfield():
    return create_task(
        os.path.join(here, "lfp_atn_simuran", "multi_runs", "run_openfield.py"),
        ["do_nothing.py"],
        num_workers=num_workers,
        dirname=dirname,
    )


def task_coherence():
    return create_task(
        os.path.join(here, "lfp_atn_simuran", "multi_runs", "run_coherence.py"),
        ["plot_coherence.py"],
        num_workers=num_workers,
        dirname=dirname,
    )


def task_lfp_plot():
    return create_task(
        os.path.join(here, "lfp_atn_simuran", "multi_runs", "run_lfp_plot.py"),
        ["plot_lfp_eg.py"],
        num_workers=num_workers,
        dirname=dirname,
    )


def task_theta_power():
    return create_task(
        os.path.join(here, "lfp_atn_simuran", "multi_runs", "run_spectra.py"),
        ["simuran_theta_power.py"],
        num_workers=num_workers,
        dirname=dirname,
    )


def task_lfp_rate():
    return create_task(
        os.path.join(here, "lfp_atn_simuran", "multi_runs", "run_lfp_rate.py"),
        ["simuran_lfp_rate.py"],
        num_workers=num_workers,
        dirname=dirname,
    )


def task_lfp_speed():
    return create_task(
        os.path.join(here, "lfp_atn_simuran", "multi_runs", "run_speed_theta.py"),
        ["speed_lfp.py"],
        num_workers=num_workers,
        dirname=dirname,
    )


def task_speed_ibi():
    base_ = os.path.join(here, "lfp_atn_simuran", "cell_lists")
    dependencies = [
        os.path.join(base_, "CTRL_Lesion_cells_filled.xlsx"),
        os.path.join(base_, "list_spike_ibi.py"),
    ]
    dir_ = os.path.join(here, "lfp_atn_simuran", "sim_results", "list_spike_ibi")
    if os.path.exists(dir_):
        targets = get_all_files_in_dir(dir_, return_absolute=True)
    else:
        targets = []

    location = os.path.abspath(os.path.join(base_, "list_spike_ibi.py"))
    action = f"python {location}"

    def clean():
        for fname_ in targets:
            if os.path.isfile(fname_):
                print("Removing file {}".format(fname_))
                os.remove(fname_)    

    return {
        "file_dep": dependencies,
        "targets": targets,
        "actions": [action],
        "clean": [clean],
        "title": title_with_actions,
        "verbosity": 0,
        "doc": action,
    }

def task_spike_lfp():
    base_ = os.path.join(here, "lfp_atn_simuran", "cell_lists")
    dependencies = [
        os.path.join(base_, "CTRL_Lesion_cells_filled.xlsx"),
        os.path.join(base_, "list_spike_lfp.py"),
    ]
    dir_ = os.path.join(here, "lfp_atn_simuran", "sim_results", "list_spike_lfp")
    if os.path.exists(dir_):
        targets = get_all_files_in_dir(dir_, return_absolute=True)
    else:
        targets = []

    location = os.path.abspath(os.path.join(base_, "list_spike_lfp.py"))
    action = f"python {location}"

    def clean():
        for fname_ in targets:
            if os.path.isfile(fname_):
                print("Removing file {}".format(fname_))
                os.remove(fname_)    

    return {
        "file_dep": dependencies,
        "targets": targets,
        "actions": [action],
        "clean": [clean],
        "title": title_with_actions,
        "verbosity": 0,
        "doc": action,
    }