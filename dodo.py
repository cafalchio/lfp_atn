# For use with Python doit.
import os

from simuran.main.doit import create_task
from skm_pyutils.py_config import read_cfg, print_cfg

here = os.path.dirname(os.path.abspath(__file__))
cfg = read_cfg(os.path.join(here, "dodo.cfg"), verbose=False)
num_workers = cfg.get("DEFAULT", "num_workers")
dirname = cfg.get("DEFAULT", "dirname")

def task_list_openfield():
    return create_task(
        os.path.join(here, "lfp_atn_simuran", "multi_runs", "CL_openfield.py"),
        ["do_nothing.py"],
        num_workers=num_workers,
        dirname=dirname,
    )

def task_coherence():
    return create_task(
        os.path.join(here, "lfp_atn_simuran", "multi_runs", "coherence_atnx.py"),
        ["plot_coherence.py"],
        num_workers=num_workers,
        dirname=dirname,
    )


def task_lfp_plot():
    return create_task(
        os.path.join(here, "lfp_atn_simuran", "multi_runs", "lfp_plot.py"),
        ["plot_lfp_eg.py"],
        num_workers=num_workers,
        dirname=dirname,
    )


# def task_lfp_difference():
#     return create_task(
#         os.path.join(here, "lfp_atn_simuran", "multi_runs", "lfp_difference.py"),
#         ["lfp_difference.py"],
#         num_workers=num_workers,
#         dirname=dirname,
#     )


def task_theta_power():
    return create_task(
        os.path.join(here, "lfp_atn_simuran", "multi_runs", "spectral_atnx.py"),
        ["simuran_theta_power.py"],
        num_workers=num_workers,
        dirname=dirname,
    )


def task_lfp_rate():
    return create_task(
        os.path.join(here, "lfp_atn_simuran", "multi_runs", "lfp_rate.py"),
        ["simuran_lfp_rate.py"],
        num_workers=num_workers,
        dirname=dirname,
    )