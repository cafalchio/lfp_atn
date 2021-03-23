# For use with Python doit.
import os


from simuran.main.doit import create_task


def task_coherence():
    return create_task(
        os.path.join("simuran", "multi_runs", "coherence_atnx.py"),
        ["plot_coherence.py"],
        num_workers=4,
    )


def task_lfp_plot():
    return create_task(
        os.path.join("simuran", "multi_runs", "lfp_plot.py"), ["plot_lfp_eg.py"], num_workers=4,
    )


def task_lfp_difference():
    return create_task(
        os.path.join("simuran", "multi_runs", "lfp_difference.py"),
        ["lfp_difference.py"],
        num_workers=4,
    )


def task_theta_power():
    return create_task(
        os.path.join("simuran", "multi_runs", "spectral_atnx.py"),
        ["simuran_theta_power.py"],
        num_workers=4,
    )
