import os
import math

import numpy as np
import matplotlib.pyplot as plt

from simuran.plot.figure import SimuranFigure
from neurochat.nc_utils import butter_filter
from lfp_atn_simuran.analysis.lfp_clean import LFPClean


def mne_plot(recording, base_dir, figures, **kwargs):
    method = kwargs.get("clean_method", "avg")
    method_kwargs = kwargs.get("clean_kwargs", {})
    min_f = kwargs.get("fmin", 1)
    max_f = kwargs.get("fmax", 100)
    img_format = kwargs.get("image_format", "png")
    lc = LFPClean(method=method, visualise=True, show_vis=False)
    result = lc.clean(recording, min_f=min_f, max_f=max_f, method_kwargs=method_kwargs)
    fig = result["fig"]

    location = os.path.splitext(recording.source_file)[0]
    out_name = "--".join(
        os.path.dirname(location)[len(base_dir + os.sep) :].split(os.sep)
    )
    figures.append(SimuranFigure(fig, out_name, dpi=100, format=img_format, done=True))

    return result["bad_channels"]
