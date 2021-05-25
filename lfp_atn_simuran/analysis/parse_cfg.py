import os
from json import loads
from skm_pyutils.py_config import read_cfg

def parse_cfg_info():
    here = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(here, "config.ini")
    cfg = read_cfg(cfg_path, False)
    clean_kwargs = {
        "pick_property": cfg.get("Picking", "Property"),
        "channels": loads(cfg.get("Picking", "Values")),
    }
    kwargs = {
        "image_format": cfg.get("Default", "ImageFormat"),
        "clean_method": cfg.get("Default", "SignalType"),
        "fmin": float(cfg.get("Default", "MinFrequency")),
        "fmax": float(cfg.get("Default", "MaxFrequency")),
        "clean_kwargs": clean_kwargs,
    }
    return kwargs