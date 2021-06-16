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
        "theta_min": float(cfg.get("Default", "ThetaMin")),
        "theta_max": float(cfg.get("Default", "ThetaMax")),
        "delta_min": float(cfg.get("Default", "DeltaMin")),
        "delta_max": float(cfg.get("Default", "DeltaMax")),
        "psd_scale": cfg.get("Default", "PsdScale"),
        "clean_kwargs": clean_kwargs,
        "cfg_base_dir" : cfg.get("Path", "BaseDir")
    }
    return kwargs

if __name__ == "__main__":
    from pprint import pprint
    pprint(parse_cfg_info())