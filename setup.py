"""How to install simuran."""

import os
from setuptools import setup, find_packages


def read(fname):
    """Read files from the main source dir."""
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


DESCRIPTION = "Lfp code in ATN for SIMURAN."

DISTNAME = "lfp_atn_simuran"
MAINTAINER = "Sean Martin"
MAINTAINER_EMAIL = "martins7@tcd.ie"
VERSION = "0.0.1"

INSTALL_REQUIRES = [
    "matplotlib >= 3.0.2",
    "numpy >= 1.15.0",
    "skm_pyutils",
    "seaborn",
    "more_itertools",
    "indexed",
    "tqdm",
    "doit",
]

CLASSIFIERS = [
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS",
    "Operating System :: Windows",
]


if __name__ == "__main__":

    setup(
        name=DISTNAME,
        author=MAINTAINER,
        author_email=MAINTAINER_EMAIL,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        license=read("LICENSE"),
        version=VERSION,
        install_requires=INSTALL_REQUIRES,
        include_package_data=True,
        packages=["lfp_atn_simuran", "lfp_atn_simuran.analysis"],
        classifiers=CLASSIFIERS,
    )