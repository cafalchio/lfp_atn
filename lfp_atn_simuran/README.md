# Analysing LFP in SUB and RSC
This is intended to be used with SIMURAN.

For example, to analyse all Control and lesion recordings for LFP power spectra, run

```
simuran -r multi_runs/spectral_atnx.py -m
```

The results will be in `sim_results\simuran_theta_power`.

## Reference code from CLA experiment
See https://github.com/seankmartin/Claustrum_Experiment

## Install the code
pip install -e .

## TODO
1. Refactor the analysis function folders.
2. Pull out the per cell analysis from SIMURAN to analyse other things
3. T-maze analysis