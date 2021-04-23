# LFP filtering and processing

## Great discussion on filtering
https://mne.tools/stable/auto_tutorials/discussions/plot_background_filtering.html#sphx-glr-auto-tutorials-discussions-plot-background-filtering-py

## Filtering concerns - especially related to time
https://www.frontiersin.org/articles/10.3389/fpsyg.2011.00365/full and its counter https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3391960/

## Rest methods
https://www.cell.com/current-biology/fulltext/S0960-9822(19)30006-5?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS0960982219300065%3Fshowall%3Dtrue

## What to report
On page 45 in Widmann et al. (2015) 7, there is a convenient list of important filter parameters that should be reported with each publication:

Filter type (high-pass, low-pass, band-pass, band-stop, FIR, IIR)

Cutoff frequency (including definition)

Filter order (or length)

Roll-off or transition bandwidth

Passband ripple and stopband attenuation

Filter delay (zero-phase, linear-phase, non-linear phase) and causality

Direction of computation (one-pass forward/reverse, or two-pass forward and reverse)

In the following, we will address how to deal with these parameters in MNE:

## MNE summary of filter choice
When filtering, there are always trade-offs that should be considered. One important trade-off is between time-domain characteristics (like ringing) and frequency-domain attenuation characteristics (like effective transition bandwidth). Filters with sharp frequency cutoffs can produce outputs that ring for a long time when they operate on signals with frequency content in the transition band. In general, therefore, the wider a transition band that can be tolerated, the better behaved the filter will be in the time domain.