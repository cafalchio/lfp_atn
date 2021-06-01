# lfp_atn
 LFP analysis

## Plot an animated example
`python -m lib.plots`

## Using doit to run sets of code
`doit list`

## Data considered

### Openfield recordings
Only small arena recordings were considered to reduce variables.
Recordings in the big square arena were not considered.

Control:
1. CSR1 - all small square recordings (not habituation) (6 days)
2. CSR2 - all small square recordings (not habituation) (6 days)
3. CSR3 - all small square recordings (not habituation) (6 days)
4. CSR4 - all small square recordings (9 days, 10 records)
5. CSR5 - all small square recordings and habituation (8 days, 9 records)
6. CSR6 - all small square recordings and habituation (9 days)

Lesion:
1. LSR1 - all small square recordings (not habituation) (6 days)
2. LSR3 - all small square recordings (not habituation) (6 days)
3. LSR4 - all small square recordings (9 days, 10 records)
4. LSR5 - all small square recordings (9 days, 10 records)
5. LSR6 - all small square recordings (8 days)
6. LSR7 - all small square recordings (9 days)

## TODO
1. Decide if using screening recordings in CL.
3. Provide the command to merge all results and plots into one folder.

## Some things to note
1. TODO list the parameters of the usual FIR filter used.
2. Delta is considered to be in the range 1.5 - 4 Hz.
3. Theta is considered to be in the range 6 - 10 Hz.