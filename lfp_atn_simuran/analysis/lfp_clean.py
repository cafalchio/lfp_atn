"""Clean LFP signals."""


class LFPClean(object):
    """
    Class to clean LFP signals.

    Attributes
    ----------
    method : string
        The method to use for cleaning.
        Currently supports "avg".
    visualise : bool
        Whether to visualise the cleaning.

    Parameters
    ----------
    method : string
        The method to use for cleaning.
        Currently supports "avg".
    visualise : bool
        Whether to visualise the cleaning.

    """

    def __init__(self, method="avg", visualise=False):
        self.method = method
        self.visualise = visualise

    @staticmethod
    def clean_lfp_signals(recording):
        """
        Clean the lfp signals in a recording.

        Parameters
        ----------
        recording : simuran.recording.Recording

        Returns
        -------
        None
        
        """
        LFPClean._clean_avg_signals(recording)

    @staticmethod
    def _clean_avg_signals(recording):
        lfp_signals = recording.get_signals()
        regions = lfp_signals.get_property("region")

        signals_grouped_by_region = lfp_signals.split_into_groups("region")

        print(signals_grouped_by_region)
        exit(-1)
