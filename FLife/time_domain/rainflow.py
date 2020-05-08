import numpy as np
import fatpack

class Rainflow(object):
    """Class for fatigue life estimation using rainflow counting method [1].

    References
    ----------
    [1] C. Amzallag et. al. Standardization of the rainflow counting method for
        fatigue analysis. International Journal of Fatigue, 16 (1994) 287-293
    """
    def __init__(self, spectral_data):
        '''Get needed values from reference object.

        :param spectral_data:  Instance of object SpectralData
        '''          
        self.spectral_data = spectral_data


    def get_life(self, C, k, nr_load_classes = 512, Su = False, range = False):
        """Calculate fatigue life with parameters C, k, as defined in [2].
 
        :param C: [int,float]
            Fatigue strength coefficient [MPa**k].
        :param k : [int,float]
            Fatigue strength exponent [/].
        :param nr_load_classes: int
            The number of intervals to divide the min-max range of the dataseries
            into. Defaults to 512.
        :param Su: [int,float]
            Ultimate tensile strength [MPa]. If specified, Goodman equivalent stress
            is used for fatigue life estimation. Defaults to False.
        :param range: bool
            If True, ranges instead of amplitudes are used for fatigue life estimation.
            Defaults to False.
        :return T: float
            Estimated fatigue life in seconds.
        """    
        t = self.spectral_data.t
        ranges,means = fatpack.find_rainflow_ranges(self.spectral_data.data, k=nr_load_classes, return_means=True)
        
        if range == False: ranges *= 0.5

        if Su: ranges = ranges/(1. - means/Su)

        d = np.sum(ranges**k / C)
        T = t / d
        
        return T
