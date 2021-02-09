import numpy as np
import fatpack
import rainflow

class Rainflow(object):
    """Class for fatigue life estimation using rainflow counting method [1].

    References
    ----------
    [1] C. Amzallag et. al. Standardization of the rainflow counting method for
        fatigue analysis. International Journal of Fatigue, 16 (1994) 287-293
    [2] Janko Slavič, Matjaž Mršnik, Martin Česnik, Jaka Javh, Miha Boltežar. 
        Vibration Fatigue by Spectral Methods, From Structural Dynamics to Fatigue Damage
        – Theory and Experiments, ISBN: 9780128221907, Elsevier, 1st September 2020
    """
    def __init__(self, spectral_data):
        '''Get needed values from reference object.

        :param spectral_data:  Instance of object SpectralData
        '''          
        self.spectral_data = spectral_data


    def get_life(self, C, k, algorithm = 'four-point',  Su = False, range = False, **kwargs):
        r"""Calculate fatigue life with parameters C, k, as defined in [2].
 
        :param C: [int,float]
            Fatigue strength coefficient [MPa**k].
        :param k : [int,float]
            Fatigue strength exponent [/].
        :param algorithm: str
            Cycle counting method. Options are `three-point` and `four-point`.
            Defaults to `four-point`.
        :param Su: [int,float]
            Ultimate tensile strength [MPa]. If specified, Goodman equivalent stress
            is used for fatigue life estimation. Defaults to False.
        :param range: bool
            If True, ranges instead of amplitudes are used for fatigue life estimation.
            Defaults to False.
        :param **nr_load_classes: int
            The number of intervals to divide the min-max range of the dataseries
            into. Used with algorithm =`four-point`. Defaults to 512.
        :returns T: float
            Estimated fatigue life in seconds.
        """    
        t = self.spectral_data.t
        cycles = self._get_cycles(algorithm = algorithm, Su=Su, **kwargs)
        ranges = cycles[0]
        means = cycles[1]
        
        if range == True: 
            pass
        elif range == False: 
            ranges *= 0.5
        else: 
            raise Exception('Unrecognized Input Error')

        if Su: ranges = ranges/(1. - means/Su)

        try :
            counts = cycles[2]  #three-point algorithm returns half cycles
            d = np.sum(counts * ranges**k / C)
        except Exception:
            d = np.sum(ranges**k / C)

        T = t / d
        
        return T


    def _get_cycles(self, algorithm = 'four-point', **kwargs):
        """
        :param algorithm: str
            Cycle counting method. Options are `three-point` and `four-point`.
            Defaults to `four-point`.
        :param **nr_load_classes: int
            The number of intervals to divide the min-max range of the dataseries
            into. Used with algorithm =`four-point`. Defaults to 512.
        :returns (ranges, means, counts): tuple of numpy.ndarray
            For algorithm = 'four-point', tuple with only two elemensts (ranges, means) is returned.
        """
        if algorithm == 'four-point':
            nr_load_classes = kwargs.get('nr_load_classes', 512)
            ranges, means = fatpack.find_rainflow_ranges(self.spectral_data.data, k=nr_load_classes, return_means=True)
            return ranges, means

        elif algorithm == 'three-point':
            cycles = np.array(list(rainflow.extract_cycles(self.spectral_data.data)))
            ranges = cycles[:,0]
            means = cycles[:,1]
            counts = cycles[:,2]
            return ranges, means, counts

        else: 
            raise ValueError('Set `algorithm` either to `three-point` or `four-point`.')