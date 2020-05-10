import numpy as np
from scipy.special import gamma


class WirschingLight(object):
    """Class for fatigue life estimation using frequency domain 
    method by Wirsching and Light [1].
   
    References
    ----------
    [1] Paul H. Wirsching and Mark C. Light. Fatigue under wide band random
        stresses. Journal of the Structural Division, 106(7):1593-1607,
        1980
    [2] Janko Slavič, Matjaž Mršnik, Martin Česnik, Jaka Javh, Miha Boltežar. 
        Vibration Fatigue by Spectral Methods, From Structural Dynamics to Fatigue Damage
        – Theory and Experiments, ISBN: 9780128221907, Elsevier, 1st September 2020
    """
    def __init__(self, spectral_data):
        '''Get needed values from reference object.

        :param spectral_data:  Instance of object SpectralData
        '''
        self.spectral_data = spectral_data
                           
    def get_life(self, C, k):
        """Calculate fatigue life with parameters C, k, as defined in [2].

        :param C: [int,float]
            Fatigue strength coefficient [MPa**k].
        :param k : [int,float]
            Fatigue strength exponent [/].
        :return T: float
            Estimated fatigue life in seconds.
        """
        m0 = self.spectral_data.moments[0]
        nu = self.spectral_data.nu
        al2 = self.spectral_data.al2

        dNB = nu * np.sqrt(2 * m0)**k * gamma(1.0 + k/2.0) / C
        ak = 0.926 - 0.033 * k
        bk = 1.587 * k - 2.323
        eps = np.sqrt(1 - al2**2)

        ro = ak + ( 1 - ak ) * ( 1 - eps )**bk
        T = 1 / (ro * dNB)
        
        return T