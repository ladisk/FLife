import numpy as np
import scipy.special as ss

class Narrowband(object):
    """Class for fatigue life estimation using frequency domain 
    method by Miles [1].
   
    References
    ----------
    [1] John W. Miles. On structural fatigue under random loading. Journal
        of the Aeronautical Sciences, 21(11):753{762, 1954.
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
            Fatigue strength coefficient [Mpa**k].
        :param k : [int,float]
            Fatigue strength exponent [/].
        :return T: float
            Estimated fatigue life in seconds.       
        """
        m0 = self.spectral_data.moments[0]
        nu = self.spectral_data.nu
        gamma = ss.gamma
        
#       D = mp * al2 * np.sqrt(2 * m0)**b * gamma(1.0 + b/2.0) # OK
        D = nu * np.sqrt(2 * m0)**k * gamma(1.0 + k/2.0) / C  #OK
        T = 1.0/D

        return T