import numpy as np
from scipy.special import gamma

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

    def get_PDF(self, s):
        '''Returns cycle PDF(Probability Density Function) as a function of stress s.

        :param s:  numpy.ndarray
            Stress vector.
        :return pdf: numpy.ndarray
        '''
        m0 = self.spectral_data.moments[0]

        pdf = (s / m0) * np.exp( - s**2 / (2 * m0)) 

        return pdf

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
        
        D = nu * np.sqrt(2 * m0)**k * gamma(1.0 + k/2.0) / C  
        T = 1.0/D

        return T