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

    def _function_PDF(self, k=False):
        '''Defines cycle PDF(Probability Density Function) function or k-th 
        statistical moment function, if k is specified.
        '''
        m0 = self.spectral_data.moments[0]

        if k==False: 
            def pdf(s):
                px = (s / m0) * np.exp( - s**2 / (2 * m0)) 
                return px
            return pdf
        else:
            if isinstance(k, (int,float)): 
                def pdf_moment(s):
                    px = (s / m0) * np.exp( - s**2 / (2 * m0)) 
                    return s**k * px
                return pdf_moment
            else:
                raise Exception('Unrecognized Input Error')

    def get_PDF(self,s):
        '''Returns cycle PDF(Probability Density Function) as a function of stress s.

        :param s:  numpy.ndarray
            Stress vector.
        :return pdf: numpy.ndarray
        '''
        return self._function_PDF()(s)

    def damage_intesity_NB(self, m0, nu, C, k):
        """Calculates narrowband damage intensity with parameters m0, nu, C, k, as defined in [2].

        :param m0: [int,float]
            Zeroth spectral moment [MPa**2].
        :param nu: [int,float]
            Frequency of positive slope zero crossing [Hz].
        :param C: [int,float]
            Fatigue strength coefficient [MPa**k].
        :param k : [int,float]
            Fatigue strength exponent [/].
        :return T: float
            Estimated fatigue life in seconds.       
        """
        d = nu * np.sqrt(2 * m0)**k * gamma(1.0 + k/2.0) / C  

        return  d

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
        
        d = self.damage_intesity_NB(m0=m0, nu=nu, C=C, k=k) 
        T = 1.0/d

        return T