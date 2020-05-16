import numpy as np
import scipy.integrate as si
import scipy.stats as ss
from scipy.integrate import quad

class Rice(object):
    """Class for fatigue life estimation using frequency domain 
    method by Rice[1].

    References
    ----------
    [1] Stephen O. Rice. Mathematical analysis of random noise. The Bell
        System Technical Journal, 24(1):46-156, 1945.
    [2] Janko Slavič, Matjaž Mršnik, Martin Česnik, Jaka Javh, Miha Boltežar. 
        Vibration Fatigue by Spectral Methods, From Structural Dynamics to Fatigue Damage
        – Theory and Experiments, ISBN: 9780128221907, Elsevier, 1st September 2020
    """

    def __init__(self, spectral_data):
        '''Get needed values from reference object.
        '''                
        self.spectral_data = spectral_data


    def get_PDF(self, s):
        '''Returns cycle PDF(Probability Density Function) as a function of stress s.

        :param s:  numpy.ndarray
            Stress vector.
        :return pdf: function pdf(s)
        '''
        m0 = self.spectral_data.moments[0]
        al2 = self.spectral_data.al2

        def pdf(s):
            px = np.sqrt(1.0 - al2**2)/np.sqrt(2.0 * np.pi * m0) * \
                np.exp( - (s**2) / (2.0 * m0 * (1.0 - al2**2))) +\
                al2*s/m0 * np.exp( - (s**2) / (2*m0)) * \
                ss.norm.cdf((al2 * s) / (np.sqrt(m0 * (1 - al2**2))))
            return px
        return pdf(s)

    def get_life(self, C, k):
        """Calculate fatigue life with parameters C, k, as defined in [2].

        :param C: [int,float]
            Fatigue strength coefficient [MPa**k].
        :param k : [int,float]
            Fatigue strength exponent [/].
        :return T: float
            Estimated fatigue life in seconds.
        """ 
        d = self.spectral_data.m_p / C * \
            quad(lambda s: s**k*self.get_PDF(s), 
                    a=0, b=np.Inf)[0]

        T = 1.0/d

        return T