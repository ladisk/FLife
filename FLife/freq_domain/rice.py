import numpy as np
import scipy.integrate as si
import scipy.stats as ss

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
        :return pdf: numpy.ndarray
        '''
        m0 = self.spectral_data.moments[0]
        al2 = self.spectral_data.al2

        pdf = np.sqrt(1.0 - al2**2)/np.sqrt(2.0 * np.pi * m0) * \
                np.exp( - (s**2) / (2.0 * m0 * (1.0 - al2**2))) + \
                al2*s/m0 * np.exp( - (s**2) / (2*m0)) * \
                ss.norm.cdf((al2 * s) / (np.sqrt(m0 * (1 - al2**2))))

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
        m_p = self.spectral_data.m_p
        m0 = self.spectral_data.moments[0]
        al2 = self.spectral_data.al2
            
        I1 = lambda s: s**k * (np.sqrt(1.0 - al2**2)/np.sqrt(2.0 * np.pi * m0)) * \
                                                        np.exp( - (s**2) / (2.0 * m0 * (1.0 - al2**2)))
        I2 = lambda s: s**k * (al2*s/m0) * np.exp( - (s**2) / (2*m0)) * ss.norm.cdf((al2 * s) / (np.sqrt(m0 * (1 - al2**2))))

        D = m_p * ( si.quad(I1, 0, np.Inf)[0] + si.quad(I2, 0, np.Inf)[0] ) / C
        T = 1.0/D

        return T