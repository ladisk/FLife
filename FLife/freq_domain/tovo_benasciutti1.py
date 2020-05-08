import numpy as np
import scipy.special as ss

class TovoBenasciutti1(object):
    """Class for fatigue life estimation using frequency domain 
    method by Tovo and Benasciutti[1, 2]. Parameter b is 
    defined by Tovo [1].
   
    References
    ----------
    [1] Roberto Tovo. Cycle distribution and fatigue damage under broadband
        random loading. International Journal of Fatigue, 24(11):1137{
        1147, 2002.
    [2] Denis Benasciutti and Roberto Tovo. Spectral methods for lifetime
        prediction under wide-band stationary random processes. International
        Journal of Fatigue, 27(8):867{877, 2005.
    [3] Janko Slavič, Matjaž Mršnik, Martin Česnik, Jaka Javh, Miha Boltežar. 
        Vibration Fatigue by Spectral Methods, From Structural Dynamics to Fatigue Damage
        – Theory and Experiments, ISBN: 9780128221907, Elsevier, 1st September 2020    """
    def __init__(self, spectral_data):
        '''Get needed values from reference object.

        :param spectral_data:  Instance of object SpectralData
        '''     
        self.spectral_data = spectral_data
    
    def get_life(self, C, k): 
        """Calculate fatigue life with parameters C, k, as defined in [3].

        :param C: [int,float]
            Fatigue strength coefficient [Mpa**k].
        :param k : [int,float]
            Fatigue strength exponent [/].
        :return T: float
            Estimated fatigue life in seconds.
        """ 
        m_p = self.spectral_data.m_p
        m0 = self.spectral_data.moments[0]
        al2, al1 = self.spectral_data.al2, self.spectral_data.al1
        
        b = min( (al1-al2) / (1.0 - al1), 1.0)
        D_nb_lcc = m_p * al2 * np.sqrt(2 * m0)**k * ss.gamma(1.0 + k/2.0) / C
        l = b + ( 1.0 - b ) * al2**(k-1.0)
        T = 1.0 / (D_nb_lcc * l)
        
        return T
        

    def get_PDF(self, s):
        '''Returns cycle PDF(Probability Density Function) as a function of stress s.'''
        al2 = self.spectral_data.al2
        al1 = self.spectral_data.al1
        sigma = self.spectral_data.moments[0]
        b = min( (al1-al2) / (1 - al1), 1)

        pdf = b * (al2 * (s / sigma) * np.exp( - s**2 / (2 * sigma))) + \
            (1 - b) * ((s / (sigma * al2**2)) * np.exp( - s**2 / (2 * al2 * sigma)))

        return pdf