from numpy import exp,sqrt,pi,empty
from scipy.special import gamma

class ZhaoBaker1(object):
    """Class for fatigue life estimation using frequency domain 
    method by Zhao and Baker [1].
   
    References
    ----------
    [1] Wangwen Zhao and Michael J. Baker. On the probability density function
        of rainflow stress range for stationary Gaussian processes. International
        Journal of Fatigue, 14(2):121-135, 1992.
    [2] Janko Slavič, Matjaž Mršnik, Martin Česnik, Jaka Javh, Miha Boltežar. 
        Vibration Fatigue by Spectral Methods, From Structural Dynamics to Fatigue Damage
        – Theory and Experiments, ISBN: 9780128221907, Elsevier, 1st September 2020
    """
    def __init__(self, spectral_data):
        '''Get needed values from reference object.

        :param spectral_data:  Instance of object SpectralData
        '''     
        self.spectral_data = spectral_data
        self.simple = True #KAJ je to, PREVERI!
    
    def _calcCoeff(self):
        """Calculate coefficients for ZhaoBaker1 method.
        
        Returns
        -------
        c = [a, b, w]
            [0, 1, 2]
        """
        gama = self.spectral_data.al2
        
        c = empty(3)
        
        c[0] = 8.0 - 7.0 * gama #a
        if gama < 0.9:
            c[1] = 1.1 #b
        else:
            c[1] = 1.1 + 9.0 * (gama - 0.9) #b
        c[2] = ( 1.0 - gama ) / ( 1.0 - sqrt(2.0/pi) * gamma(1.0 + 1.0/c[1]) * c[0]**(-1.0/c[1]) ) #w
        
        return c

    def get_PDF(self, S):
        """Returns cycle PDF(Probability Density Function) as a function of stress S.
        """
        m0 = self.spectral_data.moments[0]
        a, b, w = self._calcCoeff()
        
        pdf = w * ((a*b) / (sqrt(m0))) * ((S/sqrt(m0)))**(b-1) * exp(-a * (S/sqrt(m0))**b)+ \
            (1-w) * (S/m0) * exp(-0.5 * (S/sqrt(m0))**2)
        
        return pdf
        

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
        m_p = self.spectral_data.m_p
        a, b, w = self._calcCoeff()
               
        d = (m_p/C) * m0**(0.5*k) * ( w * a**(-k/b) * gamma(1.0+k/b) + (1.0-w) * 2**(0.5*k) * gamma(1.0+0.5*k) )
        T = 1.0 / d
        
        return T