import numpy as np
from scipy.special import gamma
from scipy.optimize import fsolve

class ZhaoBaker(object):
    """Class for fatigue life estimation using frequency domain 
    method by Zhao and Baker[1, 2].
    
    Method coefficients are defined by two versions of the Zhao-Baker 
    method: 
    - base method is tuned in simulations with material parameters 
      in the range of 2 <= k <= 6, where k is S-N curve coefficient.
    - improved method is derived for S-N curve coefficient k = 3.  
   
    References
    ----------
    [1] Wangwen Zhao and Michael J. Baker. On the probability density function
        of rainflow stress range for stationary Gaussian processes. International
        Journal of Fatigue, 14(2):121-135, 1992
    [2] Janko Slavič, Matjaž Mršnik, Martin Česnik, Jaka Javh, Miha Boltežar. 
        Vibration Fatigue by Spectral Methods, From Structural Dynamics to Fatigue Damage
        – Theory and Experiments, ISBN: 9780128221907, Elsevier, 1st September 2020
    """
    def __init__(self, spectral_data):
        '''Get needed values from reference object.

        :param spectral_data:  Instance of object SpectralData
        '''     
        self.spectral_data = spectral_data

    def _calculate_coefficients(self, method='improved'):
        """Calculate coefficients for base Zhao-Baker method.
        
        :param method:  string
            'base'/'improved'. Selects base or improved Zhao-Baker method.
        :return [a, b, w]: list
            a and b are Weibull distribution coefficients. w is weight coefficient.
        """
        if method == 'base': 
            a, b, w = self._calculate_coefficients_base()
        elif method == 'improved': 
            a, b, w = self._calculate_coefficients_improved()
        else: raise Exception('Unrecognized Input Error')
        return a, b, w


    def _calculate_coefficients_base(self):
        """Calculate coefficients for base Zhao-Baker method.
        
        :return [a, b, w]: list
            a and b are Weibull distribution coefficients. w is weight coefficient.
        """
        al2 = self.spectral_data.al2
        
        a = 8.0 - 7.0 * al2 
        if al2 < 0.9:
            b = 1.1 
        else:
            b = 1.1 + 9.0 * (al2 - 0.9) 
        w = ( 1.0 - al2 ) / ( 1.0 - np.sqrt(2.0/np.pi) * gamma(1.0 + 1.0/b) * a**(-1.0/b) ) 
        
        return [a, b, w]

    def _calculate_coefficients_improved(self):
        """Calculate coefficients for improved Zhao-Baker method.
        
        :return [a, b, w]: list
            a and b are Weibull distribution coefficients. w is weight coefficient.
        """
        al2 = self.spectral_data.al2
        a075 = self.spectral_data.al075
        
        if al2 < 0.9:
            b = 1.1 
        else:
            b = 1.1 + 9 * (al2 - 0.9) 
        
        if a075 >= 0.5:
            ro = -0.4154 + 1.392 * a075  #damage correction factor
        else:
            ro = 0.28  #damage correction factor
                
        def eq(p):
            return gamma(1.0+(3.0/b)) * (1.0-al2) * p**3.0 + \
                   3.0 * gamma(1.0+(1.0/b)) * (ro * al2 - 1.0) * p + \
                   3.0 * np.sqrt(np.pi/2.0) * al2 * (1.0 - ro)
        
        try:
            root = fsolve(eq, 0)
        except:
            root = fsolve(eq, np.random.rand()*5.0)
        
        a = root**(-b) 
        w = ( 1.0 - al2 ) / ( 1.0 - np.sqrt(2.0/np.pi) * gamma(1.0 + 1.0/b) * a**(-1.0/b) )

        return [a, b, w]

    def get_PDF(self, s, method='improved'):
        """Returns cycle PDF(Probability Density Function) as a function of stress s.

        :param s:  numpy.ndarray
            Stress vector.
        :param method:  string
            'base'/'improved'. Selects base or improved Zhao-Baker method.
        :return pdf: numpy.ndarray
        """
        m0 = self.spectral_data.moments[0]

        a, b, w = self._calculate_coefficients(method=method)
        
        pdf = w * ((a*b) / (np.sqrt(m0))) * ((s/np.sqrt(m0)))**(b-1) * np.exp(-a * (s/np.sqrt(m0))**b) +\
            (1-w) * (s/m0) * np.exp(-0.5 * (s/np.sqrt(m0))**2)
        
        return pdf
        

    def get_life(self, C, k, method='improved'):
        """Calculate fatigue life with parameters C and k, as defined in [2].

        :param C: [int,float]
            Fatigue strength coefficient [MPa**k].
        :param k : [int,float]
            Fatigue strength exponent [/].
        :param method : str
            'base'/'improved'. Selects base or improved Zhao-Baker method.
        :return T: float
            Estimated fatigue life in seconds.
        """
        m0 = self.spectral_data.moments[0]
        m_p = self.spectral_data.m_p

        a, b, w = self._calculate_coefficients(method=method)
               
        d = (m_p/C) * m0**(0.5*k) * ( w * a**(-k/b) * gamma(1.0+k/b) +\
             (1.0-w) * 2**(0.5*k) * gamma(1.0+0.5*k) )
        T = np.float(1.0 / d)
        
        return T