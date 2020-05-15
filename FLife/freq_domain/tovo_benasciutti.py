import numpy as np
import scipy.special as ss
from .narrowband import Narrowband

class TovoBenasciutti(Narrowband):
    """Class for fatigue life estimation using frequency domain 
    method by Tovo and Benasciutti[1, 2].
      
    References
    ----------
    [1] Roberto Tovo. Cycle distribution and fatigue damage under broadband
        random loading. International Journal of Fatigue, 24(11):1137{
        1147, 2002
    [2] Denis Benasciutti and Roberto Tovo. Spectral methods for lifetime
        prediction under wide-band stationary random processes. International
        Journal of Fatigue, 27(8):867{877, 2005
    [3] Janko Slavič, Matjaž Mršnik, Martin Česnik, Jaka Javh, Miha Boltežar. 
        Vibration Fatigue by Spectral Methods, From Structural Dynamics to Fatigue Damage
        – Theory and Experiments, ISBN: 9780128221907, Elsevier, 1st September 2020
        """
    def __init__(self, spectral_data):
        '''Get needed values from reference object.

        :param spectral_data:  Instance of object SpectralData
        '''     
        Narrowband.__init__(self, spectral_data)

    def _calculate_coefficient(self, method='method 2'):
        """Calculate weigthing parameter b for the Tovo-Benasciutti method. Parameter b is 
            defined by Tovo and Benasciutti [1,2].
        
        :param method:  string
            - 'method 1': `b` weighting parameter `b` is defined by Tovo[1].
            - 'method 2': `b` weighting parameter `b` is defined by Tovo and Benasciutti [2].
                          (This is the improved method)
        :return b: float
        """
        if method == 'method 1': 
            b = self._calculate_coefficient_method_1()
        elif method == 'method 2': 
            b = self._calculate_coefficient_method_2()
        else: 
            raise Exception('Unrecognized Input Error')
        return b


    def _calculate_coefficient_method_1(self):
        """Calculate weigthing parameter b Tovo-Benasciutti method. Parameter b is 
            defined by Tovo[1].
        
        :return b: float
        """
        al1 = self.spectral_data.al1
        al2 = self.spectral_data.al2

        b = min( (al1-al2) / (1.0 - al1), 1.0)
        
        return b

    def _calculate_coefficient_method_2(self):
        """Calculate weigthing parameter b for improved Tovo-Benasciutti method. Parameter b is 
            defined by Tovo and Benasciutti [2].
        
        :return b: float
        """
        al1 = self.spectral_data.al1
        al2 = self.spectral_data.al2

        b = (al1-al2) * (  1.112 * ( 1+ al1*al2 - (al1+al2)  ) * np.exp(2.11*al2) +(al1-al2) ) / ((al2-1)**2)

        return b
        
    def _function_PDF(self, method='method 2', k=False):
        '''Defines cycle PDF(Probability Density Function) function or k-th 
        pdf moment function, if k is specified.
        '''
        al2 = self.spectral_data.al2
        m0 = self.spectral_data.moments[0]

        b = self._calculate_coefficient(method=method)
        
        if k==False: 
            def pdf(s):
                px = b * ((s / m0) * np.exp( - s**2 / (2 * m0))) + \
                    (1 - b) * ((s / (m0 * al2**2)) * np.exp( - s**2 / (2 * al2**2 * m0))) 
                return px
            return pdf
        else:
            if isinstance(k, (int,float)): 
                def pdf_moment(s):
                    px = b * ((s / m0) * np.exp( - s**2 / (2 * m0))) + \
                        (1 - b) * ((s / (m0 * al2**2)) * np.exp( - s**2 / (2 * al2**2 * m0)))
                    return s**k * px
                return pdf_moment
            else:
                raise Exception('Unrecognized Input Error')

    def get_PDF(self, s, method='method 2'):
        '''Returns cycle PDF(Probability Density Function) as a function of stress s.

        :param s:  numpy.ndarray
            Stress vector.
        :param method:  string
            - 'method 1': `b` weighting parameter `b` is defined by Tovo[1].
            - 'method 2': `b` weighting parameter `b` is defined by Tovo and Benasciutti [2].
                          (This is the improved method)
        :return pdf: numpy.ndarray
        '''
        return self._function_PDF(method=method)(s)

    def get_life(self, C, k, method='method 2'): 
        """Calculate fatigue life with parameters C, k, as defined in [3].

        :param C: [int,float]
            Fatigue strength coefficient [MPa**k].
        :param k : [int,float]
            Fatigue strength exponent [/].
        :param method:  string
            - 'method 1': `b` weighting parameter `b` is defined by Tovo[1].
            - 'method 2': `b` weighting parameter `b` is defined by Tovo and Benasciutti [2].
                          (This is the improved method)
        :return T: float
            Estimated fatigue life in seconds.
        """ 
        m0 = self.spectral_data.moments[0]
        nu = self.spectral_data.nu
        al2 = self.spectral_data.al2
        
        b = self._calculate_coefficient(method=method)

        dNB = self.damage_intesity_NB(m0=m0, nu=nu, C=C, k=k) 
        l = b + ( 1.0 - b ) * al2**(k-1.0)
        T = 1.0 / (dNB * l)
        
        return T