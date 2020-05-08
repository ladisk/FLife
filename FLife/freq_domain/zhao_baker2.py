from numpy import exp,sqrt,pi,empty
from scipy.special import gamma
from scipy.optimize import fsolve
import numpy as np

class ZhaoBaker2(object):
    """Class for fatigue life estimation using frequency domain 
    method by Zhao and Baker [1].
   
    TODO:
    ----
    Implement improved version!!
    Check normalizing!!

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
        self.simple = True #KAJ je to??
    
    def _calcCoeff(self):
        """Calculate coefficients for ZhaoBaker2 method.
        
        Returns
        -------
        c = [a, b, w]
            [0, 1, 2]
        """
        a2 = self.spectral_data.al2
        a075 = self.spectral_data.al075
        
        if a2 < 0.9:
            b = 1.1
        else:
            b = 1.1 + 9 * (a2 - 0.9)
        
        if a075 >= 0.5:
            ro = -0.4154 + 1.392 * a075
        else:
            ro = 0.28
        
        #print 'b, a2, ro : %.1f, %.2f, %.2f' % (b, a2, ro)
        
        def eq(d):
            return gamma(1.0+(3.0/b)) * (1.0-a2) * d**3.0 + \
                   3.0 * gamma(1.0+(1.0/b)) * (ro * a2 - 1.0) * d + \
                   3.0 * np.sqrt(np.pi/2.0) * a2 * (1.0 - ro)
        
#        l = 0.0
#        r = 1.0
#        
#        try:
#            root = brentq(eq, l, r)
#            print 'Root (round 1) : x = %.2f' % (root, )
#        except ValueError:
#            try:
#                if l < 0:
#                    root = brentq(eq, l+0.1, r)
#                elif r > 0:
#                    root = brentq(eq, l, r - 0.1 )
#                print 'Root (round 2) : x = %.2f' % (root, )
#
#            except ValueError:
#                try:
#                    if l < 0:
#                        root = brentq(eq, l-0.1, r)
#                    elif r > 0:
#                        root = brentq(eq, l, r + 0.1 )
#                    print 'Root (round 3) : x = %.2f' % (root, )
#
#                except ValueError:         
#                    print 'Could not find root :/'
#                    root = 8 - 7 * a2
#        print 'fsolve', fsolve(eq, 0)
        
        try:
            root = fsolve(eq, 0)
        except:
            root = fsolve(eq, np.random.rand()*5.0)
        #print 'Found root at %.3f' % (root, )
        
        a = root**(-b)
        w = ( 1.0 - a2 ) / ( 1.0 - sqrt(2.0/pi) * gamma(1.0 + 1.0/b) * a**(-1.0/b) )

        return [a, b, w]

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
            Estimated fatigue life in seconds.        """
        m0 = self.spectral_data.moments[0]
        m_p = self.spectral_data.m_p
        a, b, w = self._calcCoeff()
              
        d = (m_p/C) * m0**(0.5*k) * ( w * a**(-k/b) * gamma(1.0+k/b) + (1.0-w) * 2**(0.5*k) * gamma(1.0+0.5*k) )
        T = np.float(1.0 / d)
        
        return T