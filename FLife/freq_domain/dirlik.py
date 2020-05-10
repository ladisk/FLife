import numpy as np
import scipy.special as ss

class Dirlik(object):
    """Class for fatigue life estimation using frequency domain method by Dirlik [1].
   
    References
    ----------
    [1] Turan Dirlik. Application of computers in fatigue analysis. PhD thesis,
        University of Warwick, 1985
    [2] Janko Slavič, Matjaž Mršnik, Martin Česnik, Jaka Javh, Miha Boltežar. 
        Vibration Fatigue by Spectral Methods, From Structural Dynamics to Fatigue Damage
        – Theory and Experiments, ISBN: 9780128221907, Elsevier, 1st September 2020
    """
    def __init__(self, spectral_data):
        '''Get needed values from reference object.

        :param spectral_data:  Instance of object SpectralData
        '''          
        self.spectral_data = spectral_data
        self._calculate_coefficients() 

    def _calculate_coefficients(self):
        '''Calculate coefficients for Dirlik method.
        '''
        m0,m1,m2,_,m4 = self.spectral_data.moments
        
        c = np.empty(8)
        c[0] = ( 1. / np.sqrt(m0) ) 
        c[1] = ( ( m1 / m0 ) * np.sqrt( m2 / m4 ) ) 
        c[2] = ( np.sqrt( 1. / ( m0 * m4 ) ) * m2 ) 
        c[3] = ( 2. * ( c[1] - c[2]**2 ) / ( 1. + c[2]**2 ) ) 
        c[4] = ( ( c[2] - c[1] - c[3]**2 ) / (1 - c[2] - c[3] + c[3]**2 ) ) 
        c[5] = ( ( 1. - c[2] - c[3] + c[3]**2 ) / ( 1. - c[4] ) ) 
        c[6] = ( ( 1. - c[3] - c[5] ) ) 
        c[7] = ( 1.25 * ( c[2] - c[6] - c[5] * c[4] ) / c[3] ) 
        
        self.coeff = c
            
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
        
        R = self.coeff[4]
        Q = self.coeff[7]
        G1 = self.coeff[3]
        G2 = self.coeff[5]
        G3 = self.coeff[6]
        
        T = C / ( self.spectral_data.m_p * np.sqrt(m0)**k * (  \
                                        G1 * (Q**k)*ss.gamma(1.0+k)+(np.sqrt(2.0)**k)*ss.gamma(1.+k/2.)*(G2 * abs(R)**k+G3) \
                                    ) )
    
        return T
        
    def get_PDF(self, s):
        '''Returns cycle PDF(Probability Density Function) as a function of stress s.
        '''
        m0 = self.spectral_data.moments[0]
        Z1 = self.coeff[0]
        R = self.coeff[4]
        Q = self.coeff[7]
        G1 = self.coeff[3]
        G2 = self.coeff[5]
        G3 = self.coeff[6]
        
        Z =  Z1*s
        pdf = (1.0/np.sqrt(m0)) * ( \
                            (G1/Q)*np.exp(-Z/Q) + \
                            ((G2*Z)/(R**2))*np.exp(-((Z)**2)/(2.*R**2)) + \
                            (G3*Z)*np.exp(-((Z)**2)/2.) \
                        )
        return pdf