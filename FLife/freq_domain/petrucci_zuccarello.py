import numpy as np

class PetrucciZuccarello(object):
    """Class for fatigue life estimation using frequency domain 
    method by Petrucci and Zuccarello [1].

    References
    ----------
    [1] G. Petrucci and B. Zuccarello. Fatigue life prediction under wide band
        random loading. Fatigue & Fract. Eng. Mater. & Struct., 27(12):1183-1195, 
        December 2004.
    [2] Janko Slavič, Matjaž Mršnik, Martin Česnik, Jaka Javh, Miha Boltežar. 
        Vibration Fatigue by Spectral Methods, From Structural Dynamics to Fatigue Damage
        – Theory and Experiments, ISBN: 9780128221907, Elsevier, 1st September 2020
    """
    def __init__(self, spectral_data):
        '''Get needed values from reference object.

        :param spectral_data:  Instance of object SpectralData
        '''          
        self.spectral_data = spectral_data
    
    def get_life(self, C, k, Su = 1110):
        """Calculate fatigue life with parameters C, k, as defined in [2].

        :param C: [int,float]
            Fatigue strength coefficient [MPa**k].
        :param k : [int,float]
            Fatigue strength exponent [/].
        :param Su: [int,float]
            Tensile strength [MPa].    
        :return Tf: float
            Estimated fatigue life in seconds.
        """
        m0,m1,m2,_,m4 = self.spectral_data.moments
        m_p = self.spectral_data.m_p
        
        a1, a2 = (  np.sqrt((m1**2)/(m0*m2)) ,  m2 / np.sqrt(m0*m4) )
        R = 3 * np.sqrt(m0) / Su 

        alpha_m = np.array([
                        [-1, -a2, a1, a1*a2, -a2**2, -a1**2],
                        [ 1, -a2, a1, a1*a2, a2**2, -a1**2],
                        [-1, -a2, a1, a1*a2, a2**2, -a1**2],
                        [ 1, -a2, a1, a1*a2, a2**2, -a1**2]
                        ])
        
        psi_m = np.array([
                        [1.994,  9.381, 18.349, 15.261, 1.483, 15.402],
                        [8.229, 26.510, 21.522 ,27.748, 4.338, 20.026],
                        [0.946,  8.025, 15.692, 11.867, 0.382, 13.198],
                        [8.780, 26.058, 21.628, 26.487, 5.379, 19.967]
                        ])
        
        p1 = np.dot(alpha_m[0], psi_m[0])
        p2 = np.dot(alpha_m[1], psi_m[1])
        p3 = np.dot(alpha_m[2], psi_m[2])
        p4 = np.dot(alpha_m[3], psi_m[3])
        
        psi = ( (p2-p1)/6.0 ) * (k-3.0) + p1 + ( (2.0/9.0)*(p4-p3-p2+p1)*(k-3) + \
                                                (4.0/3.0)*(p3-p1) ) * (R - 0.15)

        Tf = C / ( m_p * np.sqrt( m0**k ) * np.exp(psi) )

        return Tf