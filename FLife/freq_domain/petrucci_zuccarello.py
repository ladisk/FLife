import numpy as np
import warnings

class PetrucciZuccarello(object):
    """Deprecated - Class for fatigue life estimation using frequency domain 
    method by Petrucci and Zuccarello [1].

    References
    ----------
    [1] G. Petrucci and B. Zuccarello. Fatigue life prediction under wide band
        random loading. Fatigue & Fract. Eng. Mater. & Struct., 27(12):1183-1195, 
        December 2004.
    [2] Janko Slavič, Matjaž Mršnik, Martin Česnik, Jaka Javh, Miha Boltežar. 
        Vibration Fatigue by Spectral Methods, From Structural Dynamics to Fatigue Damage
        – Theory and Experiments, ISBN: 9780128221907, Elsevier, 1st September 2020
    
    Example
    -------
    Import modules, define time- and frequency-domain data

    >>> import FLife
    >>> import pyExSi as es
    >>> import numpy as np
    >>> from matplotlib import pyplot as plt
    >>> # time-domain data
    >>> N = 2 ** 16  # number of data points of time signal
    >>> fs = 2048  # sampling frequency [Hz]
    >>> t = np.arange(0, N) / fs  # time vector
    >>> # frequency-domain data
    >>> M = N // 2 + 1  # number of data points of frequency vector
    >>> freq = np.arange(0, M, 1) * fs / N  # frequency vector
    >>> PSD_lower = es.get_psd(freq, 20, 60, variance = 5)  # lower mode of random process
    >>> PSD_higher = es.get_psd(freq, 100, 120, variance = 2)  # higher mode of random process
    >>> PSD = PSD_lower + PSD_higher # bimodal one-sided flat-shaped PSD

    Get Gaussian stationary signal, instantiate SpectralData object and plot PSD

    >>> rg = np.random.default_rng(123) # random generator seed
    >>> x = es.random_gaussian(N, PSD, fs, rg) # Gaussian stationary signal
    >>> sd = FLife.SpectralData(input=x, dt=1/fs) # SpectralData instance
    >>> plt.plot(sd.psd[:,0], sd.psd[:,1]) 
    >>> plt.xlabel('Frequency [Hz]')
    >>> plt.ylabel('PSD')

    Define S-N curve parameters and get fatigue-life estimatate

    >>> C = 1.8e+22  # S-N curve intercept [MPa**k]
    >>> k = 7.3 # S-N curve inverse slope [/]
    >>> pz = FLife.PetrucciZuccarello(sd)
    >>> print(f'Fatigue life: {pz.get_life(C,k):.3e} s.')
    """
    def __init__(self, spectral_data):
        """Get needed values from reference object.

        :param spectral_data:  Instance of class SpectralData
        """          
        self.spectral_data = spectral_data
        warnings.warn('PetrucciZuccarello method has been deprecated since version 1.2 and will be removed in the future due to poor accuracy.')
    
    def get_life(self, C, k, Su = 1110):
        """Calculate fatigue life with parameters C, k, as defined in [2].

        :param C: [int,float]
            S-N curve intercept [MPa**k].
        :param k: [int,float]
            S-N curve inverse slope [/].
        :param Su: [int,float]
            Tensile strength [MPa].    
        :return:
            Estimated fatigue life in seconds.
        :rtype: float
        """
        m0,m1,m2, _, m4 = self.spectral_data.moments
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